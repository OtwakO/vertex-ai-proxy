# app/api/vertex_stream_handlers.py
import asyncio
import json
import time  # Import time for polling timeout
from typing import AsyncGenerator, Optional  # Added Dict

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

import app.config.settings as settings
from app.models import ChatCompletionRequest
from app.utils import (
    handle_gemini_error,
    log,
    openAI_stream_chunk,
)
from app.vertex.vertex import OpenAIMessage, OpenAIRequest
from app.vertex.vertex import chat_completions as vertex_chat_completions_impl

# client_disconnect is needed for handling client aborts
from .client_disconnect import check_client_disconnect


# Vertex 缓存响应结构 (内部使用)
# 与 nonstream_handlers.py 中的定义保持一致
class VertexCachedResponse:
    def __init__(self, text, model, total_token_count=0):
        self.text = text
        self.model = model
        self.total_token_count = (
            total_token_count if total_token_count is not None else 0
        )
        self.prompt_token_count = 0
        self.candidates_token_count = 0


async def _execute_single_fake_vertex_call(
    chat_request: ChatCompletionRequest,
    vertex_request_payload: OpenAIRequest,
    call_index: int,  # 用于日志区分调用
    log_extra_base: dict,
) -> Optional[VertexCachedResponse]:
    """
    执行单个 Vertex AI 非流式 API 调用 (用于假流式)。
    返回 VertexCachedResponse 对象 (成功且有内容时) 或 None (失败/空响应/取消/异常时)。
    """
    log_extra = {
        **log_extra_base,
        "request_type": f"vertex-fake-stream-call-{call_index}",
    }

    try:
        log("debug", f"假流式调用 #{call_index}: 发起非流式 API 请求", extra=log_extra)
        # 确保调用非流式实现
        vertex_request_payload_copy = vertex_request_payload.model_copy(deep=True)
        vertex_request_payload_copy.stream = False
        vertex_response = await vertex_chat_completions_impl(
            vertex_request_payload_copy
        )
        log(
            "debug",
            f"假流式调用 #{call_index}: 收到响应类型: {type(vertex_response)}",
            extra=log_extra,
        )

        response_text = ""
        total_tokens = 0
        response_content = None

        # --- 解析 Vertex 响应 ---
        if isinstance(vertex_response, JSONResponse):
            try:
                response_content = json.loads(vertex_response.body.decode("utf-8"))
            except (json.JSONDecodeError, AttributeError) as parse_err:
                log(
                    "error",
                    f"假流式调用 #{call_index}: 解析 Vertex JSON 响应失败: {parse_err}",
                    extra=log_extra,
                )
                return None
        elif isinstance(vertex_response, Exception):
            log(
                "error",
                f"假流式调用 #{call_index}: Vertex API 调用返回或引发异常: {vertex_response}",
                extra=log_extra,
            )
            handle_gemini_error(vertex_response, "Vertex")
            return None
        elif isinstance(vertex_response, dict):
            response_content = vertex_response  # 直接使用字典响应
        else:
            log(
                "error",
                f"假流式调用 #{call_index}: Vertex API 调用返回未知类型: {type(vertex_response)}",
                extra=log_extra,
            )
            return None

        # --- 提取内容和 token ---
        if response_content and isinstance(response_content, dict):
            choices = response_content.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                response_text = message.get("content", "")
            usage = response_content.get("usage", {})
            total_tokens = usage.get("total_tokens") if usage else 0
            total_tokens = total_tokens if total_tokens is not None else 0

        if response_text:
            response_obj = VertexCachedResponse(
                text=response_text,
                model=chat_request.model,
                total_token_count=total_tokens,
            )
            log("debug", f"假流式调用 #{call_index}: 成功获取响应对象", extra=log_extra)
            return response_obj
        else:
            log(
                "warning",
                f"假流式调用 #{call_index}: Vertex API 调用成功但返回空响应",
                extra=log_extra,
            )
            return None  # 返回 None 表示空响应

    except asyncio.CancelledError:
        log("info", f"假流式调用 #{call_index}: 任务被取消", extra=log_extra)
        return None  # 返回 None 表示任务被取消
    except Exception as e:
        log(
            "error",
            f"假流式调用 #{call_index}: 执行时发生意外错误: {e}",
            exc_info=True,
            extra=log_extra,
        )
        handle_gemini_error(e, "Vertex")
        return None  # 返回 None 表示发生异常


async def _await_and_cache_background_fake_calls(
    shielded_tasks: list[asyncio.Task],
    cache_key: str,
    response_cache_manager,
    log_extra_base: dict,
    context_label: str = "后台缓存假流任务",
):
    """
    后台任务：等待剩余的假流式 API 调用任务完成，并将成功的完整结果存入缓存。
    这些任务是被 shield 的，确保即使主请求完成或取消，它们也能继续执行以填充缓存。
    """
    if not shielded_tasks:
        return

    log_extra = {**log_extra_base, "request_type": context_label}
    log(
        "info",
        f"{context_label}: 启动，等待 {len(shielded_tasks)} 个剩余并发 API 调用完成",
        extra=log_extra,
    )

    results = []
    gather_exception = None
    timeout_duration = settings.REQUEST_TIMEOUT  # 使用全局超时设置

    try:
        # 使用 gather 等待所有后台任务完成，return_exceptions=True 使得单个任务的异常不会中断 gather
        results = await asyncio.wait_for(
            asyncio.gather(*shielded_tasks, return_exceptions=True),
            timeout=timeout_duration,
        )
    except asyncio.TimeoutError:
        log(
            "error",
            f"{context_label}: gather 等待超时 ({timeout_duration}s)",
            extra=log_extra,
        )
        gather_exception = asyncio.TimeoutError(
            "Background fake stream caching gather timed out"
        )
        results = [
            task.result() if task.done() else asyncio.TimeoutError("Task timed out")
            for task in shielded_tasks
        ]
    except Exception as e:
        log(
            "error",
            f"{context_label}: gather 发生意外错误: {e}",
            extra=log_extra,
            exc_info=True,
        )
        gather_exception = e
        results = [task.result() if task.done() else e for task in shielded_tasks]

    success_count = 0
    error_count = 0
    empty_count = 0
    cancelled_count = 0
    timeout_count = 0
    first_success_cached = False  # 确保只记录一次首次缓存成功

    for result in results:
        if isinstance(result, VertexCachedResponse):
            # 成功返回了有效结果对象
            if result.text:  # 只缓存非空结果
                # 直接存储当前成功结果，ResponseCacheManager 会处理追加逻辑
                response_cache_manager.store(cache_key, result)
                log("info", f"{context_label}: 成功结果已缓存", extra=log_extra)
                success_count += 1
            else:
                empty_count += 1  # 成功但结果为空
        elif result is None:
            # 助手函数内部处理后返回 None (例如解析失败或API返回空)
            empty_count += 1
        elif isinstance(result, asyncio.CancelledError):
            cancelled_count += 1
        elif isinstance(result, asyncio.TimeoutError):
            timeout_count += 1
        elif isinstance(result, Exception):
            # 任务执行期间发生异常
            log("error", f"{context_label}: 任务结果为异常: {result}", extra=log_extra)
            error_count += 1
        else:
            log(
                "error",
                f"{context_label}: 任务返回未知类型结果: {type(result)}",
                extra=log_extra,
            )
            error_count += 1  # 视为错误

    log_message = (
        f"{context_label}: 完成. 结果: "
        f"成功(尝试缓存首个)={success_count}, 空响应={empty_count}, 错误={error_count}, "
        f"取消={cancelled_count}, 超时/未完成={timeout_count}"
    )
    if gather_exception:
        log_message += f". Gather 异常: {type(gather_exception).__name__}"

    log("info", log_message, extra=log_extra)


async def process_vertex_stream_request(
    chat_request: ChatCompletionRequest,
    http_request: Request,
    response_cache_manager,
    active_requests_manager,
    cache_key: str,
    safety_settings: Optional[dict] = None,
    safety_settings_g2: Optional[dict] = None,
) -> StreamingResponse:
    """
    处理流式 Vertex 请求的主入口。
    根据 settings.FAKE_STREAMING 决定使用假流式（基于非流式并发和后台缓存）或真流式。
    """
    log_extra_base = {
        "cache_key": cache_key[:8],
        "model": chat_request.model,
    }

    # --- 流式响应生成器 ---
    async def stream_response_generator() -> AsyncGenerator[str, None]:
        if settings.FAKE_STREAMING:
            # --- 假流式处理逻辑 (带后台缓存) ---
            # [Fake streaming logic remains unchanged]
            log_extra = {**log_extra_base, "request_type": "vertex-fake-stream"}
            log("info", "Vertex 请求进入假流式处理模式 (带后台缓存)", extra=log_extra)

            # 1. 优先检查缓存 (如果命中且类型正确，获取并移除，然后以流式返回)
            cached_response, cache_hit = response_cache_manager.get_and_remove(
                cache_key
            )  # 使用 get_and_remove
            if cache_hit and isinstance(cached_response, VertexCachedResponse):
                log(
                    "info",
                    "Vertex 假流式请求命中缓存 (类型正确，已移除)",
                    extra=log_extra,
                )
                try:
                    yield openAI_stream_chunk(
                        model=cached_response.model,
                        content=cached_response.text,
                        finish_reason="stop",
                    )
                    return  # 成功产生响应并已移除缓存项
                except Exception as e_yield:
                    # 处理 yield 可能出现的错误 (例如连接断开)
                    log(
                        "error",
                        f"缓存命中后 yield 块时出错: {e_yield}",
                        extra=log_extra,
                    )
                    # 缓存项已被 get_and_remove 移除
                    raise  # 重新抛出异常
            elif cache_hit:
                # 缓存命中但类型不正确 (已被 get_and_remove 移除)
                log(
                    "warning",
                    f"缓存命中但类型不符 (期望 VertexCachedResponse, 得到 {type(cached_response)})，已移除错误项，视为缓存未命中",
                    extra=log_extra,
                )
                # 无需额外移除，继续执行后续逻辑
            # else: 缓存未命中，继续执行后续逻辑

            # 2. 检查是否有相同请求正在处理 (Future check, 使用锁保证原子性)
            orchestrator_pool_key = f"vertex_orchestrator:{cache_key}"
            main_orchestration_future: Optional[asyncio.Future] = None
            is_new_future = False  # 标记是否是当前请求创建了 Future
            existing_future = active_requests_manager.get(
                orchestrator_pool_key
            )  # Check before lock for early exit
            if (
                isinstance(existing_future, asyncio.Future)
                and not existing_future.done()
            ):
                log(
                    "info",
                    "发现相同请求进行中，等待其 Future (用于获取首个结果)",
                    extra=log_extra,
                )
                keep_alive_task = None
                try:
                    yield openAI_stream_chunk(
                        model=chat_request.model, content=""
                    )  # 初始保活

                    async def send_keep_alive_while_waiting():
                        while not existing_future.done():
                            await asyncio.sleep(settings.FAKE_STREAMING_INTERVAL)
                            if not existing_future.done():
                                try:
                                    yield openAI_stream_chunk(
                                        model=chat_request.model, content=""
                                    )
                                except GeneratorExit:
                                    break
                                except Exception as e_inner:
                                    log(
                                        "error",
                                        f"等待 Future 保活任务出错: {e_inner}",
                                        extra=log_extra,
                                    )
                                    break

                    keep_alive_task = asyncio.create_task(
                        send_keep_alive_while_waiting()
                    )

                    first_result_from_existing = await asyncio.wait_for(
                        existing_future, timeout=settings.REQUEST_TIMEOUT
                    )

                    if keep_alive_task and not keep_alive_task.done():
                        keep_alive_task.cancel()

                    if isinstance(first_result_from_existing, VertexCachedResponse):
                        log("info", "使用来自现有任务 Future 的结果", extra=log_extra)
                        yield openAI_stream_chunk(
                            model=first_result_from_existing.model,
                            content=first_result_from_existing.text,
                            finish_reason="stop",
                        )
                        return
                    else:
                        log(
                            "error",
                            f"现有任务 Future 返回意外类型: {type(first_result_from_existing)}",
                            extra=log_extra,
                        )
                        # 继续执行新任务

                except asyncio.TimeoutError:
                    log(
                        "warning", "等待现有任务 Future 超时", extra=log_extra
                    )  # 继续执行新任务
                except asyncio.CancelledError:
                    log("warning", "等待现有任务 Future 时被取消", extra=log_extra)
                    raise
                except Exception as e:
                    log(
                        "error", f"等待现有任务 Future 时发生错误: {e}", extra=log_extra
                    )  # 继续执行新任务
                finally:
                    if keep_alive_task and not keep_alive_task.done():
                        keep_alive_task.cancel()

            # 3. 创建新的请求处理器任务
            log(
                "info",
                "Vertex 假流式请求缓存未命中或等待失败，创建新任务组",
                extra=log_extra,
            )
            # --- Reverted Lock Logic ---
            main_orchestration_future = asyncio.Future()
            if not active_requests_manager.add(
                orchestrator_pool_key, main_orchestration_future
            ):
                log(
                    "warning",
                    f"尝试注册 Future 时发现键已存在: {orchestrator_pool_key}",
                    extra=log_extra,
                )
                # Attempt to use the existing future if add failed
                existing_future_retry = active_requests_manager.get(
                    orchestrator_pool_key
                )
                if isinstance(existing_future_retry, asyncio.Future):
                    main_orchestration_future = existing_future_retry
                else:
                    # If it's not a future or doesn't exist anymore, raise error
                    raise HTTPException(
                        status_code=500, detail="内部服务器错误：请求状态管理冲突"
                    )
            # --- End Reverted Lock Logic ---

            # --- 如果是新创建的 Future，则启动编排器 ---
            # [Block moved to after _orchestrator definition]

            # --- 内部请求编排器 ---
            async def _orchestrator():
                api_call_tasks_unshielded = []
                api_call_tasks_shielded = []
                disconnect_monitor_task = None
                all_tasks_to_wait = []
                winner_found = False
                background_cache_triggered = False
                winner_task_ref = None  # 引用获胜的核心任务

                try:
                    openai_messages = [
                        OpenAIMessage(role=m.role, content=m.content)
                        for m in chat_request.messages
                    ]
                    vertex_request_payload = OpenAIRequest(
                        model=chat_request.model,
                        messages=openai_messages,
                        stream=False,
                        temperature=chat_request.temperature,
                        max_tokens=chat_request.max_tokens,
                        top_p=chat_request.top_p,
                        top_k=chat_request.top_k,
                        stop=chat_request.stop,
                        presence_penalty=chat_request.presence_penalty,
                        frequency_penalty=chat_request.frequency_penalty,
                        seed=getattr(chat_request, "seed", None),
                        n=chat_request.n,
                    )

                    num_concurrent_requests = settings.CONCURRENT_REQUESTS
                    log(
                        "info",
                        f"假流式编排器: 发起 {num_concurrent_requests} 个并发 API 调用",
                        extra=log_extra,
                    )

                    for i in range(num_concurrent_requests):
                        core_api_task = asyncio.create_task(
                            _execute_single_fake_vertex_call(
                                chat_request=chat_request,
                                vertex_request_payload=vertex_request_payload,
                                call_index=i + 1,
                                log_extra_base=log_extra_base,
                            ),
                            name=f"VertexFakeStreamCore-{i + 1}-{cache_key[:8]}",
                        )
                        shielded_task = asyncio.shield(core_api_task)
                        api_call_tasks_unshielded.append(core_api_task)
                        api_call_tasks_shielded.append(shielded_task)
                        all_tasks_to_wait.append(core_api_task)

                    disconnect_monitor_task = asyncio.create_task(
                        check_client_disconnect(
                            http_request,
                            f"fake-stream-orchestrator-{cache_key[:8]}",
                            "fake-stream-orchestrator",
                            chat_request.model,
                        ),
                        name=f"VertexDisconnectMonitorFakeStream-{cache_key[:8]}",
                    )
                    all_tasks_to_wait.append(disconnect_monitor_task)

                    pending_tasks = all_tasks_to_wait[:]
                    first_successful_result: Optional[VertexCachedResponse] = None

                    while pending_tasks and not winner_found:
                        done, pending = await asyncio.wait(
                            pending_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        pending_tasks = list(pending)

                        for task in done:
                            if winner_found:
                                continue

                            if task is disconnect_monitor_task:
                                winner_found = True
                                log(
                                    "warning",
                                    "假流式编排器: 客户端断开连接",
                                    extra=log_extra,
                                )
                                if not main_orchestration_future.done():  # Reverted
                                    main_orchestration_future.set_exception(
                                        HTTPException(
                                            status_code=499, detail="客户端断开连接"
                                        )
                                    )
                                if (
                                    api_call_tasks_shielded
                                    and not background_cache_triggered
                                ):
                                    log(
                                        "info",
                                        "假流式编排器: 客户端断开，触发所有任务的后台缓存",
                                        extra=log_extra,
                                    )
                                    asyncio.create_task(
                                        _await_and_cache_background_fake_calls(
                                            shielded_tasks=api_call_tasks_shielded,
                                            cache_key=cache_key,
                                            response_cache_manager=response_cache_manager,
                                            log_extra_base=log_extra_base,
                                        )
                                    )
                                    background_cache_triggered = True
                                break  # 已处理首个事件，退出内层循环

                            elif task in api_call_tasks_unshielded:
                                try:
                                    result = task.result()
                                    if (
                                        isinstance(result, VertexCachedResponse)
                                        and result.text
                                    ):
                                        winner_found = True
                                        winner_task_ref = (
                                            task  # Store reference to winning task
                                        )
                                        first_successful_result = result
                                        log(
                                            "info",
                                            "假流式编排器: 收到首个成功响应",
                                            extra=log_extra,
                                        )

                                        # 设置主 Future 结果
                                        if not main_orchestration_future.done():
                                            main_orchestration_future.set_result(
                                                first_successful_result
                                            )

                                        # 取消其他非获胜的 unshielded API 任务和断开监控
                                        cancelled_count = 0
                                        for other_task in api_call_tasks_unshielded:
                                            if (
                                                other_task is not task
                                                and not other_task.done()
                                            ):
                                                other_task.cancel()
                                                cancelled_count += 1
                                        if (
                                            disconnect_monitor_task
                                            and not disconnect_monitor_task.done()
                                        ):
                                            disconnect_monitor_task.cancel()
                                            cancelled_count += 1
                                        log(
                                            "info",
                                            f"假流式编排器: 已取消 {cancelled_count} 个剩余前台任务",
                                            extra=log_extra,
                                        )

                                        # 触发后台缓存任务 (包含所有 shielded 任务)
                                        if (
                                            api_call_tasks_shielded
                                            and not background_cache_triggered
                                        ):
                                            log(
                                                "info",
                                                "假流式编排器: 触发所有任务的后台缓存",
                                                extra=log_extra,
                                            )
                                            asyncio.create_task(
                                                _await_and_cache_background_fake_calls(
                                                    shielded_tasks=api_call_tasks_shielded,
                                                    cache_key=cache_key,
                                                    response_cache_manager=response_cache_manager,
                                                    log_extra_base=log_extra_base,
                                                )
                                            )
                                            background_cache_triggered = True
                                        break  # 已处理首个事件，退出内层循环
                                    elif isinstance(result, VertexCachedResponse):
                                        log(
                                            "debug",
                                            "假流式编排器: API 调用成功但响应为空",
                                            extra=log_extra,
                                        )
                                    else:  # result is None or Exception
                                        log(
                                            "warning",
                                            f"假流式编排器: API 调用失败或被取消: {result}",
                                            extra=log_extra,
                                        )

                                except asyncio.CancelledError:
                                    log(
                                        "info",
                                        "假流式编排器: API 任务被取消",
                                        extra=log_extra,
                                    )
                                except Exception as task_exc:
                                    log(
                                        "error",
                                        f"假流式编排器: API 任务执行时发生错误: {task_exc}",
                                        exc_info=True,
                                        extra=log_extra,
                                    )

                except Exception as orch_exc:
                    log(
                        "error",
                        f"假流式编排器发生意外错误: {orch_exc}",
                        exc_info=True,
                        extra=log_extra,
                    )
                    if not main_orchestration_future.done():
                        main_orchestration_future.set_exception(orch_exc)
                finally:
                    # 确保即使编排器异常退出，后台缓存也能在可能的情况下启动
                    if api_call_tasks_shielded and not background_cache_triggered:
                        log(
                            "warning",
                            "假流式编排器: 异常退出，尝试触发后台缓存",
                            extra=log_extra,
                        )
                        asyncio.create_task(
                            _await_and_cache_background_fake_calls(
                                shielded_tasks=api_call_tasks_shielded,
                                cache_key=cache_key,
                                response_cache_manager=response_cache_manager,
                                log_extra_base=log_extra_base,
                            )
                        )
                    # 清理活跃请求池中的编排器任务本身 (如果它是由当前请求创建的)
                    # 注意：Future 本身不在此处移除，它由等待它的请求处理或超时逻辑处理
                    # if is_new_future: # This logic needs refinement based on how orchestrator task is awaited
                    #     active_requests_manager.remove(f"orchestrator_task:{cache_key}") # Example key

            # --- 启动编排器任务 (如果需要) ---
            orchestrator_task = None
            if is_new_future:  # Check if we created the future
                orchestrator_task = asyncio.create_task(_orchestrator())
                # Optionally register the orchestrator task itself if needed elsewhere
                # active_requests_manager.add(f"orchestrator_task:{cache_key}", orchestrator_task)

            # --- 等待 Future (由编排器设置) ---
            try:
                final_result = await asyncio.wait_for(
                    main_orchestration_future, timeout=settings.REQUEST_TIMEOUT + 5
                )  # Add buffer

                if isinstance(final_result, VertexCachedResponse):
                    yield openAI_stream_chunk(
                        model=final_result.model,
                        content=final_result.text,
                        finish_reason="stop",
                    )
                else:
                    # Should not happen if future is set correctly
                    yield openAI_stream_chunk(
                        model=chat_request.model,
                        content=f"错误：内部状态错误 (Future 结果类型: {type(final_result)})",
                        finish_reason="error",
                    )
            except asyncio.TimeoutError:
                log("error", "等待假流式编排器 Future 超时", extra=log_extra)
                yield openAI_stream_chunk(
                    model=chat_request.model,
                    content="错误：处理请求超时",
                    finish_reason="error",
                )
            except Exception as final_exc:
                log(
                    "error",
                    f"等待假流式编排器 Future 时发生错误: {final_exc}",
                    exc_info=True,
                    extra=log_extra,
                )
                # Check if it's the specific client disconnect exception
                status_code = getattr(final_exc, "status_code", 500)
                detail = getattr(
                    final_exc,
                    "detail",
                    f"处理请求时发生内部错误 ({type(final_exc).__name__})",
                )
                if status_code == 499:
                    log(
                        "warning", "假流式 Future 结果为客户端断开连接", extra=log_extra
                    )
                    # Don't yield error chunk if client disconnected
                else:
                    yield openAI_stream_chunk(
                        model=chat_request.model,
                        content=f"错误: {detail}",
                        finish_reason="error",
                    )
        else:
            # --- True Streaming Logic ---
            log_extra = {**log_extra_base, "request_type": "vertex-true-stream"}
            log("info", "Vertex 请求进入真流式处理模式", extra=log_extra)

            # 1. Cache Check (True Stream)
            cached_response, cache_hit = response_cache_manager.get_and_remove(
                cache_key
            )
            if cache_hit and isinstance(cached_response, VertexCachedResponse):
                log(
                    "info",
                    "Vertex 真流式请求命中缓存 (类型正确，已移除)",
                    extra=log_extra,
                )
                try:
                    yield openAI_stream_chunk(
                        model=cached_response.model,
                        content=cached_response.text,
                        finish_reason="stop",
                    )
                    return  # Cache hit, stream complete
                except Exception as e_yield:
                    log(
                        "error",
                        f"真流式缓存命中后 yield 块时出错: {e_yield}",
                        extra=log_extra,
                    )
                    raise
            elif cache_hit:
                log(
                    "warning",
                    f"真流式缓存命中但类型不符 (期望 VertexCachedResponse, 得到 {type(cached_response)})，已移除错误项，视为缓存未命中",
                    extra=log_extra,
                )

            # 2. Future Handling (True Stream)
            orchestrator_pool_key = f"vertex_orchestrator:{cache_key}"
            future_to_wait_for: Optional[asyncio.Future] = None
            we_created_the_future = False
            main_stream_future: Optional[asyncio.Future] = (
                None  # Define main_stream_future here
            )

            # Try to get existing future first
            existing_future = active_requests_manager.get(orchestrator_pool_key)
            if (
                isinstance(existing_future, asyncio.Future)
                and not existing_future.done()
            ):
                log(
                    "info", "发现相同真流式请求进行中，将等待其 Future", extra=log_extra
                )
                future_to_wait_for = existing_future
            else:
                # No existing future, or it's already done. Try to create and add ours.
                log(
                    "info",
                    "未发现相同真流式请求进行中，尝试创建新 Future",
                    extra=log_extra,
                )
                main_stream_future = asyncio.Future()  # Assign to main_stream_future
                if active_requests_manager.add(
                    orchestrator_pool_key, main_stream_future
                ):
                    log("info", "成功注册新的 Future", extra=log_extra)
                    we_created_the_future = True
                    # We don't wait for our own future here, we proceed to execute the task
                else:
                    # Race condition: another request added the future between our get and add.
                    log(
                        "warning",
                        f"真流式：尝试注册 Future 时发现键已存在 (race condition): {orchestrator_pool_key}",
                        extra=log_extra,
                    )
                    existing_future_retry = active_requests_manager.get(
                        orchestrator_pool_key
                    )
                    if (
                        isinstance(existing_future_retry, asyncio.Future)
                        and not existing_future_retry.done()
                    ):
                        log(
                            "info",
                            "真流式：使用刚刚由并发请求创建的 Future (race condition)",
                            extra=log_extra,
                        )
                        future_to_wait_for = existing_future_retry
                    else:
                        # This case is problematic - add failed, but get also failed or got a completed future.
                        log(
                            "error",
                            "真流式：注册 Future 失败且无法获取有效的现有 Future",
                            extra=log_extra,
                        )
                        raise HTTPException(
                            status_code=500, detail="内部服务器错误：请求状态管理冲突"
                        )

            # If we determined there's a future to wait for (either initially found or from race condition)
            if future_to_wait_for:
                # --- Polling Waiting Logic ---
                log("info", "开始轮询等待现有 Future (带保活)", extra=log_extra)
                start_time = time.monotonic()  # Use monotonic clock for timeout
                polling_interval = 0.1  # Seconds between checks
                try:
                    yield openAI_stream_chunk(
                        model=chat_request.model, content=""
                    )  # Initial keep-alive

                    while True:
                        if future_to_wait_for.done():
                            log("info", "轮询：检测到 Future 已完成", extra=log_extra)
                            break  # Exit loop, future is done

                        # Check for overall timeout
                        if time.monotonic() - start_time > settings.REQUEST_TIMEOUT:
                            log("warning", "轮询等待现有 Future 超时", extra=log_extra)
                            raise asyncio.TimeoutError(
                                "Polling timeout waiting for existing future"
                            )

                        # Yield keep-alive and sleep
                        try:
                            yield openAI_stream_chunk(
                                model=chat_request.model, content=""
                            )
                        except GeneratorExit:
                            log(
                                "warning",
                                "轮询等待时，keep-alive yield 发生 GeneratorExit",
                                extra=log_extra,
                            )
                            # Don't try to cancel future here, let original request handle it
                            raise  # Re-raise to stop

                        await asyncio.sleep(polling_interval)

                    # --- Future is Done ---
                    log("debug", "轮询结束，处理 Future 结果", extra=log_extra)
                    future_exception = future_to_wait_for.exception()
                    if future_exception is None:
                        # Future completed successfully, try getting result from cache
                        log(
                            "info",
                            "轮询：Future 成功完成，尝试从缓存获取结果",
                            extra=log_extra,
                        )
                        # Use get() instead of get_and_remove() here, let original request manage removal? Or remove here too?
                        # Let's try removing here for consistency, assuming the waiting request "consumes" the result.
                        cached_result, cache_hit_after_wait = (
                            response_cache_manager.get_and_remove(cache_key)
                        )
                        if cache_hit_after_wait and isinstance(
                            cached_result, VertexCachedResponse
                        ):
                            log("info", "轮询：成功从缓存获取结果", extra=log_extra)
                            yield openAI_stream_chunk(
                                model=cached_result.model,
                                content=cached_result.text,
                                finish_reason="stop",
                            )
                        elif cache_hit_after_wait:
                            log(
                                "warning",
                                f"轮询：缓存命中但类型错误 ({type(cached_result)})，视为失败",
                                extra=log_extra,
                            )
                            yield openAI_stream_chunk(
                                model=chat_request.model,
                                content="错误：并发请求状态异常 (缓存类型错误)",
                                finish_reason="error",
                            )
                        else:
                            # This might happen if the original request failed after setting future but before caching?
                            log(
                                "error",
                                "轮询：Future 成功但缓存未命中!",
                                extra=log_extra,
                            )
                            yield openAI_stream_chunk(
                                model=chat_request.model,
                                content="错误：并发请求状态异常 (缓存未命中)",
                                finish_reason="error",
                            )
                    else:
                        # Future completed with an exception
                        log(
                            "error",
                            f"轮询：Future 完成但带有异常: {type(future_exception).__name__}",
                            extra=log_extra,
                        )
                        raise future_exception  # Re-raise the exception

                    return  # Exit generator, future handled

                except asyncio.TimeoutError:
                    # Handle polling timeout
                    log(
                        "warning",
                        "轮询等待现有真流式任务 Future 超时 (捕获)",
                        extra=log_extra,
                    )
                    yield openAI_stream_chunk(
                        model=chat_request.model,
                        content="错误：等待并发请求超时",
                        finish_reason="error",
                    )
                    return  # Exit generator on timeout
                except asyncio.CancelledError:
                    log(
                        "warning",
                        "轮询等待现有真流式任务 Future 时被取消",
                        extra=log_extra,
                    )
                    raise  # Re-raise cancellation to stop the stream properly
                except Exception as e:
                    # Handle exceptions raised by future.exception() or other issues
                    log(
                        "error",
                        f"轮询等待现有真流式任务 Future 时发生错误: {e}",
                        extra=log_extra,
                    )
                    yield openAI_stream_chunk(
                        model=chat_request.model,
                        content=f"错误：等待并发请求时出错 ({type(e).__name__})",
                        finish_reason="error",
                    )
                    return  # Exit generator on other errors
                # --- End Polling Waiting Logic ---

            # 3. Execute New True Streaming Task (only if we created the future)
            if not we_created_the_future:
                # If we ended up waiting for another future, we should have returned already.
                # If we are here without creating a future, something went wrong in the logic above.
                log(
                    "error",
                    "真流式：逻辑错误，未创建 Future 但也未等待现有 Future",
                    extra=log_extra,
                )
                raise HTTPException(
                    status_code=500, detail="内部服务器错误：流处理逻辑错误"
                )

            # Ensure main_stream_future is assigned if we created it
            if main_stream_future is None:
                log(
                    "error",
                    "真流式：逻辑错误，we_created_the_future is True 但 main_stream_future is None",
                    extra=log_extra,
                )
                raise HTTPException(
                    status_code=500, detail="内部服务器错误：流处理状态错误"
                )

            # ADDED LOG: Confirming execution path for the first request
            log("debug", "真流式：作为创建者，准备执行流式调用", extra=log_extra)

            log("info", "Vertex 真流式请求：执行新的流式 API 调用", extra=log_extra)
            # The future 'main_stream_future' is the one we added successfully.

            # Prepare API Payload (True Stream)
            openai_messages = [
                OpenAIMessage(role=m.role, content=m.content)
                for m in chat_request.messages
            ]
            vertex_request_payload = OpenAIRequest(
                model=chat_request.model,
                messages=openai_messages,
                stream=True,  # Ensure stream=True
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens,
                top_p=chat_request.top_p,
                top_k=chat_request.top_k,
                stop=chat_request.stop,
                presence_penalty=chat_request.presence_penalty,
                frequency_penalty=chat_request.frequency_penalty,
                seed=getattr(chat_request, "seed", None),
                n=chat_request.n,  # Note: True streaming might handle n differently or only support n=1 effectively
            )

            # 3c. Client Disconnect Monitor & API Call Execution
            disconnect_monitor_task = None
            api_stream_generator: Optional[AsyncGenerator] = None
            full_response_text = ""
            total_tokens_from_stream = (
                0  # Placeholder for potential future token counting from stream
            )
            stream_successful = False
            error_occurred = None

            # ADDED LOG: Before the main try block
            log("debug", "真流式：即将进入主 try/finally 块", extra=log_extra)
            try:
                disconnect_monitor_task = asyncio.create_task(
                    check_client_disconnect(
                        http_request,
                        f"true-stream-{cache_key[:8]}",
                        "true-stream",
                        chat_request.model,
                    ),
                    name=f"VertexDisconnectMonitorTrueStream-{cache_key[:8]}",
                )

                log(
                    "debug",
                    "真流式：即将调用 vertex_chat_completions_impl",
                    extra=log_extra,
                )  # Existing LOG
                api_stream_generator = await vertex_chat_completions_impl(
                    vertex_request_payload
                )
                log(
                    "debug",
                    f"真流式：vertex_chat_completions_impl 调用返回，类型: {type(api_stream_generator)}",
                    extra=log_extra,
                )  # Existing LOG

                # Check if the generator is valid before iterating
                if not isinstance(api_stream_generator, AsyncGenerator):
                    log(
                        "error",
                        f"Vertex API 调用未返回预期的 AsyncGenerator，而是 {type(api_stream_generator)}",
                        extra=log_extra,
                    )
                    # Handle cases where the underlying call might return an error response directly
                    if isinstance(
                        api_stream_generator, (JSONResponse, dict, Exception)
                    ):
                        # Attempt to extract error details if possible
                        error_detail = str(api_stream_generator)
                        if isinstance(api_stream_generator, JSONResponse):
                            try:
                                error_detail = api_stream_generator.body.decode()
                            except:
                                pass
                        elif isinstance(api_stream_generator, dict):
                            error_detail = json.dumps(api_stream_generator)

                        yield openAI_stream_chunk(
                            model=chat_request.model,
                            content=f"错误：API 调用失败 ({error_detail})",
                            finish_reason="error",
                        )
                        error_occurred = Exception(f"API call failed: {error_detail}")
                    else:
                        yield openAI_stream_chunk(
                            model=chat_request.model,
                            content="错误：API 调用返回意外类型",
                            finish_reason="error",
                        )
                        error_occurred = TypeError("API call returned unexpected type")
                    # Ensure future is handled correctly on early exit
                    # Check if main_stream_future exists before using it
                    if main_stream_future and not main_stream_future.done():
                        main_stream_future.set_exception(
                            error_occurred or Exception("Unknown API call failure")
                        )
                    active_requests_manager.remove(
                        orchestrator_pool_key
                    )  # Clean up future
                    return  # Stop processing

                # 3d. Iterate, Yield, Accumulate
                log(
                    "debug",
                    "真流式：准备进入 API stream generator 迭代",
                    extra=log_extra,
                )  # Existing LOG
                chunk_count = 0  # Existing LOG
                async for chunk_str in api_stream_generator:
                    chunk_count += 1  # Existing LOG
                    log(
                        "debug", f"真流式：收到块 #{chunk_count}", extra=log_extra
                    )  # Existing LOG
                    # Check for disconnect before yielding
                    if disconnect_monitor_task.done():
                        client_disconnected_result = (
                            disconnect_monitor_task.result()
                        )  # Check if it was a disconnect
                        if client_disconnected_result is True:
                            log(
                                "warning",
                                "真流式：客户端在流传输过程中断开连接",
                                extra=log_extra,
                            )
                            error_occurred = asyncio.CancelledError(
                                "Client disconnected during stream"
                            )
                            break  # Exit the loop

                    try:
                        yield chunk_str  # Yield the raw chunk from vertex.py
                        # Attempt to parse the chunk to accumulate text (assuming OpenAI SSE format)
                        try:
                            # Chunks are like "data: {...}\n\n"
                            if chunk_str.startswith("data: "):
                                json_part = chunk_str[len("data: ") :].strip()
                                if json_part and json_part != "[DONE]":
                                    chunk_data = json.loads(json_part)
                                    choices = chunk_data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content")
                                        if content:
                                            full_response_text += content
                        except json.JSONDecodeError:
                            log(
                                "warning",
                                f"真流式：无法解析流块以累积文本: {chunk_str[:100]}...",
                                extra=log_extra,
                            )
                        except Exception as parse_exc:
                            log(
                                "error",
                                f"真流式：解析或累积流块时出错: {parse_exc}",
                                extra=log_extra,
                            )

                    except GeneratorExit:
                        log(
                            "warning",
                            "真流式：生成器在 yield 时退出 (可能由客户端断开引起)",
                            extra=log_extra,
                        )
                        error_occurred = asyncio.CancelledError(
                            "GeneratorExit during yield"
                        )
                        break
                    except Exception as e_yield_stream:
                        log(
                            "error",
                            f"真流式：yield 流块时出错: {e_yield_stream}",
                            extra=log_extra,
                        )
                        error_occurred = e_yield_stream
                        break
                log(
                    "debug",
                    f"真流式：完成迭代 API stream generator (共 {chunk_count} 块)",
                    extra=log_extra,
                )  # Existing LOG

                # Check disconnect status one last time after loop
                if not error_occurred and disconnect_monitor_task.done():
                    client_disconnected_result = disconnect_monitor_task.result()
                    if client_disconnected_result is True:
                        log(
                            "warning",
                            "真流式：客户端在流结束后断开连接",
                            extra=log_extra,
                        )
                        error_occurred = asyncio.CancelledError(
                            "Client disconnected after stream"
                        )

                if not error_occurred:
                    stream_successful = True
                    log("info", "真流式：流传输成功完成", extra=log_extra)

            except asyncio.CancelledError as ce:
                log("warning", f"真流式：任务被取消: {ce}", extra=log_extra)
                error_occurred = ce
            except Exception as e_stream:
                log(
                    "error",
                    f"真流式：处理流时发生错误: {e_stream}",
                    exc_info=True,
                    extra=log_extra,
                )
                error_occurred = e_stream
                try:
                    # Attempt to yield a final error chunk if possible
                    yield openAI_stream_chunk(
                        model=chat_request.model,
                        content=f"错误：处理流时发生内部错误 ({type(e_stream).__name__})",
                        finish_reason="error",
                    )
                except:
                    pass  # Ignore errors during final error yield

            finally:
                log("debug", "真流式：进入 finally 块", extra=log_extra)  # Existing LOG
                # Cancel disconnect monitor if it's still running
                if disconnect_monitor_task and not disconnect_monitor_task.done():
                    log(
                        "debug", "真流式：取消 disconnect_monitor_task", extra=log_extra
                    )  # Existing LOG
                    disconnect_monitor_task.cancel()

                # 3e. Caching and Future Resolution on Success
                # Ensure main_stream_future exists before using it
                if main_stream_future:
                    log(
                        "debug",
                        f"真流式：处理 Future, stream_successful={stream_successful}, error_occurred={error_occurred}",
                        extra=log_extra,
                    )  # Existing LOG
                    if stream_successful and full_response_text:
                        log(
                            "info",
                            f"真流式：成功完成，缓存完整响应 ({len(full_response_text)} chars)",
                            extra=log_extra,
                        )
                        cached_response_obj = VertexCachedResponse(
                            text=full_response_text,
                            model=chat_request.model,
                            total_token_count=total_tokens_from_stream,  # Use accumulated tokens if available
                        )
                        response_cache_manager.store(cache_key, cached_response_obj)
                        log(
                            "debug",
                            f"真流式：尝试设置 Future 结果 (当前 done: {main_stream_future.done()})",
                            extra=log_extra,
                        )  # Existing LOG
                        if not main_stream_future.done():
                            try:
                                main_stream_future.set_result(cached_response_obj)
                                log(
                                    "info",
                                    "真流式：成功设置 Future 结果",
                                    extra=log_extra,
                                )  # Existing LOG
                            except asyncio.InvalidStateError:
                                log(
                                    "warning",
                                    "真流式：尝试设置 Future 结果时状态无效 (可能已被取消或已设置)",
                                    extra=log_extra,
                                )
                        else:
                            log(
                                "warning",
                                "真流式：Future 已完成，无法再次设置结果",
                                extra=log_extra,
                            )  # Existing LOG
                        # Do NOT remove from active_requests_manager here, let it expire or be overwritten
                    elif (
                        we_created_the_future and not main_stream_future.done()
                    ):  # Only modify future if we created it
                        # Handle cases where stream finished but produced no text, or an error occurred
                        if not error_occurred and not full_response_text:
                            log(
                                "warning",
                                "真流式：流成功完成但未产生任何文本",
                                extra=log_extra,
                            )
                            empty_response = VertexCachedResponse(
                                text="", model=chat_request.model
                            )
                            try:
                                main_stream_future.set_result(empty_response)
                            except asyncio.InvalidStateError:
                                log(
                                    "warning",
                                    "真流式：尝试设置空 Future 结果时状态无效",
                                    extra=log_extra,
                                )
                        elif error_occurred:
                            log(
                                "warning",
                                f"真流式：因错误未完成，设置 Future 异常: {type(error_occurred).__name__}",
                                extra=log_extra,
                            )
                            try:
                                main_stream_future.set_exception(error_occurred)
                            except asyncio.InvalidStateError:
                                log(
                                    "warning",
                                    "真流式：尝试设置 Future 异常时状态无效",
                                    extra=log_extra,
                                )
                            # Clean up future immediately on error if we created it
                            active_requests_manager.remove(orchestrator_pool_key)
                        else:
                            # Should not happen, but set a generic error if it does
                            unknown_error = Exception("Unknown stream completion state")
                            try:
                                main_stream_future.set_exception(unknown_error)
                            except asyncio.InvalidStateError:
                                log(
                                    "warning",
                                    "真流式：尝试设置未知 Future 异常时状态无效",
                                    extra=log_extra,
                                )
                            active_requests_manager.remove(
                                orchestrator_pool_key
                            )  # Clean up future
                    elif not we_created_the_future:
                        log(
                            "debug",
                            "真流式：非创建者，不修改 Future 状态",
                            extra=log_extra,
                        )  # Existing LOG
                else:
                    log(
                        "error",
                        "真流式：在 finally 块中 main_stream_future 未定义 (仅在我们创建时应发生)",
                        extra=log_extra,
                    )
                log("debug", "真流式：退出 finally 块", extra=log_extra)  # Existing LOG

            # No further yield needed here, generator finishes naturally or via error handling above

    return StreamingResponse(
        stream_response_generator(), media_type="text/event-stream"
    )
