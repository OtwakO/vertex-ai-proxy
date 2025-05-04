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
    # log, # Use vertex_log instead
    # vertex_log as log, # Incorrect path
    # openAI_stream_chunk, # 废弃
    # openAI_from_Gemini, # 移动到正确的导入路径
)
from app.utils.logging import vertex_log as log  # Correct import path
from app.utils.response import openAI_from_Gemini  # 从正确的模块导入
from app.vertex.vertex import OpenAIMessage, OpenAIRequest
from app.vertex.vertex import chat_completions as vertex_chat_completions_impl

# client_disconnect is needed for handling client aborts
from .client_disconnect import check_client_disconnect


# Vertex 缓存响应结构 (内部使用)
# 与 nonstream_handlers.py 中的定义保持一致
class VertexCachedResponse:
    # 与 nonstream_handlers.py 中的定义保持一致
    def __init__(
        self,
        text,
        model,
        prompt_token_count=0,
        candidates_token_count=0,
        total_token_count=0,
        finish_reason="stop",
        function_call=None,
    ):
        self.text = text  # 响应文本内容
        self.model = model  # 模型名称
        # 确保 token 计数是整数，处理 None 的情况
        self.prompt_token_count = (
            int(prompt_token_count) if prompt_token_count is not None else 0
        )
        self.candidates_token_count = (
            int(candidates_token_count) if candidates_token_count is not None else 0
        )
        self.total_token_count = (
            int(total_token_count) if total_token_count is not None else 0
        )
        # finish_reason 对于流式处理很重要
        self.finish_reason = finish_reason
        # function_call 支持
        self.function_call = function_call


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
        prompt_tokens = 0
        candidates_tokens = 0
        total_tokens = 0
        response_content = None
        function_call_data = None  # 支持函数调用

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
            handle_gemini_error(vertex_response, "Vertex")  # 使用通用错误处理
            return None
        elif isinstance(vertex_response, dict):
            response_content = vertex_response  # 直接使用字典
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
                # TODO: 解析函数调用 (如果 Vertex 支持)
                # function_call_data = message.get("tool_calls") or message.get("function_call")
            usage = response_content.get("usage", {})  # 获取 usage 字段
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                candidates_tokens = usage.get(
                    "completion_tokens"
                )  # Vertex 使用 completion_tokens
                total_tokens = usage.get("total_tokens")
            # 确保 token 计数是整数或 0
            prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
            candidates_tokens = (
                int(candidates_tokens) if candidates_tokens is not None else 0
            )
            total_tokens = int(total_tokens) if total_tokens is not None else 0

        # 根据是否有有效文本内容或函数调用返回成功或空状态
        if response_text or function_call_data:
            response_obj = VertexCachedResponse(
                text=response_text,  # 文本内容
                model=chat_request.model,  # 模型名称
                prompt_token_count=prompt_tokens,  # prompt tokens
                candidates_token_count=candidates_tokens,  # completion tokens
                total_token_count=total_tokens,  # total tokens
                # finish_reason 默认为 stop，由 VertexCachedResponse 设置
                function_call=function_call_data,  # 函数调用数据
            )
            log(
                "debug",
                f"假流式调用 #{call_index}: 成功解析 Vertex 响应",
                extra=log_extra,
            )
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
        handle_gemini_error(e, "Vertex")  # 使用通用错误处理
        return None  # 返回 None 表示发生异常


async def _await_and_cache_background_fake_calls(
    shielded_tasks: list[asyncio.Task],
    cache_key: str,
    response_cache_manager,
    log_extra_base: dict,
    context_label: str = "后台缓存假流任务",
):
    """
    后台任务：等待剩余的被 shield 的假流式 API 调用任务完成，并将成功的完整结果存入缓存。
    """
    if not shielded_tasks:
        log(
            "debug",
            "后台缓存任务：没有需要等待的 shielded 任务",
            extra={"cache_key": cache_key[:8]},
        )
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
    timed_out_tasks = 0  # 记录 gather 超时但任务本身未完成的数量

    try:
        # 使用 gather 等待所有后台任务完成，return_exceptions=True 使得单个任务的异常不会中断 gather
        results = await asyncio.wait_for(
            asyncio.gather(*shielded_tasks, return_exceptions=True),
            timeout=timeout_duration,
        )
    except asyncio.TimeoutError:
        # 处理 gather 本身的超时
        log(
            "warning",
            f"{context_label}: gather 等待超时 ({timeout_duration}s)",
            extra=log_extra,
        )
        gather_exception = asyncio.TimeoutError("Background caching gather timed out")
        results = []
        # 超时后，检查哪些任务实际完成了，哪些没有
        for task in shielded_tasks:
            if task.done():
                try:
                    results.append(task.result())  # 获取已完成任务的结果或异常
                except Exception as e:
                    results.append(e)  # 记录任务本身的异常
            else:
                # 标记任务因 gather 超时而未完成
                timed_out_tasks += 1
                results.append(
                    asyncio.TimeoutError("Task did not complete within gather timeout")
                )
                task.cancel()  # 尝试取消仍在运行的任务
    except Exception as e:
        # 处理 gather 发生的其他异常
        log(
            "error",
            f"{context_label}: gather 发生意外错误: {e}",
            extra=log_extra,
            exc_info=True,
        )
        gather_exception = e
        results = []
        for task in shielded_tasks:
            if task.done():
                try:
                    results.append(task.result())
                except Exception as ex:
                    results.append(ex)
            else:
                # 标记任务因 gather 错误而未完成
                timed_out_tasks += 1
                results.append(
                    RuntimeError("Task did not complete due to gather error")
                )
                task.cancel()  # 尝试取消

    # 统计并记录后台任务的最终结果
    counts = {
        "success": 0,
        "empty": 0,
        "error": 0,
        "cancelled": 0,
        "timeout": timed_out_tasks,
    }
    log(
        "debug",
        f"{context_label}: gather 完成，开始处理 {len(results)} 个结果",
        extra=log_extra,
    )

    # 导入辅助函数 (避免循环导入)
    from .vertex_nonstream_handlers import _process_task_result

    for i, result in enumerate(results):
        task_num = i + 1
        task_context = f"{context_label} #{task_num}"
        # 使用辅助函数处理每个后台任务的结果
        status, _ = _process_task_result(
            result, cache_key, response_cache_manager, log_extra, task_context
        )
        if status in counts:
            # 'timeout' 状态已在 gather 超时处理中计数
            if status != "timeout":
                counts[status] += 1
        else:
            log(
                "error",
                f"{task_context}: 处理结果时遇到未知状态 '{status}'",
                extra=log_extra,
            )
            counts["error"] += 1  # 归为错误

    log_summary = (
        f"{context_label}: 完成. 统计: "
        f"成功缓存={counts['success']}, 空响应={counts['empty']}, 错误={counts['error']}, "
        f"取消={counts['cancelled']}, 超时/未完成={counts['timeout']}"
    )
    if gather_exception:
        log_summary += f". Gather 异常: {type(gather_exception).__name__}"

    log("info", log_summary, extra=log_extra)


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
                    # Use async for since openAI_from_Gemini will be an async generator for streams
                    async for chunk in openAI_from_Gemini(cached_response, stream=True):
                        yield chunk
                    return
                except GeneratorExit:
                    log("warning", "缓存命中后 yield 时客户端断开连接", extra=log_extra)
                    raise
                except Exception as e_yield:
                    log(
                        "error",
                        f"缓存命中后 yield 块时出错: {e_yield}",
                        extra=log_extra,
                    )
                    raise
            elif cache_hit:
                log(
                    "warning",
                    f"缓存命中但类型不符 (期望 VertexCachedResponse, 得到 {type(cached_response)})，已移除错误项，视为缓存未命中",
                    extra=log_extra,
                )

            # 2. 检查是否有相同请求正在处理 (Future check)
            # 使用 'fakestream:' 前缀区分 Future
            orchestrator_pool_key = f"fakestream:vertex_orchestrator:{cache_key}"
            main_orchestration_future: Optional[asyncio.Future] = None
            existing_future = active_requests_manager.get(orchestrator_pool_key)

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
                    yield openAI_from_Gemini(
                        model=chat_request.model, content="", stream=True
                    )  # 初始保活

                    async def send_keep_alive_while_waiting():
                        while not existing_future.done():
                            await asyncio.sleep(settings.FAKE_STREAMING_INTERVAL)
                            if not existing_future.done():
                                try:
                                    yield openAI_from_Gemini(
                                        model=chat_request.model,
                                        content="",
                                        stream=True,
                                    )
                                except GeneratorExit:
                                    log(
                                        "info",
                                        "等待 Future 保活任务: 客户端断开",
                                        extra=log_extra,
                                    )
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
                        try:
                            # Use async for since openAI_from_Gemini will be an async generator for streams
                            async for chunk in openAI_from_Gemini(
                                first_result_from_existing, stream=True
                            ):
                                yield chunk
                            return
                        except GeneratorExit:
                            log(
                                "warning",
                                "等待 Future 成功后 yield 时客户端断开连接",
                                extra=log_extra,
                            )
                            raise
                        except Exception as e_yield_future:
                            log(
                                "error",
                                f"等待 Future 成功后 yield 时出错: {e_yield_future}",
                                extra=log_extra,
                            )
                            raise
                    else:
                        log(
                            "error",
                            f"现有任务 Future 返回意外类型: {type(first_result_from_existing)}",
                            extra=log_extra,
                        )

                except asyncio.TimeoutError:
                    log("warning", "等待现有任务 Future 超时", extra=log_extra)
                except asyncio.CancelledError:
                    log("warning", "等待现有任务 Future 时被取消", extra=log_extra)
                    raise
                except Exception as e:
                    log(
                        "error", f"等待现有任务 Future 时发生错误: {e}", extra=log_extra
                    )
                finally:
                    if keep_alive_task and not keep_alive_task.done():
                        keep_alive_task.cancel()

            # 3. 创建新的请求处理器任务
            log(
                "info",
                "Vertex 假流式请求缓存未命中或等待失败，创建新任务组",
                extra=log_extra,
            )
            main_orchestration_future = asyncio.Future()
            we_created_the_future = False
            if active_requests_manager.add(
                orchestrator_pool_key, main_orchestration_future
            ):
                we_created_the_future = True
                log("info", "成功注册新的 Vertex Future (假流式)", extra=log_extra)
            else:
                log(
                    "warning",
                    f"尝试注册 Vertex Future 时发现键已存在 (竞态条件): {orchestrator_pool_key}",
                    extra=log_extra,
                )
                existing_future_retry = active_requests_manager.get(
                    orchestrator_pool_key
                )
                if isinstance(existing_future_retry, asyncio.Future):
                    main_orchestration_future = existing_future_retry
                    log(
                        "info",
                        "使用由并发请求创建的 Vertex Future (竞态条件)",
                        extra=log_extra,
                    )
                else:
                    log(
                        "error",
                        "注册 Vertex Future 失败且无法获取有效的现有 Future",
                        extra=log_extra,
                    )
                    raise HTTPException(
                        status_code=500, detail="服务器内部状态管理冲突"
                    )

            # 只有成功注册 Future 的请求才执行编排器逻辑
            if we_created_the_future:

                async def _orchestrator():
                    api_call_tasks_unshielded = []
                    api_call_tasks_shielded = []
                    disconnect_monitor_task = None
                    all_tasks_to_wait = []
                    winner_found = False
                    background_cache_triggered = False
                    winner_task_ref = None

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
                                    if not main_orchestration_future.done():
                                        main_orchestration_future.set_exception(
                                            HTTPException(
                                                status_code=499, detail="客户端断开连接"
                                            )
                                        )
                                    cancelled_count = 0
                                    for utask in api_call_tasks_unshielded:
                                        if not utask.done():
                                            utask.cancel()
                                            cancelled_count += 1
                                    log(
                                        "info",
                                        f"假流式编排器: 取消了 {cancelled_count} 个未完成的 API 调用",
                                        extra=log_extra,
                                    )
                                    if api_call_tasks_shielded:
                                        background_cache_triggered = True
                                        asyncio.create_task(
                                            _await_and_cache_background_fake_calls(
                                                shielded_tasks=api_call_tasks_shielded,
                                                cache_key=cache_key,
                                                response_cache_manager=response_cache_manager,
                                                log_extra_base=log_extra_base,
                                            )
                                        )
                                    break

                                elif task in api_call_tasks_unshielded:
                                    try:
                                        result = task.result()
                                        if (
                                            isinstance(result, VertexCachedResponse)
                                            and result.text
                                        ):
                                            winner_found = True
                                            first_successful_result = result
                                            winner_task_ref = task
                                            log(
                                                "info",
                                                "假流式编排器: 收到首个成功响应",
                                                extra=log_extra,
                                            )
                                            if not main_orchestration_future.done():
                                                main_orchestration_future.set_result(
                                                    first_successful_result
                                                )
                                            if (
                                                disconnect_monitor_task
                                                and not disconnect_monitor_task.done()
                                            ):
                                                disconnect_monitor_task.cancel()
                                            cancelled_count = 0
                                            for utask in api_call_tasks_unshielded:
                                                if (
                                                    utask is not task
                                                    and not utask.done()
                                                ):
                                                    utask.cancel()
                                                    cancelled_count += 1
                                            log(
                                                "info",
                                                f"假流式编排器: 取消了 {cancelled_count} 个其他未完成的 API 调用",
                                                extra=log_extra,
                                            )
                                            remaining_shielded = [
                                                st
                                                for st in api_call_tasks_shielded
                                                if asyncio.ensure_future(st)
                                                is not asyncio.ensure_future(
                                                    winner_task_ref
                                                )
                                            ]
                                            if remaining_shielded:
                                                background_cache_triggered = True
                                                asyncio.create_task(
                                                    _await_and_cache_background_fake_calls(
                                                        shielded_tasks=remaining_shielded,
                                                        cache_key=cache_key,
                                                        response_cache_manager=response_cache_manager,
                                                        log_extra_base=log_extra_base,
                                                    )
                                                )
                                            break
                                        elif isinstance(result, VertexCachedResponse):
                                            log(
                                                "debug",
                                                "假流式编排器: 收到空响应，继续等待",
                                                extra=log_extra,
                                            )
                                        else:
                                            log(
                                                "warning",
                                                f"假流式编排器: API 调用失败或返回 None: {result}",
                                                extra=log_extra,
                                            )
                                    except asyncio.CancelledError:
                                        log(
                                            "debug",
                                            "假流式编排器: 一个 API 调用任务被取消",
                                            extra=log_extra,
                                        )
                                    except Exception as task_exc:
                                        log(
                                            "error",
                                            f"假流式编排器: API 调用任务异常: {task_exc}",
                                            extra=log_extra,
                                        )

                        if not winner_found:
                            log(
                                "warning",
                                "假流式编排器: 所有 API 调用均未成功且客户端未断开",
                                extra=log_extra,
                            )
                            if not main_orchestration_future.done():
                                main_orchestration_future.set_exception(
                                    HTTPException(
                                        status_code=500,
                                        detail="无法从 Vertex 获取有效响应",
                                    )
                                )
                            if (
                                not background_cache_triggered
                                and api_call_tasks_shielded
                            ):
                                asyncio.create_task(
                                    _await_and_cache_background_fake_calls(
                                        shielded_tasks=api_call_tasks_shielded,
                                        cache_key=cache_key,
                                        response_cache_manager=response_cache_manager,
                                        log_extra_base=log_extra_base,
                                    )
                                )

                    except Exception as orch_exc:
                        log(
                            "error",
                            f"假流式编排器发生意外错误: {orch_exc}",
                            exc_info=True,
                            extra=log_extra,
                        )
                        if not main_orchestration_future.done():
                            main_orchestration_future.set_exception(
                                HTTPException(
                                    status_code=500, detail=f"编排器错误: {orch_exc}"
                                )
                            )
                    finally:
                        if (
                            disconnect_monitor_task
                            and not disconnect_monitor_task.done()
                        ):
                            disconnect_monitor_task.cancel()
                        log("debug", "假流式编排器任务结束", extra=log_extra)

                # 启动编排器 (仅当是创建者时)
                orchestration_task = asyncio.create_task(_orchestrator())

            # --- 等待并产生首个结果 (所有请求都执行此部分) ---
            try:
                final_result = await asyncio.wait_for(
                    main_orchestration_future, timeout=settings.REQUEST_TIMEOUT + 5
                )  # 增加缓冲

                if isinstance(final_result, VertexCachedResponse):
                    log("info", "假流式: 成功从编排器 Future 获取结果", extra=log_extra)
                    try:
                        yield openAI_from_Gemini(final_result, stream=True)
                    except GeneratorExit:
                        log(
                            "warning",
                            "假流式 Future 成功后 yield 时客户端断开连接",
                            extra=log_extra,
                        )
                        raise
                    except Exception as e_yield_final:
                        log(
                            "error",
                            f"假流式 Future 成功后 yield 时出错: {e_yield_final}",
                            extra=log_extra,
                        )
                        raise
                elif isinstance(final_result, Exception):
                    # 如果 Future 结果是异常
                    log(
                        "error",
                        f"假流式编排器 Future 完成但带有异常: {final_result}",
                        extra=log_extra,
                    )
                    raise final_result  # 重新抛出异常
                else:
                    # Future 完成但结果类型未知
                    log(
                        "error",
                        f"假流式编排器 Future 返回意外类型: {type(final_result)}",
                        extra=log_extra,
                    )
                    yield openAI_from_Gemini(
                        model=chat_request.model,
                        content="错误：服务器内部状态错误",
                        finish_reason="error",
                        stream=True,
                    )

            except asyncio.TimeoutError:
                log("error", "等待假流式编排器 Future 超时", extra=log_extra)
                yield openAI_from_Gemini(
                    model=chat_request.model,
                    content="错误：处理请求超时",
                    finish_reason="error",
                    stream=True,
                )
            except HTTPException as http_exc:
                # 重新抛出由编排器设置的 HTTPException (如客户端断开)
                log(
                    "warning",
                    f"假流式编排器 Future 引发 HTTPException: {http_exc.status_code} {http_exc.detail}",
                    extra=log_extra,
                )
                # 对于 499 Client Closed Request，不应再 yield
                if http_exc.status_code != 499:
                    try:
                        yield openAI_from_Gemini(
                            model=chat_request.model,
                            content=f"错误: {http_exc.detail}",
                            finish_reason="error",
                            stream=True,
                        )
                    except Exception as e_yield_http_err:
                        log(
                            "error",
                            f"假流式发送 HTTPException 错误块时出错: {e_yield_http_err}",
                            extra=log_extra,
                        )
                raise http_exc  # 重新抛出原始异常
            except Exception as final_exc:
                log(
                    "error",
                    f"等待假流式编排器 Future 时发生错误: {final_exc}",
                    exc_info=True,
                    extra=log_extra,
                )
                yield openAI_from_Gemini(
                    model=chat_request.model,
                    content=f"错误：处理请求时发生内部错误 ({type(final_exc).__name__})",
                    finish_reason="error",
                    stream=True,
                )
            finally:
                # 无论如何，如果此请求创建了 Future，则在完成后移除 (或设置 TTL)
                # if we_created_the_future: active_requests_manager.remove(orchestrator_pool_key) # 暂时不移除
                pass
        else:
            # --- 真流式处理逻辑 ---
            log_extra = {**log_extra_base, "request_type": "vertex-true-stream"}
            log("info", "Vertex 请求进入真流式处理模式", extra=log_extra)

            # 1. 优先检查缓存 (如果命中且类型正确，获取并移除，然后以流式返回)
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
                    yield openAI_from_Gemini(cached_response, stream=True)
                    return
                except GeneratorExit:
                    log(
                        "warning",
                        "真流式缓存命中后 yield 时客户端断开连接",
                        extra=log_extra,
                    )
                    raise
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
                    f"缓存命中但类型不符 (期望 VertexCachedResponse, 得到 {type(cached_response)})，已移除错误项，视为缓存未命中",
                    extra=log_extra,
                )

            # 2. 检查是否有相同请求正在处理 (Future check) - 真流式简化处理，不等待 Future
            log(
                "info",
                "真流式: 缓存未命中，准备启动新的 Vertex 流式调用",
                extra=log_extra,
            )

            # --- 内部类：用于累积真流式响应信息 ---
            class TrueStreamAccumulator:
                def __init__(self, model_name):
                    self.model = model_name
                    self.text = ""
                    self.prompt_token_count = 0
                    self.candidates_token_count = 0
                    self.total_token_count = 0
                    self.finish_reason = None
                    self.function_call = None  # 暂存函数调用信息

                def update_from_chunk(self, chunk_data: dict):
                    """根据 Vertex 流块更新累积器状态"""
                    choices = chunk_data.get("choices", [])
                    if not choices:
                        return False  # 无效块

                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        self.text += content  # 累积文本

                    # 解析函数调用 (如果存在)
                    tool_calls = delta.get("tool_calls")
                    if tool_calls:
                        # 假设 Vertex 的 tool_calls 结构与 OpenAI 类似或需要转换
                        # TODO: 需要根据 Vertex 实际返回调整
                        if not self.function_call:
                            self.function_call = []
                        for tc in tool_calls:
                            self.function_call.append(tc.get("function"))

                    # 解析 finishReason (Vertex 使用)
                    chunk_finish_reason = choices[0].get("finishReason")
                    if chunk_finish_reason:
                        self.finish_reason = chunk_finish_reason

                    # 解析 token 计数 (如果 Vertex 流式提供)
                    usage_metadata = chunk_data.get("usageMetadata")
                    if usage_metadata:
                        self.prompt_token_count = usage_metadata.get(
                            "promptTokenCount", self.prompt_token_count
                        )
                        self.candidates_token_count = usage_metadata.get(
                            "candidatesTokenCount", self.candidates_token_count
                        )
                        self.total_token_count = usage_metadata.get(
                            "totalTokenCount", self.total_token_count
                        )

                    return True  # 表示块被处理

                def create_final_response_obj(self) -> VertexCachedResponse:
                    """创建最终的响应对象，用于缓存"""
                    openai_finish_reason = "stop"  # 默认值
                    if self.finish_reason == "STOP":
                        openai_finish_reason = "stop"
                    elif self.finish_reason == "MAX_TOKENS":
                        openai_finish_reason = "length"
                    elif self.finish_reason == "SAFETY":
                        openai_finish_reason = "content_filter"
                    # 其他原因映射...

                    return VertexCachedResponse(
                        text=self.text,
                        model=self.model,
                        prompt_token_count=self.prompt_token_count,
                        candidates_token_count=self.candidates_token_count,
                        total_token_count=self.total_token_count,
                        finish_reason=openai_finish_reason,
                        function_call=self.function_call,
                    )

            # --- 调用 Vertex 流式 API ---
            accumulator = TrueStreamAccumulator(chat_request.model)
            disconnect_monitor_task = None  # 初始化
            try:
                # 启动客户端断开连接监控
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
                    "info",
                    "真流式: 调用 Vertex chat_completions (stream=True)",
                    extra=log_extra,
                )
                # 准备请求体
                openai_messages = [
                    OpenAIMessage(role=m.role, content=m.content)
                    for m in chat_request.messages
                ]
                vertex_request_payload = OpenAIRequest(
                    model=chat_request.model,
                    messages=openai_messages,
                    stream=True,
                    temperature=chat_request.temperature,
                    max_tokens=chat_request.max_tokens,
                    top_p=chat_request.top_p,
                    top_k=chat_request.top_k,
                    stop=chat_request.stop,
                    # TODO: 传递 safety_settings / safety_settings_g2 (如果 Vertex 实现支持)
                )

                # 获取 Vertex API 的异步生成器
                vertex_stream = await vertex_chat_completions_impl(
                    vertex_request_payload
                )

                # 检查返回类型
                if not isinstance(vertex_stream, AsyncGenerator):
                    log(
                        "error",
                        f"Vertex API 调用未返回预期的 AsyncGenerator，而是 {type(vertex_stream)}",
                        extra=log_extra,
                    )
                    raise TypeError(
                        f"Vertex API call returned unexpected type: {type(vertex_stream)}"
                    )

                # 迭代处理 Vertex 返回的流块
                async for chunk_str in vertex_stream:
                    if disconnect_monitor_task.done():
                        if disconnect_monitor_task.result() is True:
                            log(
                                "warning",
                                "真流式：客户端在流传输过程中断开连接",
                                extra=log_extra,
                            )
                            raise asyncio.CancelledError(
                                "Client disconnected during stream"
                            )

                    if not chunk_str or not chunk_str.startswith("data: "):
                        continue

                    try:
                        chunk_data = json.loads(chunk_str[len("data: ") :])
                    except json.JSONDecodeError:
                        log(
                            "error",
                            f"真流式: 解析 JSON 块失败: {chunk_str[:100]}...",
                            extra=log_extra,
                        )
                        continue

                    if not accumulator.update_from_chunk(chunk_data):
                        continue

                    # --- 格式化并 Yield OpenAI 块 ---
                    choices = chunk_data.get("choices", [])
                    delta_to_yield = {}
                    finish_reason_to_yield = None
                    usage_to_yield = None

                    if choices:
                        delta_to_yield = choices[0].get("delta", {})
                        finish_reason_to_yield = choices[0].get("finishReason")
                        if finish_reason_to_yield == "STOP":
                            finish_reason_to_yield = "stop"
                        elif finish_reason_to_yield == "MAX_TOKENS":
                            finish_reason_to_yield = "length"
                        elif finish_reason_to_yield == "SAFETY":
                            finish_reason_to_yield = "content_filter"

                    if finish_reason_to_yield:
                        usage_to_yield = {
                            "prompt_tokens": accumulator.prompt_token_count,
                            "completion_tokens": accumulator.candidates_token_count,
                            "total_tokens": accumulator.total_token_count,
                        }

                    openai_chunk_dict = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": accumulator.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta_to_yield,
                                "finish_reason": finish_reason_to_yield,
                            }
                        ],
                    }
                    if usage_to_yield:
                        openai_chunk_dict["usage"] = usage_to_yield

                    openai_chunk_sse = (
                        f"data: {json.dumps(openai_chunk_dict, ensure_ascii=False)}\n\n"
                    )

                    try:
                        yield openai_chunk_sse
                    except GeneratorExit:
                        log(
                            "warning",
                            "真流式: yield 块时客户端断开连接 (GeneratorExit)",
                            extra=log_extra,
                        )
                        raise
                    except Exception as e_yield_stream:
                        log(
                            "error",
                            f"真流式: yield 块时出错: {e_yield_stream}",
                            extra=log_extra,
                        )
                        raise

                # --- 流结束处理 ---
                log("info", "真流式: Vertex 流处理完成", extra=log_extra)
                if accumulator.text or accumulator.function_call:
                    final_response_obj = accumulator.create_final_response_obj()
                    response_cache_manager.store(cache_key, final_response_obj)
                    log("info", "真流式: 完整响应已缓存", extra=log_extra)
                else:
                    log("info", "真流式: 无有效内容生成，不缓存", extra=log_extra)

            except asyncio.CancelledError as e:
                log("warning", f"真流式: 处理被取消: {e}", extra=log_extra)
                raise
            except HTTPException as e:
                log(
                    "error",
                    f"真流式: Vertex API 调用引发 HTTPException {e.status_code}: {e.detail}",
                    extra=log_extra,
                )
                try:
                    yield openAI_from_Gemini(
                        model=chat_request.model,
                        content=f"错误: {e.detail}",
                        finish_reason="error",
                        stream=True,
                    )
                except Exception as e_yield_err:
                    log(
                        "error",
                        f"真流式: 发送 HTTPException 错误块时出错: {e_yield_err}",
                        extra=log_extra,
                    )
            except Exception as e:
                log(
                    "error",
                    f"真流式: 处理 Vertex 流时发生错误: {e}",
                    extra=log_extra,
                    exc_info=True,
                )
                handle_gemini_error(e, "Vertex")
                error_message = f"处理 Vertex 流时发生内部错误: {type(e).__name__}"
                try:
                    yield openAI_from_Gemini(
                        model=chat_request.model,
                        content=error_message,
                        finish_reason="error",
                        stream=True,
                    )
                except Exception as e_yield_err:
                    log(
                        "error",
                        f"真流式: 发送通用 Exception 错误块时出错: {e_yield_err}",
                        extra=log_extra,
                    )
            finally:
                if disconnect_monitor_task and not disconnect_monitor_task.done():
                    disconnect_monitor_task.cancel()

        # --- 生成器结束 ---
        log("debug", "流式响应生成器结束", extra=log_extra_base)

    # 返回 StreamingResponse
    return StreamingResponse(
        stream_response_generator(), media_type="text/event-stream"
    )
