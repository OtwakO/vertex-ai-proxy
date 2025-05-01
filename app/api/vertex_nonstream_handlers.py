# app/api/vertex_nonstream_handlers.py
import asyncio
import json
from typing import Any, Literal, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

import app.config.settings as settings
from app.models import ChatCompletionRequest
from app.utils import (
    handle_gemini_error,
    openAI_nonstream_response,
)
from app.utils import vertex_log as log
from app.vertex.vertex import OpenAIMessage, OpenAIRequest
from app.vertex.vertex import chat_completions as vertex_chat_completions_impl

from .client_disconnect import check_client_disconnect


# Vertex 缓存响应结构 (内部使用)
class VertexCachedResponse:
    def __init__(self, text, model, total_token_count=0):
        self.text = text
        self.model = model
        self.total_token_count = (
            total_token_count if total_token_count is not None else 0
        )
        self.prompt_token_count = 0  # 兼容 openAI_nonstream_response
        self.candidates_token_count = 0  # 兼容 openAI_nonstream_response


async def _execute_single_vertex_call(
    chat_request: ChatCompletionRequest,
    vertex_request_payload: OpenAIRequest,
) -> Tuple[str, Optional[VertexCachedResponse], Optional[Exception]]:
    """
    执行单个 Vertex AI API 调用并处理其响应或异常。

    返回:
        元组 (状态字符串, 响应对象|None, 异常对象|None)。
        状态: "success", "empty", "error", "cancelled"。
    """
    log_extra = {
        "request_type": "api-call",
        "model": chat_request.model,
    }

    try:
        # 调用实际的 Vertex API 实现
        vertex_response = await vertex_chat_completions_impl(vertex_request_payload)

        response_text = ""
        total_tokens = 0
        response_content = None

        # 解析 Vertex 返回的 JSON 响应
        if isinstance(vertex_response, JSONResponse):
            try:
                response_content = json.loads(vertex_response.body.decode("utf-8"))
            except (json.JSONDecodeError, AttributeError) as parse_err:
                log("error", f"解析 Vertex JSON 响应失败: {parse_err}", extra=log_extra)
                return "error", None, parse_err
        elif isinstance(vertex_response, Exception):
            # 处理 API 调用本身返回的异常
            log(
                "error",
                f"Vertex API 调用返回或引发异常: {vertex_response}",
                extra=log_extra,
            )
            handle_gemini_error(vertex_response, "Vertex")
            return "error", None, vertex_response

        # 从解析后的内容中提取所需信息
        if response_content and isinstance(response_content, dict):
            choices = response_content.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                response_text = message.get("content", "")
            usage = response_content.get("usage", {})
            total_tokens = usage.get("total_tokens") if usage else 0
            total_tokens = total_tokens if total_tokens is not None else 0

        # 根据是否有有效文本内容返回成功或空状态
        if response_text:
            response_obj = VertexCachedResponse(
                text=response_text,
                model=chat_request.model,
                total_token_count=total_tokens,
            )
            return "success", response_obj, None
        else:
            log("warning", f"Vertex API 调用成功但返回空响应", extra=log_extra)
            return "empty", None, None

    except asyncio.CancelledError:
        # 处理任务被取消的情况
        log("warning", f"Vertex API 调用任务被取消", extra=log_extra)
        return "cancelled", None, asyncio.CancelledError()

    except Exception as e:
        # 处理执行过程中的其他意外错误
        log(
            "error",
            f"执行 Vertex API 调用时发生意外错误: {e}",
            exc_info=True,
            extra=log_extra,
        )
        handle_gemini_error(e, "Vertex")
        return "error", None, e


def _process_task_result(
    result: Any,
    cache_key: str,
    response_cache_manager,
    log_extra: dict,
    context: str = "后台缓存任务",
) -> Tuple[str, Optional[VertexCachedResponse]]:
    """
    辅助函数：处理 API 调用任务的结果，缓存成功项并记录日志。

    返回:
        元组 (状态字符串, 响应对象|None)。
        状态: "success", "empty", "error", "cancelled", "timeout"。
    """
    status = "error"
    response_obj = None

    # 区分处理任务的各种结束状态 (取消/超时/异常/正常返回)
    if isinstance(result, (asyncio.CancelledError, asyncio.TimeoutError)):
        status = (
            "cancelled" if isinstance(result, asyncio.CancelledError) else "timeout"
        )
        log(
            "warning",
            f"{context}：API 任务未成功完成: {type(result).__name__}",
            extra=log_extra,
        )
    elif isinstance(result, Exception):
        status = "error"
        log("error", f"{context}：API 任务结果为异常: {result}", extra=log_extra)
    elif isinstance(result, tuple) and len(result) == 3:
        # 处理来自 _execute_single_vertex_call 的正常返回元组
        status, response_obj, error = result
        if status == "success" and response_obj:
            # 仅在成功时缓存结果
            response_cache_manager.store(cache_key, response_obj)
        elif status == "empty":
            log("debug", f"{context}：API 任务返回空响应", extra=log_extra)
        elif status == "error":
            log("warning", f"{context}：API 任务返回错误状态: {error}", extra=log_extra)
        elif status == "cancelled":
            log("warning", f"{context}：API 任务内部返回 'cancelled'", extra=log_extra)
        else:
            status = "error"
            log(
                "warning",
                f"{context}：API 任务返回未知状态 '{status}'",
                extra=log_extra,
            )
    else:
        # 处理非预期的结果类型
        status = "error"
        log(
            "error", f"{context}：API 任务返回格式不符的结果: {result}", extra=log_extra
        )

    return status, response_obj if status == "success" else None


async def _await_and_cache_shielded(
    shielded_tasks: list[asyncio.Task],
    cache_key: str,
    response_cache_manager,
    model: str,
):
    """
    后台任务：等待剩余的 API 调用任务完成，并将成功的结果存入缓存。
    这些任务是被 shield 的，确保即使主请求完成或取消，它们也能继续执行以填充缓存。
    """
    if not shielded_tasks:
        return

    log_extra = {
        "cache_key": cache_key[:8],
        "model": model,
        "request_type": "background-cache",
    }
    log(
        "info",
        f"后台缓存任务启动，等待 {len(shielded_tasks)} 个剩余并发 API 调用完成",
        extra=log_extra,
    )

    results = []
    timed_out_tasks = 0
    gather_exception = None
    timeout_duration = settings.REQUEST_TIMEOUT
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
            f"后台缓存任务: gather 等待超时 ({timeout_duration}s)",
            extra=log_extra,
        )
        gather_exception = asyncio.TimeoutError("Background caching gather timed out")
        results = []
        # 超时后，检查哪些任务实际完成了，哪些没有
        for task in shielded_tasks:
            if task.done():
                try:
                    results.append(task.result())
                except Exception as e:
                    results.append(e)  # 记录已完成任务的异常
            else:
                timed_out_tasks += 1
                results.append(
                    asyncio.TimeoutError(f"Task did not complete within gather timeout")
                )
    except Exception as e:
        # 处理 gather 发生的其他异常
        log("error", f"后台缓存任务: gather 发生意外错误: {e}", extra=log_extra)
        gather_exception = e
        results = []
        for task in shielded_tasks:
            if task.done():
                try:
                    results.append(task.result())
                except Exception as ex:
                    results.append(ex)
            else:
                timed_out_tasks += 1
                results.append(
                    RuntimeError(f"Task did not complete due to gather error")
                )

    # 统计并记录后台任务的最终结果
    counts = {
        "success": 0,
        "empty": 0,
        "error": 0,
        "cancelled": 0,
        "timeout": timed_out_tasks,
    }

    for i, result in enumerate(results):
        if i >= len(shielded_tasks):
            log("error", f"后台缓存任务: 结果数量与任务数量不匹配", extra=log_extra)
            counts["error"] += 1
            continue

        # 使用辅助函数处理每个后台任务的结果
        status, _ = _process_task_result(
            result, cache_key, response_cache_manager, log_extra, "后台缓存任务"
        )

        if status in counts:
            if status != "timeout":
                counts[status] += 1
        else:
            log(
                "error",
                f"后台缓存任务：处理结果时遇到未知状态 '{status}'",
                extra=log_extra,
            )
            counts["error"] += 1

    log_message = (
        f"后台缓存任务完成. 结果: "
        f"成功缓存={counts['success']}, 空响应={counts['empty']}, 错误={counts['error']}, "
        f"取消={counts['cancelled']}, 超时/未完成={counts['timeout']}"
    )
    if gather_exception:
        log_message += f". Gather 异常: {type(gather_exception).__name__}"

    log("info", log_message, extra=log_extra)


async def process_vertex_request(
    chat_request: ChatCompletionRequest,
    http_request: Request,
    request_type: Literal["non-stream"],
    response_cache_manager,
    active_requests_manager,
    cache_key: str,
    safety_settings: Optional[dict] = None,
    safety_settings_g2: Optional[dict] = None,
):
    """
    处理非流式 Vertex 请求的主入口。
    管理并发 API 调用、缓存和客户端断开连接。
    """
    log_extra = {
        "cache_key": cache_key[:8],
        "model": chat_request.model,
        "request_type": request_type,
    }

    # 1. 优先检查缓存
    cached_response, cache_hit = response_cache_manager.get_and_remove(cache_key)
    if cache_hit:
        log("info", f"Vertex 请求命中缓存", extra=log_extra)
        return openAI_nonstream_response(cached_response)

    # 2. 检查是否有相同请求正在处理 (防止重复调用)
    # 使用 Future 来同步等待正在进行的相同请求的结果
    orchestrator_pool_key = f"vertex_orchestrator:{cache_key}"
    existing_future = active_requests_manager.get(orchestrator_pool_key)
    if isinstance(existing_future, asyncio.Future) and not existing_future.done():
        log("info", f"发现相同请求进行中，等待其 Future", extra=log_extra)
        try:
            # 等待现有任务的 Future 完成
            first_result = await asyncio.wait_for(
                existing_future, timeout=settings.REQUEST_TIMEOUT
            )
            if isinstance(first_result, VertexCachedResponse):
                log("info", f"使用来自现有任务 Future 的结果", extra=log_extra)
                # 理论上，现有任务应该已将结果存入缓存，再次尝试获取
                cached_response_again, cache_hit_again = (
                    response_cache_manager.get_and_remove(cache_key)
                )
                if cache_hit_again:
                    return openAI_nonstream_response(cached_response_again)
                else:
                    # 容错：如果缓存意外未命中，仍使用 Future 的结果
                    log(
                        "warning",
                        "现有任务 Future 完成但缓存未命中，使用 Future 结果",
                        extra=log_extra,
                    )
                    return openAI_nonstream_response(first_result)
            else:
                log(
                    "error",
                    f"现有任务 Future 返回意外类型: {type(first_result)}",
                    extra=log_extra,
                )
        except asyncio.TimeoutError:
            log("warning", f"等待现有任务 Future 超时", extra=log_extra)
        except asyncio.CancelledError:
            log("warning", f"等待现有任务 Future 时被取消", extra=log_extra)
            raise
        except Exception as e:
            log("error", f"等待现有任务 Future 时发生错误: {e}", extra=log_extra)
        # 若等待失败 (超时/错误)，则继续创建新任务

    # 3. 创建新的非流式编排器任务
    log("info", f"Vertex 请求缓存未命中，创建新任务组", extra=log_extra)
    # 创建一个 Future，用于当前请求等待第一个成功结果
    first_result_future = asyncio.Future()
    # 将 Future 注册到管理器，以便后续相同请求可以等待它
    active_requests_manager.add(orchestrator_pool_key, first_result_future)

    async def _orchestrator():
        """
        内部非流式编排器:
        - 并发启动 API 调用和客户端断开监控。
        - 处理首个成功响应或客户端断开。
        - 触发后台缓存任务。
        """
        api_call_tasks_unshielded = []  # 用于 asyncio.wait，可被取消
        api_call_tasks_shielded = []  # 用于后台缓存，不可被轻易取消
        disconnect_monitor_task = None
        all_tasks_to_wait = []  # 初始等待的任务列表

        # 准备 Vertex API 请求体
        openai_messages = [
            OpenAIMessage(role=m.role, content=m.content) for m in chat_request.messages
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
            logprobs=getattr(chat_request, "logprobs", None),
            response_logprobs=getattr(chat_request, "response_logprobs", None),
            n=chat_request.n,
        )

        num_concurrent_requests = settings.CONCURRENT_REQUESTS
        log(
            "info",
            f"非流式编排器: 发起 {num_concurrent_requests} 个并发 API 调用 (含后台缓存)",
            extra=log_extra,
        )

        # 创建 N 个并发 API 调用任务
        for _ in range(num_concurrent_requests):
            core_api_task = asyncio.create_task(
                _execute_single_vertex_call(
                    chat_request=chat_request,
                    vertex_request_payload=vertex_request_payload,
                )
            )
            # 使用 shield 保护核心任务，防止在后台缓存阶段被轻易取消
            shielded_task = asyncio.shield(core_api_task)
            api_call_tasks_shielded.append(shielded_task)
            # 未受保护的任务用于初始等待，以便能快速取消
            api_call_tasks_unshielded.append(core_api_task)
            all_tasks_to_wait.append(core_api_task)

        # 创建客户端断开连接监控任务
        disconnect_monitor_task = asyncio.create_task(
            check_client_disconnect(
                http_request,
                f"orchestrator-{cache_key[:8]}",
                "non-stream-orchestrator",
                chat_request.model,
            )
        )
        all_tasks_to_wait.append(disconnect_monitor_task)

        first_event_handled = False
        pending_tasks = all_tasks_to_wait[:]

        try:
            # 循环等待，直到处理了第一个重要事件 (API 成功或客户端断开)
            while pending_tasks and not first_event_handled:
                # 等待任意一个任务完成
                done, pending = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                pending_tasks = list(pending)  # 更新待处理任务列表

                for task in done:
                    if first_event_handled:
                        continue  # 如果已处理过首个事件，忽略其他已完成任务

                    if task is disconnect_monitor_task:
                        # 情况 1: 客户端先断开连接
                        first_event_handled = True
                        log("warning", "非流式编排器: 客户端断开连接", extra=log_extra)
                        cancelled_api_count = 0
                        # 取消所有仍在运行的 unshielded API 任务
                        for api_task in api_call_tasks_unshielded:
                            if not api_task.done():
                                api_task.cancel()
                                cancelled_api_count += 1
                        log(
                            "info",
                            f"非流式编排器: 已取消 {cancelled_api_count} 个挂起的 API 任务",
                            extra=log_extra,
                        )
                        # 设置主 Future 异常，通知等待者客户端已断开
                        if not first_result_future.done():
                            first_result_future.set_exception(
                                HTTPException(status_code=499, detail="客户端断开连接")
                            )
                        # 即使断开，也启动后台任务等待 shielded 任务完成以供缓存
                        if api_call_tasks_shielded:
                            log(
                                "info",
                                "非流式编排器: 客户端断开，仍启动后台任务等待剩余 API 调用以供缓存",
                                extra=log_extra,
                            )
                            asyncio.create_task(
                                _await_and_cache_shielded(
                                    shielded_tasks=api_call_tasks_shielded,
                                    cache_key=cache_key,
                                    response_cache_manager=response_cache_manager,
                                    model=chat_request.model,
                                )
                            )
                        break  # 已处理首个事件，退出内层循环

                    elif task in api_call_tasks_unshielded:
                        # 情况 2: 某个 API 调用先完成
                        try:
                            # 处理该 API 任务的结果
                            status, response_obj = _process_task_result(
                                task.result(),
                                cache_key,
                                response_cache_manager,
                                log_extra,
                                "非流式编排器",
                            )

                            if status == "success":
                                # 情况 2a: API 调用成功
                                first_event_handled = True
                                # log("info", f"非流式编排器: 收到首个成功响应", extra=log_extra) # Redundant log removed

                                # 取消不再需要的客户端断开监控任务
                                if (
                                    disconnect_monitor_task
                                    and not disconnect_monitor_task.done()
                                ):
                                    disconnect_monitor_task.cancel()

                                # 设置主 Future 的结果，通知等待者已获得成功响应
                                if not first_result_future.done():
                                    first_result_future.set_result(response_obj)
                                else:
                                    log(
                                        "warning",
                                        f"非流式编排器: Future 已完成，无法设置首个成功结果",
                                        extra=log_extra,
                                    )

                                # 找出除当前成功任务外的其他 shielded 任务
                                corresponding_shielded = next(
                                    (
                                        st
                                        for st, core_t in zip(
                                            api_call_tasks_shielded,
                                            api_call_tasks_unshielded,
                                        )
                                        if core_t is task
                                    ),
                                    None,
                                )
                                remaining_shielded_tasks = [
                                    st
                                    for st in api_call_tasks_shielded
                                    if st is not corresponding_shielded
                                ]

                                # 启动后台任务处理剩余的 shielded 任务以供缓存
                                if remaining_shielded_tasks:
                                    log(
                                        "info",
                                        f"非流式编排器: 启动后台任务以缓存 {len(remaining_shielded_tasks)} 个剩余 API 调用",
                                        extra=log_extra,
                                    )
                                    asyncio.create_task(
                                        _await_and_cache_shielded(
                                            shielded_tasks=remaining_shielded_tasks,
                                            cache_key=cache_key,
                                            response_cache_manager=response_cache_manager,
                                            model=chat_request.model,
                                        )
                                    )
                                else:
                                    log(
                                        "info",
                                        "非流式编排器: 首个成功响应是最后一个 API 任务，无需后台缓存",
                                        extra=log_extra,
                                    )

                                break  # 已处理首个事件，退出内层循环

                            elif status in ["error", "empty", "cancelled", "timeout"]:
                                # 情况 2b: API 调用失败/空/取消/超时
                                if task in all_tasks_to_wait:
                                    all_tasks_to_wait.remove(task)
                                # 检查是否所有 API 任务都已失败 (且尚未处理过事件)
                                if not any(
                                    t in all_tasks_to_wait
                                    for t in api_call_tasks_unshielded
                                ):
                                    if not first_event_handled:
                                        log(
                                            "error",
                                            "非流式编排器: 所有 API 调用均失败或空响应",
                                            extra=log_extra,
                                        )
                                        # 设置主 Future 异常
                                        if not first_result_future.done():
                                            first_result_future.set_exception(
                                                HTTPException(
                                                    status_code=503,
                                                    detail="所有后端 Vertex 调用均失败",
                                                )
                                            )
                                        first_event_handled = True
                                        break

                        except asyncio.CancelledError:
                            # 处理 API 任务在等待过程中被取消的情况
                            log(
                                "warning",
                                f"非流式编排器: API 任务在处理器中被取消",
                                extra=log_extra,
                            )
                            if task in all_tasks_to_wait:
                                all_tasks_to_wait.remove(task)
                            if (
                                not any(
                                    t in all_tasks_to_wait
                                    for t in api_call_tasks_unshielded
                                )
                                and not first_event_handled
                            ):
                                log(
                                    "error",
                                    "非流式编排器: 所有 API 调用均失败或被取消",
                                    extra=log_extra,
                                )
                                if not first_result_future.done():
                                    first_result_future.set_exception(
                                        HTTPException(
                                            status_code=503,
                                            detail="所有后端 Vertex 调用均失败或被取消",
                                        )
                                    )
                                first_event_handled = True
                                break
                        except Exception as e:
                            # 处理结果处理过程中的异常
                            log(
                                "error",
                                f"非流式编排器: 处理 API 任务结果时出错: {e}",
                                exc_info=True,
                                extra=log_extra,
                            )
                            if task in all_tasks_to_wait:
                                all_tasks_to_wait.remove(task)
                            if (
                                not any(
                                    t in all_tasks_to_wait
                                    for t in api_call_tasks_unshielded
                                )
                                and not first_event_handled
                            ):
                                log(
                                    "error",
                                    "非流式编排器: 所有 API 调用均失败或出错",
                                    extra=log_extra,
                                )
                                if not first_result_future.done():
                                    first_result_future.set_exception(
                                        HTTPException(
                                            status_code=500,
                                            detail=f"处理后端响应时出错: {e}",
                                        )
                                    )
                                first_event_handled = True
                                break

                # 如果已处理首个事件，则退出外层循环
                if first_event_handled:
                    break
            # 外层循环结束

            # 如果循环正常结束但未处理任何事件 (理论上不太可能，除非所有任务都以非成功状态结束)
            if not first_event_handled:
                log("error", "非流式编排器: 所有任务完成但未成功处理", extra=log_extra)
                if not first_result_future.done():
                    first_result_future.set_exception(
                        HTTPException(
                            status_code=503, detail="所有后端 Vertex 调用均失败或超时"
                        )
                    )

        except Exception as e:
            # 捕获非流式编排器顶层的意外错误
            log(
                "error",
                f"非流式编排器发生意外错误: {e}",
                exc_info=True,
                extra=log_extra,
            )
            if not first_result_future.done():
                first_result_future.set_exception(
                    HTTPException(status_code=500, detail=f"非流式编排器内部错误: {e}")
                )
            # 尝试取消所有子任务
            if disconnect_monitor_task and not disconnect_monitor_task.done():
                disconnect_monitor_task.cancel()
            for api_task in api_call_tasks_unshielded:
                if not api_task.done():
                    api_task.cancel()

        finally:
            # 确保所有资源被清理
            if disconnect_monitor_task and not disconnect_monitor_task.done():
                disconnect_monitor_task.cancel()
            # 确保主 Future 最终被设置 (以防万一)
            if not first_result_future.done():
                log("error", "非流式编排器退出但 Future 未设置 (异常)", extra=log_extra)
                first_result_future.set_exception(
                    HTTPException(status_code=500, detail="非流式编排器未知错误")
                )
            # 从管理器中移除 Future，允许后续相同请求创建新任务
            active_requests_manager.remove(orchestrator_pool_key)

    # 启动后台非流式编排器任务
    asyncio.create_task(_orchestrator())

    # 4. 等待主 Future 的结果 (由 _orchestrator 设置)
    try:
        log(
            "info",
            f"等待首个 Vertex 响应 Future (超时: {settings.REQUEST_TIMEOUT}s)...",
            extra=log_extra,
        )
        first_result = await asyncio.wait_for(
            first_result_future, timeout=settings.REQUEST_TIMEOUT
        )

        if isinstance(first_result, VertexCachedResponse):
            log("info", "收到首个成功响应，返回给用户", extra=log_extra)
            # 再次检查缓存，理论上应该命中
            cached_response_final, cache_hit_final = (
                response_cache_manager.get_and_remove(cache_key)
            )
            if cache_hit_final:
                return openAI_nonstream_response(cached_response_final)
            else:
                # 容错
                log(
                    "warning",
                    "非流式编排器完成但缓存未命中，使用 Future 结果",
                    extra=log_extra,
                )
                return openAI_nonstream_response(first_result)
        else:
            # Future 被设置了非预期类型的结果
            log("error", f"Future 返回意外类型: {type(first_result)}", extra=log_extra)
            raise HTTPException(
                status_code=500, detail="内部服务器错误：非流式编排器状态异常"
            )

    except asyncio.TimeoutError:
        # 等待 Future 超时
        log(
            "error",
            f"等待首个 Vertex 响应 Future 超时 ({settings.REQUEST_TIMEOUT}s)",
            extra=log_extra,
        )
        raise HTTPException(
            status_code=504, detail="请求超时，等待后端 Vertex 响应时间过长"
        )
    except asyncio.CancelledError:
        # 主请求被取消 (例如，FastAPI 请求被中断)
        log("warning", "等待 Future 时主请求被取消", extra=log_extra)
        raise
    except Exception as e:
        # 处理等待 Future 或后续处理中发生的异常
        log(
            "error",
            f"等待或处理 Future 时发生错误: {e}",
            exc_info=True,
            extra=log_extra,
        )
        if isinstance(e, HTTPException):
            raise e  # 重新抛出已知的 HTTP 异常
        else:
            raise HTTPException(status_code=500, detail=f"处理请求时发生内部错误")
