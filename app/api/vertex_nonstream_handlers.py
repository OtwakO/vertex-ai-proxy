# app/api/vertex_nonstream_handlers.py
import asyncio
import json
from typing import Any, Literal, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

import app.config.settings as settings
from app.models import ChatCompletionRequest
from app.utils import handle_gemini_error
from app.utils.logging import vertex_log as log

# from app.utils.response import openAI_nonstream_response # 废弃，使用 openAI_from_Gemini
from app.utils.response import openAI_from_Gemini  # 导入新的响应格式化函数
from app.vertex.vertex import OpenAIMessage, OpenAIRequest
from app.vertex.vertex import chat_completions as vertex_chat_completions_impl

from .client_disconnect import check_client_disconnect


class VertexCachedResponse:
    """Vertex 缓存响应结构 (内部使用)。"""

    def __init__(
        self,
        text,
        model,
        prompt_token_count=0,
        candidates_token_count=0,
        total_token_count=0,
        function_call=None,
    ):
        self.text = text
        self.model = model
        self.prompt_token_count = (
            int(prompt_token_count) if prompt_token_count is not None else 0
        )
        self.candidates_token_count = (
            int(candidates_token_count) if candidates_token_count is not None else 0
        )
        self.total_token_count = (
            int(total_token_count) if total_token_count is not None else 0
        )
        # 添加 finish_reason 以便 openAI_from_Gemini 正确处理
        self.finish_reason = "stop"  # 非流式通常是 stop
        self.function_call = function_call


async def _execute_single_vertex_call(
    chat_request: ChatCompletionRequest,
    vertex_request_payload: OpenAIRequest,
) -> Tuple[str, Optional[VertexCachedResponse], Optional[Exception]]:
    """
    执行单次 Vertex AI API 调用，处理响应或异常。

    返回:
        包含状态、响应对象（成功时）或异常对象（失败时）的元组。
        状态可能为 'success', 'empty', 'error', 'cancelled'。
    """
    log_extra = {
        "request_type": "vertex-api-call",
        "model": chat_request.model,
    }

    try:
        log("debug", "发起 Vertex API 调用", extra=log_extra)
        vertex_response = await vertex_chat_completions_impl(vertex_request_payload)
        log(
            "debug",
            f"收到 Vertex API 响应，类型: {type(vertex_response)}",
            extra=log_extra,
        )

        response_text = ""
        prompt_tokens = 0
        candidates_tokens = 0
        total_tokens = 0
        response_content = None
        function_call_data = None  # 支持函数调用

        if isinstance(vertex_response, JSONResponse):
            try:
                response_content = json.loads(vertex_response.body.decode("utf-8"))
            except (json.JSONDecodeError, AttributeError) as parse_err:
                log("error", f"解析 Vertex JSON 响应失败: {parse_err}", extra=log_extra)
                return "error", None, parse_err
        elif isinstance(vertex_response, Exception):
            log(
                "error",
                f"Vertex API 调用返回或引发异常: {vertex_response}",
                extra=log_extra,
            )
            handle_gemini_error(vertex_response, "Vertex")  # 使用通用错误处理
            return "error", None, vertex_response
        elif isinstance(vertex_response, dict):
            response_content = vertex_response
        else:
            log(
                "error",
                f"Vertex API 调用返回未知类型: {type(vertex_response)}",
                extra=log_extra,
            )
            return (
                "error",
                None,
                TypeError(
                    f"Unexpected response type from Vertex API: {type(vertex_response)}"
                ),
            )

        if response_content and isinstance(response_content, dict):
            choices = response_content.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message", {})
                response_text = message.get("content", "")
                # TODO: 解析函数调用 (如果 Vertex 支持)
                # function_call_data = message.get("tool_calls") or message.get("function_call")
            usage = response_content.get("usage", {})
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                candidates_tokens = usage.get(
                    "completion_tokens"
                )  # Vertex 使用 completion_tokens
                total_tokens = usage.get("total_tokens")
            prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else 0
            candidates_tokens = (
                int(candidates_tokens) if candidates_tokens is not None else 0
            )
            total_tokens = int(total_tokens) if total_tokens is not None else 0

        if response_text or function_call_data:
            response_obj = VertexCachedResponse(
                text=response_text,
                model=chat_request.model,
                prompt_token_count=prompt_tokens,
                candidates_token_count=candidates_tokens,
                total_token_count=total_tokens,
                # finish_reason 默认为 stop，由 VertexCachedResponse 设置
                function_call=function_call_data,
            )
            log("debug", "成功解析 Vertex 响应", extra=log_extra)
            return "success", response_obj, None
        else:
            log(
                "warning",
                "Vertex API 调用成功但返回空响应 (无文本或函数调用)",
                extra=log_extra,
            )
            return "empty", None, None

    except asyncio.CancelledError:
        log("warning", "Vertex API 调用任务被取消", extra=log_extra)
        return "cancelled", None, asyncio.CancelledError("API call was cancelled")

    except Exception as e:
        log(
            "error",
            f"执行 Vertex API 调用时发生意外错误: {e}",
            exc_info=True,
            extra=log_extra,
        )
        handle_gemini_error(e, "Vertex")  # 使用通用错误处理
        return "error", None, e


def _process_task_result(
    result: Any,
    cache_key: str,
    response_cache_manager,
    log_extra: dict,
    context: str = "后台缓存任务",
) -> Tuple[str, Optional[VertexCachedResponse]]:
    """
    处理 API 调用任务的结果，缓存成功响应并记录日志。

    Args:
        result: API 调用的结果 (可能来自 gather 或 _execute_single_vertex_call)。
        cache_key: 用于缓存的键。
        response_cache_manager: 缓存管理器实例。
        log_extra: 附加的日志信息。
        context: 日志上下文信息 (例如, "后台缓存任务")。

    返回:
        包含状态和响应对象（成功时）的元组。
        状态可能为 'success', 'empty', 'error', 'cancelled', 'timeout'。
    """
    status = "error"
    response_obj = None

    if isinstance(result, (asyncio.CancelledError, asyncio.TimeoutError)):
        status = (
            "cancelled" if isinstance(result, asyncio.CancelledError) else "timeout"
        )
        log(
            "warning",
            f"{context}: API 任务未成功完成: {type(result).__name__}",
            extra=log_extra,
        )
    elif isinstance(result, Exception):
        status = "error"
        log("error", f"{context}: API 任务结果为异常: {result}", extra=log_extra)
    elif isinstance(result, tuple) and len(result) == 3:
        call_status, response_obj, error = result
        status = call_status
        if status == "success" and response_obj:
            log("debug", f"{context}: API 调用成功，准备缓存", extra=log_extra)
            response_cache_manager.store(cache_key, response_obj)
            log("info", f"{context}: 成功结果已缓存", extra=log_extra)
        elif status == "empty":
            log("debug", f"{context}: API 调用返回空响应", extra=log_extra)
        elif status == "error":
            log("warning", f"{context}: API 调用返回错误状态: {error}", extra=log_extra)
        elif status == "cancelled":
            # 这个状态理论上不应由 _execute_single_vertex_call 直接返回，而是通过异常捕获
            log(
                "warning",
                f"{context}: API 调用内部返回 'cancelled' 状态",
                extra=log_extra,
            )
        else:
            status = "error"
            log(
                "error",
                f"{context}: API 调用返回未知状态 '{call_status}'",
                extra=log_extra,
            )
    else:
        status = "error"
        log(
            "error",
            f"{context}: API 任务返回格式不符的结果: {type(result)}",
            extra=log_extra,
        )

    return status, response_obj if status == "success" else None


async def _await_and_cache_shielded(
    shielded_tasks: list[asyncio.Task],
    cache_key: str,
    response_cache_manager,
    model: str,
):
    """
    后台任务：等待被 `shield` 保护的 API 调用完成，并将成功结果存入缓存。
    """
    if not shielded_tasks:
        log(
            "debug",
            "后台缓存任务：没有需要等待的 shielded 任务",
            extra={"cache_key": cache_key[:8]},
        )
        return

    log_extra = {
        "cache_key": cache_key[:8],  # 日志中显示部分缓存键
        "model": model,
        "request_type": "vertex-background-cache",
    }
    log(
        "info",
        f"后台缓存任务启动，等待 {len(shielded_tasks)} 个剩余并发 API 调用完成",
        extra=log_extra,
    )

    results = []
    timed_out_tasks = 0  # 记录 gather 超时但任务本身未完成的数量
    gather_exception = None
    timeout_duration = settings.REQUEST_TIMEOUT
    try:
        # 使用 gather 等待所有后台任务完成，return_exceptions=True 使得单个任务的异常不会中断 gather
        results = await asyncio.wait_for(
            asyncio.gather(*shielded_tasks, return_exceptions=True),
            timeout=timeout_duration,
        )
    except asyncio.TimeoutError:
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
                    results.append(e)
            else:
                timed_out_tasks += 1
                results.append(
                    asyncio.TimeoutError("Task did not complete within gather timeout")
                )
                task.cancel()  # 尝试取消仍在运行的任务
    except Exception as e:
        log(
            "error",
            f"后台缓存任务: gather 发生意外错误: {e}",
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
                timed_out_tasks += 1
                results.append(
                    RuntimeError("Task did not complete due to gather error")
                )
                task.cancel()  # 尝试取消

    counts = {
        "success": 0,
        "empty": 0,
        "error": 0,
        "cancelled": 0,
        "timeout": timed_out_tasks,
    }
    log(
        "debug",
        f"后台缓存任务: gather 完成，开始处理 {len(results)} 个结果",
        extra=log_extra,
    )

    for i, result in enumerate(results):
        task_num = i + 1
        task_context = f"后台缓存任务 #{task_num}"
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
            counts["error"] += 1

    log_summary = (
        f"后台缓存任务完成. 统计: "
        f"成功缓存={counts['success']}, 空响应={counts['empty']}, 错误={counts['error']}, "
        f"取消={counts['cancelled']}, 超时/未完成={counts['timeout']}"
    )
    if gather_exception:
        log_summary += f". Gather 异常: {type(gather_exception).__name__}"

    log("info", log_summary, extra=log_extra)


async def process_vertex_request(
    chat_request: ChatCompletionRequest,
    http_request: Request,
    request_type: Literal["non-stream"],
    response_cache_manager,
    active_requests_manager,
    cache_key: str,
    safety_settings: Optional[dict] = None,  # TODO: Vertex 是否使用 safety settings?
    safety_settings_g2: Optional[dict] = None,  # TODO: Vertex 是否使用 safety settings?
):
    """
    处理非流式 Vertex 请求的主函数。

    负责管理并发 API 调用、缓存、Future 共享以及处理客户端断开连接。
    """
    log_extra = {
        "cache_key": cache_key[:8],  # 日志中显示部分缓存键
        "model": chat_request.model,
        "request_type": request_type,
    }

    # 优先检查缓存
    cached_response, cache_hit = response_cache_manager.get_and_remove(cache_key)
    if cache_hit and isinstance(
        cached_response, VertexCachedResponse
    ):  # 确保缓存类型正确
        log("info", "Vertex 请求命中缓存", extra=log_extra)
        return openAI_from_Gemini(
            cached_response, stream=False
        )  # 使用新的响应格式化函数
    elif cache_hit:
        log(
            "warning",
            f"Vertex 缓存命中但类型不符: {type(cached_response)}，忽略缓存",
            extra=log_extra,
        )
        # 类型不符，继续执行，如同未命中

    # 检查是否有相同请求正在处理 (防止重复调用)
    # 使用 Future 来同步等待正在进行的相同请求的结果
    # 使用 'nonstream:' 前缀区分 Future，避免与流式请求冲突
    orchestrator_pool_key = f"nonstream:vertex_orchestrator:{cache_key}"
    existing_future = active_requests_manager.get(orchestrator_pool_key)
    if isinstance(existing_future, asyncio.Future) and not existing_future.done():
        log("info", "发现相同请求进行中，等待其 Future", extra=log_extra)
        try:
            first_result = await asyncio.wait_for(
                existing_future, timeout=settings.REQUEST_TIMEOUT
            )
            if isinstance(first_result, VertexCachedResponse):
                log("info", "使用来自现有任务 Future 的结果", extra=log_extra)
                # 理论上，现有任务应该已将结果存入缓存，再次尝试获取
                cached_response_again, cache_hit_again = (
                    response_cache_manager.get_and_remove(cache_key)
                )
                if cache_hit_again and isinstance(
                    cached_response_again, VertexCachedResponse
                ):
                    log("info", "从 Future 等待后成功获取缓存", extra=log_extra)
                    return openAI_from_Gemini(cached_response_again, stream=False)
                else:
                    # 容错：如果缓存意外未命中或类型错误，仍使用 Future 的结果
                    log(
                        "warning",
                        "现有任务 Future 完成但缓存未命中或类型错误，直接使用 Future 结果",
                        extra=log_extra,
                    )
                    return openAI_from_Gemini(first_result, stream=False)
            else:
                log(
                    "error",
                    f"现有任务 Future 返回意外类型: {type(first_result)}",
                    extra=log_extra,
                )
        except asyncio.TimeoutError:
            log("warning", "等待现有任务 Future 超时", extra=log_extra)
        except asyncio.CancelledError:
            log("warning", "等待现有任务 Future 时被取消", extra=log_extra)
            raise
        except Exception as e:
            log("error", f"等待现有任务 Future 时发生错误: {e}", extra=log_extra)
        # 若等待失败 (超时/错误)，则继续创建新任务

    log("info", "Vertex 请求缓存未命中，创建新任务组", extra=log_extra)
    # 创建一个 Future，用于当前请求等待第一个成功结果
    first_result_future = asyncio.Future()
    # 将 Future 注册到管理器，以便后续相同请求可以等待它
    active_requests_manager.add(orchestrator_pool_key, first_result_future)

    async def _orchestrator():
        """
        内部非流式编排器函数：
        - 并发启动 API 调用与客户端断开监控。
        - 处理首个成功响应或客户端断开事件。
        - 触发后台缓存任务。
        """
        api_call_tasks_unshielded = []  # 用于 asyncio.wait，可被取消
        api_call_tasks_shielded = []  # 用于后台缓存，不可被轻易取消 (通过 asyncio.shield)
        disconnect_monitor_task = None
        all_tasks_to_wait = []  # 初始等待的任务列表 (包含 unshielded API 任务和断开监控)

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
                done, pending = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                pending_tasks = list(pending)

                for task in done:
                    if first_event_handled:
                        continue  # 如果已处理过首个事件，忽略其他已完成任务

                    if task is disconnect_monitor_task:
                        # 情况 1: 客户端先断开连接
                        first_event_handled = True
                        log("warning", "非流式编排器: 客户端断开连接", extra=log_extra)
                        cancelled_api_count = 0
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
                                        "非流式编排器: Future 已完成，无法设置首个成功结果",
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
                            log(
                                "warning",
                                "非流式编排器: API 任务在处理器中被取消",
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

            # 如果循环正常结束但未处理任何事件 (理论上不太可能)
            if not first_event_handled:
                log("error", "非流式编排器: 所有任务完成但未成功处理", extra=log_extra)
                if not first_result_future.done():
                    first_result_future.set_exception(
                        HTTPException(
                            status_code=503, detail="所有后端 Vertex 调用均失败或超时"
                        )
                    )

        except Exception as e:
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

    asyncio.create_task(_orchestrator())

    # 等待主 Future 的结果 (由 _orchestrator 设置)
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
            if cache_hit_final and isinstance(
                cached_response_final, VertexCachedResponse
            ):
                log("info", "从缓存获取最终响应", extra=log_extra)
                return openAI_from_Gemini(cached_response_final, stream=False)
            else:
                # 容错: 缓存未命中或类型错误，使用 Future 结果
                log(
                    "warning",
                    "非流式编排器完成但缓存未命中/类型错误，使用 Future 结果",
                    extra=log_extra,
                )
                return openAI_from_Gemini(first_result, stream=False)
        elif isinstance(first_result, Exception):
            log(
                "error",
                f"编排器 Future 完成但带有异常: {first_result}",
                extra=log_extra,
            )
            raise first_result
        else:
            log(
                "error",
                f"编排器 Future 返回意外类型: {type(first_result)}",
                extra=log_extra,
            )
            raise HTTPException(
                status_code=500, detail="内部服务器错误：非流式编排器状态异常"
            )

    except asyncio.TimeoutError:
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
        log(
            "error",
            f"等待或处理 Future 时发生错误: {e}",
            exc_info=True,
            extra=log_extra,
        )
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail="处理请求时发生内部错误")
