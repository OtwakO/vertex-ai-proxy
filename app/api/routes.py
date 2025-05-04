import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

import app.config.settings as settings
from app.models import ChatCompletionRequest, ChatCompletionResponse, ModelList
from app.models.schemas import ChatCompletionResponse
from app.services import GeminiClient
from app.utils import (
    generate_cache_key,
    generate_cache_key_all,
    log,  # Keep generic log for Aistudio/shared parts
    protect_from_abuse,
)
from app.utils.logging import vertex_log  # Import vertex_log specifically
from app.utils.response import openAI_from_Gemini

# 导入拆分后的模块
from .auth import verify_password
from .nonstream_handlers import (
    process_request as process_aistudio_request,  # Rename for clarity
)
from .stream_handlers import (
    process_stream_request as process_aistudio_stream_request,  # Rename for clarity
)

# 导入 Vertex 处理器
from .vertex_nonstream_handlers import process_vertex_request
from .vertex_stream_handlers import process_vertex_stream_request

# 创建路由器
router = APIRouter()

# 全局变量引用 - 这些将在main.py中初始化并传递给路由
key_manager = None
response_cache_manager = None
active_requests_manager = None
safety_settings = None
safety_settings_g2 = None
current_api_key = None
FAKE_STREAMING = None
FAKE_STREAMING_INTERVAL = None
PASSWORD = None
MAX_REQUESTS_PER_MINUTE = None
MAX_REQUESTS_PER_DAY_PER_IP = None


# 初始化路由器的函数
def init_router(
    _key_manager,
    _response_cache_manager,
    _active_requests_manager,
    _safety_settings,
    _safety_settings_g2,
    _current_api_key,
    _fake_streaming,
    _fake_streaming_interval,
    _password,
    _max_requests_per_minute,
    _max_requests_per_day_per_ip,
):
    global key_manager, response_cache_manager, active_requests_manager
    global safety_settings, safety_settings_g2, current_api_key
    global FAKE_STREAMING, FAKE_STREAMING_INTERVAL
    global PASSWORD, MAX_REQUESTS_PER_MINUTE, MAX_REQUESTS_PER_DAY_PER_IP

    key_manager = _key_manager
    response_cache_manager = _response_cache_manager
    active_requests_manager = _active_requests_manager
    safety_settings = _safety_settings
    safety_settings_g2 = _safety_settings_g2
    current_api_key = _current_api_key
    FAKE_STREAMING = _fake_streaming
    FAKE_STREAMING_INTERVAL = _fake_streaming_interval
    PASSWORD = _password
    MAX_REQUESTS_PER_MINUTE = _max_requests_per_minute
    MAX_REQUESTS_PER_DAY_PER_IP = _max_requests_per_day_per_ip


# 自定义密码验证依赖
async def custom_verify_password(request: Request):
    await verify_password(request, settings.PASSWORD)


async def verify_user_agent(request: Request):
    if not settings.WHITELIST_USER_AGENT:
        return
    if request.headers.get("User-Agent") not in settings.WHITELIST_USER_AGENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed client"
        )


def get_cached(cache_key, is_stream: bool):
    # 检查缓存是否存在，如果存在，返回缓存
    cached_response, cache_hit = response_cache_manager.get_and_remove(cache_key)

    if cache_hit and cached_response:
        log(
            "info",
            f"缓存命中: {cache_key[:8]}...",
            extra={"request_type": "non-stream", "model": cached_response.model},
        )

        if is_stream:
            chunk = openAI_from_Gemini(cached_response, stream=True)
            return StreamingResponse(chunk, media_type="text/event-stream")
        else:
            return openAI_from_Gemini(cached_response, stream=False)

    return None


@router.get("/aistudio/models", response_model=ModelList)
async def aistudio_list_models(
    _=Depends(custom_verify_password), _2=Depends(verify_user_agent)
):
    # 使用原有的Gemini实现
    if settings.PUBLIC_MODE:
        filtered_models = ["gemini-2.5-pro-exp-03-25", "gemini-2.5-flash-preview-04-17"]
    elif settings.WHITELIST_MODELS:
        filtered_models = [
            model
            for model in GeminiClient.AVAILABLE_MODELS
            if model in settings.WHITELIST_MODELS
        ]
    else:
        filtered_models = [
            model
            for model in GeminiClient.AVAILABLE_MODELS
            if model not in settings.BLOCKED_MODELS
        ]
    return ModelList(
        data=[
            {
                "id": model,
                "object": "model",
                "created": 1678888888,
                "owned_by": "organization-owner",
            }
            for model in filtered_models
        ]
    )


@router.get("/vertex/models", response_model=ModelList)
async def vertex_list_models(
    _=Depends(custom_verify_password), _2=Depends(verify_user_agent)
):
    # 使用Vertex AI实现
    from app.vertex.vertex import list_models as vertex_list_models

    # 调用Vertex AI实现
    vertex_response = await vertex_list_models(api_key=current_api_key)

    # 转换为ModelList格式
    return ModelList(data=vertex_response.get("data", []))


# API路由
@router.get("/v1/models", response_model=ModelList)
@router.get("/models", response_model=ModelList)
async def list_models(_=Depends(custom_verify_password), _2=Depends(verify_user_agent)):
    if settings.ENABLE_VERTEX:
        return await vertex_list_models(_, _2)
    return await aistudio_list_models(_, _2)


@router.post("/aistudio/chat/completions", response_model=ChatCompletionResponse)
async def aistudio_chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _=Depends(custom_verify_password),
    _2=Depends(verify_user_agent),
):
    global current_api_key

    # 生成缓存键 - 用于匹配请求内容对应缓存
    if settings.PRECISE_CACHE:
        cache_key = generate_cache_key_all(request)
    else:
        cache_key = generate_cache_key(request, 8)

    # 请求前基本检查
    await protect_from_abuse(
        http_request,
        settings.MAX_REQUESTS_PER_MINUTE,
        settings.MAX_REQUESTS_PER_DAY_PER_IP,
    )

    if request.model not in GeminiClient.AVAILABLE_MODELS:
        log("error", "无效的模型", extra={"model": request.model, "status_code": 400})
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="无效的模型"
        )

    # 记录请求缓存键信息
    log(
        "info",
        f"请求缓存键: {cache_key[:8]}...",
        extra={"request_type": "non-stream", "model": request.model},
    )

    # 检查缓存是否存在，如果存在，返回缓存
    cached_response = get_cached(cache_key, is_stream=request.stream)
    if cached_response:
        return cached_response

    if not settings.PUBLIC_MODE:
        # 构建包含缓存键的活跃请求池键
        pool_key = f"{cache_key}"

        # 查找所有使用相同缓存键的活跃任务
        active_task = active_requests_manager.get(pool_key)
        if active_task and not active_task.done():
            log(
                "info",
                "发现相同请求的进行中任务",
                extra={
                    "request_type": "stream" if request.stream else "non-stream",
                    "model": request.model,
                },
            )

            # 等待已有任务完成
            try:
                # 设置超时，避免无限等待
                await asyncio.wait_for(active_task, timeout=180)

                # 使用任务结果
                if active_task.done() and not active_task.cancelled():
                    result = active_task.result()
                    active_requests_manager.remove(pool_key)
                    if result:
                        return result

            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                # 任务超时或被取消的情况下，记录日志然后让代码继续执行
                error_type = "超时" if isinstance(e, asyncio.TimeoutError) else "被取消"
                log(
                    "warning",
                    f"等待已有任务{error_type}: {pool_key}",
                    extra={"request_type": "non-stream", "model": request.model},
                )

                # 从活跃请求池移除该任务
                if active_task.done() or active_task.cancelled():
                    active_requests_manager.remove(pool_key)
                    log(
                        "info",
                        f"已从活跃请求池移除{error_type}任务: {pool_key}",
                        extra={"request_type": "non-stream"},
                    )

    if request.stream:
        # 流式请求处理任务
        process_task = asyncio.create_task(
            process_aistudio_stream_request(  # Use renamed import
                chat_request=request,
                key_manager=key_manager,
                response_cache_manager=response_cache_manager,
                safety_settings=safety_settings,
                safety_settings_g2=safety_settings_g2,
                cache_key=cache_key,
            )
        )

    else:
        # 创建非流式请求处理任务
        process_task = asyncio.create_task(
            process_aistudio_request(  # Use renamed import
                chat_request=request,
                key_manager=key_manager,
                response_cache_manager=response_cache_manager,
                safety_settings=safety_settings,
                safety_settings_g2=safety_settings_g2,
                cache_key=cache_key,
            )
        )

    if not settings.PUBLIC_MODE:
        # 将任务添加到活跃请求池
        active_requests_manager.add(pool_key, process_task)

    # 等待任务完成
    try:
        response = await process_task
        if not settings.PUBLIC_MODE:
            active_requests_manager.remove(pool_key)

        return response
    except Exception:
        if not settings.PUBLIC_MODE:
            # 如果任务失败，从活跃请求池中移除
            active_requests_manager.remove(pool_key)

        # 检查是否已有缓存的结果（可能是由另一个任务创建的）
        cached_response = get_cached(cache_key, is_stream=request.stream)
        if cached_response:
            return cached_response

        # 发送错误信息给客户端
        raise HTTPException(status_code=500, detail=" hajimi 服务器内部处理时发生错误")


@router.post("/vertex/chat/completions", response_model=ChatCompletionResponse)
async def vertex_chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _=Depends(custom_verify_password),
    _2=Depends(verify_user_agent),
):
    """处理 Vertex AI 聊天补全请求，包含缓存和并发管理"""
    global current_api_key  # Vertex 可能仍需要 key，尽管 handlers 内部可能处理

    # TODO: 确定 Vertex 模型的验证方式。假设它们以 'vertex/' 开头或在特定列表中。
    # if not request.model.startswith("vertex/"):
    #     log("error", "无效的 Vertex 模型", extra={"model": request.model, "status_code": 400})
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="无效的 Vertex 模型")

    # 生成缓存键
    if settings.PRECISE_CACHE:
        cache_key = generate_cache_key_all(request)
    else:
        cache_key = generate_cache_key(request, 8)

    # 请求前基本检查 (复用 aistudio 的逻辑)
    await protect_from_abuse(
        http_request,
        settings.MAX_REQUESTS_PER_MINUTE,
        settings.MAX_REQUESTS_PER_DAY_PER_IP,
    )

    log_extra = {
        "request_type": "vertex-" + ("stream" if request.stream else "non-stream"),
        "model": request.model,
    }
    # Use vertex_log for Vertex specific messages
    vertex_log("info", f"Vertex 请求缓存键: {cache_key[:8]}...", extra=log_extra)

    # 检查缓存 (get_cached 内部使用 openAI_from_Gemini，应能处理 VertexCachedResponse)
    # get_cached uses the generic 'log' for cache hits, which is acceptable for a shared function.
    cached_response = get_cached(cache_key, is_stream=request.stream)
    if cached_response:
        return cached_response

    # 检查活跃请求 (复用 aistudio 的逻辑)
    if not settings.PUBLIC_MODE:
        pool_key = f"vertex:{cache_key}"  # 使用 'vertex:' 前缀区分
        active_task = active_requests_manager.get(pool_key)
        if active_task and not active_task.done():
            vertex_log(
                "info", "发现相同 Vertex 请求进行中", extra=log_extra
            )  # Use vertex_log
            try:
                await asyncio.wait_for(
                    active_task, timeout=settings.REQUEST_TIMEOUT + 5
                )  # 增加超时缓冲
                if active_task.done() and not active_task.cancelled():
                    result = active_task.result()
                    active_requests_manager.remove(pool_key)  # 成功后移除
                    if result:
                        # 如果任务成功返回结果 (可能是 StreamingResponse 或 JSONResponse)
                        return result
                    else:
                        # 任务成功但返回 None (可能内部处理了错误或空响应)
                        vertex_log(
                            "warning",
                            "等待的 Vertex 任务成功完成但返回 None",
                            extra=log_extra,
                        )  # Use vertex_log
                        # 重新检查缓存，以防万一
                        cached_response_after_wait = get_cached(
                            cache_key, is_stream=request.stream
                        )
                        if cached_response_after_wait:
                            return cached_response_after_wait
                        else:  # 如果还是没有，则抛出错误
                            raise HTTPException(
                                status_code=500,
                                detail="等待的 Vertex 任务未产生有效结果或缓存",
                            )

            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                error_type = "超时" if isinstance(e, asyncio.TimeoutError) else "被取消"
                vertex_log(
                    "warning",
                    f"等待已有 Vertex 任务{error_type}: {pool_key}",
                    extra=log_extra,
                )  # Use vertex_log
                if active_task.done() or active_task.cancelled():
                    active_requests_manager.remove(pool_key)
            except Exception as e:
                vertex_log(
                    "error", f"等待已有 Vertex 任务时发生未知错误: {e}", extra=log_extra
                )  # Use vertex_log
                if active_task.done() or active_task.cancelled():
                    active_requests_manager.remove(pool_key)
            # 如果等待失败，继续创建新任务

    # 创建新的 Vertex 请求处理任务
    if request.stream:
        process_task = asyncio.create_task(
            process_vertex_stream_request(  # 调用 Vertex 流式处理器
                chat_request=request,
                http_request=http_request,  # 传递 http_request
                response_cache_manager=response_cache_manager,
                active_requests_manager=active_requests_manager,  # 传递 active_requests_manager
                cache_key=cache_key,
                safety_settings=safety_settings,  # 传递 safety_settings
                safety_settings_g2=safety_settings_g2,  # 传递 safety_settings_g2
            )
        )
    else:
        process_task = asyncio.create_task(
            process_vertex_request(  # 调用 Vertex 非流式处理器
                chat_request=request,
                http_request=http_request,  # 传递 http_request
                request_type="non-stream",  # 指定类型
                response_cache_manager=response_cache_manager,
                active_requests_manager=active_requests_manager,  # 传递 active_requests_manager
                cache_key=cache_key,
                safety_settings=safety_settings,  # 传递 safety_settings
                safety_settings_g2=safety_settings_g2,  # 传递 safety_settings_g2
            )
        )

    if not settings.PUBLIC_MODE:
        active_requests_manager.add(pool_key, process_task)

    # 等待任务完成并处理结果/异常
    try:
        response = await process_task
        if not settings.PUBLIC_MODE:
            # 仅在任务成功且非空时才移除 future，以便后续请求可以重用缓存
            # 注意：流式响应 StreamingResponse 本身不代表成功完成，需要让其自然结束
            if not isinstance(response, StreamingResponse) and response is not None:
                active_requests_manager.remove(pool_key)
            elif isinstance(response, StreamingResponse):
                # 对于流式响应，我们不立即移除 future，让其在后台缓存任务完成后自行处理或超时
                pass
        # Correctly indented return statement inside the try block
        return response
    # Correctly indented except block aligned with try
    except Exception as e:
        vertex_log(
            "error",
            f"处理 Vertex 请求任务时捕获到异常: {e}",
            extra=log_extra,
            exc_info=True,
        )  # Use vertex_log
        if not settings.PUBLIC_MODE:
            active_requests_manager.remove(pool_key)  # 失败时移除

        # 再次检查缓存，以防后台任务已完成
        cached_response_on_error = get_cached(cache_key, is_stream=request.stream)
        if cached_response_on_error:
            vertex_log(
                "info", "Vertex 任务失败后从缓存中获取到结果", extra=log_extra
            )  # Use vertex_log
            return cached_response_on_error

        # 如果没有缓存，则抛出 HTTP 异常
        status_code = getattr(e, "status_code", 500)
        detail = getattr(
            e, "detail", f"处理 Vertex 请求时发生内部错误: {type(e).__name__}"
        )
        raise HTTPException(status_code=status_code, detail=detail)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _=Depends(custom_verify_password),
    _2=Depends(verify_user_agent),
):
    """处理API请求的主函数，根据需要处理流式或非流式请求"""
    if settings.ENABLE_VERTEX:
        return await vertex_chat_completions(request, http_request, _, _2)
    return await aistudio_chat_completions(request, http_request, _, _2)
