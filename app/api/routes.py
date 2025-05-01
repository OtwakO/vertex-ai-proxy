import asyncio
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

import app.config.settings as settings
from app.models import ChatCompletionRequest, ChatCompletionResponse, ModelList
from app.models.schemas import ChatCompletionResponse, Choice, Message
from app.services import GeminiClient
from app.utils import (
    generate_cache_key,
    generate_cache_key_all,
    log,
    openAI_nonstream_response,
    openAI_stream_chunk,
    protect_from_abuse,
)
from app.utils.logging import log, vertex_log
from app.utils.response import openAI_from_Gemini
from app.vertex.vertex import list_models as list_models_vertex

# 导入拆分后的模块
from .auth import verify_password
from .nonstream_handlers import process_request
from .stream_handlers import process_stream_request

# Vertex AI 处理器
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


@router.get("/aistudio/models", response_model=ModelList)
async def aistudio_list_models():
    # 使用原有的Gemini实现
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
async def vertex_list_models():
    # 使用Vertex AI实现
    from app.vertex.vertex import list_models as vertex_list_models

    # 调用Vertex AI实现
    vertex_response = await vertex_list_models(api_key=current_api_key)

    # 转换为ModelList格式
    return ModelList(data=vertex_response.get("data", []))


# API路由
@router.get("/v1/models", response_model=ModelList)
@router.get("/models", response_model=ModelList)
async def list_models():
    if settings.ENABLE_VERTEX:
        return await vertex_list_models()
    return await aistudio_list_models()


@router.post("/aistudio/chat/completions", response_model=ChatCompletionResponse)
async def aistudio_chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _: None = Depends(custom_verify_password),
):
    global current_api_key
    # 使用原有的Gemini实现

    # 生成缓存键 - 用于匹配请求内容对应缓存
    if settings.PRECISE_CACHE:
        cache_key = generate_cache_key_all(request)
    else:
        cache_key = generate_cache_key(request, 8)

    # 请求前基本检查
    protect_from_abuse(
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
    cached_response, cache_hit = response_cache_manager.get_and_remove(cache_key)

    if cache_hit and not request.stream:
        log(
            "info",
            f"缓存命中: {cache_key[:8]}...",
            extra={"request_type": "non-stream", "model": request.model},
        )
        return openAI_from_Gemini(cached_response, stream=False)

    if cache_hit and request.stream:
        log(
            "info",
            f"缓存命中: {cache_key[:8]}...",
            extra={"request_type": "non-stream", "model": request.model},
        )

        chunk = openAI_stream_chunk(
            model=cached_response.model,
            content=cached_response.text,
            finish_reason="stop",
        )

        return StreamingResponse(chunk, media_type="text/event-stream")

    # 构建包含缓存键的活跃请求池键
    pool_key = f"cache:{cache_key}"

    # 查找所有使用相同缓存键的活跃任务
    active_task = active_requests_manager.get(pool_key)
    if active_task and not active_task.done():
        log(
            "info",
            f"发现相同请求的进行中任务",
            extra={"request_type": "non-stream", "model": request.model},
        )

        # 等待已有任务完成
        try:
            # 设置超时，避免无限等待
            await asyncio.wait_for(active_task, timeout=180)

            # 使用任务结果
            if active_task.done() and not active_task.cancelled():
                result = active_task.result()
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
            process_stream_request(
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
            process_request(
                chat_request=request,
                http_request=http_request,
                request_type="non-stream",
                key_manager=key_manager,
                response_cache_manager=response_cache_manager,
                active_requests_manager=active_requests_manager,
                safety_settings=safety_settings,
                safety_settings_g2=safety_settings_g2,
                cache_key=cache_key,
            )
        )

    # 将任务添加到活跃请求池
    active_requests_manager.add(pool_key, process_task)

    # 等待任务完成
    try:
        response = await process_task
        return response
    except Exception as e:
        # 如果任务失败，从活跃请求池中移除
        active_requests_manager.remove(pool_key)

        # 检查是否已有缓存的结果（可能是由另一个任务创建的）
        cached_response, cache_hit = response_cache_manager.get_and_remove(cache_key)
        if cache_hit:
            log(
                "info",
                f"任务失败但找到缓存，使用缓存结果: {cache_key[:8]}...",
                extra={"request_type": "non-stream", "model": request.model},
            )
            return cached_response

        # 发送错误信息给客户端
        raise HTTPException(status_code=500, detail=f" hajimi 服务器内部处理时发生错误")


@router.post("/vertex/chat/completions", response_model=ChatCompletionResponse)
async def vertex_chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _: None = Depends(custom_verify_password),
):
    # global current_api_key

    # --- Vertex 聊天完成处理 ---

    # 生成缓存键
    if settings.PRECISE_CACHE:
        cache_key = generate_cache_key_all(request)
    else:
        cache_key = generate_cache_key(request, 4)

    # 请求限流检查
    protect_from_abuse(
        http_request,
        settings.MAX_REQUESTS_PER_MINUTE,
        settings.MAX_REQUESTS_PER_DAY_PER_IP,
    )

    # TODO: (可选) 添加 Vertex 模型有效性验证
    # 例如:
    # available_vertex_models = await vertex_list_models()
    # if request.model not in [m['id'] for m in available_vertex_models.get("data", [])]:
    #     log('error', "无效的 Vertex 模型", extra={'model': request.model, 'status_code': 400})
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="无效的 Vertex 模型")

    # 记录 Vertex 请求信息
    log_request_type = "stream" if request.stream else "non-stream"
    vertex_log(
        "info",
        f"Vertex 请求缓存键: {cache_key[:8]}...",
        extra={"request_type": log_request_type, "model": request.model},
    )

    # 检查并获取缓存 (若命中则移除)
    cached_response, cache_hit = response_cache_manager.get_and_remove(cache_key)

    # 缓存命中 (非流式)
    if cache_hit and not request.stream:
        vertex_log(
            "info",
            f"Vertex 缓存命中: {cache_key[:8]}...",
            extra={"request_type": "non-stream", "model": request.model},
        )
        return openAI_nonstream_response(cached_response)

    # 缓存命中 (流式)
    if cache_hit and request.stream:
        vertex_log(
            "info",
            f"Vertex 缓存命中: {cache_key[:8]}...",
            extra={"request_type": "stream", "model": request.model},
        )
        chunk = openAI_stream_chunk(
            model=getattr(
                cached_response, "model", request.model
            ),  # 使用缓存中的模型或请求中的模型作为后备
            content=getattr(cached_response, "text", ""),
            finish_reason="stop",
        )
        return StreamingResponse(chunk, media_type="text/event-stream")

    # 检查相同缓存键的进行中任务
    pool_key = f"vertex_cache:{cache_key}"  # 使用 'vertex_cache:' 前缀区分
    active_task = active_requests_manager.get(pool_key)
    if active_task and not active_task.done():
        vertex_log(
            "info",
            f"发现相同 Vertex 请求的进行中任务: {pool_key}",
            extra={"request_type": log_request_type, "model": request.model},
        )
        try:
            # 等待进行中的相同任务完成
            await asyncio.wait_for(active_task, timeout=180)
            if active_task.done() and not active_task.cancelled():
                result = active_task.result()
                if result:
                    return result  # 使用已完成任务的结果
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            # 等待任务超时或被取消
            error_type = "超时" if isinstance(e, asyncio.TimeoutError) else "被取消"
            vertex_log(
                "warning",
                f"等待已有 Vertex 任务{error_type}: {pool_key}",
                extra={"request_type": log_request_type, "model": request.model},
            )
            # 从活跃请求池移除已结束的任务
            if active_task.done() or active_task.cancelled():
                active_requests_manager.remove(pool_key)
                vertex_log(
                    "info",
                    f"已从活跃请求池移除{error_type} Vertex 任务: {pool_key}",
                    extra={"request_type": log_request_type},
                )
        # 若等待失败或无有效结果，则继续创建新任务

    # 创建 Vertex 处理任务
    if request.stream:
        process_task = asyncio.create_task(
            process_vertex_stream_request(
                chat_request=request,
                http_request=http_request,
                response_cache_manager=response_cache_manager,
                active_requests_manager=active_requests_manager,
                cache_key=cache_key,
                safety_settings=safety_settings,
                safety_settings_g2=safety_settings_g2,
            )
        )
    else:
        process_task = asyncio.create_task(
            process_vertex_request(
                chat_request=request,
                http_request=http_request,
                request_type="non-stream",
                response_cache_manager=response_cache_manager,
                active_requests_manager=active_requests_manager,
                cache_key=cache_key,
                safety_settings=safety_settings,
                safety_settings_g2=safety_settings_g2,
            )
        )

    # 将处理任务添加到活跃请求池
    active_requests_manager.add(pool_key, process_task)

    # 等待处理任务完成
    try:
        response = await process_task
        return response
    except Exception as e:
        # 处理任务失败，从活跃请求池移除
        active_requests_manager.remove(pool_key)
        vertex_log(
            "error",
            f"Vertex 请求处理时发生错误: {str(e)}",
            extra={
                "request_type": log_request_type,
                "model": request.model,
                "cache_key": cache_key[:8],
            },
        )

        # 任务失败后，再次检查缓存 (可能由其他并发任务写入)
        cached_response, cache_hit = response_cache_manager.get_and_remove(cache_key)
        if cache_hit:
            vertex_log(
                "info",
                f"Vertex 任务失败但找到缓存，使用缓存结果: {cache_key[:8]}...",
                extra={"request_type": log_request_type, "model": request.model},
            )
            if request.stream:
                chunk = openAI_stream_chunk(
                    model=getattr(cached_response, "model", request.model),
                    content=getattr(cached_response, "text", ""),
                    finish_reason="stop",
                )
                return StreamingResponse(chunk, media_type="text/event-stream")
            else:
                return openAI_nonstream_response(cached_response)

        # 无缓存且任务失败，返回错误
        raise HTTPException(
            status_code=500, detail=f"Vertex 服务器内部处理时发生错误: {str(e)}"
        )
    # --- 结束 Vertex 聊天完成处理 ---


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    _: None = Depends(custom_verify_password),
):
    """处理API请求的主函数，根据需要处理流式或非流式请求"""
    if settings.ENABLE_VERTEX:
        return await vertex_chat_completions(request, http_request, _)
    return await aistudio_chat_completions(request, http_request, _)
