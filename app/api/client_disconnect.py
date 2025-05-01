import asyncio
from fastapi import Request, HTTPException
from app.models import ChatCompletionRequest
from app.utils import update_api_call_stats

from app.utils.logging import log
import app.config.settings as settings


# 客户端断开检测函数
async def check_client_disconnect(
    http_request: Request, current_api_key: str, request_type: str, model: str
):
    """检查客户端是否断开连接"""
    while True:
        if await http_request.is_disconnected():
            # extra_log = {'key': current_api_key[:8], 'request_type': request_type, 'model': model}
            # log('info', "客户端连接已中断，等待API请求完成", extra=extra_log)
            return True
        await asyncio.sleep(0.5)


# 客户端断开处理函数
async def handle_client_disconnect(
    gemini_task: asyncio.Task,
    chat_request: ChatCompletionRequest,
    request_type: str,
    current_api_key: str,
    response_cache_manager,
    cache_key: str,
    model: str,
    key: str,
):
    try:
        # 等待API任务完成，使用shield防止它被取消
        response_content = await asyncio.shield(gemini_task)

        # 更新API调用统计

        # 检查响应文本是否为空
        if response_content is None or response_content.text == "":
            log(
                "info",
                "客户端断开后 Gemini API 返回空响应，不进行缓存",
                extra={
                    "key": current_api_key[:8],
                    "request_type": request_type,
                    "model": chat_request.model,
                },
            )

            return False
        await update_api_call_stats(
            settings.api_call_stats, key, model, response_content.total_token_count
        )

        # 存入缓存 (追加到deque)
        response_cache_manager.store(cache_key, response_content)
        log(
            "info",
            f"请求成功完成，缓存响应",
            extra={
                "key": current_api_key[:8],
                "request_type": request_type,
                "model": chat_request.model,
            },
        )

        return True
    # except asyncio.CancelledError:
    #     # 对于取消异常，仍然尝试继续完成任务
    #     log('info', "客户端断开后任务被取消，但我们仍会尝试完成",
    #         extra={'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model})

    #     # 检查任务是否已经完成
    #     if gemini_task.done() and not gemini_task.cancelled():
    #         try:
    #             response_content = gemini_task.result()

    #             # 进行缓存
    #             # response_cache_manager.store(cache_key, response)

    #             # 更新API调用统计
    #             update_api_call_stats(settings.api_call_stats,key,model)

    #             return response
    #         except Exception as inner_e:
    #             log('error', f"客户端断开后从已完成任务获取结果失败: {str(inner_e)}",
    #                 extra={'key': current_api_key[:8], 'request_type': request_type, 'model': chat_request.model})

    #             # # 删除缓存，因为出现错误
    #             # if cache_key and cache_key in response_cache_manager.cache:
    #             #     log('info', f"因任务获取结果失败，删除缓存: {cache_key[:8]}...",
    #             #         extra={'cache_operation': 'remove-on-error', 'request_type': request_type})
    #             #     del response_cache_manager.cache[cache_key]

    #     return None
    except Exception as e:
        # 处理API任务异常
        error_msg = str(e[:10])
        extra_log = {
            "key": current_api_key[:8],
            "request_type": request_type,
            "model": chat_request.model,
            "error_message": error_msg,
        }
        log("error", f"客户端断开后处理API响应时出错: {error_msg}", extra=extra_log)

        # 向客户端抛出异常
        raise HTTPException(status_code=500, detail="hajimi 服务器内部处理时发生错误")
