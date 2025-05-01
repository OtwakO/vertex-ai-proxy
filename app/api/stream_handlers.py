import asyncio
import json
import time
import random

# from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from app.models import ChatCompletionRequest
from app.services import GeminiClient, OpenAIClient
from app.utils import (
    handle_gemini_error,
    update_api_call_stats,
    log,
    openAI_stream_chunk,
)
from app.utils.response import openAI_from_Gemini
from app.utils.stats import get_api_key_usage
import app.config.settings as settings


# 流式请求处理函数
async def process_stream_request(
    chat_request: ChatCompletionRequest,
    key_manager,
    response_cache_manager,
    safety_settings,
    safety_settings_g2,
    cache_key: str,
) -> StreamingResponse:
    """处理流式API请求"""

    # 创建一个直接流式响应的生成器函数
    async def stream_response_generator():
        # 转换消息格式
        contents, system_instruction = GeminiClient.convert_messages(
            GeminiClient, chat_request.messages, model=chat_request.model
        )

        # 重置已尝试的密钥
        key_manager.reset_tried_keys_for_request()

        # 获取所有可用的API密钥
        all_keys = key_manager.api_keys.copy()
        random.shuffle(all_keys)  # 随机打乱密钥顺序

        # 检查每个API密钥的调用次数，过滤掉超过限制的密钥
        valid_keys = []
        for api_key in all_keys:
            # 获取API密钥的调用次数
            usage = await get_api_key_usage(settings.api_call_stats, api_key)
            # 如果调用次数小于限制，则添加到有效密钥列表
            if usage < settings.API_KEY_DAILY_LIMIT:
                valid_keys.append(api_key)
            else:
                log(
                    "warning",
                    f"API密钥 {api_key[:8]}... 已达到每日调用限制 ({usage}/{settings.API_KEY_DAILY_LIMIT})",
                    extra={
                        "key": api_key[:8],
                        "request_type": "stream",
                        "model": chat_request.model,
                    },
                )

        # 如果没有有效密钥，则随机使用一个密钥
        if not valid_keys:
            log(
                "warning",
                "所有API密钥已达到每日调用限制，将随机使用一个密钥",
                extra={"request_type": "stream", "model": chat_request.model},
            )
            valid_keys = [random.choice(all_keys)]

        # 设置初始并发数
        current_concurrent = settings.CONCURRENT_REQUESTS
        max_retry_num = settings.MAX_RETRY_NUM

        # 如果可用密钥数量小于并发数，则使用所有可用密钥
        if len(valid_keys) < current_concurrent:
            current_concurrent = len(valid_keys)

        # 当前请求次数
        current_try_num = 0

        # 空响应计数
        empty_response_count = 0

        # (假流式) 尝试使用不同API密钥，直到所有密钥都尝试过
        while (
            valid_keys
            and settings.FAKE_STREAMING
            and (current_try_num < max_retry_num)
            and (empty_response_count < settings.MAX_EMPTY_RESPONSES)
        ):
            # 获取当前批次的密钥
            batch_num = min(max_retry_num - current_try_num, current_concurrent)

            current_batch = valid_keys[:batch_num]
            valid_keys = valid_keys[batch_num:]

            # 创建并发任务
            tasks = []
            tasks_map = {}
            for api_key in current_batch:
                # 假流式模式的处理逻辑
                log(
                    "info",
                    f"假流式请求开始，使用密钥: {api_key[:8]}...",
                    extra={
                        "key": api_key[:8],
                        "request_type": "fake-stream",
                        "model": chat_request.model,
                    },
                )

                task = asyncio.create_task(
                    handle_fake_streaming(
                        api_key,
                        chat_request,
                        contents,
                        response_cache_manager,
                        system_instruction,
                        safety_settings,
                        safety_settings_g2,
                        cache_key,
                    )
                )

                tasks.append((api_key, task))
                tasks_map[task] = api_key

            # 等待所有任务完成或找到成功响应
            success = False
            while tasks and not success:
                # 等待任务完成
                done, pending = await asyncio.wait(
                    [task for _, task in tasks],
                    timeout=settings.FAKE_STREAMING_INTERVAL,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # 如果没有任务完成，发送保活消息
                if not done:
                    yield openAI_stream_chunk(model=chat_request.model, content="")
                    continue

                # 检查已完成的任务是否成功
                for task in done:
                    api_key = tasks_map[task]
                    if not task.cancelled():
                        try:
                            status = task.result()
                            # 如果有成功响应内容
                            if status == "success":
                                success = True
                                log(
                                    "info",
                                    f"假流式请求成功",
                                    extra={
                                        "request_type": "fake-stream",
                                        "model": chat_request.model,
                                    },
                                )

                                cached_response, cache_hit = (
                                    response_cache_manager.get_and_remove(cache_key)
                                )

                                yield openAI_stream_chunk(
                                    model=chat_request.model,
                                    content=cached_response.text,
                                    finish_reason="stop",
                                )

                                break
                            elif status == "empty":
                                # 增加空响应计数
                                empty_response_count += 1
                                log(
                                    "warning",
                                    f"空响应计数: {empty_response_count}/{settings.MAX_EMPTY_RESPONSES}",
                                    extra={
                                        "key": api_key[:8],
                                        "request_type": "stream",
                                        "model": chat_request.model,
                                    },
                                )

                        except Exception as e:
                            error_detail = handle_gemini_error(e, api_key)
                            log(
                                "error",
                                f"请求失败: {error_detail}",
                                extra={
                                    "key": api_key[:8],
                                    "request_type": "stream",
                                    "model": chat_request.model,
                                },
                            )

                # 如果找到成功的响应，跳出循环
                if success:
                    return

                # 如果空响应次数达到限制，跳出循环
                if empty_response_count >= settings.MAX_EMPTY_RESPONSES:
                    log(
                        "warning",
                        f"空响应次数达到限制 ({empty_response_count}/{settings.MAX_EMPTY_RESPONSES})，停止轮询",
                        extra={"request_type": "stream", "model": chat_request.model},
                    )
                    break

                # 更新任务列表，移除已完成的任务
                tasks = [(k, t) for k, t in tasks if not t.done()]

            # 如果所有请求都失败或返回空响应，增加并发数并继续尝试
            if not success and valid_keys:
                # 增加并发数，但不超过最大并发数
                current_concurrent = min(
                    current_concurrent + settings.INCREASE_CONCURRENT_ON_FAILURE,
                    settings.MAX_CONCURRENT_REQUESTS,
                )
                log(
                    "info",
                    f"所有假流式请求失败或返回空响应，增加并发数至: {current_concurrent}",
                    extra={"request_type": "stream", "model": chat_request.model},
                )

        # (真流式) 尝试使用不同API密钥，直到所有密钥都尝试过或达到尝试上限
        while (
            valid_keys
            and not settings.FAKE_STREAMING
            and (current_try_num < max_retry_num)
            and (empty_response_count < settings.MAX_EMPTY_RESPONSES)
        ):
            # 获取密钥
            api_key = valid_keys[0]
            valid_keys = valid_keys[1:]
            current_try_num += 1

            success = False
            try:
                client = GeminiClient(api_key)

                # 获取流式响应
                stream_generator = client.stream_chat(
                    chat_request,
                    contents,
                    safety_settings_g2
                    if "gemini-2.5-pro" in chat_request.model
                    else safety_settings,
                    system_instruction,
                )
                token = 0
                # 处理流式响应
                async for chunk in stream_generator:
                    if chunk:
                        if chunk.total_token_count:
                            token += int(chunk.total_token_count)
                        success = True
                        data = openAI_from_Gemini(chunk, stream=True)
                        # log('info', f"流式响应发送数据: {data}")
                        yield data

                    else:
                        log(
                            "warning",
                            f"流式响应: API密钥 {api_key[:8]}... 返回空响应",
                            extra={
                                "key": api_key[:8],
                                "request_type": "stream",
                                "model": chat_request.model,
                            },
                        )
                        # 增加空响应计数
                        empty_response_count += 1
                        log(
                            "warning",
                            f"空响应计数: {empty_response_count}/{settings.MAX_EMPTY_RESPONSES}",
                            extra={
                                "key": api_key[:8],
                                "request_type": "stream",
                                "model": chat_request.model,
                            },
                        )
                        break

            except Exception as e:
                error_detail = handle_gemini_error(e, api_key)
                log(
                    "error",
                    f"流式响应: API密钥 {api_key[:8]}... 请求失败: {error_detail}",
                    extra={
                        "key": api_key[:8],
                        "request_type": "stream",
                        "model": chat_request.model,
                    },
                )
                return
            finally:
                # 更新API调用统计
                if success:
                    await update_api_call_stats(
                        settings.api_call_stats,
                        endpoint=api_key,
                        model=chat_request.model,
                        token=token,  # 添加token参数
                    )
                    return

                # 如果空响应次数达到限制，跳出循环
                if empty_response_count >= settings.MAX_EMPTY_RESPONSES:
                    log(
                        "warning",
                        f"空响应次数达到限制 ({empty_response_count}/{settings.MAX_EMPTY_RESPONSES})，停止轮询",
                        extra={"request_type": "stream", "model": chat_request.model},
                    )
                    break

        # 所有API密钥都尝试失败的处理
        log(
            "error",
            "所有API密钥均请求失败，请稍后重试",
            extra={"key": "ALL", "request_type": "stream", "model": chat_request.model},
        )

        # 发送错误信息给客户端
        yield openAI_stream_chunk(
            model=chat_request.model,
            content="所有API密钥均请求失败，请稍后重试",
            finish_reason="stop",
        )

    # 处理假流式模式
    async def handle_fake_streaming(
        api_key,
        chat_request,
        contents,
        response_cache_manager,
        system_instruction,
        safety_settings,
        safety_settings_g2,
        cache_key,
    ):
        # 使用非流式请求内容
        gemini_client = GeminiClient(api_key)

        api_call_future = asyncio.create_task(
            asyncio.to_thread(
                gemini_client.complete_chat,
                chat_request,
                contents,
                safety_settings_g2
                if "gemini-2.5-pro" in chat_request.model
                else safety_settings,
                system_instruction,
            )
        )
        gemini_task = asyncio.shield(api_call_future)

        try:
            # 获取响应内容
            response_content = await gemini_task

            log(
                "info",
                f"假流式成功获取响应，进行缓存",
                extra={
                    "key": api_key[:8],
                    "request_type": "fake-stream",
                    "model": chat_request.model,
                },
            )

            # 更新API调用统计
            await update_api_call_stats(
                settings.api_call_stats,
                endpoint=api_key,
                model=chat_request.model,
                token=response_content.total_token_count,
            )

            # 检查响应内容是否为空
            if not response_content or not response_content.text:
                log(
                    "warning",
                    f"请求返回空响应",
                    extra={
                        "key": api_key[:8],
                        "request_type": "fake-stream",
                        "model": chat_request.model,
                    },
                )
                return "empty"

            # 缓存
            response_cache_manager.store(cache_key, response_content)

            return "success"

        except Exception as e:
            handle_gemini_error(e, api_key)
            # log('error', f"假流式模式: API密钥 {api_key[:8]}... 请求失败: {error_detail}",
            #     extra={'key': api_key[:8], 'request_type': 'fake-stream', 'model': chat_request.model})
            return "error"

    return StreamingResponse(
        stream_response_generator(), media_type="text/event-stream"
    )
