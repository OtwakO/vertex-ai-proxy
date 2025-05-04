import json
import time


def openAI_from_text(
    model="gemini", content=None, finish_reason=None, total_token_count=0, stream=True
):
    """
    根据传入参数，创建 OpenAI 标准响应对象块
    """

    now_time = int(time.time())
    content_chunk = {}
    formatted_chunk = {
        "id": f"chatcmpl-{now_time}",
        "created": now_time,
        "model": model,
        "choices": [{"index": 0, "finish_reason": finish_reason}],
    }

    if content:
        content_chunk = {"role": "assistant", "content": content}

    if finish_reason:
        formatted_chunk["usage"] = {"total_tokens": total_token_count}

    if stream:
        formatted_chunk["choices"][0]["delta"] = content_chunk
        formatted_chunk["object"] = "chat.completion.chunk"
        return f"data: {json.dumps(formatted_chunk, ensure_ascii=False)}\n\n"
    else:
        formatted_chunk["choices"][0]["message"] = content_chunk
        formatted_chunk["object"] = "chat.completion"
        return formatted_chunk


def openAI_from_Gemini(response, stream=True):
    """
    根据 GeminiResponseWrapper 对象创建 OpenAI 标准响应对象块。

    Args:
        response: GeminiResponseWrapper 对象，包含响应数据。

    Returns:
        OpenAI 标准响应
    """
    now_time = int(time.time())
    chunk_id = f"chatcmpl-{now_time}"  # 使用时间戳生成唯一 ID
    content_chunk = {}
    formatted_chunk = {
        "id": chunk_id,
        "created": now_time,
        "model": response.model,
        "choices": [{"index": 0, "finish_reason": response.finish_reason}],
    }

    # 准备 usage 数据，使用 getattr 获取并提供默认值 0 ( API 返回 None 时使用)
    prompt_tokens = getattr(response, "prompt_token_count", 0)
    candidates_tokens = getattr(response, "candidates_token_count", 0)
    total_tokens = getattr(response, "total_token_count", 0)

    usage_data = {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(candidates_tokens),
        "total_tokens": int(total_tokens),
    }

    if response.function_call:
        tool_calls = []
        # 处理函数调用的每一部分
        for part in response.function_call:
            function_name = part.get("name")
            # Gemini 的 args 是 dict, OpenAI 需要 string
            function_args_str = json.dumps(part.get("args", {}), ensure_ascii=False)

            tool_call_id = f"call_{function_name}"  # 编码函数名到 ID
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": function_args_str,
                    },
                }
            )

        content_chunk = {
            "role": "assistant",
            "content": None,  # 函数调用时 content 为 null
            "tool_calls": tool_calls,
        }
    elif response.text:
        # 处理普通文本响应
        content_chunk = {"role": "assistant", "content": response.text}

    if stream:
        formatted_chunk["choices"][0]["delta"] = content_chunk
        formatted_chunk["object"] = "chat.completion.chunk"
        # 仅在流结束时添加 usage 字段
        if response.finish_reason:
            formatted_chunk["usage"] = usage_data
    else:
        formatted_chunk["choices"][0]["message"] = content_chunk
        formatted_chunk["object"] = "chat.completion"
        # 非流式响应总是包含 usage 字段，以满足 response_model 验证
        formatted_chunk["usage"] = usage_data

    if stream:
        return f"data: {json.dumps(formatted_chunk, ensure_ascii=False)}\n\n"
    else:
        return formatted_chunk
