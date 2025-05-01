import json
import time
from app.utils.logging import log


def openAI_stream_chunk(
    model="gemini", content=None, finish_reason=None, total_token_count=0
):
    """
    创建 OpenAI 流式标准响应对象块 (SSE 格式)
    """

    now_time = int(time.time())
    formatted_chunk = {
        "id": f"chatcmpl-{now_time}",
        "object": "chat.completion.chunk",
        "created": now_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }

    if finish_reason:
        formatted_chunk["usage"] = {"total_tokens": total_token_count}

    return f"data: {json.dumps(formatted_chunk, ensure_ascii=False)}\n\n"


# 已废弃，将在后续版本中删除
def openAI_nonstream_response(response):
    """
    使用 gemini 非流式响应对象(提取后),
    创建 OpenAI 非流式标准响应对象
    """
    return {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.prompt_token_count,
            "completion_tokens": response.candidates_token_count,
            "total_tokens": response.total_token_count,
        },
    }


def openAI_from_Gemini(response, stream=True):
    """
    根据 GeminiResponseWrapper 对象创建 OpenAI 流式标准响应对象块 (SSE 格式)。

    Args:
        response: GeminiResponseWrapper 对象，包含响应数据。

    Returns:
        格式化后的 SSE 字符串 ("data: {...}\n\n")。
    """
    now_time = int(time.time())
    chunk_id = f"chatcmpl-{now_time}"  # 使用时间戳生成唯一 ID

    formatted_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": now_time,
        "model": response.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": response.finish_reason}],
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

        formatted_chunk["choices"][0]["delta"] = {
            "role": "assistant",
            "content": None,  # 函数调用时 content 为 null
            "tool_calls": tool_calls,
        }
    elif response.text:
        # 处理普通文本响应
        formatted_chunk["choices"][0]["delta"] = {
            "role": "assistant",
            "content": response.text,
        }

    if response.finish_reason == "STOP":
        formatted_chunk["usage"] = {
            "prompt_tokens": response.prompt_token_count,
            "completion_tokens": response.candidates_token_count,
            "total_tokens": response.total_token_count,
        }

    if stream:
        # 返回 SSE 格式的流式块
        return f"data: {json.dumps(formatted_chunk, ensure_ascii=False)}\n\n"
    else:
        # 构建非流式响应
        formatted_chunk["object"] = "chat.completion"
        # 将 'delta' 键重命名为 'message'
        formatted_chunk["choices"][0]["message"] = formatted_chunk["choices"][0].pop(
            "delta"
        )
        return formatted_chunk
