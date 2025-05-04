from datetime import datetime
from app.utils.logging import log
from app.config.settings import stats_lock


def clean_expired_stats(api_call_stats):
    """清理过期统计数据的函数"""
    now = datetime.now()

    # 清理24小时前的调用记录
    api_call_stats["calls"] = [
        call
        for call in api_call_stats["calls"]
        if (now - call["timestamp"]).total_seconds() <= 24 * 3600  # 保留24小时内的记录
    ]


async def update_api_call_stats(api_call_stats, endpoint=None, model=None, token=None):
    """
    更新API调用统计的函数

    参数:
    - api_call_stats: 统计数据字典
    - endpoint: API端点（API密钥）
    - model: 模型名称
    - token: token数量
    """
    # 使用异步锁
    async with stats_lock:
        now = datetime.now()

        # 如果提供了端点和模型，记录调用信息
        if endpoint and model:
            # 创建调用记录
            call_record = {
                "api_key": endpoint,
                "model": model,
                "timestamp": now,
                "tokens": token if token is not None else 0,
            }

            # 添加到调用列表
            api_call_stats["calls"].append(call_record)

            # 记录日志
            log_message = f"API调用已记录: 秘钥 '{endpoint[:8]}', 模型 '{model}', 令牌: {token if token is not None else 0}"
            log("info", log_message)
        else:
            log("warning", "API调用记录不完整: 缺少秘钥或模型信息")


async def get_api_key_usage(api_call_stats, api_key, model=None):
    """
    获取API密钥的调用次数

    参数:
    - api_call_stats: 统计数据字典
    - api_key: API密钥
    - model: 模型名称，如果为None则统计所有模型的调用次数

    返回:
    - 24小时内的调用次数
    """
    # 使用异步锁保护并发访问
    async with stats_lock:
        # 检查并清理过期统计
        clean_expired_stats(api_call_stats)

        # 如果提供了模型，则只统计该模型的调用次数
        if model:
            # 统计指定模型的调用次数
            usage = sum(
                1
                for call in api_call_stats["calls"]
                if call["api_key"] == api_key and call["model"] == model
            )
            return usage
        else:
            # 统计所有模型的调用次数
            total_usage = sum(
                1 for call in api_call_stats["calls"] if call["api_key"] == api_key
            )
            return total_usage
