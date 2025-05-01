from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from app.utils import (
    log_manager,
    clean_expired_stats,
)
import app.config.settings as settings
from app.services import GeminiClient
from app.utils.auth import verify_web_password
from app.utils.maintenance import api_call_stats_clean
from app.utils.logging import log, vertex_log_manager
from app.config.persistence import save_settings

# 创建路由器
dashboard_router = APIRouter(prefix="/api", tags=["dashboard"])

# 全局变量引用，将在init_dashboard_router中设置
key_manager = None
response_cache_manager = None
active_requests_manager = None


def init_dashboard_router(key_mgr, cache_mgr, active_req_mgr):
    """初始化仪表盘路由器"""
    global key_manager, response_cache_manager, active_requests_manager
    key_manager = key_mgr
    response_cache_manager = cache_mgr
    active_requests_manager = active_req_mgr
    return dashboard_router


@dashboard_router.get("/dashboard-data")
async def get_dashboard_data():
    """获取仪表盘数据的API端点，用于动态刷新"""
    # 先清理过期数据，确保统计数据是最新的
    clean_expired_stats(settings.api_call_stats)
    response_cache_manager.clean_expired()  # 使用管理器清理缓存
    active_requests_manager.clean_completed()  # 使用管理器清理活跃请求

    # 获取当前统计数据
    now = datetime.now()

    # 计算过去24小时的调用总数
    last_24h_calls = len(settings.api_call_stats["calls"])

    # 计算过去一小时内的调用总数
    one_hour_ago = now - timedelta(hours=1)
    hourly_calls = sum(
        1
        for call in settings.api_call_stats["calls"]
        if call["timestamp"] >= one_hour_ago
    )

    # 计算过去一分钟内的调用总数
    one_minute_ago = now - timedelta(minutes=1)
    minute_calls = sum(
        1
        for call in settings.api_call_stats["calls"]
        if call["timestamp"] >= one_minute_ago
    )

    # 计算时间序列数据（过去30分钟，每分钟一个数据点）
    time_series_data = []
    tokens_time_series = []

    # 过去30分钟的每分钟API调用统计
    for i in range(30, -1, -1):
        minute_start = now - timedelta(minutes=i)
        minute_end = now - timedelta(minutes=i - 1) if i > 0 else now

        # 这一分钟的调用次数
        calls_in_minute = sum(
            1
            for call in settings.api_call_stats["calls"]
            if minute_start <= call["timestamp"] < minute_end
        )

        # 这一分钟的token使用量
        tokens_in_minute = sum(
            call["tokens"]
            for call in settings.api_call_stats["calls"]
            if minute_start <= call["timestamp"] < minute_end
        )

        # 添加到时间序列
        time_series_data.append(
            {"time": minute_start.strftime("%H:%M"), "value": calls_in_minute}
        )

        tokens_time_series.append(
            {"time": minute_start.strftime("%H:%M"), "value": tokens_in_minute}
        )

    # 获取API密钥使用统计
    api_key_stats = []
    for api_key in key_manager.api_keys:
        # 获取API密钥前8位作为标识
        api_key_id = api_key[:8]

        # 计算24小时内的调用次数、token数和按模型分类的统计
        calls_24h = 0
        total_tokens = 0
        model_stats = {}

        # 筛选该API密钥的调用记录
        api_key_calls = [
            call
            for call in settings.api_call_stats["calls"]
            if call["api_key"] == api_key
        ]

        # 按模型分类统计
        for call in api_key_calls:
            model = call["model"]
            tokens = call["tokens"]

            calls_24h += 1
            total_tokens += tokens

            if model not in model_stats:
                model_stats[model] = {"calls": 0, "tokens": 0}

            model_stats[model]["calls"] += 1
            model_stats[model]["tokens"] += tokens

        # 计算使用百分比
        usage_percent = (
            (calls_24h / settings.API_KEY_DAILY_LIMIT) * 100
            if settings.API_KEY_DAILY_LIMIT > 0
            else 0
        )

        # 添加到结果列表
        api_key_stats.append(
            {
                "api_key": api_key_id,
                "calls_24h": calls_24h,
                "total_tokens": total_tokens,
                "limit": settings.API_KEY_DAILY_LIMIT,
                "usage_percent": round(usage_percent, 2),
                "model_stats": model_stats,  # 现在包含调用次数和token数
            }
        )

    # 按使用百分比降序排序
    api_key_stats.sort(key=lambda x: x["usage_percent"], reverse=True)

    # 根据ENABLE_VERTEX设置决定返回哪种日志
    if settings.ENABLE_VERTEX:
        recent_logs = vertex_log_manager.get_recent_logs(500)  # 获取最近500条Vertex日志
    else:
        recent_logs = log_manager.get_recent_logs(500)  # 获取最近500条普通日志

    # 获取缓存统计
    total_cache = response_cache_manager.cur_cache_num

    # 获取活跃请求统计
    active_count = len(active_requests_manager.active_requests)
    active_done = sum(
        1 for task in active_requests_manager.active_requests.values() if task.done()
    )
    active_pending = active_count - active_done

    # 返回JSON格式的数据
    return {
        "key_count": len(key_manager.api_keys),
        "model_count": len(GeminiClient.AVAILABLE_MODELS),
        "retry_count": settings.MAX_RETRY_NUM,
        "last_24h_calls": last_24h_calls,
        "hourly_calls": hourly_calls,
        "minute_calls": minute_calls,
        "calls_time_series": time_series_data,  # 添加API调用时间序列
        "tokens_time_series": tokens_time_series,  # 添加Token使用时间序列
        "current_time": datetime.now().strftime("%H:%M:%S"),
        "logs": recent_logs,
        "api_key_stats": api_key_stats,
        # 添加配置信息
        "max_requests_per_minute": settings.MAX_REQUESTS_PER_MINUTE,
        "max_requests_per_day_per_ip": settings.MAX_REQUESTS_PER_DAY_PER_IP,
        # 添加版本信息
        "local_version": settings.version["local_version"],
        "remote_version": settings.version["remote_version"],
        "has_update": settings.version["has_update"],
        # 添加流式响应配置
        "fake_streaming": settings.FAKE_STREAMING,
        "fake_streaming_interval": settings.FAKE_STREAMING_INTERVAL,
        # 添加随机字符串配置
        "random_string": settings.RANDOM_STRING,
        "random_string_length": settings.RANDOM_STRING_LENGTH,
        # 添加联网搜索配置
        "search_mode": settings.search["search_mode"],
        "search_prompt": settings.search["search_prompt"],
        # 添加缓存信息
        "cache_entries": total_cache,
        "cache_expiry_time": settings.CACHE_EXPIRY_TIME,
        "max_cache_entries": settings.MAX_CACHE_ENTRIES,
        # 添加活跃请求池信息
        "active_count": active_count,
        "active_done": active_done,
        "active_pending": active_pending,
        # 添加并发请求配置
        "concurrent_requests": settings.CONCURRENT_REQUESTS,
        "increase_concurrent_on_failure": settings.INCREASE_CONCURRENT_ON_FAILURE,
        "max_concurrent_requests": settings.MAX_CONCURRENT_REQUESTS,
        # 启用vertex
        "enable_vertex": settings.ENABLE_VERTEX,
    }


@dashboard_router.post("/reset-stats")
async def reset_stats(password_data: dict):
    """
    重置API调用统计数据

    Args:
        password_data (dict): 包含密码的字典

    Returns:
        dict: 操作结果
    """
    try:
        if not isinstance(password_data, dict):
            raise HTTPException(status_code=422, detail="请求体格式错误：应为JSON对象")

        password = password_data.get("password")
        if not password:
            raise HTTPException(status_code=400, detail="缺少密码参数")

        if not isinstance(password, str):
            raise HTTPException(status_code=422, detail="密码参数类型错误：应为字符串")

        if not verify_web_password(password):
            raise HTTPException(status_code=401, detail="密码错误")

        # 调用重置函数
        await api_call_stats_clean()

        return {"status": "success", "message": "API调用统计数据已重置"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置失败：{str(e)}")


@dashboard_router.post("/update-config")
async def update_config(config_data: dict):
    """
    更新配置项

    Args:
        config_data (dict): 包含配置项和密码的字典

    Returns:
        dict: 操作结果
    """
    try:
        if not isinstance(config_data, dict):
            raise HTTPException(status_code=422, detail="请求体格式错误：应为JSON对象")

        password = config_data.get("password")
        if not password:
            raise HTTPException(status_code=400, detail="缺少密码参数")

        if not isinstance(password, str):
            raise HTTPException(status_code=422, detail="密码参数类型错误：应为字符串")

        if not verify_web_password(password):
            raise HTTPException(status_code=401, detail="密码错误")

        # 获取要更新的配置项
        config_key = config_data.get("key")
        config_value = config_data.get("value")

        if not config_key:
            raise HTTPException(status_code=400, detail="缺少配置项键名")

        # 根据配置项类型进行类型转换和验证
        if config_key == "max_requests_per_minute":
            try:
                value = int(config_value)
                if value <= 0:
                    raise ValueError("每分钟请求限制必须大于0")
                settings.MAX_REQUESTS_PER_MINUTE = value
                log("info", f"每分钟请求限制已更新为：{value}")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"参数类型错误：{str(e)}")

        elif config_key == "max_requests_per_day_per_ip":
            try:
                value = int(config_value)
                if value <= 0:
                    raise ValueError("每IP每日请求限制必须大于0")
                settings.MAX_REQUESTS_PER_DAY_PER_IP = value
                log("info", f"每IP每日请求限制已更新为：{value}")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"参数类型错误：{str(e)}")

        elif config_key == "fake_streaming":
            if not isinstance(config_value, bool):
                raise HTTPException(status_code=422, detail="参数类型错误：应为布尔值")
            settings.FAKE_STREAMING = config_value
            log("info", f"假流式请求已更新为：{config_value}")
        elif config_key == "fake_streaming_interval":
            try:
                value = float(config_value)
                if value <= 0:
                    raise ValueError("假流式间隔必须大于0")
                settings.FAKE_STREAMING_INTERVAL = value
                log("info", f"假流式间隔已更新为：{value}")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"参数类型错误：{str(e)}")

        elif config_key == "random_string":
            if not isinstance(config_value, bool):
                raise HTTPException(status_code=422, detail="参数类型错误：应为布尔值")
            settings.RANDOM_STRING = config_value
            log("info", f"随机字符串已更新为：{config_value}")
        elif config_key == "random_string_length":
            try:
                value = int(config_value)
                if value <= 0:
                    raise ValueError("随机字符串长度必须大于0")
                settings.RANDOM_STRING_LENGTH = value
                log("info", f"随机字符串长度已更新为：{value}")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"参数类型错误：{str(e)}")

        elif config_key == "search_mode":
            if not isinstance(config_value, bool):
                raise HTTPException(status_code=422, detail="参数类型错误：应为布尔值")
            settings.search["search_mode"] = config_value
            log("info", f"联网搜索模式已更新为：{config_value}")

            # 在切换search_mode时，重新获取一次可用模型列表
            try:
                # 重置密钥栈以确保随机性
                key_manager._reset_key_stack()
                # 获取一个随机API密钥
                for key in key_manager.api_keys:
                    log("info", f"使用API密钥 {key[:8]}... 刷新可用模型列表")
                    # 使用随机密钥获取可用模型
                    all_models = await GeminiClient.list_available_models(key)
                    GeminiClient.AVAILABLE_MODELS = [
                        model.replace("models/", "") for model in all_models
                    ]
                    if len(GeminiClient.AVAILABLE_MODELS) > 0:
                        log(
                            "info",
                            f"可用模型列表已更新，当前模型数量：{len(GeminiClient.AVAILABLE_MODELS)}",
                        )
                        break
                else:
                    log("warning", "没有可用的API密钥，无法刷新可用模型列表")
            except Exception as e:
                log("warning", f"刷新可用模型列表时发生错误: {str(e)}")

        elif config_key == "concurrent_requests":
            try:
                value = int(config_value)
                if value <= 0:
                    raise ValueError("并发请求数必须大于0")
                settings.CONCURRENT_REQUESTS = value
                log("info", f"并发请求数已更新为：{value}")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"参数类型错误：{str(e)}")

        elif config_key == "increase_concurrent_on_failure":
            try:
                value = int(config_value)
                if value < 0:
                    raise ValueError("失败时增加的并发数不能为负数")
                settings.INCREASE_CONCURRENT_ON_FAILURE = value
                log("info", f"失败时增加的并发数已更新为：{value}")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"参数类型错误：{str(e)}")

        elif config_key == "max_concurrent_requests":
            try:
                value = int(config_value)
                if value <= 0:
                    raise ValueError("最大并发请求数必须大于0")
                settings.MAX_CONCURRENT_REQUESTS = value
                log("info", f"最大并发请求数已更新为：{value}")
            except ValueError as e:
                raise HTTPException(status_code=422, detail=f"参数类型错误：{str(e)}")

        elif config_key == "enable_vertex":
            if not isinstance(config_value, bool):
                raise HTTPException(status_code=422, detail="参数类型错误：应为布尔值")
            settings.ENABLE_VERTEX = config_value
            log("info", f"Vertex AI 已更新为：{config_value}")

        elif config_key == "gemini_api_keys":
            if not isinstance(config_value, str):
                raise HTTPException(
                    status_code=422, detail="参数类型错误：API密钥应为逗号分隔的字符串"
                )

            # 分割并清理API密钥
            new_keys = [key.strip() for key in config_value.split(",") if key.strip()]
            if not new_keys:
                raise HTTPException(status_code=400, detail="未提供有效的API密钥")

            # 添加到现有的API密钥字符串中
            current_keys = (
                settings.GEMINI_API_KEYS.split(",") if settings.GEMINI_API_KEYS else []
            )
            current_keys = [key.strip() for key in current_keys if key.strip()]

            # 合并新旧密钥并去重
            all_keys = list(set(current_keys + new_keys))
            settings.GEMINI_API_KEYS = ",".join(all_keys)

            # 计算新添加的密钥数量
            added_key_count = 0
            for key in new_keys:
                if key not in key_manager.api_keys:
                    key_manager.api_keys.append(key)
                    added_key_count += 1

            # 更新MAX_RETRY_NUM
            if added_key_count > 0:
                settings.MAX_RETRY_NUM += added_key_count
                log("info", f"MAX_RETRY_NUM已更新为：{settings.MAX_RETRY_NUM}")

            # 重置密钥栈
            key_manager._reset_key_stack()

            # 如果可用模型为空，尝试获取模型列表
            if not GeminiClient.AVAILABLE_MODELS:
                try:
                    # 使用新添加的密钥之一尝试获取可用模型
                    for key in new_keys:
                        log(
                            "info", f"使用新添加的API密钥 {key[:8]}... 获取可用模型列表"
                        )
                        all_models = await GeminiClient.list_available_models(key)
                        GeminiClient.AVAILABLE_MODELS = [
                            model.replace("models/", "") for model in all_models
                        ]
                        if GeminiClient.AVAILABLE_MODELS:
                            log(
                                "info",
                                f"成功获取可用模型列表，共 {len(GeminiClient.AVAILABLE_MODELS)} 个模型",
                            )
                            break
                except Exception as e:
                    log("warning", f"获取可用模型列表时发生错误: {str(e)}")

            log(
                "info",
                f"已添加 {added_key_count} 个新API密钥，当前共有 {len(key_manager.api_keys)} 个",
            )

        else:
            raise HTTPException(status_code=400, detail=f"不支持的配置项：{config_key}")
        save_settings()
        return {"status": "success", "message": f"配置项 {config_key} 已更新"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失败：{str(e)}")
