import random
import re
import os
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from app.utils.logging import format_log_message
import app.config.settings as settings

logger = logging.getLogger("my_logger")


class APIKeyManager:
    def __init__(self):
        self.api_keys = re.findall(r"AIzaSy[a-zA-Z0-9_-]{33}", settings.GEMINI_API_KEYS)
        # 加载更多 GEMINI_API_KEYS
        for i in range(1, 99):
            if keys := os.environ.get(f"GEMINI_API_KEYS_{i}", ""):
                self.api_keys += re.findall(r"AIzaSy[a-zA-Z0-9_-]{33}", keys)
            else:
                break

        self.key_stack = []  # 初始化密钥栈
        self._reset_key_stack()  # 初始化时创建随机密钥栈
        # self.api_key_blacklist = set()
        # self.api_key_blacklist_duration = 60
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self.tried_keys_for_request = set()  # 用于跟踪当前请求尝试中已试过的 key

    def _reset_key_stack(self):
        """创建并随机化密钥栈"""
        shuffled_keys = self.api_keys[:]  # 创建 api_keys 的副本以避免直接修改原列表
        random.shuffle(shuffled_keys)
        self.key_stack = shuffled_keys

    def get_available_key(self):
        """从栈顶获取密钥，若栈空则重新生成"""
        while self.key_stack:
            key = self.key_stack.pop()
            # if key not in self.api_key_blacklist and key not in self.tried_keys_for_request:
            if key not in self.tried_keys_for_request:
                self.tried_keys_for_request.add(key)
                return key

        # 栈空，重新生成密钥栈
        self._reset_key_stack()

        # 再次尝试从新栈中获取密钥 (迭代一次)
        while self.key_stack:
            key = self.key_stack.pop()
            # if key not in self.api_key_blacklist and key not in self.tried_keys_for_request:
            if key not in self.tried_keys_for_request:
                self.tried_keys_for_request.add(key)
                return key

        if not self.api_keys:
            log_msg = format_log_message("ERROR", "没有配置任何 API 密钥！")
            logger.error(log_msg)
            return None

        return None

    def show_all_keys(self):
        log_msg = format_log_message(
            "INFO", f"当前可用API key个数: {len(self.api_keys)} "
        )
        logger.info(log_msg)
        for i, api_key in enumerate(self.api_keys):
            log_msg = format_log_message(
                "INFO", f"API Key{i}: {api_key[:8]}...{api_key[-3:]}"
            )
            logger.info(log_msg)

    # def blacklist_key(self, key):
    #     log_msg = format_log_message('WARNING', f"{key[:8]} → 暂时禁用 {self.api_key_blacklist_duration} 秒")
    #     logger.warning(log_msg)
    #     self.api_key_blacklist.add(key)
    #     self.scheduler.add_job(lambda: self.api_key_blacklist.discard(key), 'date',
    #                            run_date=datetime.now() + timedelta(seconds=self.api_key_blacklist_duration))

    def reset_tried_keys_for_request(self):
        """在新的请求尝试时重置已尝试的 key 集合"""
        self.tried_keys_for_request = set()


async def test_api_key(api_key: str) -> bool:
    """
    测试 API 密钥是否有效。
    """
    try:
        import httpx

        url = "https://generativelanguage.googleapis.com/v1beta/models?key={}".format(
            api_key
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return True
    except Exception:
        return False
