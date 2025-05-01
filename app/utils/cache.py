import time
import hashlib
import json
from typing import Dict, Any, Optional, List, Tuple
import logging
from collections import deque
from app.utils.logging import log

logger = logging.getLogger("my_logger")

# 定义缓存项的结构
CacheItem = Dict[str, Any]


class ResponseCacheManager:
    """管理API响应缓存的类，一个键可以对应多个缓存项（使用deque）"""

    def __init__(
        self,
        expiry_time: int,
        max_entries: int,
        cache_dict: Dict[str, deque[CacheItem]] = None,
    ):
        """
        初始化缓存管理器。

        Args:
            expiry_time (int): 缓存项的过期时间（秒）。
            max_entries (int): 缓存中允许的最大总条目数。
            cache_dict (Dict[str, deque[CacheItem]], optional): 初始缓存字典。默认为 None。
        """
        self.cache: Dict[str, deque[CacheItem]] = (
            cache_dict if cache_dict is not None else {}
        )
        self.expiry_time = expiry_time
        self.max_entries = max_entries  # 总条目数限制
        self.cur_cache_num = 0  # 当前条目数

    def get(self, cache_key: str) -> Tuple[Optional[Any], bool]:
        """获取指定键的第一个有效缓存项（不删除）"""
        now = time.time()
        if cache_key in self.cache:
            cache_deque = self.cache[cache_key]
            # 查找第一个未过期的项
            for item in cache_deque:
                if now < item.get("expiry_time", 0):
                    return item["response"], True  # 返回响应，不删除
        return None, False

    def get_and_remove(self, cache_key: str) -> Tuple[Optional[Any], bool]:
        """获取并删除指定键的第一个有效缓存项。"""
        now = time.time()
        if cache_key in self.cache:
            cache_deque = self.cache[cache_key]
            item_to_remove = None
            response = None

            # 查找第一个未过期的项
            for item in cache_deque:
                if now < item.get("expiry_time", 0):
                    item_to_remove = item
                    response = item["response"]
                    break  # 找到第一个就停止
                else:
                    cache_deque.remove(item)

            if item_to_remove:
                try:
                    cache_deque.remove(item_to_remove)  # 从deque中删除
                    # log('info', f"从缓存获取并删除项: {cache_key[:8]}...")
                    self.cur_cache_num -= 1
                    if not cache_deque:  # 如果deque变空，则删除该键
                        del self.cache[cache_key]
                    return response, True
                except ValueError:
                    # 理论上不应该发生，因为我们刚从deque中找到了它
                    log(
                        "warning", f"尝试删除缓存项时出错（未找到）: {cache_key[:8]}..."
                    )
                    pass  # 继续执行到函数末尾返回 False

        return None, False

    def store(self, cache_key: str, response: Any):
        """存储响应到缓存（追加到键对应的deque）。"""
        now = time.time()
        new_item: CacheItem = {
            "response": response,
            "expiry_time": now + self.expiry_time,
            "created_at": now,
        }

        if cache_key not in self.cache:
            self.cache[cache_key] = deque()
        # log('info', f"开始存储响应到deque末尾，当前cache：{self.cache}")

        self.cache[cache_key].append(new_item)  # 追加到deque末尾

        # log('info', f"响应已缓存: {cache_key[:8]}...")
        self.cur_cache_num += 1
        # 如果缓存总条目数超过限制，清理最旧的
        self.clean_if_needed()

    def clean_expired(self):
        """清理所有缓存项中已过期的项。"""
        now = time.time()
        keys_to_remove = []
        for key, cache_deque in self.cache.items():
            # 创建一个新的deque只包含未过期的项
            valid_items = deque(
                item for item in cache_deque if now < item.get("expiry_time", 0)
            )

            if len(valid_items) < len(cache_deque):
                clean_num = len(cache_deque) - len(valid_items)
                log("info", f"清理键 {key[:8]}... 的过期缓存项 {clean_num} 个。")
                self.cur_cache_num -= clean_num

            if not valid_items:
                keys_to_remove.append(key)  # 标记此键以便稍后删除
            else:
                self.cache[key] = valid_items  # 替换为只包含有效项的deque

        # 删除所有项都已过期的键
        for key in keys_to_remove:
            del self.cache[key]
            log("info", f"缓存键 {key[:8]}... 的所有项均已过期，移除该键。")

    def clean_if_needed(self):
        """如果缓存总条目数超过限制，清理全局最旧的项目。"""
        if self.cur_cache_num <= self.max_entries:
            return

        items_to_remove_count = self.cur_cache_num - self.max_entries
        log(
            "info",
            f"缓存总数 {self.cur_cache_num} 超过限制 {self.max_entries}，需要清理 {items_to_remove_count} 个最旧项。",
        )

        # 收集所有缓存项及其元数据（键、创建时间、项本身）
        all_items_meta = []
        for key, cache_deque in self.cache.items():
            for item in cache_deque:
                all_items_meta.append(
                    {"key": key, "created_at": item.get("created_at", 0), "item": item}
                )

        # 按创建时间排序（升序，最旧的在前）
        all_items_meta.sort(key=lambda x: x["created_at"])

        # 确定要删除的最旧项
        items_actually_removed = 0
        keys_potentially_empty = set()
        for i in range(min(items_to_remove_count, len(all_items_meta))):
            item_meta = all_items_meta[i]
            key_to_clean = item_meta["key"]
            item_to_clean = item_meta["item"]

            if key_to_clean in self.cache:
                try:
                    self.cache[key_to_clean].remove(item_to_clean)
                    items_actually_removed += 1
                    self.cur_cache_num -= 1
                    log(
                        "debug",
                        f"因容量限制，删除键 {key_to_clean[:8]}... 的旧缓存项 (创建于 {item_meta['created_at']})。",
                    )
                    keys_potentially_empty.add(key_to_clean)
                except ValueError:
                    # 可能在处理过程中已被其他操作删除，忽略
                    log(
                        "warning",
                        f"尝试因容量限制删除缓存项时未找到: {key_to_clean[:8]}...",
                    )
                    pass

        # 检查是否有deque变空
        for key in keys_potentially_empty:
            if key in self.cache and not self.cache[key]:
                del self.cache[key]
                log(
                    "info", f"因容量限制清理后，键 {key[:8]}... 的deque已空，移除该键。"
                )

        if items_actually_removed > 0:
            log("info", f"因容量限制，共清理了 {items_actually_removed} 个旧缓存项。")


# 根据模型名称和全部消息，生成请求的唯一缓存键。
def generate_cache_key_all(chat_request) -> str:
    """
    根据模型名称和全部消息，生成请求的唯一缓存键。
    Args:
        chat_request: 包含模型和消息列表的请求对象 (符合OpenAI格式)。
    Returns:
        一个代表该请求的唯一缓存键字符串 (MD5哈希值)。
    """
    # 创建包含请求关键信息的字典
    request_data = {"model": chat_request.model, "messages": []}

    # 添加消息内容
    for msg in chat_request.messages:
        if isinstance(msg.content, str):
            message_data = {"role": msg.role, "content": msg.content}
            request_data["messages"].append(message_data)
        elif isinstance(msg.content, list):
            content_list = []
            for item in msg.content:
                if item.get("type") == "text":
                    content_list.append({"type": "text", "text": item.get("text")})
                # 对于图像数据，只使用标识符而不是全部数据
                elif item.get("type") == "image_url":
                    image_data = item.get("image_url", {}).get("url", "")
                    if image_data.startswith("data:image/"):
                        # 对于base64图像，使用前32字符的哈希作为标识符
                        content_list.append(
                            {
                                "type": "image_url",
                                "hash": hashlib.md5(
                                    image_data[:32].encode()
                                ).hexdigest(),
                            }
                        )
                    else:
                        content_list.append({"type": "image_url", "url": image_data})
            request_data["messages"].append({"role": msg.role, "content": content_list})

    # 将字典转换为JSON字符串并计算哈希值
    json_data = json.dumps(request_data, sort_keys=True)
    return hashlib.md5(json_data.encode()).hexdigest()


def generate_cache_key(chat_request, last_n_user_messages: int = 4) -> str:
    """
    根据模型名称和最后 N 条消息生成请求的唯一缓存键。
    Args:
        chat_request: 包含模型和消息列表的请求对象 (符合OpenAI格式)。
        last_n_user_messages: 需要包含在缓存键计算中的最后消息的数量。
    Returns:
        一个代表该请求的唯一缓存键字符串 (MD5哈希值)。
    """
    if last_n_user_messages <= 0:
        # 如果不考虑任何消息，只基于模型生成key
        key_data = {"model": chat_request.model}
        json_data = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(json_data.encode()).hexdigest()

    selected_user_messages_reversed = []  # 临时存放反向找到的消息
    user_messages_found = 0

    # 从消息列表末尾开始向前查找
    for msg in reversed(chat_request.messages):
        processed_content = None

        if isinstance(msg.content, str):
            processed_content = msg.content
        elif isinstance(msg.content, list):
            # 处理图文混合内容
            content_list = []
            for item in msg.content:
                item_type = item.get("type") if hasattr(item, "get") else None
                if item_type == "text":
                    text = item.get("text", "") if hasattr(item, "get") else ""
                    content_list.append({"type": "text", "text": text})
                elif item_type == "image_url":
                    image_url_data = (
                        item.get("image_url", {}) if hasattr(item, "get") else {}
                    )
                    url = (
                        image_url_data.get("url", "")
                        if hasattr(image_url_data, "get")
                        else ""
                    )
                    if url.startswith("data:image/"):
                        # 对于base64图像，使用前32字符的哈希作为标识符
                        img_hash = hashlib.md5(url[:32].encode()).hexdigest()
                        content_list.append({"type": "image_url", "hash": img_hash})
                    else:
                        content_list.append({"type": "image_url", "url": url})

            processed_content = content_list

        # 如果内容成功处理，则添加到列表中
        if processed_content is not None:
            selected_user_messages_reversed.append(
                {"role": msg.role, "content": processed_content}
            )
            user_messages_found += 1
            # 如果已找到足够数量的消息，停止遍历
            if user_messages_found >= last_n_user_messages:
                break

    # 创建用于生成缓存键的数据结构，只包含模型和选定的消息（反序）
    key_data = {
        "model": chat_request.model,
        "last_user_messages": selected_user_messages_reversed,  # 使用特定键名区分
    }
    # 将字典转换为排序后的JSON字符串并计算哈希值
    # 使用 sort_keys=True 确保相同内容但顺序不同的字典能生成相同的key
    json_data = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(json_data.encode()).hexdigest()
