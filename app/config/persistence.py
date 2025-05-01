import json
import os
import inspect
import pathlib
from app.config import settings
from app.utils.logging import log


def save_settings():
    """
    将settings中所有的从os.environ.get获取的配置保存到JSON文件中
    """
    if settings.ENABLE_STORAGE:
        # 确保存储目录存在
        storage_dir = pathlib.Path(settings.STORAGE_DIR)
        storage_dir.mkdir(parents=True, exist_ok=True)

        # 设置JSON文件路径
        settings_file = storage_dir / "settings.json"

        # 获取settings模块中的所有变量
        settings_dict = {}
        for name, value in inspect.getmembers(settings):
            # 跳过内置和私有变量，以及函数/模块/类
            if (
                not name.startswith("_")
                and not inspect.isfunction(value)
                and not inspect.ismodule(value)
                and not inspect.isclass(value)
            ):
                # 尝试将可序列化的值添加到字典中
                try:
                    json.dumps({name: value})  # 测试是否可序列化
                    settings_dict[name] = value
                except (TypeError, OverflowError):
                    # 如果不可序列化，则跳过
                    continue
        log("info", f"保存设置到JSON文件: {settings_file}")

        # 保存到JSON文件
        with open(settings_file, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, ensure_ascii=False, indent=4)

        return settings_file


def load_settings():
    """
    从JSON文件中加载设置并更新settings模块
    """
    if settings.ENABLE_STORAGE:
        # 设置JSON文件路径
        storage_dir = pathlib.Path(settings.STORAGE_DIR)
        settings_file = storage_dir / "settings.json"

        # 如果文件不存在，则返回
        if not settings_file.exists():
            return False

        # 从JSON文件中加载设置
        try:
            with open(settings_file, "r", encoding="utf-8") as f:
                loaded_settings = json.load(f)

            # 更新settings模块中的变量
            for name, value in loaded_settings.items():
                if hasattr(settings, name):
                    setattr(settings, name, value)
            log("info", f"加载设置成功")
            return True
        except Exception as e:
            print(f"加载设置时出错: {e}")
            return False
