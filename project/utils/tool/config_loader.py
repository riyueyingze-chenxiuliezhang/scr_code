import yaml
from typing import Any


class ConfigMeta(type):
    """
    元类核心逻辑：
    1. 通过类属性直接访问配置 (Config.valve.port)
    2. 首次访问时合并加载所有配置文件
    3. 自动处理嵌套字典的链式访问
    """
    _merged_data = None  # 类级配置存储

    def __getattr__(cls, name: str) -> Any:
        # 拦截未定义属性的访问
        if cls._merged_data is None:
            # 惰性加载机制：第一次访问时初始化
            cls._load_configs()

        value = cls._merged_data.get(name)
        if value is None:
            raise AttributeError(f"'{cls.__name__}' 对象没有属性 '{name}'")

        # 嵌套字典转 ConfigDict 以便链式访问
        return ConfigDict(value) if isinstance(value, dict) else value

    def _load_configs(cls):
        """加载并合并所有配置文件"""
        if not cls.config_paths:
            raise ValueError("必须通过 ConfigLoader.config_paths 指定配置文件路径")

        merged = {}
        for path in cls.config_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    merged.update(yaml.safe_load(f) or {})
            except FileNotFoundError:
                raise RuntimeError(f"配置文件不存在: {path}")
            except yaml.YAMLError as e:
                raise RuntimeError(f"YAML解析失败 [{path}]: {e}")

        if not merged:
            raise ValueError("所有配置文件内容均为空")
        cls._merged_data = merged


class ConfigDict:
    """字典包装器，支持点操作符访问嵌套字典"""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        value = self._data.get(name)
        if value is None:
            raise AttributeError(f"配置项 '{name}' 不存在")
        return ConfigDict(value) if isinstance(value, dict) else value

    def __repr__(self):
        return f"ConfigDict({self._data})"


class ConfigLoader(metaclass=ConfigMeta):
    """
    用法示例：
    """
    config_paths = []  # 通过这里指定配置文件路径


if __name__ == '__main__':
    # 示例访问测试
    print("阀门配置项:", ConfigLoader)
    # 假设配置中valve为字典，例如: valve: {port: 8080}
    print("阀门端口:", ConfigLoader.valve.port)
    # ConfigLoader.config_path = Path(r"C:\Users\admi\Desktop\aaa\project\utils\tool\config\config.yaml")
    #
    # class Test:
    #     def __init__(self, config):
    #         print(config.valve)
    #
    # Test(ConfigLoader)
