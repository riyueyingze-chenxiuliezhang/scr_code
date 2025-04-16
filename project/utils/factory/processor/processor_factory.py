"""
# @Time: 2025/4/14 19:00
# @File: processor_factory.py
"""
from project.utils.impl.processor import MinMaxProcessor, NormalProcessor, NoProcessor


class ProcessorFactory:
    """经验回放工厂"""

    def __init__(self):
        # 注册支持的网络类型 (可扩展)
        self.processor_registry = {
            "minmax": MinMaxProcessor,
            "normal": NormalProcessor,
            "none": NoProcessor
        }

    def build_processor(
            self,
            config,
            processor_name: str = "none"
    ):
        """
        根据配置构建网络
        Args:
            config: 配置项
            processor_name: 使用数据处理的名称
        """
        # 参数校验
        if processor_name not in self.processor_registry:
            raise KeyError(f"不支持的 processor_name: {processor_name}，可选: {list(self.processor_registry.keys())}")

        # 获取经验池类
        processor_class = self.processor_registry[processor_name]

        # 构造参数
        init_args = {
            "config": config,
        }

        # 创建网络实例
        return processor_class(**init_args)


def data_processor(config):
    processor_name = config.data.get("processor_name", "minmax")
    factory = ProcessorFactory()
    return factory.build_processor(config, processor_name)
