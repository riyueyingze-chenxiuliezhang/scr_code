"""
# @Time: 2025/3/28 13:55
# @File: argument_parser.py
"""
import argparse


class ArgumentParser:
    def __init__(self, description=None):
        """
        初始化参数解析器
        :param description: 程序的描述信息
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None  # 存储解析后的参数

    def add_argument(self, *name_or_flags, **kwargs):
        """
        添加命令行参数
        :param name_or_flags: 参数名称，例如 "--input" 或短选项 "-i"
        :param kwargs: 其他参数配置（type, help, default, required等）
        """
        self.parser.add_argument(*name_or_flags, **kwargs)

    def parse_args(self):
        """解析命令行参数并返回结果"""
        self.args = self.parser.parse_args()
        return self.args

    def get_args(self):
        """获取解析后的参数（以字典形式返回）"""
        return vars(self.args)


# 使用示例
if __name__ == "__main__":
    # 创建解析器实例
    parser = ArgumentParser(description="示例程序：命令行参数解析演示")

    # 添加参数
    parser.add_argument("--input", "-i", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output", "-o", type=str, default="output.txt", help="输出文件路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细模式")
    parser.add_argument("--count", "-c", type=int, default=1, help="重复次数")

    # 解析参数
    args = parser.parse_args()

    # 获取参数字典
    args_dict = parser.get_args()

    # 演示输出
    print("解析后的参数对象:", args)
    print("参数字典形式:", args_dict)
    print("输入文件:", args.input)
    print("输出文件:", args.output)
    print("详细模式:", args.verbose)
    print("重复次数:", args.count)
