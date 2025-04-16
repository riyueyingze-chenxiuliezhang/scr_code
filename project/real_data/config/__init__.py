"""
# @Time: 2025/3/26 14:07
# @File: __init__.py
"""
from pathlib import Path

from project.utils import ConfigLoader, ArgumentParser

PROJECT_DIR = Path(__file__).parents[1]

parser = ArgumentParser(description="配置训练参数")
parser.add_argument("--config", "-c", type=str, default="param_1_config.yaml", help="训练配置文件名")
parser.add_argument("--epochs", "-e", type=int, default=150, help="训练配置文件名")
args = parser.parse_args()

ConfigLoader.config_paths = [
    PROJECT_DIR.parent / "experience_config" / "real_data_config" / args.config,
    PROJECT_DIR.parent / "experience_config" / "global_config.yaml"
]
# 训练模型保存文件夹
MODEL_SAVE_DIR = PROJECT_DIR / "result" / "model" / ConfigLoader.name
MODEL_SAVE_DIR.mkdir(exist_ok=True, parents=True)
# 训练状态保存文件夹
STAT_SAVE_DIR = PROJECT_DIR / "result" / "stat" / ConfigLoader.name
STAT_SAVE_DIR.mkdir(exist_ok=True, parents=True)

print(f"当前加载配置文件 {ConfigLoader.config_paths}")

__all__ = [
    "args",
    "ConfigLoader",
    "PROJECT_DIR", "MODEL_SAVE_DIR", "STAT_SAVE_DIR"
]
