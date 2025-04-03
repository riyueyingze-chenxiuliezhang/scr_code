"""
# @Time: 2025/4/1 12:38
# @File: assert_check.py
"""
from typing import Tuple, Union, Optional, Any


def type_check(v: Any, t: type) -> Any:
    """类型检查函数

    Args:
        v: 要检查的值
        t: 期望的类型

    Returns:
        通过检查的原值

    Raises:
        TypeError: 当值的类型不符合预期时
    """
    if type(v) is not t:  # 严格类型检查（不包含子类）
        raise TypeError(f"类型必须为 {t.__name__}，当前类型：{type(v).__name__}")
    return v


def range_check(
        v: Union[int, float],
        r: Tuple[Optional[Union[int, float]], Optional[Union[int, float]]]
) -> Union[int, float]:
    """范围检查函数

    Args:
        v: 要检查的数值
        r: (最小值, 最大值) 元组，None 表示无限制

    Returns:
        通过检查的原值

    Raises:
        ValueError: 当范围无效或数值越界时
    """
    min_val, max_val = r

    # 校验范围有效性（仅当两个边界都存在时比较）
    if (min_val is not None
            and max_val is not None
            and min_val > max_val):
        raise ValueError(
            f"无效范围：下限 {min_val} 大于上限 {max_val}"
        )

    # 下限检查
    if min_val is not None and v < min_val:
        raise ValueError(
            f"当前值 {v} 小于下限 {min_val}"
        )

    # 上限检查
    if max_val is not None and v > max_val:
        raise ValueError(
            f"当前值为 {v} 大于上限 {max_val}"
        )

    return v


if __name__ == '__main__':
    range_check(type_check(1, int), (1, None))
