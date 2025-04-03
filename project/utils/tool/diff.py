"""
# @Time: 2025/4/3 14:17
# @File: diff.py
"""


def print_differences(differences):
    for key, (val1, val2) in differences.items():
        if val1 is None:
            print(f"键 '{key}' 仅存在于第二个字典中，值为: {val2}")
        elif val2 is None:
            print(f"键 '{key}' 仅存在于第一个字典中，值为: {val1}")
        else:
            print(f"键 '{key}' 的值不同：")
            print(f"    第一个字典中的值: {val1}")
            print(f"    第二个字典中的值: {val2}")


def deep_compare_dicts(dict1, dict2, path=""):
    differences = []
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        current_path = f"{path}.{key}" if path else key
        in_dict1 = key in dict1
        in_dict2 = key in dict2

        if not in_dict1:
            differences.append((current_path, None, dict2[key]))
        elif not in_dict2:
            differences.append((current_path, dict1[key], None))
        else:
            val1 = dict1[key]
            val2 = dict2[key]

            # 递归处理嵌套字典
            if isinstance(val1, dict) and isinstance(val2, dict):
                differences += deep_compare_dicts(val1, val2, current_path)
            elif val1 != val2:
                differences.append((current_path, val1, val2))

    return differences


if __name__ == '__main__':
    a = {"a": 1, "b": {"c": [2, 2, 3, 4, 5, [4, 5, 6]]}}
    b = {"a": 1, "b": {"C": [2, 2, [4, 5, 6], 3, 4, 5]}}
    dif = deep_compare_dicts(a, b)
    print(dif)
