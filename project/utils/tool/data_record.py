"""
# @Time: 2025/3/27 15:25
# @File: data_record.py
"""
import pickle
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Union, List, Any

import numpy as np


class DataRecorder:
    def __init__(self, save_path: Union[str, Path], chunk_size: int = 100000, file_fmt: str = 'txt'):
        self._save_path = Path(save_path)
        self._chunk_size = chunk_size
        self._file_format = file_fmt.lower()
        self._data_buffers = {}  # 名称: 数据缓冲区
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._counters = defaultdict(int)  # 每个名称的块计数器
        self._counter_lock = Lock()
        self._futures = []  # 保存所有提交的Future对象
        self._futures_lock = Lock()

    def add_data(self, name: str, value: Union[int, float, str]) -> None:
        """添加数据到缓冲区，触发自动保存"""
        if not isinstance(value, (int, float, str)):
            raise TypeError(f"值必须是 int/float/str. 得到的类型为 {type(value)}")

        if name not in self._data_buffers:
            self._data_buffers[name] = []
        buffer = self._data_buffers[name]
        buffer.append(value)

        if len(buffer) >= self._chunk_size:
            self._submit_save_task(name, buffer)
            self._data_buffers[name] = []

    def flush(self) -> None:
        """强制保存所有剩余数据"""
        for name in list(self._data_buffers.keys()):
            buffer = self._data_buffers.get(name, [])
            if buffer:
                self._submit_save_task(name, buffer)
                self._data_buffers[name] = []

        for name in list(self._data_buffers.keys()):
            self.merge_files(name)

    def _submit_save_task(self, name: str, data: List[Any]) -> None:
        """提交保存任务到线程池"""
        with self._counter_lock:
            part_num = self._counters[name]
            self._counters[name] += 1
        future = self._executor.submit(
            self._save_chunk,
            self._save_path,
            name,
            part_num,
            data.copy(),
            self._file_format
        )
        with self._futures_lock:
            self._futures.append(future)

    @staticmethod
    def _save_chunk(save_dir: Path, name: str, part_num: int, chunk: List[Any], fmt: str) -> None:
        """保存数据块到文件"""
        try:
            save_dir.mkdir(parents=True, exist_ok=True)

            if fmt == 'txt':
                content = '\n'.join(map(str, chunk))
                file_path = save_dir / f"{name}_{part_num}.txt"
                file_path.write_text(content, encoding='utf-8')
            elif fmt == 'pkl':
                file_path = save_dir / f"{name}_{part_num}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(chunk, f)
            elif fmt == 'npy':
                if all(isinstance(x, (int, float)) for x in chunk):
                    file_path = save_dir / f"{name}_{part_num}.npy"
                    np.save(file_path, np.array(chunk))
                else:
                    raise ValueError("npy格式只支持int/float类型数据")
            else:
                raise ValueError(f"不支持的类型: {fmt}")
        except Exception as e:
            print(f"保存失败 {name} data: {e}")

    def wait(self) -> None:
        """等待所有保存任务完成"""
        with self._futures_lock:
            futures = list(self._futures)
            self._futures.clear()
        for future in futures:
            future.result()

    def merge_files(self, name: str, delete_chunks: bool = True) -> None:
        """合并指定名称的所有分块文件，并可选择删除原分块"""
        self.wait()  # 确保所有保存任务完成
        ext = self._get_extension(self._file_format)
        pattern = re.compile(re.escape(name) + r'_(\d+)\.' + re.escape(ext) + '$')

        files = []
        for f in self._save_path.iterdir():
            if not f.is_file():
                continue
            match = pattern.fullmatch(f.name)
            if match:
                part_num = int(match.group(1))
                files.append((part_num, f))

        files.sort(key=lambda x: x[0])
        if not files:
            return  # 没有找到分块文件

        merged_data = []
        for part_num, file_path in files:
            chunk_data = self._load_chunk(file_path, self._file_format)
            merged_data.extend(chunk_data)

        merged_filename = f"{name}.{ext}"
        merged_file_path = self._save_path / merged_filename

        try:
            if self._file_format == 'txt':
                content = '\n'.join(map(str, merged_data))
                merged_file_path.write_text(content, encoding='utf-8')
            elif self._file_format == 'pkl':
                with open(merged_file_path, 'wb') as f:
                    pickle.dump(merged_data, f)
            elif self._file_format == 'npy':
                if all(isinstance(x, (int, float)) for x in merged_data):
                    np.save(merged_file_path, np.array(merged_data))
                else:
                    raise ValueError("npy格式只支持int/float类型数据")
            else:
                raise ValueError(f"不支持的格式: {self._file_format}")
        except Exception as e:
            print(f"合并文件失败: {e}")
            return

        if delete_chunks:
            for part_num, file_path in files:
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"删除分块文件 {file_path} 失败: {e}")

    @staticmethod
    def load_data(save_dir: Union[str, Path], name: str, file_format: str) -> List[Any]:
        """加载指定名称的所有数据"""
        save_dir = Path(save_dir)
        file_format = file_format.lower()
        ext = DataRecorder._get_extension(file_format)
        pattern = re.compile(re.escape(name) + r'_(\d+)\.' + re.escape(ext) + '$')

        files = []
        for f in save_dir.iterdir():
            if not f.is_file():
                continue
            match = pattern.fullmatch(f.name)
            if match:
                part_num = int(match.group(1))
                files.append((part_num, f))

        files.sort(key=lambda x: x[0])
        data = []
        for part_num, file_path in files:
            chunk_data = DataRecorder._load_chunk(file_path, file_format)
            data.extend(chunk_data)
        return data

    @staticmethod
    def _load_chunk(file_path: Path, fmt: str) -> List[Any]:
        """加载单个数据块"""
        try:
            if fmt == 'txt':
                content = file_path.read_text(encoding='utf-8')
                return [line.strip() for line in content.splitlines() if line.strip()]
            elif fmt == 'pkl':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif fmt == 'npy':
                arr = np.load(file_path)
                return arr.tolist()
            else:
                raise ValueError(f"不支持的格式: {fmt}")
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
            return []

    @staticmethod
    def _get_extension(fmt: str) -> str:
        """获取文件扩展名"""
        if fmt == 'txt':
            return 'txt'
        elif fmt == 'pkl':
            return 'pkl'
        elif fmt == 'npy':
            return 'npy'
        else:
            raise ValueError(f"不支持的格式: {fmt}")

    def __del__(self):
        self._executor.shutdown(wait=True)


if __name__ == '__main__':
    data_record = DataRecorder("./result/test", 100, "txt")
    for i in range(950):
        data_record.add_data("num", i)
        data_record.add_data("math", i * 2 + 1)
    data_record.flush()
    data_record.merge_files("math", True)
