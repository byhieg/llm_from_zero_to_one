from enum import Enum
import os
import numpy as np
import struct


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices
    Matched with Megatron-LM specifications.
    """

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: "DType") -> int:
        """Get the code from the numpy dtype"""
        if value == np.uint8:
            return cls.uint8.value
        elif value == np.int8:
            return cls.int8.value
        elif value == np.int16:
            return cls.int16.value
        elif value == np.int32:
            return cls.int32.value
        elif value == np.int64:
            return cls.int64.value
        elif value == np.float64:
            return cls.float64.value
        elif value == np.float32:
            return cls.float32.value
        elif value == np.uint16:
            return cls.uint16.value
        else:
            raise ValueError(f"Unsupported dtype: {value}")

    @classmethod
    def dtype_from_code(cls, value: int) -> "np.dtype":
        """Get the numpy dtype from the code"""
        if value == cls.uint8.value:
            return np.uint8
        elif value == cls.int8.value:
            return np.int8
        elif value == cls.int16.value:
            return np.int16
        elif value == cls.int32.value:
            return np.int32
        elif value == cls.int64.value:
            return np.int64
        elif value == cls.float64.value:
            return np.float64
        elif value == cls.float32.value:
            return np.float32
        elif value == cls.uint16.value:
            return np.uint16
        else:
            raise ValueError(f"Unsupported code: {value}")


class IndexDataset:
    """
    索引数据集封装类。
    """

    def __init__(self, idx_path: str, bin_path: str, seq_len: int = 1024):
        ...
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"Index file {idx_path} not found.")

        with open(idx_path, "rb") as f:
            # 按照 tools/llm_data_processor.py 中的定义读取头部(共 36 字节)
            magic_bytes = f.read(4)
            version_bytes = f.read(8)
            dtype_size_bytes = f.read(1)
            lens_size_bytes = f.read(8)
            doc_count_bytes = f.read(8)

            if magic_bytes != b"MMID":
                raise ValueError("Invalid index file format.")

            # 读index file
            self.bin_path = bin_path
            self.dtype_val = struct.unpack("<B", dtype_size_bytes)[0]
            self.total_docs = np.frombuffer(lens_size_bytes, dtype=np.uint64)[0]
            self.sizes = np.frombuffer(f.read(self.total_docs * 4), dtype=np.uint32)
            self.pointers = np.frombuffer(
                f.read((self.total_docs + 1) * 8), dtype=np.uint64
            )

            self.total_tokens = self.pointers[-1]

        # 预先根据类型进行一次全局 memmap 初始化，避免在 __getitem__ 时重复 mmap

        self.bin_buffer = np.memmap(
            self.bin_path, dtype=self.get_dtype_val(), mode="r", offset=0
        )

    def get_bin_path(self):
        return self.bin_path

    def get_dtype_val(self) -> DType:
        return DType.dtype_from_code(self.dtype_val)

    def get_total_docs(self):
        return self.total_docs

    def get_total_tokens(self):
        return self.total_tokens

    def get_doc_idx(self, idx):
        return self.pointers[int(idx)]

    def get_doc_len(self, idx):
        return self.sizes[int(idx)]

    def get(self, idx, offset=0, length=None):
        """
        读取第 idx 个文档，从 offset 开始读取 length 个 token
        """
        idx = int(idx)
        # pointer 存着 doc id 在 bin中的起始位置
        ptr = self.pointers[idx]
        if length is None:
            length = self.sizes[idx] - offset

        start = ptr + offset
        end = start + length

        # 直接使用预初始化的内存映射对象进行切片，速度极快
        # 注意: 切片的索引必须是 Python 原生的 int 类型，否则某些 numpy 版本会报错
        return self.bin_buffer[int(start) : int(end)]


if __name__ == "__main__":
    idx_path = "data/tiny_stories.idx"
    bin_path = "data/tiny_stories.bin"
    dataset = IndexDataset(idx_path, bin_path, seq_len=1024)
    print(dataset.get_bin_path())
    print(dataset.get_dtype_val())
    print(dataset.get_total_docs())
    print(dataset.get_total_tokens())
