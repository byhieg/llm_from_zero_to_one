"""MiniMind 数据预处理工具 —— 将原始文本数据集转换为 Megatron 格式的 .bin + .idx 文件

本脚本模仿 Megatron-LM 数据处理模块，对 MiniMind 原始文本数据集进行分词预处理，
产出可直接用于训练的二进制文件：

- **.bin 文件**：分词后的 token 序列，以 numpy memmap 格式存储（支持 uint16/uint32 等 dtype）
- **.idx 文件**：索引文件，记录每个文档在 .bin 中的字节偏移位置

输出格式兼容 Megatron-LM 的 IndexedDataset 规范，可被 SimpleMegatronDataset 直接加载。

用法::

    python tools/llm_data_processor.py
"""

import os
import sys
from pathlib import Path
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
import struct
from transformers import TokenizersBackend

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset.simple_megatron_dataset import DType  # noqa: E402


"""
index

字节偏移 (Offset)   长度 (Bytes)  数据类型   说明 (Description)
0,                  4/9,        char[],     Magic Number: 用于验证文件格式是否正确。
4/9,                8,          uint64,     Version: 索引格式版本（通常是 1）。
12/17,              1,          uint8,      DType Enum: 记录 .bin 里数据的类型（如 uint16 是 8）。
13/18,              8,          uint64,     Lens Size: 记录总共有多少个文档（Documents）。
21/26,              8,          uint64,     Doc Count: 记录总共有多少个 Token（还是样本，视版本定）。
29/34~ ...,         L×4,        uint32,     Sizes: 一个数组，记录每个文档的 Token 数量。
...~ ...,          (L+1)×8,     uint64,     Pointers: 一个数组，记录每个文档在 .bin 里的字节起始偏移。

"""


def run(
    dataset: Dataset,
    column_name: str,
    output_name: str,
    output_dir: str,
    tokenizer: TokenizersBackend,
    add_bos_id=False,
    add_eos_id=False,
    dtype=np.int32,
):
    """将 HuggingFace 数据集转换为 Megatron 格式的 .bin + .idx 文件

    Args:
        dataset: HuggingFace Dataset 对象
        column_name: 文本字段名（默认 "text"）
        output_name: 输出文件名（不含扩展名）
        output_dir: 输出目录
        tokenizer: 分词器实例
        add_bos_id: 是否在开头添加 BOS token
        add_eos_id: 是否在结尾添加 EOS token
        dtype: .bin 文件的数据类型（如 np.uint32，需与词表大小匹配）
    """
    os.makedirs(output_dir, exist_ok=True)

    def process(example) -> dict:
        text = example[column_name]
        # allowed_special in tiktoken.encode expects string literals, not token ids.
        # But we actually want to append the eot_token ID at the end manually
        # rather than parsing `<|endoftext|>` from the text.
        ids: list[int] = tokenizer(text, add_special_tokens=False).input_ids
        if add_bos_id:
            ids.insert(0, tokenizer.bos_token_id)
        if add_eos_id:
            ids.append(tokenizer.eos_token_id)

        return {"ids": ids, "len": len(ids)}

    tokenized = dataset.map(
        process,
        desc=f"Tokenizing {output_name}",
        num_proc=os.cpu_count(),
    )

    total_docs = len(tokenized)
    sizes = np.array(tokenized["len"], dtype=np.uint32)

    pointers = np.zeros(total_docs + 1, dtype=np.uint64)
    # 计算前缀和作为偏移量
    np.cumsum(sizes, out=pointers[1:], dtype=np.uint64)

    total_tokens = pointers[-1]
    print(f"datasets has {total_tokens} tokens from {total_docs} docs")

    bin_path = os.path.join(output_dir, f"{output_name}.bin")
    idx_path = os.path.join(output_dir, f"{output_name}.idx")

    # Create memory-mapped file
    arr = np.memmap(bin_path, dtype=dtype, mode="w+", shape=(total_tokens,))

    # 优化后的写入逻辑
    current_idx = 0
    for batch in tqdm(
        tokenized.iter(batch_size=1024),
        total=len(tokenized) // 1024 + 1,
        desc="Writing",
    ):
        for ids in batch["ids"]:
            length = len(ids)
            arr[current_idx : current_idx + length] = ids
            current_idx += length

    arr.flush()

    # 写入 index 文件
    with open(idx_path, "wb") as f:
        # Magic Number: 'MMID'
        f.write(struct.pack("<4s", b"MMID"))
        # Version
        f.write(np.array([1], dtype=np.uint64).tobytes())
        # DType Enum
        f.write(struct.pack("<B", DType.code_from_dtype(dtype)))
        # Lens Size
        f.write(np.array([total_docs], dtype=np.uint64).tobytes())
        # Doc Count
        f.write(np.array([0], dtype=np.uint64).tobytes())
        # Sizes
        f.write(sizes.tobytes())
        # Pointers
        f.write(pointers.tobytes())

    print(f"Successfully created {bin_path} and {idx_path}")


if __name__ == "__main__":
    """默认入口：使用 MiniMind 数据集和分词器生成 .bin + .idx 文件"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("jingyaogong/minimind-3")

    dataset = load_dataset(
        "jingyaogong/minimind_dataset",
        "default",
        split="train",
        data_files={"train": "pretrain_t2t_mini.jsonl"},
    )

    run(
        dataset,
        "text",
        "minimind_dataset",
        "train_data",
        tokenizer,
        dtype=np.uint32,
        add_eos_id=True,
    )
