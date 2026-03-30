import os
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
import struct
from transformers import TokenizersBackend
from util import DType


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
    """
    模仿 megatron 数据模块，对原始文本进行处理。产出 idx 和 bin 文件。

    bin 文件是直接 tokenizer 之后的 token 序列。numpy 格式。需要根据词典大小指定 dtype

    index 文件是 bin 文件的索引，他记录了每个原始文本在 bin 文件中的位置。

    这里是处理单数据集的，如果多数据集只需要多次调用这个脚本即可。

    这里原始文档，我们成为 doc。doc 是数据集中的一条记录，他可以是一段话，或者一段代码。

    同时我们可以指定是否添加 special tokens。
    如果添加，我们需要指定 special tokens 的列表。
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    dataset = load_dataset("roneneldan/TinyStories", "default", split="train")

    run(dataset, "text", "tiny_stories", "train_data", tokenizer, dtype=np.uint16, add_eos_id=True)
