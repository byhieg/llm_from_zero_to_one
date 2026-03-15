from dataclasses import dataclass
from datasets import load_dataset
import os
import numpy as np
import tiktoken
from tqdm import tqdm
import json
from glob import glob

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = "/root/autodl-tmp"


@dataclass
class DatasetConfig:
    name: str = "/root/autodl-tmp/fineweb-edu"
    tokens_per_shard: int = 100 * 1000 * 1000
    tokenizer_model: str = "gpt2"
    max_files: int = 0

    def get_data_dir(self):
        return f"{current_dir}/data"


def preprocess_data(config: DatasetConfig):
    all_files = sorted(glob(f"{config.name}/*.parquet"))

    if config.max_files > 0:
        all_files = all_files[: config.max_files]

    if not all_files:
        raise FileNotFoundError(f"未找到 parquet 文件: {config.name}/*.parquet")

    print(f"📁 找到 {len(all_files)} 个 parquet 文件")
    print(f"📦 每个 shard 包含 {config.tokens_per_shard:,} tokens")

    os.makedirs(config.get_data_dir(), exist_ok=True)

    tokenizer = tiktoken.get_encoding(config.tokenizer_model)
    allowed_special = tokenizer.special_tokens_set

    shard_id = 0
    current_tokens = []
    total_tokens = 0
    total_samples = 0
    shard_infos = {}

    for file_path in tqdm(all_files, desc="处理文件"):
        dataset = load_dataset(
            "parquet",
            data_files=file_path,
            split="train",
        )

        for sample in tqdm(
            dataset, desc=f"Tokenizing {os.path.basename(file_path)}", leave=False
        ):
            tokens = tokenizer.encode(sample["text"], allowed_special=allowed_special)
            current_tokens.extend(tokens)
            total_samples += 1

            while len(current_tokens) >= config.tokens_per_shard:
                shard_tokens = current_tokens[: config.tokens_per_shard]
                current_tokens = current_tokens[config.tokens_per_shard :]

                shard_path = f"{config.get_data_dir()}/shard_{shard_id:04d}.npy"
                np.save(shard_path, np.array(shard_tokens, dtype=np.int32))

                shard_infos[shard_id] = {
                    "tokens": len(shard_tokens),
                    "type": "int16",
                    "file_path": f"shard_{shard_id:04d}.npy",
                }

                print(
                    f"✅ Shard {shard_id}: {len(shard_tokens):,} tokens → {shard_path}"
                )
                total_tokens += len(shard_tokens)
                shard_id += 1

    if current_tokens:
        shard_path = f"{config.get_data_dir()}/shard_{shard_id:04d}.npy"
        np.save(shard_path, np.array(current_tokens, dtype=np.int32))

        shard_infos[shard_id] = {
            "tokens": len(current_tokens),
            "type": "int32",
            "file_path": f"shard_{shard_id:04d}.npy",
        }

        print(f"✅ Shard {shard_id}: {len(current_tokens):,} tokens → {shard_path}")
        total_tokens += len(current_tokens)

    with open(f"{config.get_data_dir()}/meta.json", "w", encoding="utf-8") as f:
        json.dump(shard_infos, f, indent=2, ensure_ascii=False)

    print(f"\n🎉 完成！")
    print(f"   📊 总样本: {total_samples:,}")
    print(f"   🔢 总 tokens: {total_tokens:,}")
    print(f"   📦 总 shards: {len(shard_infos)}")
    print(f"   📝 meta.json 已保存至 {config.get_data_dir()}/meta.json")


if __name__ == "__main__":
    os.makedirs(f"{current_dir}/data", exist_ok=True)
    preprocess_data(config=DatasetConfig())
