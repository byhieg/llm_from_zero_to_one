from dataclasses import dataclass
from datasets import load_dataset
import os
import numpy as np
import tiktoken
from multiprocessing import Pool
from tqdm import tqdm
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
current_dir = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DatasetConfig:
    name: str = "roneneldan/TinyStories"
    shard: int = 8
    tokenizer_model: str = "gpt2"
    split: str = "train"
    cache_dir: str = f"{current_dir}/cache"
    data_dir: str = f"{current_dir}/data"


def _per_shard_preprocess_data(config: DatasetConfig, shard_id: int):
    dataset = load_dataset(
        path=config.name, cache_dir=config.cache_dir, split=config.split
    )
    shard_ds = dataset.shard(num_shards=config.shard, index=shard_id)

    tokenizer = tiktoken.get_encoding(config.tokenizer_model)
    EOT_TOKEN = tokenizer.eot_token
    print(f"Shard {shard_id}: {len(shard_ds):,} samples")

    all_tokens = []
    for sample in tqdm(
        shard_ds, position=shard_id, desc=f"Shard {shard_id}", leave=True
    ):
        # 每个样本后面加<|endoftext|>
        tokens = tokenizer.encode(sample["text"]) + [EOT_TOKEN]
        all_tokens.extend(tokens)

    os.makedirs(config.data_dir, exist_ok=True)
    # 保存
    shard_path = f"{config.data_dir}/shard_{shard_id:04d}.npy"
    np.save(shard_path, np.array(all_tokens, dtype=np.int32))

    return {
        "shard_id": shard_id,
        "num_tokens": len(all_tokens),
        "num_samples": len(shard_ds),
        "file_path": f"shard_{shard_id:04d}.npy",
    }


def preprocess_data(config: DatasetConfig):
    shard_infos = [(config, i) for i in range(config.shard)]
    with Pool(processes=config.shard) as pool:
        results = pool.starmap(_per_shard_preprocess_data, shard_infos)

    # 汇总
    total_tokens = sum(r["num_tokens"] for r in results)
    print(f"\n完成！总 tokens: {total_tokens:,}")
    meta = {}
    for r in sorted(results, key=lambda x: x["shard_id"]):
        print(f"  Shard {r['shard_id']}: {r['num_tokens']:,} tokens")
        meta[r["shard_id"]] = {
            "tokens": r["num_tokens"],
            "samples": r["num_samples"],
            "type": "int32",
            "file_path": r["file_path"],
        }

    with open(f"{config.data_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    preprocess_data(config=DatasetConfig())
