import torch
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any, List
import random
from torch.utils.data import Dataset


class ShardIndexDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int, seed: int = 42):
        super().__init__()
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.seed = seed
        assert self.data_path.exists(), f"{data_path} need exist"
        with Path(self.data_path / "meta.json").open("r", encoding="utf-8") as f:
            self.data_meta: Dict[str, Dict[str, Any]] = json.load(f)

        self.length = sum(v["tokens"] for v in self.data_meta.values())
        self.total_shard = int(max(self.data_meta.keys())) + 1
        self.shard_sample_range: Dict[str, int] = {
            k: v["tokens"] for k, v in self.data_meta.items()
        }
        self.shard_data: Dict[str, np.ndarray] = {}

        self.shard_order: List[str] = list(self.shard_sample_range.keys())
        self.epoch = 0

    def __len__(self) -> int:
        return self.length // self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 根据样本索引计算 token 起始位置（无重叠切分）
        token_start = idx * self.seq_len
        result = self._get_shard_by_index(token_start)  # type: ignore[return-value]
        shard_id = str(result[0])
        local_index = int(result[1])
        # Ensure shard_id is a string for dict key access and compatibility with dynamic typing
        shard_id = str(shard_id)
        # 加载当前 shard（如果尚未加载）
        if shard_id not in self.shard_data:
            shard_data = np.load(
                Path(self.data_path / self.data_meta[shard_id]["file_path"]),
                mmap_mode="r",
            )
            self.shard_data[shard_id] = shard_data

        # 读取所需的 token 数量（seq_len + 1，用于构建 input 和 label）
        needed_tokens = self.seq_len + 1
        current_shard_size = self.data_meta[shard_id]["tokens"]
        current_shard_remaining = current_shard_size - local_index

        if current_shard_remaining >= needed_tokens:
            # 当前 shard 可以一次性提供所需 tokens
            tokens = self.shard_data[shard_id][
                local_index : local_index + needed_tokens
            ]
        else:
            current_idx = self.shard_order.index(shard_id)
            next_shard = self.shard_order[(current_idx + 1) % self.total_shard]

            if next_shard not in self.shard_data:
                shard_data = np.load(
                    Path(self.data_path / self.data_meta[next_shard]["file_path"]),
                    mmap_mode="r",
                )
                self.shard_data[next_shard] = shard_data

            # 从当前 shard 读取剩余部分，从下一个 shard 读取需要的部分
            tokens_from_current = self.shard_data[shard_id][local_index:]
            tokens_from_next = self.shard_data[next_shard][
                : needed_tokens - current_shard_remaining
            ]
            tokens = np.concatenate([tokens_from_current, tokens_from_next])

        tokens = torch.from_numpy(tokens.astype(self.data_meta[shard_id]["type"]))
        return tokens[:-1], tokens[1:]

    def shuffle_shard(self, epoch: int) -> None:
        """打乱 shard 顺序。DDP 环境下使用相同 seed+epoch 确保所有进程一致。"""
        self.epoch = epoch
        rng = random.Random(self.seed + epoch)
        self.shard_order = list(self.shard_sample_range.keys())
        rng.shuffle(self.shard_order)

    def _get_shard_by_index(self, index: int) -> tuple[str, int]:
        assert 0 <= index < self.length, f"{index} need between zero and {self.length}"
        cur = 0
        for shard in self.shard_order:
            sample_num = self.shard_sample_range[shard]
            if cur <= index < cur + sample_num:
                return shard, index - cur
            cur += sample_num

        raise ValueError(f"can not get index {index} in data shard ")
