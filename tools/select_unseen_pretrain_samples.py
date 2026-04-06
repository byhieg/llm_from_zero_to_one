"""从 MiniMind 全量预训练数据中筛选未见过样本，用于评估模型泛化能力

本脚本用于从 MiniMind 全量预训练数据集（full）中，过滤掉已在 mini 训练集中出现过的文本，
随机采样出「未见过」(unseen) 的样本，用于评测模型在未见数据上的泛化性能。

典型使用场景：
    全量数据集 pretrain_t2t.jsonl 包含大量文本，而 mini 训练集 pretrain_t2t_mini.jsonl
    仅使用了其中一小部分。为了公平评估模型的真正泛化能力，需要确保评测样本
    不在训练集中出现过。本工具通过集合去重 + 蓄水池抽样实现这一目标。

用法示例::

    python tools/select_unseen_pretrain_samples.py \\
        --full-jsonl dataset/pretrain_t2t.jsonl \\
        --mini-jsonl dataset/pretrain_t2t_mini.jsonl \\
        --output-jsonl train_data/pretrain_t2t_eval_unseen_1024.jsonl \\
        --sample-count 1024 \\
        --text-key text \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="从 MiniMind 全量数据中筛选 mini 训练集未覆盖的样本，用于 unseen 评测"
    )
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--mini-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-smaller", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: str | Path):
    """逐行迭代 JSONL 文件，返回 (行号, 解析后的字典) 元组"""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_seen_texts(path: str | Path, text_key: str) -> set[str]:
    """加载 mini 训练集的文本内容，构建「已见过」文本集合，用于后续去重过滤"""
    seen_texts = set()
    for line_no, record in iter_jsonl(path):
        if text_key not in record:
            raise KeyError(f"{path}:{line_no} missing key '{text_key}'")
        seen_texts.add(str(record[text_key]))
    return seen_texts


def sample_unseen_records(
    full_path: str | Path,
    seen_texts: set[str],
    sample_count: int,
    text_key: str,
    seed: int,
) -> tuple[list[dict], int, int]:
    """从全量数据集中使用蓄水池抽样算法采样未见过样本

    Args:
        full_path: MiniMind 全量预训练数据集 JSONL 路径
        seen_texts: mini 训练集已出现过的文本集合（用于去重）
        sample_count: 目标采样数量
        text_key: 文本字段名（默认 "text"）
        seed: 随机种子，保证采样可复现

    Returns:
        (采样记录列表, 未见过样本总数, 跳过的已见样本数)
    """
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    rng = random.Random(seed)
    reservoir = []
    unseen_count = 0
    skipped_seen_count = 0
    for line_no, record in iter_jsonl(full_path):
        if text_key not in record:
            raise KeyError(f"{full_path}:{line_no} missing key '{text_key}'")
        text = str(record[text_key])
        if text in seen_texts:
            skipped_seen_count += 1
            continue
        unseen_count += 1
        if len(reservoir) < sample_count:
            reservoir.append(record)
            continue
        replace_index = rng.randrange(unseen_count)
        if replace_index < sample_count:
            reservoir[replace_index] = record
    return reservoir, unseen_count, skipped_seen_count


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    """将记录列表写入 JSONL 文件"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """主流程：加载 mini 已见文本 → 从全量数据中过滤并采样未见过样本 → 输出结果"""
    args = parse_args()
    seen_texts = load_seen_texts(args.mini_jsonl, args.text_key)
    records, unseen_count, skipped_seen_count = sample_unseen_records(
        full_path=args.full_jsonl,
        seen_texts=seen_texts,
        sample_count=args.sample_count,
        text_key=args.text_key,
        seed=args.seed,
    )
    if unseen_count < args.sample_count and not args.allow_smaller:
        raise ValueError(
            f"only found {unseen_count} unseen samples, fewer than requested {args.sample_count}"
        )
    write_jsonl(args.output_jsonl, records)
    print(f"mini seen texts: {len(seen_texts)}")
    print(f"full unseen texts: {unseen_count}")
    print(f"skipped seen texts: {skipped_seen_count}")
    print(f"selected samples: {len(records)}")
    print(f"output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
