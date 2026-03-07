from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-jsonl", required=True)
    parser.add_argument("--mini-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-smaller", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def load_seen_texts(path: str | Path, text_key: str) -> set[str]:
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
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
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
