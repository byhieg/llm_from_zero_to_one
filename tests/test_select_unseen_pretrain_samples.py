import json

from tools.select_unseen_pretrain_samples import (
    load_seen_texts,
    sample_unseen_records,
)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_sample_unseen_records_filters_seen_texts(tmp_path):
    mini_path = tmp_path / "mini.jsonl"
    full_path = tmp_path / "full.jsonl"
    write_jsonl(mini_path, [{"text": "a"}, {"text": "b"}])
    write_jsonl(
        full_path,
        [
            {"text": "a"},
            {"text": "c"},
            {"text": "d"},
            {"text": "b"},
            {"text": "e"},
        ],
    )

    seen_texts = load_seen_texts(mini_path, "text")
    records, unseen_count, skipped_seen_count = sample_unseen_records(
        full_path=full_path,
        seen_texts=seen_texts,
        sample_count=2,
        text_key="text",
        seed=123,
    )

    assert unseen_count == 3
    assert skipped_seen_count == 2
    assert len(records) == 2
    assert all(record["text"] not in {"a", "b"} for record in records)


def test_sample_unseen_records_raises_for_missing_text_key(tmp_path):
    mini_path = tmp_path / "mini.jsonl"
    full_path = tmp_path / "full.jsonl"
    write_jsonl(mini_path, [{"text": "a"}])
    write_jsonl(full_path, [{"content": "missing"}])

    seen_texts = load_seen_texts(mini_path, "text")

    try:
        sample_unseen_records(
            full_path=full_path,
            seen_texts=seen_texts,
            sample_count=1,
            text_key="text",
            seed=1,
        )
    except KeyError as exc:
        assert "missing key 'text'" in str(exc)
    else:
        raise AssertionError("expected KeyError")
