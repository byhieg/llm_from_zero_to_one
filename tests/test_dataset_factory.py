from dataset.dataset_factory import create_dataset


class DummyPaddingDataset:
    def __init__(self, tokenizer, max_seq, dataset, dataset_config):
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.dataset = dataset
        self.dataset_config = dataset_config


def test_create_padding_dataset_supports_dataset_path_and_name(monkeypatch):
    calls = {}
    fake_dataset = [{"text": "hello"}]

    def fake_load_dataset(path, name, split, data_files=None, **kwargs):
        calls["path"] = path
        calls["name"] = name
        calls["split"] = split
        calls["data_files"] = data_files
        return fake_dataset

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path):
            calls["tokenizer_path"] = path
            return object()

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr("transformers.AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        "dataset.padding_dataset.PretrainPaddingDataset", DummyPaddingDataset
    )

    dataset = create_dataset(
        data_strategy="padding",
        dataset_config={
            "dataset_path": "jingyaogong/minimind_dataset",
            "dataset_name": "default",
            "data_files": ["pretrain_t2t_mini.jsonl"],
            "split": "train",
            "tokenizer_path": "jingyaogong/minimind-3",
            "seq_len": 128,
            "col_name": "text",
        },
    )

    assert calls == {
        "path": "jingyaogong/minimind_dataset",
        "name": "default",
        "split": "train",
        "data_files": ["pretrain_t2t_mini.jsonl"],
        "tokenizer_path": "jingyaogong/minimind-3",
    }
    assert dataset.max_seq == 128
    assert dataset.dataset is fake_dataset
