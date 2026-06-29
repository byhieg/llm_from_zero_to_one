import torch

from checkpoint_manager import Checkpoint
import evaluator.checkpoint_evaluator as evaluator_module
from evaluator import PretrainEvaluator
from evaluator.eval_args import EvalArgs, EvalCheckpointConfig, EvalConfig
from trainer.common_args import ModelConfig


class DummyTokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def __call__(
        self,
        text,
        add_special_tokens=False,
        max_length=None,
        truncation=False,
    ):
        tokens = [11, 12] if text == "hello" else [21, 22]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return type("Tokenized", (), {"input_ids": tokens})()


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def select(self, indices):
        return FakeDataset([self.rows[index] for index in indices])


class DummyModel:
    def __init__(self):
        self.loaded_state_dict = None
        self.device = None
        self.forward_calls = 0

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, labels=None):
        self.forward_calls += 1
        vocab_size = 256
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), vocab_size),
            -100.0,
            device=input_ids.device,
        )
        if labels is not None:
            safe_labels = labels.masked_fill(labels == -100, 0)
            logits.scatter_(2, safe_labels.unsqueeze(-1), 100.0)
        return logits, None


def test_pretrain_evaluator_loads_checkpoint_and_scores_dataset(monkeypatch):
    captured = {}
    model = DummyModel()

    class FakeCheckpointManager:
        def __init__(self, checkpoint_config, model_name):
            captured["checkpoint_dir"] = checkpoint_config.checkpoint_dir
            captured["model_name"] = model_name

        def get_checkpoint(self, step=None):
            captured["checkpoint_step"] = step
            return Checkpoint(
                model_state_dict={"weight": 1},
                optimizer_state_dict={},
                metadata={"global_step": 12},
            )

    monkeypatch.setattr(evaluator_module, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(evaluator_module, "create_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        evaluator_module,
        "load_dataset",
        lambda path, name, split, data_files=None: (
            captured.update(
                {
                    "dataset_path": path,
                    "dataset_name": name,
                    "split": split,
                    "data_files": data_files,
                }
            )
            or FakeDataset([{"text": "hello"}, {"text": "world"}])
        ),
    )

    def fake_from_pretrained(path):
        captured["tokenizer_path"] = path
        return DummyTokenizer()

    monkeypatch.setattr(
        evaluator_module.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    args = EvalArgs(
        name="minimind_61m_pretrain",
        checkpoint=EvalCheckpointConfig(checkpoint_dir="checkpoints/pretrain"),
        eval=EvalConfig(
            dataset_path="wikitext",
            dataset_name="wikitext-2-raw-v1",
            split="test",
            text_column="text",
            tokenizer_path="from-infer",
            add_bos_id=True,
            add_eos_id=True,
            max_samples=2,
            batch_size=2,
            checkpoint_step=123,
        ),
        model=ModelConfig(name="gpt2", config={"vocab_size": 32}),
    )

    metrics = PretrainEvaluator(args).run()

    assert captured["checkpoint_dir"] == "checkpoints/pretrain"
    assert captured["model_name"] == "minimind_61m_pretrain"
    assert captured["checkpoint_step"] == 123
    assert captured["dataset_path"] == "wikitext"
    assert captured["dataset_name"] == "wikitext-2-raw-v1"
    assert captured["split"] == "test"
    assert captured["data_files"] is None
    assert captured["tokenizer_path"] == "from-infer"
    assert model.loaded_state_dict == {"weight": 1}
    assert model.forward_calls == 1
    assert metrics["sample_count"] == 2.0
    assert metrics["token_count"] > 0
    assert metrics["loss"] < 1e-6
    assert abs(metrics["perplexity"] - 1.0) < 1e-6


def test_pretrain_evaluator_can_score_existing_model(monkeypatch):
    captured = {}
    model = DummyModel()

    monkeypatch.setattr(
        evaluator_module,
        "load_dataset",
        lambda path, name, split, data_files=None: (
            captured.update(
                {
                    "dataset_path": path,
                    "dataset_name": name,
                    "split": split,
                    "data_files": data_files,
                }
            )
            or FakeDataset([{"text": "hello"}])
        ),
    )

    def fake_from_pretrained(path):
        captured["tokenizer_path"] = path
        return DummyTokenizer()

    monkeypatch.setattr(
        evaluator_module.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )

    args = EvalArgs(
        checkpoint=EvalCheckpointConfig(checkpoint_dir="checkpoints/pretrain"),
        eval=EvalConfig(
            dataset_path="wikitext",
            dataset_name="wikitext-2-raw-v1",
            split="validation",
            text_column="text",
            tokenizer_path="from-infer",
            add_bos_id=False,
            add_eos_id=True,
            max_samples=1,
            batch_size=1,
        ),
        model=ModelConfig(name="gpt2", config={"vocab_size": 32}),
    )

    metrics = PretrainEvaluator(args).evaluate_model(
        model=model,
        device=torch.device("cpu"),
    )

    assert captured["dataset_path"] == "wikitext"
    assert captured["dataset_name"] == "wikitext-2-raw-v1"
    assert captured["split"] == "validation"
    assert captured["data_files"] is None
    assert captured["tokenizer_path"] == "from-infer"
    assert model.forward_calls == 1
    assert metrics["sample_count"] == 1.0
    assert metrics["loss"] < 1e-6
