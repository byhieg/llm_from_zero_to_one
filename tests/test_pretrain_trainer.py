import random
import pickle

import torch
import torch.nn as nn

import trainer.pretrain.pretrain as pretrain_module
from trainer.pretrain.pretrain import PreTrainTrainer
from trainer.train_args import DataConfig, ModelConfig, OptimizerConfig, PretrainArgs, TrainingConfig


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 123

    def __getitem__(self, index):
        return index


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)


def test_get_dataset_config_inherits_training_seq_len():
    args = PretrainArgs(
        training=TrainingConfig(seq_len=2048),
        data=DataConfig(
            data_strategy="padding",
            dataset_config={
                "data_path": "jingyaogong/minimind_dataset",
                "data_files": "pretrain_t2t_mini.jsonl",
                "tokenizer_path": "jingyaogong/minimind-3",
            },
        ),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    dataset_config = trainer._get_dataset_config()

    assert dataset_config["seq_len"] == 2048
    assert args.data.dataset_config.get("seq_len") is None


def test_build_dataloader_uses_data_seed_and_config():
    args = PretrainArgs(
        training=TrainingConfig(batch_size=8, seed=42),
        data=DataConfig(
            data_strategy="padding",
            dataset_config={"data_path": "demo"},
            dataloader_config={
                "seed": 123,
                "shuffle": True,
                "num_workers": 0,
                "drop_last": True,
            },
        ),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    dataloader = trainer._build_dataloader(DummyDataset())

    assert dataloader.batch_size == 8
    assert dataloader.drop_last is True
    assert dataloader.generator.initial_seed() == 123


def test_build_dataloader_worker_init_fn_is_picklable_when_num_workers_positive():
    args = PretrainArgs(
        training=TrainingConfig(batch_size=8, seed=42),
        data=DataConfig(
            data_strategy="padding",
            dataset_config={"data_path": "demo"},
            dataloader_config={
                "seed": 123,
                "shuffle": True,
                "num_workers": 2,
            },
        ),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    dataloader = trainer._build_dataloader(DummyDataset())

    assert dataloader.worker_init_fn is not None
    pickle.dumps(dataloader.worker_init_fn)


def test_build_optimizer_uses_adamw_config():
    args = PretrainArgs(
        training=TrainingConfig(learning_rate=1e-3),
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
        optimizer=OptimizerConfig(
            name="adamw",
            weight_decay=0.1,
            betas=[0.8, 0.95],
            eps=1e-6,
        ),
    )
    trainer = PreTrainTrainer(args)

    optimizer = trainer._build_optimizer(DummyModel())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 0.1
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["eps"] == 1e-6


def test_build_optimizer_supports_adam():
    args = PretrainArgs(
        training=TrainingConfig(learning_rate=5e-4),
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
        optimizer=OptimizerConfig(name="adam"),
    )
    trainer = PreTrainTrainer(args)

    optimizer = trainer._build_optimizer(DummyModel())

    assert isinstance(optimizer, torch.optim.Adam)

def test_maybe_compile_model_uses_compile_for_mps(monkeypatch):
    calls = {}

    def fake_compile(model):
        calls["compiled"] = True
        return model

    monkeypatch.setattr(pretrain_module.torch, "compile", fake_compile, raising=False)

    args = PretrainArgs(
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)
    model = DummyModel()

    compiled_model = trainer._maybe_compile_model(model, torch.device("mps"))

    assert compiled_model is model
    assert calls["compiled"] is True


def test_get_dataloader_seed_falls_back_to_training_seed():
    args = PretrainArgs(
        training=TrainingConfig(seed=999),
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    assert trainer._get_dataloader_seed() == 999


def test_set_seed_controls_random_and_torch():
    args = PretrainArgs(
        training=TrainingConfig(seed=123),
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    trainer._set_seed(321)
    random_value_1 = random.randint(0, 1000)
    torch_value_1 = torch.randint(0, 1000, (1,)).item()

    trainer._set_seed(321)
    random_value_2 = random.randint(0, 1000)
    torch_value_2 = torch.randint(0, 1000, (1,)).item()

    assert random_value_1 == random_value_2
    assert torch_value_1 == torch_value_2


def test_init_swanlab_skips_when_disabled():
    args = PretrainArgs(
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    trainer._init_swanlab(
        torch.device("cpu"),
        DummyDataset(),
        trainer._build_dataloader(DummyDataset()),
    )

    assert trainer._swanlab is None


def test_init_swanlab_runs_when_enabled(monkeypatch):
    calls = {}

    class FakeSwanlab:
        def init(self, **kwargs):
            calls["init"] = kwargs

        def log(self, data):
            calls.setdefault("log", []).append(data)

        def finish(self):
            calls["finished"] = True

    monkeypatch.setattr(pretrain_module, "import_module", lambda name: FakeSwanlab())

    args = PretrainArgs(
        training=TrainingConfig(batch_size=2),
        data=DataConfig(
            data_strategy="padding",
            dataset_config={"data_path": "demo"},
            dataloader_config={"seed": 7},
        ),
        model=ModelConfig(name="gpt2", config={}),
    )
    args.swanlab.enabled = True
    args.swanlab.project = "demo-project"
    args.swanlab.experiment_name = "demo-exp"
    args.swanlab.tags = ["unit"]
    trainer = PreTrainTrainer(args)
    dataloader = trainer._build_dataloader(DummyDataset())

    trainer._init_swanlab(torch.device("cpu"), DummyDataset(), dataloader)
    trainer._log_swanlab({"train/epoch": 0})
    trainer._finish_swanlab()

    assert calls["init"]["project"] == "demo-project"
    assert calls["init"]["experiment_name"] == "demo-exp"
    assert calls["init"]["tags"] == ["unit"]
    assert calls["init"]["config"]["training"]["batch_size"] == 2
    assert calls["init"]["config"]["data"]["dataloader_config"]["seed"] == 7
    assert calls["init"]["config"]["data"]["dataset_config"]["seq_len"] == args.training.seq_len
    assert "swanlab" not in calls["init"]["config"]
    assert calls["init"]["config"]["runtime"]["device"] == "cpu"
    assert calls["log"] == [{"train/epoch": 0}]
    assert calls["finished"] is True
