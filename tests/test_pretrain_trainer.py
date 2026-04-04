import random
import pickle

import torch
import torch.nn as nn

import trainer.pretrain.pretrain as pretrain_module
from trainer.pretrain.pretrain import PreTrainTrainer
from trainer.train_args import (
    CheckpointConfig,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    PretrainArgs,
    TrainingConfig,
)


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 123

    def __getitem__(self, index):
        return index


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)


class TrainStepModel(nn.Module):
    def __init__(self, processed_batches):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.processed_batches = processed_batches

    def forward(self, x, y):
        self.processed_batches.append(int(x.reshape(-1)[0].item()))
        loss = self.weight * 0 + x.float().mean() * 0
        return x, loss


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


def test_run_builds_optimizer_before_loading_optimizer_state(monkeypatch):
    calls = []

    class FakeCheckpointManager:
        def __init__(self, checkpoint_config, model_name):
            pass

        def get_checkpoint(self):
            return {
                "linear.weight": torch.ones((2, 4)),
                "linear.bias": torch.zeros(2),
            }, {
                "step": 7,
                "optimizer_state_dict": {
                    "state": {},
                    "param_groups": [{"lr": 1e-3, "params": [0, 1]}],
                },
            }

        def save_checkpoint(self, checkpoint, step, metadata):
            calls.append(("save_checkpoint", step, metadata["epoch"]))

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

        def load_state_dict(self, state_dict):
            calls.append(("optimizer.load_state_dict", state_dict))

        def zero_grad(self):
            pass

        def state_dict(self):
            return {"optimizer": "state"}

    monkeypatch.setattr(pretrain_module, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(
        pretrain_module, "create_dataset", lambda **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        pretrain_module, "create_model", lambda *args, **kwargs: DummyModel()
    )

    args = PretrainArgs(
        training=TrainingConfig(epoch_num=0),
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: [])
    monkeypatch.setattr(
        trainer, "_init_swanlab", lambda device, dataset, dataloader: None
    )
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(
        trainer,
        "_maybe_compile_model",
        lambda model, device: calls.append(("compile", device.type)) or model,
    )

    original_load_state_dict = DummyModel.load_state_dict
    original_to = DummyModel.to

    def fake_load_state_dict(self, state_dict, *args, **kwargs):
        calls.append(("model.load_state_dict", sorted(state_dict.keys())))
        return original_load_state_dict(self, state_dict, *args, **kwargs)

    def fake_to(self, device, *args, **kwargs):
        calls.append(("model.to", str(device)))
        return original_to(self, device, *args, **kwargs)

    monkeypatch.setattr(DummyModel, "load_state_dict", fake_load_state_dict)
    monkeypatch.setattr(DummyModel, "to", fake_to)

    def fake_build_optimizer(model):
        calls.append(("build_optimizer", next(model.parameters()).device.type))
        return FakeOptimizer()

    monkeypatch.setattr(trainer, "_build_optimizer", fake_build_optimizer)

    trainer.run()

    expected_optimizer_state = {
        "state": {},
        "param_groups": [{"lr": 1e-3, "params": [0, 1]}],
    }

    assert calls[0] == ("model.load_state_dict", ["linear.bias", "linear.weight"])
    assert calls[1][0] == "model.to"
    assert calls[2] == ("build_optimizer", calls[1][1])
    assert calls[3] == ("optimizer.load_state_dict", expected_optimizer_state)
    assert calls[4] == ("compile", calls[1][1])
    assert calls[5] == ("save_checkpoint", 7, 0)


def test_run_skips_consumed_micro_batches_when_resuming(monkeypatch):
    processed_batches = []

    class FakeCheckpointManager:
        def __init__(self, checkpoint_config, model_name):
            self.saved_checkpoints = []

        def get_checkpoint(self):
            model = TrainStepModel([])
            return model.state_dict(), {
                "global_step": 3,
                "epoch": 0,
                "micro_step_in_epoch": 2,
            }

        def save_checkpoint(self, checkpoint, step, metadata):
            self.saved_checkpoints.append((step, metadata))

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

        def step(self):
            pass

        def zero_grad(self):
            pass

        def state_dict(self):
            return {"optimizer": "state"}

    dataloader = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1]), torch.tensor([1])),
        (torch.tensor([2]), torch.tensor([2])),
        (torch.tensor([3]), torch.tensor([3])),
    ]

    monkeypatch.setattr(pretrain_module, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(
        pretrain_module, "create_dataset", lambda **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        pretrain_module,
        "create_model",
        lambda *args, **kwargs: TrainStepModel(processed_batches),
    )

    args = PretrainArgs(
        training=TrainingConfig(epoch_num=1, accumulation_steps=1, log_steps=100),
        checkpoint=CheckpointConfig(checkpoint_dir="checkpoints/test-pretrain"),
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: dataloader)
    monkeypatch.setattr(
        trainer, "_init_swanlab", lambda device, dataset, dataloader: None
    )
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(trainer, "_maybe_compile_model", lambda model, device: model)
    monkeypatch.setattr(trainer, "_build_optimizer", lambda model: FakeOptimizer())
    monkeypatch.setattr(trainer, "_save_checkpoint_if_needed", lambda **kwargs: None)

    trainer.run()

    assert processed_batches == [2, 3]


def test_save_training_checkpoint_persists_resume_position(monkeypatch):
    saved_checkpoints = []
    trainer = PreTrainTrainer(
        PretrainArgs(
            checkpoint=CheckpointConfig(
                checkpoint_dir="checkpoints/test-pretrain", save_steps=1
            ),
            data=DataConfig(
                data_strategy="padding",
                dataset_config={"data_path": "demo"},
            ),
            model=ModelConfig(name="gpt2", config={}),
        )
    )
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    monkeypatch.setattr(
        trainer.checkpoint_manager,
        "save_checkpoint",
        lambda checkpoint, step, metadata: saved_checkpoints.append(
            (checkpoint, step, metadata)
        ),
    )

    trainer._save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=6,
        epoch=0,
        micro_step_in_epoch=4,
        dataloader_length=4,
    )

    checkpoint, step, metadata = saved_checkpoints[0]

    assert step == 6
    assert checkpoint.keys() == model.state_dict().keys()
    assert metadata["global_step"] == 6
    assert metadata["step"] == 6
    assert metadata["epoch"] == 1
    assert metadata["micro_step_in_epoch"] == 0
    assert "optimizer_state_dict" in metadata


def test_run_saves_final_checkpoint_even_without_updates(monkeypatch):
    saved_checkpoints = []

    class FakeCheckpointManager:
        def __init__(self, checkpoint_config, model_name):
            pass

        def get_checkpoint(self):
            return None, None

        def save_checkpoint(self, checkpoint, step, metadata):
            saved_checkpoints.append((checkpoint, step, metadata))

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

        def zero_grad(self):
            pass

        def state_dict(self):
            return {"optimizer": "state"}

    monkeypatch.setattr(pretrain_module, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(
        pretrain_module, "create_dataset", lambda **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        pretrain_module, "create_model", lambda *args, **kwargs: DummyModel()
    )

    args = PretrainArgs(
        training=TrainingConfig(epoch_num=0),
        checkpoint=CheckpointConfig(checkpoint_dir="checkpoints/test-pretrain"),
        data=DataConfig(data_strategy="padding", dataset_config={"data_path": "demo"}),
        model=ModelConfig(name="gpt2", config={}),
    )
    trainer = PreTrainTrainer(args)

    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: [])
    monkeypatch.setattr(
        trainer, "_init_swanlab", lambda device, dataset, dataloader: None
    )
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(trainer, "_maybe_compile_model", lambda model, device: model)
    monkeypatch.setattr(trainer, "_build_optimizer", lambda model: FakeOptimizer())

    trainer.run()

    assert len(saved_checkpoints) == 1
    _, step, metadata = saved_checkpoints[0]
    assert step == 0
    assert metadata["global_step"] == 0
    assert metadata["epoch"] == 0
    assert metadata["micro_step_in_epoch"] == 0


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
    assert (
        calls["init"]["config"]["data"]["dataset_config"]["seq_len"]
        == args.training.seq_len
    )
    assert "swanlab" not in calls["init"]["config"]
    assert calls["init"]["config"]["runtime"]["device"] == "cpu"
    assert calls["log"] == [{"train/epoch": 0}]
    assert calls["finished"] is True
