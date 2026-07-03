import json
import pickle
import random

import torch
import torch.nn as nn

import trainer.pretrain.deepspeed_train as deepspeed_train_module
import trainer.pretrain.naive_train as naive_train_module
import trainer.pretrain.pretrain as pretrain_module
from trainer.common_args import ModelConfig
from trainer.pretrain.pretrain import PreTrainTrainer, ResumableDistributedSampler
from trainer.pretrain.pretrain_args import (
    PreTrainArgs,
    PreTrainCheckpointConfig,
    PreTrainDataConfig,
    PreTrainEvalConfig,
    PreTrainEvalDataConfig,
    PreTrainTrainConfig,
    PreTrainTrainDataConfig,
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


class CountingDataset(torch.utils.data.Dataset):
    def __init__(self, size: int):
        self.size = size
        self.visited_indices = []

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.visited_indices.append(index)
        return torch.tensor(index), torch.tensor(index)


def make_pretrain_args(**overrides) -> PreTrainArgs:
    args = PreTrainArgs(
        experiment=overrides.pop("experiment", None) or PreTrainArgs().experiment,
        model=overrides.pop("model", ModelConfig(name="gpt2", config={})),
        train=overrides.pop(
            "train",
            PreTrainTrainConfig(
                batch_size=8,
                seq_len=1024,
                seed=42,
                log_steps=10,
                naive_config={"learning_rate": 1e-3},
            ),
        ),
        eval=overrides.pop("eval", PreTrainEvalConfig()),
        checkpoint=overrides.pop(
            "checkpoint",
            PreTrainCheckpointConfig(save_checkpoint_dir="checkpoints/pretrain"),
        ),
        data=overrides.pop(
            "data",
            PreTrainDataConfig(
                train=PreTrainTrainDataConfig(
                    data_strategy="padding",
                    dataset_config={"dataset_path": "demo"},
                    dataloader_config={"seed": 123, "shuffle": True, "num_workers": 0},
                ),
                eval=PreTrainEvalDataConfig(
                    dataset_config={
                        "dataset_path": "json",
                        "text_column": "text",
                        "tokenizer_path": "demo-tokenizer",
                    }
                ),
            ),
        ),
    )
    args.experiment.name = "demo-exp"
    for field_name, value in overrides.items():
        setattr(args, field_name, value)
    return args


def test_is_deepspeed_backend_reflects_train_backend():
    """测试 DeepSpeed 后端判断入口。"""

    deepspeed_trainer = PreTrainTrainer(
        make_pretrain_args(train=PreTrainTrainConfig(backend="deepspeed"))
    )
    naive_trainer = PreTrainTrainer(
        make_pretrain_args(train=PreTrainTrainConfig(backend="naive"))
    )

    assert deepspeed_trainer._is_deepspeed_backend() is True
    assert naive_trainer._is_deepspeed_backend() is False


def test_get_train_dataset_config_inherits_seq_len():
    args = make_pretrain_args(
        train=PreTrainTrainConfig(seq_len=2048),
        data=PreTrainDataConfig(
            train=PreTrainTrainDataConfig(
                data_strategy="padding",
                dataset_config={
                    "dataset_path": "jingyaogong/minimind_dataset",
                    "tokenizer_path": "jingyaogong/minimind-3",
                },
            ),
            eval=PreTrainEvalDataConfig(),
        ),
    )
    trainer = PreTrainTrainer(args)

    dataset_config = trainer._get_train_dataset_config()

    assert dataset_config["seq_len"] == 2048
    assert args.data.train.dataset_config.get("seq_len") is None


def test_build_dataloader_uses_train_data_config():
    args = make_pretrain_args(
        train=PreTrainTrainConfig(batch_size=8, seed=42),
        data=PreTrainDataConfig(
            train=PreTrainTrainDataConfig(
                data_strategy="padding",
                dataset_config={"dataset_path": "demo"},
                dataloader_config={
                    "seed": 123,
                    "shuffle": True,
                    "num_workers": 0,
                    "drop_last": True,
                },
            ),
            eval=PreTrainEvalDataConfig(),
        ),
    )
    trainer = PreTrainTrainer(args)

    dataloader = trainer._build_dataloader(DummyDataset())

    assert dataloader.batch_size == 8
    assert dataloader.drop_last is True
    assert isinstance(dataloader.sampler, ResumableDistributedSampler)


def test_build_epoch_iterator_skips_batches_via_sampler_offset():
    args = make_pretrain_args(
        train=PreTrainTrainConfig(batch_size=2),
        data=PreTrainDataConfig(
            train=PreTrainTrainDataConfig(
                data_strategy="padding",
                dataset_config={"dataset_path": "demo"},
                dataloader_config={"seed": 123, "shuffle": False, "num_workers": 0},
            ),
            eval=PreTrainEvalDataConfig(),
        ),
    )
    trainer = PreTrainTrainer(args)
    dataset = CountingDataset(8)
    dataloader = trainer._build_dataloader(dataset)

    iterator = trainer._build_epoch_iterator(dataloader, micro_step_offset=2)
    first_batch = next(iterator)

    assert dataset.visited_indices == [4, 5]
    assert first_batch[0].tolist() == [4, 5]


def test_build_dataloader_worker_init_fn_is_picklable():
    args = make_pretrain_args(
        data=PreTrainDataConfig(
            train=PreTrainTrainDataConfig(
                data_strategy="padding",
                dataset_config={"dataset_path": "demo"},
                dataloader_config={"seed": 123, "shuffle": True, "num_workers": 2},
            ),
            eval=PreTrainEvalDataConfig(),
        )
    )
    trainer = PreTrainTrainer(args)

    dataloader = trainer._build_dataloader(DummyDataset())

    assert dataloader.worker_init_fn is not None
    pickle.dumps(dataloader.worker_init_fn)


def test_build_optimizer_uses_default_adamw():
    trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(naive_config={"learning_rate": 1e-3})
        )
    )

    optimizer = trainer._build_optimizer(DummyModel())

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 0.0
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8


def test_load_deepspeed_config_supports_dict_and_json(monkeypatch, tmp_path):
    config = {
        "gradient_accumulation_steps": 2,
        "zero_optimization": {"stage": 2},
    }
    config_path = tmp_path / "deepspeed_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class FakeDeepSpeed:
        def init_distributed(self):
            return None

    monkeypatch.setattr(
        deepspeed_train_module, "import_module", lambda name: FakeDeepSpeed()
    )

    path_trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(
                backend="deepspeed",
                deepspeed_config=str(config_path),
            )
        )
    )
    dict_trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(
                backend="deepspeed",
                deepspeed_config=config,
            )
        )
    )

    assert path_trainer._init_deepspeed().config == str(config_path)
    assert dict_trainer._init_deepspeed().config == config


def test_run_uses_deepspeed_backend_runtime(monkeypatch, tmp_path):
    calls = []
    deepspeed_config = {
        "gradient_accumulation_steps": 2,
        "zero_optimization": {"stage": 2},
    }
    config_path = tmp_path / "deepspeed_config.json"
    config_path.write_text(json.dumps(deepspeed_config), encoding="utf-8")

    class FakeEngine(nn.Module):
        def __init__(self, model, optimizer):
            super().__init__()
            self.module = model
            self.optimizer = optimizer

        def forward(self, x, y):
            return self.module(x, y)

        def backward(self, loss):
            calls.append(("engine.backward", float(loss.detach().item())))
            loss.backward()

        def step(self):
            calls.append(("engine.step",))

        def zero_grad(self):
            calls.append(("engine.zero_grad",))

        def get_global_grad_norm(self):
            return 0.5

        def gradient_accumulation_steps(self):
            return 2

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

        def zero_grad(self):
            calls.append(("optimizer.zero_grad",))

    class FakeDeepSpeed:
        def init_distributed(self):
            return None

        def initialize(self, **kwargs):
            model_parameters = list(kwargs["model_parameters"])
            optimizer = FakeOptimizer()
            calls.append(
                (
                    "deepspeed.initialize",
                    kwargs["config"],
                    "optimizer" in kwargs,
                    len(model_parameters),
                )
            )
            return (
                FakeEngine(kwargs["model"], optimizer),
                optimizer,
                None,
                None,
            )

    monkeypatch.setattr(
        deepspeed_train_module, "import_module", lambda name: FakeDeepSpeed()
    )
    monkeypatch.setattr(
        pretrain_module, "create_dataset", lambda **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        pretrain_module, "create_model", lambda *args, **kwargs: DummyModel()
    )

    trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(
                epoch_num=0,
                backend="deepspeed",
                deepspeed_config=str(config_path),
                naive_config={"learning_rate": 5e-4},
            ),
        )
    )
    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: [])
    monkeypatch.setattr(trainer, "_init_swanlab", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(
        trainer,
        "_build_optimizer",
        lambda model: (_ for _ in ()).throw(
            AssertionError("deepspeed backend should use optimizer from config")
        ),
    )

    trainer.run()

    assert calls == [
        ("deepspeed.initialize", str(config_path), False, 2),
    ]


def test_run_delegates_deepspeed_step_and_boundary_to_engine(monkeypatch, tmp_path):
    calls = []
    saved_steps = []
    processed_batches = []
    dataloader = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1]), torch.tensor([1])),
        (torch.tensor([2]), torch.tensor([2])),
        (torch.tensor([3]), torch.tensor([3])),
    ]
    deepspeed_config = {
        "gradient_accumulation_steps": 2,
        "zero_optimization": {"stage": 2},
    }
    config_path = tmp_path / "deepspeed_config.json"
    config_path.write_text(json.dumps(deepspeed_config), encoding="utf-8")

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

    class FakeEngine(nn.Module):
        def __init__(self, model, optimizer):
            super().__init__()
            self.module = model
            self.optimizer = optimizer
            self.micro_step = 0

        def forward(self, x, y):
            return self.module(x, y)

        def backward(self, loss):
            calls.append(("engine.backward", float(loss.detach().item())))
            loss.backward()

        def step(self):
            self.micro_step += 1
            calls.append(("engine.step", self.micro_step))

        def is_gradient_accumulation_boundary(self):
            boundary = self.micro_step % 2 == 0
            calls.append(("engine.boundary", self.micro_step, boundary))
            return boundary

        def get_global_grad_norm(self):
            return 0.5

        def gradient_accumulation_steps(self):
            return 2

    class FakeDeepSpeed:
        def init_distributed(self):
            return None

        def initialize(self, **kwargs):
            optimizer = FakeOptimizer()
            return (
                FakeEngine(kwargs["model"], optimizer),
                optimizer,
                None,
                None,
            )

    monkeypatch.setattr(
        deepspeed_train_module, "import_module", lambda name: FakeDeepSpeed()
    )
    monkeypatch.setattr(
        pretrain_module, "create_dataset", lambda **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        pretrain_module,
        "create_model",
        lambda *args, **kwargs: TrainStepModel(processed_batches),
    )

    trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(
                epoch_num=1,
                log_steps=100,
                backend="deepspeed",
                deepspeed_config=str(config_path),
            )
        )
    )
    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: dataloader)
    monkeypatch.setattr(trainer, "_init_swanlab", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(
        trainer,
        "_save_checkpoint",
        lambda checkpoint_dir, global_step, epoch, micro_step_in_epoch, tag=None: (
            saved_steps.append(global_step) if micro_step_in_epoch != 0 else None
        ),
    )

    trainer.run()

    assert processed_batches == [0, 1, 2, 3]
    assert saved_steps == [1, 2]
    assert calls == [
        ("engine.backward", 0.0),
        ("engine.step", 1),
        ("engine.boundary", 1, False),
        ("engine.backward", 0.0),
        ("engine.step", 2),
        ("engine.boundary", 2, True),
        ("engine.backward", 0.0),
        ("engine.step", 3),
        ("engine.boundary", 3, False),
        ("engine.backward", 0.0),
        ("engine.step", 4),
        ("engine.boundary", 4, True),
    ]


def test_deepspeed_grad_norm_none_falls_back_to_zero(monkeypatch, tmp_path):
    config = {
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 0},
    }
    config_path = tmp_path / "deepspeed_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

    class FakeEngine(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.optimizer = FakeOptimizer()

        def forward(self, x, y):
            loss = self.weight * 0 + x.float().mean() * 0
            return x, loss

        def backward(self, loss):
            loss.backward()

        def step(self):
            return None

        def is_gradient_accumulation_boundary(self):
            return True

        def get_global_grad_norm(self):
            return None

        def gradient_accumulation_steps(self):
            return 1

    class FakeDeepSpeed:
        def init_distributed(self):
            return None

        def initialize(self, **kwargs):
            engine = FakeEngine()
            return engine, engine.optimizer, None, None

    monkeypatch.setattr(
        deepspeed_train_module, "import_module", lambda name: FakeDeepSpeed()
    )

    runtime = deepspeed_train_module.DeepSpeedPretrainRuntime(
        make_pretrain_args(
            train=PreTrainTrainConfig(
                backend="deepspeed",
                deepspeed_config=str(config_path),
            )
        )
    )
    runtime.prepare(DummyModel())

    result = runtime.train(
        x=torch.tensor([0]),
        y=torch.tensor([0]),
        device=torch.device("cpu"),
    )

    assert result is not None
    assert result["grad_norm"].item() == 0.0


def test_get_amp_dtype_and_grad_scaler(monkeypatch):
    calls = []

    class FakeGradScaler:
        def __init__(self, device_type):
            calls.append(device_type)

    monkeypatch.setattr(torch.amp, "GradScaler", FakeGradScaler)

    bf16_trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(naive_config={"amp": True, "amp_dtype": "bf16"})
        )
    )
    fp16_trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(naive_config={"amp": True, "amp_dtype": "fp16"})
        )
    )

    assert naive_train_module.get_amp_dtype(bf16_trainer.args) == torch.bfloat16
    assert naive_train_module.get_amp_dtype(fp16_trainer.args) == torch.float16
    assert (
        naive_train_module.build_grad_scaler(bf16_trainer.args, torch.device("cuda"))
        is None
    )
    assert (
        naive_train_module.build_grad_scaler(fp16_trainer.args, torch.device("cpu"))
        is None
    )
    assert (
        naive_train_module.build_grad_scaler(fp16_trainer.args, torch.device("cuda"))
        is not None
    )
    assert calls == ["cuda"]


def test_run_builds_optimizer_without_checkpoint_resume(monkeypatch):
    calls = []

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

        def zero_grad(self):
            pass

        def state_dict(self):
            return {"optimizer": "state"}

    monkeypatch.setattr(
        pretrain_module, "create_dataset", lambda **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        pretrain_module, "create_model", lambda *args, **kwargs: DummyModel()
    )

    trainer = PreTrainTrainer(
        make_pretrain_args(train=PreTrainTrainConfig(epoch_num=0))
    )

    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: [])
    monkeypatch.setattr(
        trainer,
        "_init_swanlab",
        lambda device, dataset, dataloader, run_id=None: calls.append(
            ("init_swanlab", run_id)
        ),
    )
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(
        naive_train_module,
        "maybe_compile_model",
        lambda model, device: calls.append(("compile", device.type)) or model,
    )
    original_to = DummyModel.to

    def fake_to(self, device, *args, **kwargs):
        calls.append(("model.to", str(device)))
        return original_to(self, device, *args, **kwargs)

    monkeypatch.setattr(DummyModel, "to", fake_to)
    monkeypatch.setattr(
        trainer,
        "_build_optimizer",
        lambda model: (
            calls.append(("build_optimizer", next(model.parameters()).device.type))
            or FakeOptimizer()
        ),
    )

    trainer.run()

    assert calls[0] == ("init_swanlab", None)
    assert calls[1][0] == "model.to"
    assert calls[2] == ("build_optimizer", calls[1][1])
    assert calls[3] == ("compile", calls[1][1])


def test_run_starts_from_scratch_when_checkpoint_is_disabled(monkeypatch):
    processed_batches = []
    dataloader = [
        (torch.tensor([0]), torch.tensor([0])),
        (torch.tensor([1]), torch.tensor([1])),
        (torch.tensor([2]), torch.tensor([2])),
        (torch.tensor([3]), torch.tensor([3])),
    ]

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

        def step(self):
            pass

        def zero_grad(self):
            pass

        def state_dict(self):
            return {"optimizer": "state"}

    monkeypatch.setattr(
        pretrain_module, "create_dataset", lambda **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        pretrain_module,
        "create_model",
        lambda *args, **kwargs: TrainStepModel(processed_batches),
    )

    trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(
                epoch_num=1,
                log_steps=100,
                naive_config={"accumulation_steps": 1},
            ),
            checkpoint=PreTrainCheckpointConfig(
                save_checkpoint_dir="checkpoints/test-pretrain"
            ),
        )
    )
    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: dataloader)
    monkeypatch.setattr(trainer, "_init_swanlab", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(
        naive_train_module, "maybe_compile_model", lambda model, device: model
    )
    monkeypatch.setattr(trainer, "_build_optimizer", lambda model: FakeOptimizer())
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda *args, **kwargs: None)

    trainer.run()

    assert processed_batches == [0, 1, 2, 3]


def test_run_does_not_save_final_checkpoint_when_disabled(monkeypatch):
    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 0.0}]

        def zero_grad(self):
            pass

        def state_dict(self):
            return {"optimizer": "state"}

    monkeypatch.setattr(
        pretrain_module, "create_dataset", lambda **kwargs: DummyDataset()
    )
    monkeypatch.setattr(
        pretrain_module, "create_model", lambda *args, **kwargs: DummyModel()
    )

    trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(epoch_num=0),
            checkpoint=PreTrainCheckpointConfig(
                save_checkpoint_dir="checkpoints/test-pretrain"
            ),
        )
    )
    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: [])
    monkeypatch.setattr(trainer, "_init_swanlab", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(
        naive_train_module, "maybe_compile_model", lambda model, device: model
    )
    monkeypatch.setattr(trainer, "_build_optimizer", lambda model: FakeOptimizer())

    trainer.run()


def test_get_dataloader_seed_and_set_seed():
    trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(seed=999),
            data=PreTrainDataConfig(
                train=PreTrainTrainDataConfig(
                    data_strategy="padding",
                    dataset_config={"dataset_path": "demo"},
                ),
                eval=PreTrainEvalDataConfig(),
            ),
        )
    )

    assert trainer._get_dataloader_seed() == 999

    trainer._set_seed(321)
    random_value_1 = random.randint(0, 1000)
    torch_value_1 = torch.randint(0, 1000, (1,)).item()
    trainer._set_seed(321)
    random_value_2 = random.randint(0, 1000)
    torch_value_2 = torch.randint(0, 1000, (1,)).item()

    assert random_value_1 == random_value_2
    assert torch_value_1 == torch_value_2


def test_init_swanlab_uses_experiment_and_train_data(monkeypatch):
    calls = {}

    class FakeRunPublic:
        run_id = "run-123"

    class FakeRun:
        public = FakeRunPublic()

    class FakeSwanlab:
        def init(self, **kwargs):
            calls["init"] = kwargs
            return FakeRun()

        def log(self, data):
            calls.setdefault("log", []).append(data)

        def finish(self):
            calls["finished"] = True

    monkeypatch.setattr(pretrain_module, "import_module", lambda name: FakeSwanlab())

    args = make_pretrain_args(
        train=PreTrainTrainConfig(batch_size=2),
        data=PreTrainDataConfig(
            train=PreTrainTrainDataConfig(
                data_strategy="padding",
                dataset_config={"dataset_path": "demo"},
                dataloader_config={"seed": 7},
            ),
            eval=PreTrainEvalDataConfig(),
        ),
    )
    args.experiment.name = "demo-exp"
    args.experiment.swanlab.enabled = True
    args.experiment.swanlab.project = "demo-project"
    args.experiment.swanlab.tags = ["unit"]
    trainer = PreTrainTrainer(args)
    dataloader = trainer._build_dataloader(DummyDataset())

    trainer._init_swanlab(torch.device("cpu"), DummyDataset(), dataloader)
    trainer._log_swanlab({"train/epoch": 0})
    trainer._finish_swanlab()

    assert calls["init"]["project"] == "demo-project"
    assert calls["init"]["experiment_name"] == "demo-exp"
    assert calls["init"]["tags"] == ["unit"]
    assert calls["init"]["config"]["train"]["batch_size"] == 2
    assert calls["init"]["config"]["data"]["train"]["dataloader_config"]["seed"] == 7
    assert calls["init"]["config"]["data"]["train"]["dataset_config"]["seq_len"] == 1024
    assert "experiment" not in calls["init"]["config"]
    assert calls["init"]["config"]["runtime"]["device"] == "cpu"
    assert trainer._swanlab_run_id == "run-123"
    assert calls["log"] == [{"train/epoch": 0}]
    assert calls["finished"] is True
