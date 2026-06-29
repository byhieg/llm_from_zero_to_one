import pickle
import random

import torch
import torch.nn as nn

import trainer.pretrain.pretrain as pretrain_module
from checkpoint_manager import Checkpoint
from trainer.common_args import ModelConfig
from trainer.pretrain.pretrain import PreTrainTrainer, ResumableDistributedSampler
from trainer.pretrain.pretrain_args import (
    PreTrainArgs,
    PreTrainCheckpointConfig,
    PreTrainDataConfig,
    PreTrainEvalConfig,
    PreTrainEvalDataConfig,
    PreTrainOptimizerConfig,
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
            PreTrainCheckpointConfig(checkpoint_dir="checkpoints/pretrain"),
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
        optimizer=overrides.pop("optimizer", PreTrainOptimizerConfig()),
    )
    args.experiment.name = "demo-exp"
    for field_name, value in overrides.items():
        setattr(args, field_name, value)
    return args


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


def test_build_optimizer_supports_adamw_and_adam():
    adamw_trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(naive_config={"learning_rate": 1e-3}),
            optimizer=PreTrainOptimizerConfig(
                name="adamw",
                weight_decay=0.1,
                betas=[0.8, 0.95],
                eps=1e-6,
            ),
        )
    )
    adam_trainer = PreTrainTrainer(
        make_pretrain_args(
            train=PreTrainTrainConfig(naive_config={"learning_rate": 5e-4}),
            optimizer=PreTrainOptimizerConfig(name="adam"),
        )
    )

    adamw = adamw_trainer._build_optimizer(DummyModel())
    adam = adam_trainer._build_optimizer(DummyModel())

    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.defaults["lr"] == 1e-3
    assert adamw.defaults["weight_decay"] == 0.1
    assert adamw.defaults["betas"] == (0.8, 0.95)
    assert adamw.defaults["eps"] == 1e-6
    assert isinstance(adam, torch.optim.Adam)


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

    assert bf16_trainer._get_amp_dtype() == torch.bfloat16
    assert fp16_trainer._get_amp_dtype() == torch.float16
    assert bf16_trainer._build_grad_scaler(torch.device("cuda")) is None
    assert fp16_trainer._build_grad_scaler(torch.device("cpu")) is None
    assert fp16_trainer._build_grad_scaler(torch.device("cuda")) is not None
    assert calls == ["cuda"]


def test_pretrain_trainer_uses_experiment_name_for_checkpoint_manager(monkeypatch):
    calls = {}

    class FakeCheckpointManager:
        def __init__(self, checkpoint_config, model_name):
            calls["checkpoint_dir"] = checkpoint_config.checkpoint_dir
            calls["model_name"] = model_name

    monkeypatch.setattr(pretrain_module, "CheckpointManager", FakeCheckpointManager)

    args = make_pretrain_args(
        checkpoint=PreTrainCheckpointConfig(checkpoint_dir="checkpoints/pretrain")
    )
    args.experiment.name = "minimind_61m_pretrain"
    trainer = PreTrainTrainer(args)

    assert trainer.checkpoint_manager is not None
    assert calls == {
        "checkpoint_dir": "checkpoints/pretrain",
        "model_name": "minimind_61m_pretrain",
    }


def test_run_builds_optimizer_before_loading_optimizer_state(monkeypatch):
    calls = []

    class FakeCheckpointManager:
        def __init__(self, checkpoint_config, model_name):
            pass

        def get_checkpoint(self):
            return None

        def save_checkpoint(self, checkpoint, step):
            calls.append(("save_checkpoint", step, checkpoint.metadata["epoch"]))

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

    trainer = PreTrainTrainer(
        make_pretrain_args(train=PreTrainTrainConfig(epoch_num=0))
    )
    monkeypatch.setattr(
        trainer.checkpoint_manager,
        "get_checkpoint",
        lambda: Checkpoint(
            model_state_dict={
                "linear.weight": torch.ones((2, 4)),
                "linear.bias": torch.zeros(2),
            },
            optimizer_state_dict={
                "state": {},
                "param_groups": [{"lr": 1e-3, "params": [0, 1]}],
            },
            metadata={
                "global_step": 7,
                "resume_config": trainer._get_checkpoint_resume_config(),
                "swanlab_run_id": "run-321",
            },
        ),
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
    monkeypatch.setattr(
        trainer,
        "_build_optimizer",
        lambda model: (
            calls.append(("build_optimizer", next(model.parameters()).device.type))
            or FakeOptimizer()
        ),
    )

    trainer.run()

    assert calls[0] == ("model.load_state_dict", ["linear.bias", "linear.weight"])
    assert calls[1] == ("init_swanlab", "run-321")
    assert calls[2][0] == "model.to"
    assert calls[3] == ("build_optimizer", calls[2][1])
    assert calls[4][0] == "optimizer.load_state_dict"
    assert calls[5] == ("compile", calls[2][1])
    assert calls[6] == ("save_checkpoint", 7, 0)


def test_run_skips_consumed_micro_batches_when_resuming(monkeypatch):
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
                checkpoint_dir="checkpoints/test-pretrain"
            ),
        )
    )
    monkeypatch.setattr(
        trainer.checkpoint_manager,
        "get_checkpoint",
        lambda: Checkpoint(
            model_state_dict=TrainStepModel([]).state_dict(),
            optimizer_state_dict={},
            metadata={
                "global_step": 3,
                "epoch": 0,
                "micro_step_in_epoch": 2,
                "resume_config": trainer._get_checkpoint_resume_config(),
            },
        ),
    )
    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: dataloader)
    monkeypatch.setattr(trainer, "_init_swanlab", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(trainer, "_maybe_compile_model", lambda model, device: model)
    monkeypatch.setattr(trainer, "_build_optimizer", lambda model: FakeOptimizer())
    monkeypatch.setattr(trainer, "_save_checkpoint_if_needed", lambda **kwargs: None)

    trainer.run()

    assert processed_batches == [2, 3]


def test_save_training_checkpoint_persists_resume_position(monkeypatch):
    saved_checkpoints = []
    trainer = PreTrainTrainer(
        make_pretrain_args(
            checkpoint=PreTrainCheckpointConfig(
                checkpoint_dir="checkpoints/test-pretrain", save_steps=1
            )
        )
    )
    model = DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    monkeypatch.setattr(
        trainer.checkpoint_manager,
        "save_checkpoint",
        lambda checkpoint, step: saved_checkpoints.append((checkpoint, step)),
    )
    trainer._swanlab_run_id = "run-789"

    trainer._save_training_checkpoint(
        model=model,
        optimizer=optimizer,
        global_step=6,
        epoch=0,
        micro_step_in_epoch=4,
        dataloader_length=4,
    )

    checkpoint, step = saved_checkpoints[0]

    assert step == 6
    assert checkpoint.metadata["global_step"] == 6
    assert checkpoint.metadata["epoch"] == 1
    assert checkpoint.metadata["micro_step_in_epoch"] == 0
    assert (
        checkpoint.metadata["resume_config"] == trainer._get_checkpoint_resume_config()
    )
    assert checkpoint.metadata["swanlab_run_id"] == "run-789"


def test_run_saves_final_checkpoint_even_without_updates(monkeypatch):
    saved_checkpoints = []

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
                checkpoint_dir="checkpoints/test-pretrain"
            ),
        )
    )
    monkeypatch.setattr(
        trainer.checkpoint_manager,
        "save_checkpoint",
        lambda checkpoint, step: saved_checkpoints.append((checkpoint, step)),
    )
    monkeypatch.setattr(trainer.checkpoint_manager, "get_checkpoint", lambda: None)
    monkeypatch.setattr(trainer, "_init_seed", lambda: None)
    monkeypatch.setattr(trainer, "_build_dataloader", lambda dataset: [])
    monkeypatch.setattr(trainer, "_init_swanlab", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer, "_finish_swanlab", lambda: None)
    monkeypatch.setattr(trainer, "_maybe_compile_model", lambda model, device: model)
    monkeypatch.setattr(trainer, "_build_optimizer", lambda model: FakeOptimizer())

    trainer.run()

    checkpoint, step = saved_checkpoints[0]
    assert len(saved_checkpoints) == 1
    assert step == 0
    assert checkpoint.metadata["global_step"] == 0
    assert checkpoint.metadata["epoch"] == 0
    assert checkpoint.metadata["micro_step_in_epoch"] == 0


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


def test_run_eval_if_needed_logs_metrics(monkeypatch):
    trainer = PreTrainTrainer(
        make_pretrain_args(
            eval=PreTrainEvalConfig(steps=2),
            data=PreTrainDataConfig(
                train=PreTrainTrainDataConfig(
                    data_strategy="padding",
                    dataset_config={"dataset_path": "demo"},
                ),
                eval=PreTrainEvalDataConfig(
                    dataset_config={
                        "dataset_path": "json",
                        "text_column": "text",
                        "tokenizer_path": "demo-tokenizer",
                    }
                ),
            ),
        )
    )
    logged = []

    class FakeEvaluator:
        def evaluate_model(self, model, device, checkpoint_step):
            assert str(device) == "cpu"
            assert checkpoint_step == 2
            return {
                "loss": 1.5,
                "perplexity": 4.48,
                "token_count": 128.0,
                "sample_count": 8.0,
            }

    monkeypatch.setattr(trainer, "_get_pretrain_evaluator", lambda: FakeEvaluator())
    monkeypatch.setattr(trainer, "_log_swanlab", lambda data: logged.append(data))

    skipped = trainer._run_eval_if_needed(DummyModel(), torch.device("cpu"), 1)
    elapsed = trainer._run_eval_if_needed(DummyModel(), torch.device("cpu"), 2)

    assert skipped == 0.0
    assert elapsed >= 0.0
    assert logged == [
        {
            "eval/step": 2,
            "eval/loss": 1.5,
            "eval/perplexity": 4.48,
            "eval/token_count": 128.0,
            "eval/sample_count": 8.0,
        }
    ]
