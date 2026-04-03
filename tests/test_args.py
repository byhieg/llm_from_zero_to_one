import os
import pytest
import tempfile
from pathlib import Path
from trainer.train_args import (
    PretrainArgs,
    TrainingArgs,
    TrainingConfig,
    CheckpointConfig,
    DataConfig,
    InferenceConfig,
    OptimizerConfig,
    SwanlabConfig,
    load_args_from_yaml,
    register_args,
    get_args_class,
    list_modes,
    generate_default_config,
    _substitute_env_vars,
    _resolve_config_path,
)
from dataclasses import dataclass


def test_list_modes():
    modes = list_modes()
    assert "pretrain" in modes


def test_get_args_class():
    args_cls = get_args_class("pretrain")
    assert args_cls.__name__ == "PretrainArgs"


def test_get_args_class_invalid():
    with pytest.raises(ValueError, match="Unknown mode"):
        get_args_class("invalid_mode")


def test_from_yaml_basic():
    yaml_content = """
training:
  batch_size: 32
  learning_rate: 0.001
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, mode = load_args_from_yaml("pretrain", f.name, validate=False)

    assert args.training.batch_size == 32
    assert args.training.learning_rate == 0.001
    assert mode == "pretrain"
    assert args.training.seq_len == 1024
    assert args.training.epoch_num == 1


def test_from_yaml_all_params():
    yaml_content = """
training:
  batch_size: 32
  seq_len: 512
  epoch_num: 10
  learning_rate: 0.001
  warmup_steps: 100
  grad_clip: 0.5
  accumulation_steps: 2
  log_steps: 50

checkpoint:
  save_steps: 500
  checkpoint_dir: checkpoints/test
  resume_from_checkpoint: null

data:
  data_path: /path/to/data

inference:
  steps: 1000
  tokens: 50
  topk: 200
  temperature: 0.8
  prompt: "Hello world"

optimizer:
  name: adamw
  weight_decay: 0.1
  betas: [0.8, 0.95]
  eps: 1.0e-6

swanlab:
  enabled: true
  project: demo-project
  experiment_name: exp-1
  tags: ["demo"]
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, mode = load_args_from_yaml("pretrain", f.name, validate=False)

    assert args.training.batch_size == 32
    assert args.training.seq_len == 512
    assert args.training.epoch_num == 10
    assert args.data.data_path == "/path/to/data"
    assert args.training.learning_rate == 0.001
    assert args.training.warmup_steps == 100
    assert args.training.grad_clip == 0.5
    assert args.training.accumulation_steps == 2
    assert args.training.log_steps == 50
    assert args.checkpoint.save_steps == 500
    assert args.checkpoint.checkpoint_dir == "checkpoints/test"
    assert args.inference.steps == 1000
    assert args.inference.tokens == 50
    assert args.inference.topk == 200
    assert args.inference.temperature == 0.8
    assert args.inference.prompt == "Hello world"
    assert args.optimizer.name == "adamw"
    assert args.optimizer.weight_decay == 0.1
    assert tuple(args.optimizer.betas) == (0.8, 0.95)
    assert args.optimizer.eps == 1.0e-6
    assert args.swanlab.enabled is True
    assert args.swanlab.project == "demo-project"
    assert args.swanlab.experiment_name == "exp-1"
    assert args.swanlab.tags == ["demo"]


def test_to_yaml():
    args = PretrainArgs(
        training=TrainingConfig(batch_size=64, learning_rate=0.0001, epoch_num=5),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        args.to_yaml(f.name)
        loaded, _ = load_args_from_yaml("pretrain", f.name, validate=False)

    assert loaded.training.batch_size == 64
    assert loaded.training.learning_rate == 0.0001
    assert loaded.training.epoch_num == 5


def test_swanlab_config_defaults():
    args = PretrainArgs()

    assert isinstance(args.optimizer, OptimizerConfig)
    assert args.optimizer.name == "adamw"
    assert isinstance(args.swanlab, SwanlabConfig)
    assert args.swanlab.enabled is False
    assert args.swanlab.project == "llm-training"


def test_to_dict():
    args = PretrainArgs(training=TrainingConfig(batch_size=32))
    d = args.to_dict()

    assert isinstance(d, dict)
    assert d["training"]["batch_size"] == 32
    assert "learning_rate" in d["training"]
    assert "steps" in d["inference"]


def test_from_dict():
    d = {
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
        },
    }
    args = PretrainArgs.from_dict(d)

    assert args.training.batch_size == 64
    assert args.training.learning_rate == 0.001


def test_load_args_from_yaml_factory():
    yaml_content = """
training:
  batch_size: 128
  epoch_num: 20
data:
  data_path: /data
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, mode = load_args_from_yaml("pretrain", f.name, validate=False)

    assert isinstance(args, PretrainArgs)
    assert args.training.batch_size == 128
    assert args.training.epoch_num == 20
    assert mode == "pretrain"


def test_register_custom_args():
    @dataclass
    class CustomArgs(TrainingArgs):
        custom_param: int = 100

    register_args("custom", CustomArgs)
    assert "custom" in list_modes()

    yaml_content = """
training:
  batch_size: 32
data:
  data_path: /data
custom_param: 200
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, mode = load_args_from_yaml("custom", f.name, validate=False)

    assert args.custom_param == 200
    assert args.training.batch_size == 32


def test_optional_param():
    yaml_content = """
checkpoint:
  resume_from_checkpoint: path/to/checkpoint
data:
  data_path: /data
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, _ = load_args_from_yaml("pretrain", f.name, validate=False)

    assert args.checkpoint.resume_from_checkpoint == "path/to/checkpoint"

    yaml_content2 = """
training:
  batch_size: 16
data:
  data_path: /data
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content2)
        f.flush()
        args2, _ = load_args_from_yaml("pretrain", f.name, validate=False)

    assert args2.checkpoint.resume_from_checkpoint is None


class TestAutoModeDetection:
    def test_mode_from_yaml(self):
        yaml_content = """
mode: pretrain
training:
  batch_size: 64
data:
  data_path: /data/train
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            args, mode = load_args_from_yaml(config_path=f.name, validate=False)

        assert mode == "pretrain"
        assert args.training.batch_size == 64
        assert isinstance(args, PretrainArgs)

    def test_mode_override(self):
        yaml_content = """
mode: pretrain
training:
  batch_size: 64
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            args, mode = load_args_from_yaml(
                mode="pretrain", config_path=f.name, validate=False
            )

        assert mode == "pretrain"

    def test_no_mode_error(self):
        yaml_content = """
training:
  batch_size: 64
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError, match="mode not specified"):
                load_args_from_yaml(config_path=f.name, validate=False)

    def test_no_config_no_mode_error(self):
        with pytest.raises(ValueError, match="Either 'mode' or 'config_path'"):
            load_args_from_yaml()


class TestEnvVarSubstitution:
    def test_simple_substitution(self):
        os.environ["TEST_DATA_PATH"] = "/my/data"
        data = {"path": "${TEST_DATA_PATH}"}
        result = _substitute_env_vars(data)
        assert result["path"] == "/my/data"
        del os.environ["TEST_DATA_PATH"]

    def test_default_value(self):
        data = {"path": "${UNDEFINED_VAR:-/default/path}"}
        result = _substitute_env_vars(data)
        assert result["path"] == "/default/path"

    def test_nested_substitution(self):
        os.environ["BASE_DIR"] = "/base"
        data = {
            "train": {
                "data_path": "${BASE_DIR}/train",
                "output_path": "${BASE_DIR}/output",
            },
            "paths": ["${BASE_DIR}/a", "${BASE_DIR}/b"],
        }
        result = _substitute_env_vars(data)
        assert result["train"]["data_path"] == "/base/train"
        assert result["train"]["output_path"] == "/base/output"
        assert result["paths"] == ["/base/a", "/base/b"]
        del os.environ["BASE_DIR"]

    def test_preserve_undefined(self):
        data = {"path": "${TOTALLY_UNDEFINED_VAR}"}
        result = _substitute_env_vars(data)
        assert result["path"] == "${TOTALLY_UNDEFINED_VAR}"


class TestValidation:
    def test_valid_config(self):
        args = PretrainArgs(
            data=DataConfig(
                data_strategy='padding',
                dataset_config={'data_path': 'roneneldan/TinyStories'},
                dataloader_config={'num_workers': 0}
            ),
            training=TrainingConfig(batch_size=32, learning_rate=0.001),
        )
        errors = args.validate()
        assert errors == []

    def test_missing_data_path(self):
        args = PretrainArgs(
            training=TrainingConfig(epoch_num=10),
            data=DataConfig(data_strategy='padding', dataset_config={})
        )
        errors = args.validate()
        assert any("data_path" in e for e in errors)

    def test_invalid_batch_size(self):
        args = PretrainArgs(
            training=TrainingConfig(batch_size=0),
            data=DataConfig(data_strategy='padding', dataset_config={'data_path': '/data'})
        )
        errors = args.validate()
        assert any("batch_size" in e for e in errors)

    def test_invalid_learning_rate(self):
        args = PretrainArgs(
            training=TrainingConfig(learning_rate=-0.001),
            data=DataConfig(data_strategy='padding', dataset_config={'data_path': '/data'}),
        )
        errors = args.validate()
        assert any("learning_rate" in e for e in errors)

    def test_invalid_dataloader_num_workers(self):
        args = PretrainArgs(
            data=DataConfig(
                data_strategy='padding',
                dataset_config={'data_path': '/data'},
                dataloader_config={'num_workers': -1},
            )
        )
        errors = args.validate()
        assert any("num_workers" in e for e in errors)

    def test_invalid_optimizer_name(self):
        args = PretrainArgs()
        args.optimizer.name = "sgd"
        errors = args.validate()
        assert any("optimizer.name" in e for e in errors)

    def test_invalid_weight_decay(self):
        args = PretrainArgs()
        args.optimizer.weight_decay = -0.1
        errors = args.validate()
        assert any("optimizer.weight_decay" in e for e in errors)

    def test_invalid_optimizer_betas(self):
        args = PretrainArgs()
        args.optimizer.betas = [0.9]
        errors = args.validate()
        assert any("optimizer.betas" in e for e in errors)

    def test_validation_on_load(self):
        yaml_content = """
training:
  epoch_num: 10
data:
  data_path: ""
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ValueError, match="validation"):
                load_args_from_yaml("pretrain", f.name, validate=True)

            args, _ = load_args_from_yaml("pretrain", f.name, validate=False)
            assert args.training.epoch_num == 10


class TestGenerateDefaultConfig:
    def test_generate_to_temp(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = generate_default_config("pretrain", f.name)

        assert path.exists()

        args, mode = load_args_from_yaml(config_path=path, validate=False)
        assert args.training.batch_size == 16
        assert mode == "pretrain"

    def test_generate_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "config.yaml"
            result = generate_default_config("pretrain", path)

            assert result.exists()
            assert result.parent.exists()

    def test_generated_config_has_mode(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = generate_default_config("pretrain", f.name)

        content = path.read_text()
        assert "mode: pretrain" in content


class TestResolveConfigPath:
    def test_explicit_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("training:\n  batch_size: 32")
            f.flush()

            result = _resolve_config_path("pretrain", f.name)
            assert result == Path(f.name)

    def test_missing_explicit_path(self):
        with pytest.raises(FileNotFoundError):
            _resolve_config_path("pretrain", "/nonexistent/config.yaml")

    def test_default_path_missing(self):
        with pytest.raises(FileNotFoundError, match="Tip:"):
            _resolve_config_path("nonexistent_mode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
