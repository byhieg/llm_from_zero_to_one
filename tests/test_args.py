import os
import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass

from trainer.train_args import (
    load_args_from_yaml,
    register_args,
    get_args_class,
    list_modes,
    generate_default_config,
    _substitute_env_vars,
    _resolve_config_path,
)
from trainer.common_args import (
    TrainingArgs,
    ModelConfig,
    DataConfig,
    CheckpointConfig,
    ExperimentConfig,
)
from trainer.pretrain.pretrain_args import (
    PretrainArgs,
    PretrainTrainingConfig,
    PretrainEvalConfig,
    PretrainOptimizerConfig,
)
from evaluator.eval_args import EvalArgs, EvalConfig


def test_list_modes():
    modes = list_modes()
    assert "eval" in modes
    assert "pretrain" in modes


def test_get_args_class():
    args_cls = get_args_class("pretrain")
    assert args_cls.__name__ == "PretrainArgs"

    eval_args_cls = get_args_class("eval")
    assert eval_args_cls.__name__ == "EvalArgs"


def test_get_args_class_invalid():
    with pytest.raises(ValueError, match="Unknown mode"):
        get_args_class("invalid_mode")


def test_from_yaml_basic():
    yaml_content = """
training:
  batch_size: 32
  naive_config:
    learning_rate: 0.001
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, mode = load_args_from_yaml("pretrain", f.name, validate=False)

    assert args.training.batch_size == 32
    assert args.training.naive_config.get("learning_rate") == 0.001
    assert mode == "pretrain"
    assert args.training.seq_len == 1024
    assert args.training.epoch_num == 1


def test_from_yaml_all_params():
    yaml_content = """
name: minimind_61m_pretrain

training:
  batch_size: 32
  seq_len: 512
  epoch_num: 10
  naive_config:
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
  dataset_config:
    dataset_path: /path/to/data

eval:
  steps: 100
  dataset_path: wikitext
  dataset_name: wikitext-2-raw-v1
  data_files:
    validation: /tmp/eval.jsonl
  split: validation
  text_column: text
  tokenizer_path: demo-tokenizer
  max_samples: 50
  batch_size: 4
  add_bos_id: true
  add_eos_id: false

optimizer:
  name: adamw
  weight_decay: 0.1
  betas: [0.8, 0.95]
  eps: 1.0e-6

experiment:
  enabled: true
  project: demo-project
  experiment_name: exp-1
  tags: ["demo"]
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, _ = load_args_from_yaml("pretrain", f.name, validate=False)

    assert args.training.batch_size == 32
    assert args.name == "minimind_61m_pretrain"
    assert args.training.seq_len == 512
    assert args.training.epoch_num == 10
    assert args.data.dataset_config["dataset_path"] == "/path/to/data"
    assert args.training.naive_config.get("learning_rate") == 0.001
    assert args.training.naive_config.get("warmup_steps") == 100
    assert args.training.naive_config.get("grad_clip") == 0.5
    assert args.training.naive_config.get("accumulation_steps") == 2
    assert args.training.log_steps == 50
    assert args.checkpoint.save_steps == 500
    assert args.checkpoint.checkpoint_dir == "checkpoints/test"
    assert args.eval.steps == 100
    assert args.eval.dataset_path == "wikitext"
    assert args.eval.dataset_name == "wikitext-2-raw-v1"
    assert args.eval.data_files == {"validation": "/tmp/eval.jsonl"}
    assert args.eval.split == "validation"
    assert args.eval.text_column == "text"
    assert args.eval.tokenizer_path == "demo-tokenizer"
    assert args.eval.max_samples == 50
    assert args.eval.batch_size == 4
    assert args.eval.add_bos_id is True
    assert args.eval.add_eos_id is False
    assert args.optimizer.name == "adamw"
    assert args.optimizer.weight_decay == 0.1
    assert tuple(args.optimizer.betas) == (0.8, 0.95)
    assert args.optimizer.eps == 1.0e-6
    assert args.experiment.enabled is True
    assert args.experiment.project == "demo-project"
    assert args.experiment.experiment_name == "exp-1"
    assert args.experiment.tags == ["demo"]


def test_load_eval_args_from_yaml():
    yaml_content = """
mode: eval
name: minimind_61m_eval
model:
  name: gpt2
  config:
    vocab_size: 100
checkpoint:
  checkpoint_dir: checkpoints/pretrain
eval:
  dataset_path: "wikitext"
  data_files:
    test: "/tmp/eval.jsonl"
  text_column: "text"
  tokenizer_path: "demo-tokenizer"
  checkpoint_step: 42
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, mode = load_args_from_yaml(config_path=f.name, validate=True)

    assert mode == "eval"
    assert isinstance(args, EvalArgs)
    assert args.name == "minimind_61m_eval"
    assert args.eval.dataset_path == "wikitext"
    assert args.eval.data_files == {"test": "/tmp/eval.jsonl"}
    assert args.eval.text_column == "text"
    assert args.eval.tokenizer_path == "demo-tokenizer"
    assert args.eval.checkpoint_step == 42


def test_to_yaml():
    args = PretrainArgs(
        training=PretrainTrainingConfig(batch_size=64, epoch_num=5),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        args.to_yaml(f.name)
        loaded, _ = load_args_from_yaml("pretrain", f.name, validate=False)

    assert loaded.training.batch_size == 64
    assert loaded.training.epoch_num == 5


def test_experiment_config_defaults():
    args = PretrainArgs()

    assert isinstance(args.optimizer, PretrainOptimizerConfig)
    assert args.optimizer.name == "adamw"
    assert isinstance(args.experiment, ExperimentConfig)
    assert args.experiment.enabled is False
    assert args.experiment.project == "llm-training"


def test_to_dict():
    args = PretrainArgs(training=PretrainTrainingConfig(batch_size=32))
    d = args.to_dict()

    assert isinstance(d, dict)
    assert d["training"]["batch_size"] == 32
    assert "naive_config" in d["training"]
    assert "dataset_path" in d["eval"]


def test_from_dict():
    d = {
        "training": {
            "batch_size": 64,
            "naive_config": {
                "learning_rate": 0.001,
            },
        },
    }
    args = PretrainArgs.from_dict(d)

    assert args.training.batch_size == 64
    assert args.training.naive_config.get("learning_rate") == 0.001


def test_load_args_from_yaml_factory():
    yaml_content = """
training:
  batch_size: 128
  epoch_num: 20
data:
  dataset_config:
    dataset_path: /data
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
  dataset_config:
    dataset_path: /data
custom_param: 200
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, _ = load_args_from_yaml("custom", f.name, validate=False)

    assert args.custom_param == 200


def test_optional_param():
    yaml_content = """
checkpoint:
  resume_from_checkpoint: path/to/checkpoint
data:
  dataset_config:
    dataset_path: /data
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
  dataset_config:
    dataset_path: /data
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
  dataset_config:
    dataset_path: /data/train
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
            _, mode = load_args_from_yaml(
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
                "dataset_path": "${BASE_DIR}/train",
                "output_path": "${BASE_DIR}/output",
            },
            "paths": ["${BASE_DIR}/a", "${BASE_DIR}/b"],
        }
        result = _substitute_env_vars(data)
        assert result["train"]["dataset_path"] == "/base/train"
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
                data_strategy="padding",
                dataset_config={"dataset_path": "roneneldan/TinyStories"},
                dataloader_config={"num_workers": 0},
            ),
            training=PretrainTrainingConfig(batch_size=32, naive_config={"learning_rate": 0.001}),
        )
        errors = args.validate()
        assert errors == []

    def test_missing_dataset_path(self):
        args = PretrainArgs(
            training=PretrainTrainingConfig(epoch_num=10),
            data=DataConfig(data_strategy="padding", dataset_config={}),
        )
        errors = args.validate()
        assert any("dataset_path" in e for e in errors)

    def test_invalid_batch_size(self):
        args = PretrainArgs(
            training=PretrainTrainingConfig(batch_size=0),
            data=DataConfig(
                data_strategy="padding", dataset_config={"dataset_path": "/data"}
            ),
        )
        errors = args.validate()
        assert any("batch_size" in e for e in errors)

    def test_invalid_learning_rate(self):
        args = PretrainArgs(
            training=PretrainTrainingConfig(naive_config={"learning_rate": -0.001}),
            data=DataConfig(
                data_strategy="padding", dataset_config={"dataset_path": "/data"}
            ),
        )
        errors = args.validate()
        assert any("learning_rate" in e for e in errors)

    def test_invalid_dataloader_num_workers(self):
        args = PretrainArgs(
            data=DataConfig(
                data_strategy="padding",
                dataset_config={"dataset_path": "/data"},
                dataloader_config={"num_workers": -1},
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

    def test_eval_requires_required_fields(self):
        args = EvalArgs(
            checkpoint=CheckpointConfig(checkpoint_dir="checkpoints/pretrain"),
            model=ModelConfig(name="gpt2", config={}),
        )
        errors = args.validate()
        assert any("eval.dataset_path" in e for e in errors)
        assert any("eval.tokenizer_path" in e for e in errors)

    def test_pretrain_eval_requires_fields_when_enabled(self):
        args = PretrainArgs(eval=PretrainEvalConfig(steps=100))
        errors = args.validate()
        assert any("eval.dataset_path" in e for e in errors)
        assert any("eval.tokenizer_path" in e for e in errors)

    def test_validation_on_load(self):
        yaml_content = """
training:
  epoch_num: 10
data:
  dataset_config:
    dataset_path: ""
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