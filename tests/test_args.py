import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from evaluator.eval_args import EvalArgs, EvalCheckpointConfig
from trainer.common_args import ModelConfig, TrainingArgs
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
from trainer.train_args import (
    _resolve_config_path,
    _substitute_env_vars,
    generate_default_config,
    get_args_class,
    list_modes,
    load_args_from_yaml,
    register_args,
)


def _write_temp_yaml(content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        handle.write(content)
        handle.flush()
        return handle.name


def test_list_modes_contains_eval_and_pretrain():
    assert {"eval", "pretrain"} <= set(list_modes())


def test_get_args_class_returns_refactored_pretrain_args():
    assert get_args_class("pretrain") is PreTrainArgs
    assert get_args_class("eval") is EvalArgs


def test_get_args_class_invalid_mode():
    with pytest.raises(ValueError, match="Unknown mode"):
        get_args_class("invalid_mode")


def test_load_pretrain_args_from_demo_shape():
    yaml_path = _write_temp_yaml(
        """
experiment:
  mode: pretrain
  name: minimind_dataset_gpt2_61M
  swanlab:
    enabled: true
    project: llm-pretrain
    tags: [demo]
model:
  name: gpt2
  config:
    vocab_size: 6400
train:
  batch_size: 32
  seq_len: 340
  epoch_num: 2
  backend: deepspeed
  deepspeed_config: xxx/deepspeed_config.json
eval:
  steps: 100
  max_samples: 1024
  batch_size: 16
checkpoint:
  save_steps: 1000
  checkpoint_dir: checkpoints/pretrain
data:
  train:
    data_strategy: padding
    dataset_config:
      dataset_path: json
      data_files:
        train: /tmp/train.jsonl
    dataloader_config:
      num_workers: 8
  eval:
    dataset_config:
      dataset_path: json
      text_column: text
      tokenizer_path: demo-tokenizer
optimizer:
  name: adamw
"""
    )

    args, mode = load_args_from_yaml(config_path=yaml_path, validate=False)

    assert mode == "pretrain"
    assert isinstance(args, PreTrainArgs)
    assert args.experiment.mode == "pretrain"
    assert args.experiment.name == "minimind_dataset_gpt2_61M"
    assert args.experiment.swanlab.enabled is True
    assert args.experiment.swanlab.project == "llm-pretrain"
    assert args.train.batch_size == 32
    assert args.train.backend == "deepspeed"
    assert args.train.deepspeed_config == "xxx/deepspeed_config.json"
    assert args.eval.steps == 100
    assert args.data.train.dataset_config["dataset_path"] == "json"
    assert args.data.eval.dataset_config["tokenizer_path"] == "demo-tokenizer"


def test_load_pretrain_args_from_legacy_shape():
    yaml_path = _write_temp_yaml(
        """
mode: pretrain
name: legacy-exp
swanlab:
  enabled: true
  project: legacy-project
  tags: [legacy]
training:
  batch_size: 8
  learning_rate: 0.001
  amp: true
  amp_dtype: fp16
data:
  data_strategy: padding
  dataset_config:
    dataset_path: json
  dataloader_config:
    seed: 7
eval:
  steps: 10
  dataset_path: json
  text_column: text
  tokenizer_path: demo-tokenizer
"""
    )

    args, mode = load_args_from_yaml(config_path=yaml_path, validate=False)

    assert mode == "pretrain"
    assert args.experiment.name == "legacy-exp"
    assert args.experiment.swanlab.project == "legacy-project"
    assert args.train.batch_size == 8
    assert args.train.naive_config["learning_rate"] == 0.001
    assert args.train.naive_config["amp"] is True
    assert args.train.naive_config["amp_dtype"] == "fp16"
    assert args.data.train.dataloader_config["seed"] == 7
    assert args.data.eval.dataset_config["dataset_path"] == "json"
    assert args.data.eval.dataset_config["text_column"] == "text"


def test_pretrain_to_yaml_renders_demo_structure():
    args = PreTrainArgs()
    args.experiment.name = "demo-exp"
    args.experiment.swanlab.enabled = True
    args.experiment.swanlab.project = "demo-project"
    args.train.batch_size = 64
    args.data.train.dataset_config["dataset_path"] = "json"
    args.data.eval.dataset_config.update(
        {"dataset_path": "json", "text_column": "text", "tokenizer_path": "demo"}
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        args.to_yaml(handle.name)
        content = Path(handle.name).read_text()
        loaded, mode = load_args_from_yaml(config_path=handle.name, validate=False)

    assert "experiment:" in content
    assert "  mode: pretrain" in content
    assert "train:" in content
    assert "data:" in content
    assert "  train:" in content
    assert "  eval:" in content
    assert "mode: pretrain" not in content.splitlines()[0]
    assert mode == "pretrain"
    assert loaded.experiment.name == "demo-exp"
    assert loaded.train.batch_size == 64


def test_generate_default_config_for_pretrain_uses_experiment_mode():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        path = generate_default_config("pretrain", handle.name)

    content = path.read_text()
    loaded, mode = load_args_from_yaml(config_path=path, validate=False)

    assert "experiment:" in content
    assert "  mode: pretrain" in content
    assert "train:" in content
    assert mode == "pretrain"
    assert isinstance(loaded, PreTrainArgs)


def test_load_eval_args_from_yaml():
    yaml_path = _write_temp_yaml(
        """
mode: eval
name: minimind_61m_eval
model:
  name: gpt2
  config:
    vocab_size: 100
checkpoint:
  checkpoint_dir: checkpoints/pretrain
eval:
  dataset_path: wikitext
  data_files:
    test: /tmp/eval.jsonl
  text_column: text
  tokenizer_path: demo-tokenizer
  checkpoint_step: 42
"""
    )

    args, mode = load_args_from_yaml(config_path=yaml_path, validate=True)

    assert mode == "eval"
    assert isinstance(args, EvalArgs)
    assert args.name == "minimind_61m_eval"
    assert args.eval.dataset_path == "wikitext"
    assert args.eval.data_files == {"test": "/tmp/eval.jsonl"}
    assert args.eval.checkpoint_step == 42


def test_mode_can_be_detected_from_experiment_block():
    yaml_path = _write_temp_yaml(
        """
experiment:
  mode: pretrain
data:
  train:
    dataset_config:
      dataset_path: /data/train
"""
    )

    args, mode = load_args_from_yaml(config_path=yaml_path, validate=False)

    assert mode == "pretrain"
    assert isinstance(args, PreTrainArgs)


def test_no_mode_raises_error():
    yaml_path = _write_temp_yaml("train:\n  batch_size: 64\n")

    with pytest.raises(ValueError, match="mode not specified"):
        load_args_from_yaml(config_path=yaml_path, validate=False)


def test_no_config_and_no_mode_raises_error():
    with pytest.raises(ValueError, match="Either 'mode' or 'config_path'"):
        load_args_from_yaml()


def test_substitute_env_vars_supports_nested_values(monkeypatch):
    monkeypatch.setenv("BASE_DIR", "/base")

    result = _substitute_env_vars(
        {
            "train": {"dataset_path": "${BASE_DIR}/train"},
            "paths": ["${BASE_DIR}/a", "${UNDEFINED_VAR:-/fallback}"],
        }
    )

    assert result["train"]["dataset_path"] == "/base/train"
    assert result["paths"] == ["/base/a", "/fallback"]


def test_register_custom_args():
    @dataclass
    class CustomArgs(TrainingArgs):
        custom_param: int = 100

    register_args("custom", CustomArgs)

    yaml_path = _write_temp_yaml("custom_param: 200\n")
    args, _ = load_args_from_yaml("custom", yaml_path, validate=False)

    assert args.custom_param == 200


def test_pretrain_validation_accepts_valid_new_structure():
    args = PreTrainArgs(
        train=PreTrainTrainConfig(
            batch_size=32,
            naive_config={"learning_rate": 0.001},
        ),
        data=PreTrainDataConfig(
            train=PreTrainTrainDataConfig(
                data_strategy="padding",
                dataset_config={"dataset_path": "json"},
                dataloader_config={"num_workers": 0},
            ),
            eval=PreTrainEvalDataConfig(
                dataset_config={
                    "dataset_path": "json",
                    "text_column": "text",
                    "tokenizer_path": "demo-tokenizer",
                }
            ),
        ),
        eval=PreTrainEvalConfig(steps=10),
    )

    assert args.validate() == []


def test_pretrain_validation_requires_train_dataset_path():
    args = PreTrainArgs(
        train=PreTrainTrainConfig(epoch_num=10),
        data=PreTrainDataConfig(train=PreTrainTrainDataConfig(dataset_config={})),
    )

    errors = args.validate()

    assert any("data.train.dataset_config.dataset_path" in error for error in errors)


def test_pretrain_validation_requires_eval_dataset_fields():
    args = PreTrainArgs(eval=PreTrainEvalConfig(steps=100))

    errors = args.validate()

    assert any("data.eval.dataset_config.dataset_path" in error for error in errors)
    assert any("data.eval.dataset_config.tokenizer_path" in error for error in errors)


def test_pretrain_validation_requires_experiment_name_when_swanlab_enabled():
    args = PreTrainArgs()
    args.experiment.swanlab.enabled = True
    args.experiment.swanlab.project = "demo-project"

    errors = args.validate()

    assert any("experiment.name" in error for error in errors)


def test_eval_validation_requires_required_fields():
    args = EvalArgs(
        checkpoint=EvalCheckpointConfig(checkpoint_dir="checkpoints/pretrain"),
        model=ModelConfig(name="gpt2", config={}),
    )

    errors = args.validate()

    assert any("eval.dataset_path" in error for error in errors)
    assert any("eval.tokenizer_path" in error for error in errors)


def test_resolve_config_path_supports_explicit_and_missing_paths():
    yaml_path = _write_temp_yaml("custom_param: 1\n")

    assert _resolve_config_path("custom", yaml_path) == Path(yaml_path)

    with pytest.raises(FileNotFoundError):
        _resolve_config_path("custom", "/nonexistent/config.yaml")

    with pytest.raises(FileNotFoundError, match="Tip:"):
        _resolve_config_path("nonexistent_mode")


def test_pretrain_defaults_use_prefixed_types():
    args = PreTrainArgs(
        checkpoint=PreTrainCheckpointConfig(),
        optimizer=PreTrainOptimizerConfig(),
    )

    assert isinstance(args, PreTrainArgs)
    assert isinstance(args.train, PreTrainTrainConfig)
    assert isinstance(args.checkpoint, PreTrainCheckpointConfig)
    assert isinstance(args.optimizer, PreTrainOptimizerConfig)
