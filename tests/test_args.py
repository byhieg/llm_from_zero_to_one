"""测试 args.py 的解析功能"""
import pytest
from trainer.args import parse_args, register_args, get_args_class, list_modes, TrainingArgs
from dataclasses import dataclass


def test_list_modes():
    """测试列出所有可用模式"""
    modes = list_modes()
    assert "pretrain" in modes


def test_get_args_class():
    """测试获取参数类"""
    args_cls = get_args_class("pretrain")
    assert args_cls.__name__ == "PretrainArgs"


def test_get_args_class_invalid():
    """测试获取不存在的模式"""
    with pytest.raises(ValueError, match="Unknown mode"):
        get_args_class("invalid_mode")


def test_parse_args_basic():
    """测试基本参数解析"""
    args = parse_args("pretrain", ["--batch-size", "32", "--learning-rate", "0.001"])
    assert args.batch_size == 32
    assert args.learning_rate == 0.001


def test_parse_args_with_underscores():
    """测试使用下划线格式的参数名"""
    args = parse_args("pretrain", ["--batch_size", "64"])
    assert args.batch_size == 64


def test_parse_args_default_values():
    """测试默认值"""
    args = parse_args("pretrain", [])
    assert args.batch_size == 16  # 默认值
    assert args.seq_len == 1024  # 默认值
    assert args.epoch_num == 1  # 默认值


def test_parse_args_all_types():
    """测试各种类型的参数"""
    args = parse_args("pretrain", [
        "--batch-size", "32",
        "--seq-len", "512",
        "--epoch-num", "10",
        "--data-path", "/path/to/data",
        "--learning-rate", "0.001",
        "--warmup-steps", "100",
        "--grad-clip", "0.5",
        "--accumulation-steps", "2",
        "--log-steps", "50",
        "--save-steps", "500",
        "--checkpoint-dir", "checkpoints/test",
    ])

    assert args.batch_size == 32
    assert args.seq_len == 512
    assert args.epoch_num == 10
    assert args.data_path == "/path/to/data"
    assert args.learning_rate == 0.001
    assert args.warmup_steps == 100
    assert args.grad_clip == 0.5
    assert args.accumulation_steps == 2
    assert args.log_steps == 50
    assert args.save_steps == 500
    assert args.checkpoint_dir == "checkpoints/test"


def test_register_custom_args():
    """测试注册自定义参数类"""
    @dataclass
    class CustomArgs(TrainingArgs):
        custom_param: int = 100

    register_args("custom", CustomArgs)
    assert "custom" in list_modes()

    args = parse_args("custom", ["--custom-param", "200"])
    assert args.custom_param == 200


def test_parse_args_pretrain_specific():
    """测试 PretrainArgs 特有参数"""
    args = parse_args("pretrain", [
        "--inference-steps", "1000",
        "--inference-tokens", "50",
        "--inference-topk", "200",
        "--inference-temperature", "0.8",
        "--inference-prompt", "Hello world",
    ])

    assert args.inference_steps == 1000
    assert args.inference_tokens == 50
    assert args.inference_topk == 200
    assert args.inference_temperature == 0.8
    assert args.inference_prompt == "Hello world"


def test_parse_args_optional():
    """测试可选参数"""
    args = parse_args("pretrain", ["--resume-from-checkpoint", "path/to/checkpoint"])
    assert args.resume_from_checkpoint == "path/to/checkpoint"

    args2 = parse_args("pretrain", [])
    assert args2.resume_from_checkpoint is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
