"""测试 train_args.py 的 YAML 功能"""
import pytest
import tempfile
from pathlib import Path
from trainer.train_args import (
    PretrainArgs, 
    TrainingArgs,
    load_args_from_yaml, 
    register_args, 
    get_args_class, 
    list_modes
)
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


def test_from_yaml_basic():
    """测试从 YAML 加载基本参数"""
    yaml_content = """
batch_size: 32
learning_rate: 0.001
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args = PretrainArgs.from_yaml(f.name)
        
    assert args.batch_size == 32
    assert args.learning_rate == 0.001
    # 默认值
    assert args.seq_len == 1024
    assert args.epoch_num == 1


def test_from_yaml_all_params():
    """测试从 YAML 加载所有参数"""
    yaml_content = """
batch_size: 32
seq_len: 512
epoch_num: 10
data_path: /path/to/data
learning_rate: 0.001
warmup_steps: 100
grad_clip: 0.5
accumulation_steps: 2
log_steps: 50
save_steps: 500
checkpoint_dir: checkpoints/test
resume_from_checkpoint: null
inference_steps: 1000
inference_tokens: 50
inference_topk: 200
inference_temperature: 0.8
inference_prompt: "Hello world"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args = PretrainArgs.from_yaml(f.name)

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
    assert args.inference_steps == 1000
    assert args.inference_tokens == 50
    assert args.inference_topk == 200
    assert args.inference_temperature == 0.8
    assert args.inference_prompt == "Hello world"


def test_to_yaml():
    """测试保存到 YAML"""
    args = PretrainArgs(
        batch_size=64,
        learning_rate=0.0001,
        epoch_num=5,
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        args.to_yaml(f.name)
        # 重新加载
        loaded = PretrainArgs.from_yaml(f.name)
        
    assert loaded.batch_size == 64
    assert loaded.learning_rate == 0.0001
    assert loaded.epoch_num == 5


def test_to_dict():
    """测试转换为字典"""
    args = PretrainArgs(batch_size=32)
    d = args.to_dict()
    
    assert isinstance(d, dict)
    assert d["batch_size"] == 32
    assert "learning_rate" in d
    assert "inference_steps" in d


def test_from_dict():
    """测试从字典创建"""
    d = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "invalid_key": "should be ignored",
    }
    args = PretrainArgs.from_dict(d)
    
    assert args.batch_size == 64
    assert args.learning_rate == 0.001
    # 无效 key 被过滤
    assert not hasattr(args, "invalid_key")


def test_load_args_from_yaml_factory():
    """测试使用工厂函数加载"""
    yaml_content = """
batch_size: 128
epoch_num: 20
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args = load_args_from_yaml("pretrain", f.name)
        
    assert isinstance(args, PretrainArgs)
    assert args.batch_size == 128
    assert args.epoch_num == 20


def test_register_custom_args():
    """测试注册自定义参数类"""
    @dataclass
    class CustomArgs(TrainingArgs):
        custom_param: int = 100

    register_args("custom", CustomArgs)
    assert "custom" in list_modes()
    
    # 测试从 YAML 加载自定义参数
    yaml_content = """
custom_param: 200
batch_size: 32
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args = load_args_from_yaml("custom", f.name)
        
    assert args.custom_param == 200
    assert args.batch_size == 32


def test_optional_param():
    """测试可选参数"""
    yaml_content = """
resume_from_checkpoint: path/to/checkpoint
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args = PretrainArgs.from_yaml(f.name)
        
    assert args.resume_from_checkpoint == "path/to/checkpoint"
    
    # 不提供时为 None
    yaml_content2 = "batch_size: 16"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content2)
        f.flush()
        args2 = PretrainArgs.from_yaml(f.name)
        
    assert args2.resume_from_checkpoint is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
