"""测试 train_args.py 的 YAML 功能"""
import os
import pytest
import tempfile
from pathlib import Path
from trainer.train_args import (
    PretrainArgs, 
    TrainingArgs,
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
        args, mode = load_args_from_yaml("pretrain", f.name, validate=False)
        
    assert args.batch_size == 32
    assert args.learning_rate == 0.001
    assert mode == "pretrain"
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
        args, mode = load_args_from_yaml("pretrain", f.name, validate=False)

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
        loaded, _ = load_args_from_yaml("pretrain", f.name, validate=False)
        
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
data_path: /data
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, mode = load_args_from_yaml("pretrain", f.name, validate=False)
        
    assert isinstance(args, PretrainArgs)
    assert args.batch_size == 128
    assert args.epoch_num == 20
    assert mode == "pretrain"


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
data_path: /data
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, mode = load_args_from_yaml("custom", f.name, validate=False)
        
    assert args.custom_param == 200
    assert args.batch_size == 32


def test_optional_param():
    """测试可选参数"""
    yaml_content = """
resume_from_checkpoint: path/to/checkpoint
data_path: /data
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        args, _ = load_args_from_yaml("pretrain", f.name, validate=False)
        
    assert args.resume_from_checkpoint == "path/to/checkpoint"
    
    # 不提供时为 None
    yaml_content2 = "batch_size: 16\ndata_path: /data"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content2)
        f.flush()
        args2, _ = load_args_from_yaml("pretrain", f.name, validate=False)
        
    assert args2.resume_from_checkpoint is None


class TestAutoModeDetection:
    """测试从 YAML 自动识别模式"""
    
    def test_mode_from_yaml(self):
        """测试从 YAML 中读取 mode"""
        yaml_content = """
mode: pretrain
batch_size: 64
data_path: /data/train
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            args, mode = load_args_from_yaml(config_path=f.name, validate=False)
            
        assert mode == "pretrain"
        assert args.batch_size == 64
        assert isinstance(args, PretrainArgs)
    
    def test_mode_override(self):
        """测试命令行 mode 覆盖 YAML mode"""
        yaml_content = """
mode: pretrain
batch_size: 64
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            # 即使 YAML 中有 mode，命令行指定的优先
            args, mode = load_args_from_yaml(mode="pretrain", config_path=f.name, validate=False)
            
        assert mode == "pretrain"
    
    def test_no_mode_error(self):
        """测试没有指定 mode 时报错"""
        yaml_content = "batch_size: 64"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            with pytest.raises(ValueError, match="mode not specified"):
                load_args_from_yaml(config_path=f.name, validate=False)
    
    def test_no_config_no_mode_error(self):
        """测试没有 config 也没有 mode 时报错"""
        with pytest.raises(ValueError, match="Either 'mode' or 'config_path'"):
            load_args_from_yaml()


class TestEnvVarSubstitution:
    """测试环境变量替换"""
    
    def test_simple_substitution(self):
        """测试简单环境变量替换"""
        os.environ["TEST_DATA_PATH"] = "/my/data"
        data = {"path": "${TEST_DATA_PATH}"}
        result = _substitute_env_vars(data)
        assert result["path"] == "/my/data"
        del os.environ["TEST_DATA_PATH"]
    
    def test_default_value(self):
        """测试默认值"""
        data = {"path": "${UNDEFINED_VAR:-/default/path}"}
        result = _substitute_env_vars(data)
        assert result["path"] == "/default/path"
    
    def test_nested_substitution(self):
        """测试嵌套结构中的替换"""
        os.environ["BASE_DIR"] = "/base"
        data = {
            "train": {
                "data_path": "${BASE_DIR}/train",
                "output_path": "${BASE_DIR}/output"
            },
            "paths": ["${BASE_DIR}/a", "${BASE_DIR}/b"]
        }
        result = _substitute_env_vars(data)
        assert result["train"]["data_path"] == "/base/train"
        assert result["train"]["output_path"] == "/base/output"
        assert result["paths"] == ["/base/a", "/base/b"]
        del os.environ["BASE_DIR"]
    
    def test_preserve_undefined(self):
        """测试未定义的变量保持原值"""
        data = {"path": "${TOTALLY_UNDEFINED_VAR}"}
        result = _substitute_env_vars(data)
        assert result["path"] == "${TOTALLY_UNDEFINED_VAR}"


class TestValidation:
    """测试配置验证"""
    
    def test_valid_config(self):
        """测试有效配置"""
        args = PretrainArgs(
            data_path="/data/train",
            batch_size=32,
            learning_rate=0.001
        )
        errors = args.validate()
        assert errors == []
    
    def test_missing_data_path(self):
        """测试缺少 data_path"""
        args = PretrainArgs(epoch_num=10, data_path="")
        errors = args.validate()
        assert any("data_path" in e for e in errors)
    
    def test_invalid_batch_size(self):
        """测试无效的 batch_size"""
        args = PretrainArgs(batch_size=0, data_path="/data")
        errors = args.validate()
        assert any("batch_size" in e for e in errors)
    
    def test_invalid_learning_rate(self):
        """测试无效的 learning_rate"""
        args = PretrainArgs(learning_rate=-0.001, data_path="/data")
        errors = args.validate()
        assert any("learning_rate" in e for e in errors)
    
    def test_validation_on_load(self):
        """测试加载时验证"""
        yaml_content = """
epoch_num: 10
data_path: ""
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            # 应该抛出验证错误
            with pytest.raises(ValueError, match="validation"):
                load_args_from_yaml("pretrain", f.name, validate=True)
            
            # 跳过验证时不应抛出错误
            args, _ = load_args_from_yaml("pretrain", f.name, validate=False)
            assert args.epoch_num == 10


class TestGenerateDefaultConfig:
    """测试生成默认配置"""
    
    def test_generate_to_temp(self):
        """测试生成到临时文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            path = generate_default_config("pretrain", f.name)
            
        assert path.exists()
        
        # 验证可以加载并包含 mode
        args, mode = load_args_from_yaml(config_path=path, validate=False)
        assert args.batch_size == 16  # 默认值
        assert mode == "pretrain"
    
    def test_generate_creates_directory(self):
        """测试生成时创建目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "config.yaml"
            result = generate_default_config("pretrain", path)
            
            assert result.exists()
            assert result.parent.exists()
    
    def test_generated_config_has_mode(self):
        """测试生成的配置包含 mode 字段"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            path = generate_default_config("pretrain", f.name)
        
        content = path.read_text()
        assert "mode: pretrain" in content


class TestResolveConfigPath:
    """测试配置路径解析"""
    
    def test_explicit_path(self):
        """测试显式指定的路径"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("batch_size: 32")
            f.flush()
            
            result = _resolve_config_path("pretrain", f.name)
            assert result == Path(f.name)
    
    def test_missing_explicit_path(self):
        """测试显式指定但不存在的路径"""
        with pytest.raises(FileNotFoundError):
            _resolve_config_path("pretrain", "/nonexistent/config.yaml")
    
    def test_default_path_missing(self):
        """测试默认路径不存在"""
        with pytest.raises(FileNotFoundError, match="Tip:"):
            _resolve_config_path("nonexistent_mode")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
