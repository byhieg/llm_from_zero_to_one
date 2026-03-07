"""测试模型注册表"""
import pytest
import torch

from models import (
    BaseModel,
    BaseModelConfig,
    register_model,
    register_config,
    get_model_class,
    get_config_class,
    list_models,
    is_registered,
)
from models.gpt2 import GPT2, GPT2Config


def test_gpt2_config():
    """测试 GPT2 配置"""
    config = GPT2Config()
    assert config.block_size == 1024
    assert config.vocab_size == 50257
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.n_embd == 768
    
    # 自定义配置
    config2 = GPT2Config(n_layer=6, n_head=8)
    assert config2.n_layer == 6
    assert config2.n_head == 8


def test_gpt2_config_serialization():
    """测试配置序列化"""
    config = GPT2Config(n_layer=6)
    
    # to_dict
    d = config.to_dict()
    assert d["n_layer"] == 6
    
    # from_dict
    config2 = GPT2Config.from_dict({"n_layer": 6, "n_head": 8})
    assert config2.n_layer == 6
    assert config2.n_head == 8


def test_gpt2_registration():
    """测试 GPT2 注册"""
    assert is_registered("gpt2")
    assert "gpt2" in list_models()
    
    # 获取类
    gpt2_cls = get_model_class("gpt2")
    assert gpt2_cls == GPT2
    
    gpt2_config_cls = get_config_class("gpt2")
    assert gpt2_config_cls == GPT2Config


def test_gpt2_model_creation():
    """测试 GPT2 模型创建"""
    config = GPT2Config(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
    )
    
    model = GPT2(config)
    
    # 测试参数统计
    n_params = model.count_parameters()
    assert n_params > 0
    print(f"\nGPT2 params: {n_params:,}")
    
    # 测试前向传播
    x = torch.randint(0, 1000, (2, 16))  # batch=2, seq=16
    logits, loss = model(x, labels=x)
    
    assert logits.shape == (2, 16, 1000)
    assert loss is not None
    assert loss.item() > 0


def test_gpt2_save_load_checkpoint(tmp_path):
    """测试保存和加载检查点"""
    config = GPT2Config(n_layer=2, n_head=4, n_embd=128)
    model = GPT2(config)
    
    # 保存
    checkpoint_path = tmp_path / "model.pt"
    model.save_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()
    
    # 加载
    loaded_model, checkpoint = GPT2.load_checkpoint(
        checkpoint_path, config, device="cpu"
    )
    
    assert isinstance(loaded_model, GPT2)
    assert "config" in checkpoint
    
    # 验证输出一致
    x = torch.randint(0, 1000, (1, 8))
    model.eval()
    loaded_model.eval()
    
    with torch.no_grad():
        logits1, _ = model(x, labels=x)
        logits2, _ = loaded_model(x, labels=x)
    
    assert torch.allclose(logits1, logits2)


def test_model_factory():
    """测试模型工厂"""
    from models.model_factory import create_model
    
    # 使用工厂创建模型
    model = create_model(
        "gpt2",
        config_dict={
            "block_size": 128,
            "vocab_size": 1000,
            "n_layer": 2,
            "n_head": 4,
            "n_embd": 128,
        }
    )
    
    assert isinstance(model, GPT2)
    assert model.count_parameters() > 0
    
    # 使用 kwargs
    model2 = create_model("gpt2", n_layer=2, n_head=4, n_embd=128, vocab_size=1000, block_size=128)
    assert isinstance(model2, GPT2)
    
    # 混合使用（kwargs 覆盖 config_dict）
    model3 = create_model(
        "gpt2",
        config_dict={"n_layer": 6, "n_head": 8},
        n_layer=2  # 覆盖
    )
    assert model3.config.n_layer == 2
    assert model3.config.n_head == 8


def test_unregistered_model_error():
    """测试未注册模型无法创建"""
    from models.model_factory import create_model
    
    # 尝试创建未注册的模型
    with pytest.raises(ValueError, match="Unknown config 'nonexistent'"):
        create_model("nonexistent")


def test_custom_model_registration():
    """测试自定义模型注册"""
    @dataclass
    class CustomConfig(BaseModelConfig):
        hidden_size: int = 256
    
    @register_config("custom")
    class CustomModelConfig(CustomConfig):
        pass  # 不需要额外参数
    
    @register_model("custom")
    class CustomModel(BaseModel):
        def __init__(self, config: CustomModelConfig):
            super().__init__(config)
            self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        
        def forward(self, x, labels=None):
            return self.linear(x), None
    
    # 验证注册
    assert "custom" in list_models()
    assert is_registered("custom")
    
    # 创建实例
    config = CustomModelConfig(hidden_size=128)
    model = CustomModel(config)
    
    x = torch.randn(2, 10, 128)
    out, _ = model(x)
    assert out.shape == (2, 10, 128)


from dataclasses import dataclass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
