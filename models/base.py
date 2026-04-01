"""模型基类定义"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json
from pathlib import Path

import torch
from torch import nn


@dataclass
class BaseModelConfig:
    """所有模型配置的基类"""
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self, path: str | Path) -> None:
        """保存到 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseModelConfig":
        """从字典创建实例"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str | Path) -> "BaseModelConfig":
        """从 JSON 文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class BaseModel(nn.Module):
    """所有模型的基类"""
    
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token IDs [batch_size, seq_len]
            labels: 标签（用于计算损失）[batch_size, seq_len]
            
        Returns:
            (logits, loss) - logits: [batch_size, seq_len, vocab_size], loss: 标量或 None
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        device: str = "cpu",
    ) -> list[int]:
        """
        生成文本（自回归推理）
        
        Args:
            prompt_tokens: 输入 token 列表
            max_new_tokens: 最多生成的新 token 数量
            temperature: 温度参数（越高越随机）
            top_k: Top-K 采样参数
            device: 设备
            
        Returns:
            生成的 token 列表（包含输入）
        """
        self.eval()
        tokens = prompt_tokens.copy()
        
        for _ in range(max_new_tokens):
            # 截断到最大长度
            if hasattr(self.config, 'block_size'):
                context = tokens[-self.config.block_size:]
            else:
                context = tokens
            
            # 前向传播
            x = torch.tensor([context], dtype=torch.long, device=device)
            logits, _ = self(x, None)
            
            # 取最后一个位置的 logits
            logits = logits[0, -1, :] / temperature
            
            # Top-K 采样
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
            probs = torch.softmax(top_k_values, dim=-1)
            idx_next = top_k_indices[torch.multinomial(probs, num_samples=1)]
            
            tokens.append(idx_next.item())
        
        self.train()
        return tokens
    
    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int | None = None,
        step: int | None = None,
        **kwargs
    ) -> None:
        """
        保存检查点
        
        Args:
            path: 保存路径
            optimizer: 优化器（可选）
            epoch: 当前 epoch（可选）
            step: 当前步数（可选）
            **kwargs: 其他要保存的信息
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
        }
        
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if step is not None:
            checkpoint["step"] = step
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        # 添加额外信息
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        config: BaseModelConfig,
        optimizer: torch.optim.Optimizer | None = None,
        device: str = "cpu",
    ) -> tuple["BaseModel", dict]:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            config: 模型配置
            optimizer: 优化器（可选）
            device: 加载设备
            
        Returns:
            (model, checkpoint_dict) - 加载的模型和检查点字典
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"✅ Checkpoint loaded from {path}")
        return model, checkpoint
    
    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config}, params={self.count_parameters():,})"
