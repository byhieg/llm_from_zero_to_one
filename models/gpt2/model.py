"""GPT2 模型实现"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from jaxtyping import Float
from torch import Tensor

from models.base import BaseModel
from models.gpt2.config import GPT2Config
from models.registry import register_model
from ..common_module import LayerNorm




class CausalSelfAttention(nn.Module):
    """因果自注意力"""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: Float[Tensor, "B T n_embd"]) -> Float[Tensor, "B T n_embd"]:
        B, T, n_embd = x.shape
        assert n_embd % self.config.n_head == 0
        head_dim = n_embd // self.config.n_head
        
        # QKV 投影
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embd, dim=-1)
        
        # 重塑为多头
        q = q.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, head_dim).transpose(1, 2)

        # 使用 PyTorch 的高效实现
        output = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(B, T, n_embd)
        
        return self.c_proj(output)


class MLP(nn.Module):
    """前馈网络"""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_embd"]:
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_embd"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@register_model("gpt2")
class GPT2(BaseModel):
    """GPT2 模型"""

    def __init__(self, config: GPT2Config):
        super().__init__(config)
        
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": LayerNorm(config.n_embd),
        })
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重共享
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None
    ) -> tuple[Float[Tensor, "B T V"], Optional[Float[Tensor, ""]]]:
        """
        前向传播
        
        Args:
            input_ids: [B, T] 输入 token IDs
            labels: [B, T] 标签（用于计算损失）
            
        Returns:
            (logits, loss)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token 和位置嵌入
        tok_emb = self.transformer.wte(input_ids)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 最终层归一化
        x = self.transformer.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)

        # 计算损失
        loss = None
        if labels is not None:
            logits_flat = logits.view(-1, self.config.vocab_size)
            labels_flat = labels.view(-1).long()
            loss = nn.functional.cross_entropy(logits_flat, labels_flat)

        return logits, loss
