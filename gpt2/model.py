from torch import nn
import torch
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class LayerNorm(nn.Module):
    """
    Layer Normalization (LayerNorm)

    对每个样本的特征维度进行归一化，公式为：

    μ = (1/d) * Σᵢ xᵢ

    σ² = (1/d) * Σᵢ (xᵢ - μ)²

    x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

    yᵢ = γᵢ * x̂ᵢ + βᵢ

    或简写为：

    LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β

    其中：
        - μ: 特征均值
        - σ²: 特征方差
        - ε: 数值稳定性常数 (eps)
        - γ (gemma): 可学习缩放参数
        - β (beta): 可学习平移参数
        - ⊙: 逐元素乘法 (Hadamard积)
    """

    def __init__(self, config: GPTConfig, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gemma = nn.Parameter(torch.ones(config.n_embd))
        self.beta = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_embd"]:
        x_mean: Float[Tensor, "... 1"] = torch.mean(x, dim=-1, keepdim=True)
        x_var: Float[Tensor, "... 1"] = torch.var(x, dim=-1, keepdim=True)
        normalized: Float[Tensor, "... n_embd"] = (x - x_mean) / torch.sqrt(
            x_var + self.eps
        )

        return self.gemma * normalized + self.beta


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: Float[Tensor, "B T n_embd"]) -> Float[Tensor, "B T n_embd"]:
        """
        Scaled Dot-Product Attention (点积缩放注意力)

        公式:
            Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

        其中:
            - Q, K, V: Query, Key, Value 矩阵
            - d_k: Key 的维度 (head_dim)
            - QK^T: Query 和 Key 的点积，得到注意力分数矩阵
            - / sqrt(d_k): 缩放因子，防止点积过大导致 softmax 梯度消失
            - softmax: 归一化为概率分布
            - @ V: 用注意力权重对 Value 加权求和

        Causal Mask:
            对于自回归生成，使用上三角 mask 防止看到未来的 token
            mask[i,j] = True if j > i (位置 j 在位置 i 之后)
        """
        # qkv 合并投影
        B, T, n_embd = x.shape
        assert n_embd % self.config.n_head == 0
        head_dim = n_embd // self.config.n_head
        qkv: Float[Tensor, "B T triple_n_embd"] = self.c_attn(x)
        # 实际形状: (..., seq_len, 3 * n_embd)

        # 拆分成 q, k, v
        q: Float[Tensor, "B T n_embd"]
        k: Float[Tensor, "B T n_embd"]
        v: Float[Tensor, "B T n_embd"]
        q, k, v = qkv.split(self.config.n_embd, dim=-1)
        q: Float[Tensor, "B n_head T head_dim"] = q.view(
            B, T, self.config.n_head, head_dim
        ).transpose(1, 2)
        k: Float[Tensor, "B n_head T head_dim"] = k.view(
            B, T, self.config.n_head, head_dim
        ).transpose(1, 2)
        v: Float[Tensor, "B n_head T head_dim"] = v.view(
            B, T, self.config.n_head, head_dim
        ).transpose(1, 2)

        # score:Float[Tensor, "B n_head T T"] = q @ k.transpose(2,3) / (head_dim ** 0.5)
        # mask = torch.triu(torch.ones(T,T),diagonal=1).bool()
        # # mask:
        # # [[False,  True,  True,  True],
        # #  [False, False,  True,  True],
        # #  [False, False, False,  True],
        # #  [False, False, False, False]]
        # score = torch.masked_fill(score,mask,float('-inf')) # 掩码，将上三角的分数都设置为最小，这样softmax之后就是0
        # attn_weights:Float[Tensor,"B n_head T T"]  = nn.functional.softmax(score,dim=-1)

        # output:Float[Tensor, "B n_head T head_dim"] =  attn_weights @ v
        output = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        output: Float[Tensor, "B T n_embd"] = (
            output.transpose(1, 2).contiguous().view(B, T, n_embd)
        )
        return self.c_proj(output)


class MLP(nn.Module):
    """
    输入: x (batch, seq_len, n_embd)
           ↓
    ┌─────────────────┐
    │   Linear        │  n_embd → 4 * n_embd  (扩展4倍)
    │  (c_fc)         │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │      GELU       │  激活函数
    │   (非线性)       │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │   Linear        │  4 * n_embd → n_embd  (投影回原维度)
    │  (c_proj)       │
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │    Dropout      │  正则化
    └────────┬────────┘
             ↓
    输出: (batch, seq_len, n_embd)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_embd"]:
        x = self.c_fc(x)
        x = self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLP(config)

    def forward(self, x: Float[Tensor, "... n_embd"]) -> Float[Tensor, "... n_embd"]:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2(nn.Module):
    """
                        Input
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
    ┌───────┐    ┌────────┐    ┌───────┐
    │ Layer │    │ Layer  │    │  Add  │
    │ Norm  │───→│ Attention│──→│& Norm │──┐
    │ (ln1) │    │ (attn) │    │       │  │
    └───────┘    └────────┘    └───────┘  │
        ↑                    │            │
        └────────────────────┘            │
                                          │
        ┌─────────────────────────────────┘
        ↓
    ┌───────┐    ┌──────────┐    ┌───────┐
    │ Layer │    │  Feed    │    │  Add  │
    │ Norm  │───→│ Forward  │───→│& Norm │──→ Output
    │ (ln2) │    │  (ffn)   │    │       │
    └───────┘    └──────────┘    └───────┘
        ↑                    │
        └────────────────────┘
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": LayerNorm(config),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # todo 初始化权重

    def forward(
        self,
        x: torch.LongTensor,
        y: torch.LongTensor,
    ) -> tuple[Float[Tensor, "batch_size seq_len vocab_size"], Float]:
        B, T = x.shape

        tok_emb = self.transformer.wte(x)
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits: Float[Tensor, "B T vocab_size"] = self.lm_head(x)

        logits_flat: Float[Tensor, "B*T vocab_size"] = logits.view(
            -1, self.config.vocab_size
        )
        y_flat: torch.LongTensor = y.view(-1).long()  # 确保标签是 Long 类型
        loss = nn.functional.cross_entropy(logits_flat, y_flat)
        return logits, loss
