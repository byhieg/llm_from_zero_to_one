# AI Agent 项目指南

> 本文件帮助 AI 编程助手（Cursor / GitHub Copilot / Claude / Windsurf 等）快速理解项目约定，生成符合规范的代码。

---

## 项目概述

**llm-from-zero-to-one** 是一个从零手写实现的大语言模型（LLM）训练框架，纯 PyTorch 实现，不依赖 HuggingFace Transformers 的模型代码。灵感来源于 [MiniMind](https://github.com/jingyaogong/minimind) 开源项目。

- **语言**: Python 3.10+
- **包管理**: uv
- **依赖管理**: pyproject.toml
- **版本**: 0.1.0

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch >= 2.10.0 |
| 数据处理 | HuggingFace Datasets, Transformers |
| 类型注解 | jaxtyping (张量形状标注) |
| 配置管理 | YAML + dataclass |
| 代码规范 | ruff (格式化 + lint) |
| 测试 | pytest |
| 实验跟踪 | SwanLab (可选) |

## 项目结构

```
src/
├── models/                  # 模型模块（注册表模式）
│   ├── gpt2/               #   GPT-2 Decoder-only Transformer
│   ├── base.py             #   BaseModel 基类（含 generate/save/load）
│   ├── registry.py         #   @register_model / @register_config 装饰器
│   └── model_factory.py    #   create_model() 工厂函数
├── trainer/                 # 训练模块
│   ├── pretrain/pretrain.py #   PreTrainTrainer 核心训练循环
│   ├── train.py            #   CLI 入口（--config 模式分发）
│   └── train_args.py       #   PretrainArgs / EvalArgs dataclass
├── dataset/                 # 数据加载
│   ├── dataset_factory.py   #   create_dataset() 工厂
│   ├── padding_dataset.py  #   Padding 策略（边训边分词）
│   └── simple_megatron_dataset/  # Megatron 策略（.bin/.idx 内存映射）
├── checkpoint_manager/      # 检查点保存/恢复/断点续训
├── evaluator/               # Loss / Perplexity 评估
└── logger/                  # 日志模块
tools/                       # 辅助脚本
configs/                     # YAML 训练配置文件
tests/                       # pytest 测试套件
```

## 核心架构模式

### 1. 注册表模式（模型扩展）

```python
# 注册新模型只需两步：
@register_config("my_model")
class MyModelConfig(BaseModelConfig):
    hidden_size: int = 256

@register_model("my_model")
class MyModel(BaseModel):
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        # ... 模型实现
```

### 2. YAML 驱动配置

所有训练参数通过 YAML 文件统一管理，支持环境变量替换 `${VAR:-default}`：

```yaml
mode: pretrain
model:
  name: gpt2
  config:
    vocab_size: 6400
training:
  batch_size: 16
  seq_len: 1024
```

### 3. 双数据加载策略

| 策略 | 适用场景 | 特点 |
|------|---------|------|
| `padding` | 小规模实验 | 边训练边分词，HuggingFace Datasets 集成 |
| `megatron` | 大规模预训练 | 预先分词为 .bin/.idx，内存映射高效读取 |

---

## 编码规范（必须遵守）

### 1. 注释规范

- **所有类和公开方法必须有 docstring 注释**
- **注释优先使用中文**
- 符合顶级开源项目的注释规范（Google/Sphinx 风格）

```python
class CausalSelfAttention(nn.Module):
    """因果自注意力模块

    实现多头因果自注意力机制，用于 GPT-2 的 Transformer Block。
    使用 PyTorch 原生 scaled_dot_product_attention 实现高效计算。
    """

    def forward(self, x: Float[Tensor, "B T n_embd"]) -> Float[Tensor, "B T n_embd"]:
        """前向传播

        Args:
            x: 输入张量 [batch, seq_len, embed_dim]

        Returns:
            注意力输出张量，形状与输入相同
        """
```

### 2. 运行命令规范

**激活虚拟环境后运行**：

```bash
# 训练
uv run python3 -m main --config configs/pretrain_padding_example.yaml

# 评估
uv run python3 -m main --config configs/eval_minimind_unseen_1024.yaml

# 测试
uv run python3 -m pytest tests/ -v
```

### 3. 修改代码后的必做步骤

每次修改代码后，**必须按顺序执行**以下操作：

```
① 编写/更新对应测试 → ② ruff 格式化 → ③ pytest 通过 → ④ git commit
```

具体命令：

```bash
# 步骤 1: 确保有对应测试
# 步骤 2: ruff 格式化 + lint
uv run ruff format .
uv run ruff check --fix .

# 步骤 3: 运行测试（必须全部通过）
uv run python3 -m pytest tests/ -v

# 步骤 4: 提交代码
git add <changed_files>
git commit -m "type: 简洁描述改动"
```

### 4. Git Commit 规范

使用 Conventional Commits 格式：

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feat:` | 新功能 | `feat: 支持 Adam 优化器` |
| `fix:` | Bug 修复 | `fix: 修复 checkpoint 恢复时步数偏移问题` |
| `refactor:` | 重构（不改变行为） | `refactor: 将 importlib 改为常规 import` |
| `docs:` | 文档/注释 | `docs: 为 dataset_factory 添加完整中文注释` |
| `test:` | 测试相关 | `test: 增加 Megatron 数据集边界条件测试` |
| `chore:` | 构建/工具链 | `chore: 升级 ruff 到最新版本` |

### 5. Import 顺序

由 ruff/isort 自动管理，规则：
1. 标准库
2. 第三方库
3. 本地模块 (`from xxx import yyy`)
4. 相对导入 (`from .xxx import yyy`)

**注意**: `tools/` 目录下的脚本因需先设置 `sys.path`，允许在非顶部位置 import 并添加 `# noqa: E402`。

### 6. 类型安全

- 使用 `jaxtyping` 进行张量形状注解：`Float[Tensor, "B T n_embd"]`
- 使用 `dataclass` 统一配置管理
- 函数签名尽量完整（含返回类型）

---

## 关键设计决策

| 决策 | 原因 |
|------|------|
| 纯 PyTorch 实现 | 不依赖 HF Transformers 模型代码，便于学习底层原理 |
| 注册表模式 | 扩展新模型架构只需装饰器，无需修改工厂代码 |
| YAML 配置驱动 | 超参与代码分离，便于实验管理和复现 |
| dataclass 参数 | 类型安全 + 自动序列化（to_dict/from_dict/from_yaml/to_yaml） |
| 双数据策略 | 兼容小规模快速实验（padding）和大规模高效训练（megatron） |

## 数据集信息

- **训练数据**: [minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset) (HuggingFace)
- **分词器**: [minimind-3](https://huggingface.co/jingyaogong/minimind-3) (HuggingFace)
- **评测数据**: 从全量数据中筛选 mini 训练集未覆盖的 unseen 样本（见 `tools/select_unseen_pretrain_samples.py`）

## 当前版本能力 (v0.1.0)

- ✅ GPT-2 模型从零实现
- ✅ Pre-training 全流程（AdamW/Cosine LR/梯度累积/AMP/torch.compile）
- ✅ 双数据加载策略（Padding / Megatron .bin+.idx）
- ✅ 检查点管理与断点续训
- ✅ 训练中评估（Loss / Perplexity）
- ✅ SwanLab 实验跟踪集成
- ✅ 数据预处理工具链

## 路线图 (Roadmap)

- v0.2.0: DDP 分布式训练
- v0.3.0: SFT/DPO 微调、Qwen/DeepSeek 架构、HF 格式互转
- v0.4.0: DeepSpeed/FSDP 大规模训练

---

## 常见任务速查

### 添加新模型

1. 在 `src/models/` 下创建新目录
2. 定义 `XXXConfig(BaseModelConfig)` 和 `XXXModel(BaseModel)`
3. 用 `@register_config("xxx")` 和 `@register_model("xxx")` 注册
4. 在 `tests/test_models.py` 添加测试

### 添加新的数据加载策略

1. 在 `src/dataset/` 下创建新 Dataset 类
2. 在 `dataset_factory.py` 的 `create_dataset()` 中添加新分支
3. 在 `tests/` 下添加对应测试

### 修改训练逻辑

1. 修改 `src/trainer/pretrain/pretrain.py`
2. 如需新参数，在 `src/trainer/train_args.py` 添加字段
3. 更新 `configs/*.yaml` 示例配置
4. 添加/更新 `tests/test_pretrain_trainer.py` 测试
