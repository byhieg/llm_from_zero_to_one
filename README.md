<h1 align="center">🚀 LLM From Zero to One</h1>

<p align="center">
  <strong>从零开始构建大语言模型训练框架 —— 纯 PyTorch 实现，面向教学与生产级 LLM 训练</strong>
</p>

<p align="center">
  <a href="#-功能特性">功能特性</a> •
  <a href="#-版本能力矩阵">版本能力</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-项目结构">项目结构</a> •
  <a href="#-配置说明">配置说明</a> •
  <a href="#-路线图">路线图</a>
</p>

---

## 📖 简介

`llm-from-zero-to-one` 是一个**从零手写实现的大语言模型（LLM）训练框架**，旨在帮助开发者深入理解 LLM 训练的每一个核心环节。项目采用纯 PyTorch 实现，不依赖 HuggingFace Transformers 的模型代码，覆盖从数据预处理、模型定义到训练循环的核心流程。

本项目适合：
- **学习者**：想理解 Transformer/GPT 模型底层原理和训练细节的开发者
- **研究者**：需要灵活定制训练流程的研究人员
- **工程师**：希望掌握 LLM 工程化实践的技术人员

## 🙏 致谢

### 灵感来源

本项目的设计理念和架构深受 **[MiniMind](https://github.com/jingyaogong/minimind)** 项目的启发。MiniMind 是一个优秀的开源项目，致力于以极低的成本（3 元 GPU 费用 + 2 小时训练时间）从零训练超小语言模型（~25.8M 参数），其"大道至简"的哲学——使用纯 PyTorch 原生实现所有核心算法、不依赖第三方框架封装——与本项目的初衷高度一致。

> **MiniMind 项目亮点**：
> - 🧠 全阶段开源复现：Pretrain → SFT → LoRA → DPO → RLHF (PPO/GRPO) → 模型蒸馏
> - 🔬 核心代码从零手写，帮助理解 LLM 每一个细节
> - 📚 完整教程导向，适合 LLM 入门学习

### 数据集

本项目示例配置中使用的预训练数据集来自 **[minimind_dataset](https://huggingface.co/datasets/jingyaogong/minimind_dataset)**（HuggingFace），分词器采用 **[minimind-3](https://huggingface.co/jingyaogong/minimind-3)**。感谢 MiniMind 社区开放的高质量数据资源。

## ✨ 功能特性

### 🔧 核心能力

| 模块 | 能力 | 说明 |
|------|------|------|
| **模型架构** | GPT-2 从零实现 | 完整的 Decoder-only Transformer，包含因果自注意力、MLP、Pre-Norm、位置编码 |
| **预训练** | 全量训练循环 | 支持 AdamW/Adam 优化器、余弦学习率调度、梯度累积、梯度裁剪 |
| **混合精度** | AMP (fp16/bf16) | CUDA 上支持自动混合精度训练，提升训练效率与显存利用率 |
| **编译加速** | torch.compile | 自动启用 PyTorch 2.x 编译加速（CUDA/MPS） |
| **数据加载** | 双策略支持 | Padding（边训边分词）/ Megatron（预先分词 .bin/.idx）两种模式 |
| **检查点管理** | 断点续训 | 支持按步保存、自动恢复最新检查点、配置兼容性校验 |
| **实验跟踪** | SwanLab 集成 | 可选接入 SwanLab 进行实验可视化与参数记录 |
| **数据处理工具** | 数据预处理 | 提供文本分词 → `.bin` + `.idx` 文件生成的完整流水线 |

### 🏗️ 架构设计亮点

- **注册表模式**：模型与配置通过装饰器注册，扩展新架构只需几行代码
- **YAML 驱动配置**：所有超参统一通过 YAML 管理，支持环境变量替换 `${VAR:-default}`
- **类型安全**：使用 `jaxtyping` 进行张量形状注解，`dataclass` 统一配置管理
- **确定性训练**：完整的种子控制（全局/Dataloader Worker），保证实验可复现

## 📊 版本能力矩阵

### v0.1.0 (当前版本)

| 能力维度 | 支持状态 | 详细说明 |
|----------|:--------:|----------|
| **模型架构** | ✅ 已支持 | GPT-2 (Decoder-only Transformer) |
| **训练模式** | ✅ 已支持 | Pre-training（语言模型预训练） |
| **优化器** | ✅ 已支持 | AdamW, Adam |
| **学习率调度** | ✅ 已支持 | Cosine Annealing + Linear Warmup |
| **混合精度训练 (AMP)** | ✅ 已支持 | fp16, bf16 (CUDA) |
| **torch.compile 加速** | ✅ 已支持 | CUDA / MPS 自动启用 |
| **梯度累积** | ✅ 已支持 | 可配置 accumulation_steps |
| **梯度裁剪** | ✅ 已支持 | 全局范数裁剪 |
| **数据加载 - Padding 策略** | ✅ 已支持 | 边训练边分词，HuggingFace Datasets 集成 |
| **数据加载 - Megatron 策略** | ✅ 已支持 | 预先分词 (.bin + .idx 内存映射) |
| **检查点保存/恢复** | ✅ 已支持 | 按步保存 + 断点续训 + 配置兼容校验 |
| **SwanLab 实验跟踪** | ✅ 已支持 | 参数/指标可视化 |
| **YAML 配置管理** | ✅ 已支持 | 环境变量替换 + 校验 + 默认生成 |
| **数据处理工具链** | ✅ 已支持 | 分词 → .bin/.idx 生成 |
| **单元测试** | ✅ 已覆盖 | 核心模块 pytest 测试 |

### 🗺️ 路线图 (Roadmap)

| 能力维度 | 计划版本 | 说明 |
|----------|:--------:|------|
| 分布式训练 (DDP) | v0.2.0 | 单机多卡数据并行 |
| SFT / DPO 微调 | v0.3.0 | 监督微调 + 偏好对齐 |
| Qwen / DeepSeek 模型架构 | v0.3.0 | Dense & MoE 架构支持 |
| HuggingFace 格式互转 | v0.3.0 | 导出/导入 HF 格式权重 |
| DeepSpeed / FSDP | v0.4.0 | 大规模分布式训练加速 |

## 🚀 快速开始

### 环境要求

- Python >= 3.10
- PyTorch >= 2.10.0（建议使用机器镜像或外部训练环境中已安装的版本）
- CUDA (推荐) 或 MPS (Apple Silicon)

### 安装

```bash
# 克隆项目
git clone <repo-url>
cd llm_from_zero_to_one

# 安装项目默认依赖；默认不会安装或替换 PyTorch/DeepSpeed
uv sync

# 如果当前环境还没有 PyTorch，可以显式安装 PyTorch extra
uv sync --extra torch

# 如果需要 DeepSpeed 后端，可以显式安装 DeepSpeed extra
uv sync --extra deepspeed

# 如果需要 SwanLab 实验跟踪，可以显式安装 SwanLab extra
uv sync --extra swanlab

# 激活虚拟环境
source .venv/bin/activate
```

> 说明：`torch` 与 `deepspeed` 依赖 CUDA、驱动和机器镜像版本，项目默认不在
> `pyproject.toml` 的基础依赖中固定它们。复用已有训练环境时，请先激活该环境，
> 再使用 `uv run --active ...` 运行项目命令。

### 运行预训练

```bash
# 使用内置配置文件启动预训练
uv run python3 -m main --config configs/pretrain_padding_example.yaml

# 生成默认配置模板
uv run python3 -m main --config my_config.yaml --generate-config
```

### 数据预处理（Megatron 格式）

```bash
# 将原始文本数据集处理为 .bin + .idx 格式
uv run python3 tools/llm_data_processor.py
```

## 📁 项目结构

```
llm_from_zero_to_one/
├── configs/                      # YAML 配置文件
│   ├── pretrain_padding_example.yaml    # Padding 策略预训练示例
│   └── pretrain_megatron_example.yaml   # Megatron 策略预训练示例
├── src/
│   ├── models/                   # 模型模块
│   │   ├── gpt2/                        # GPT-2 模型实现
│   │   │   ├── model.py                 #   CausalSelfAttention, MLP, Block, GPT2
│   │   │   └── config.py                #   GPT2Config
│   │   ├── base.py                      # BaseModel 基类（含 generate/save/load）
│   │   ├── common_module.py             # LayerNorm 等公共组件
│   │   ├── registry.py                  # 模型注册表（装饰器）
│   │   └── model_factory.py             # 模型工厂
│   ├── trainer/                  # 训练模块
│   │   ├── pretrain/pretrain.py         # PreTrainTrainer 核心训练逻辑
│   │   ├── train.py                     # CLI 入口 & 模式分发
│   │   └── train_args.py               # YAML 配置 dataclass 定义
│   ├── dataset/                  # 数据加载模块
│   │   ├── dataset_factory.py           # 数据策略工厂
│   │   ├── padding_dataset.py          # Padding 策略数据集
│   │   └── simple_megatron_dataset/     # Megatron 策略数据集 (.bin/.idx)
│   └── logger/                   # 日志模块
├── tools/                       # 辅助工具脚本
│   ├── llm_data_processor.py            # 数据预处理（生成 .bin/.idx）
│   ├── check_idx.py                     # 索引文件检查
│   └── select_unseen_pretrain_samples.py # 未见过样本筛选
├── tests/                       # 测试套件
├── main.py                      # 程序入口
└── pyproject.toml               # 项目配置
```

## ⚙️ 配置说明

项目通过 **YAML 配置文件** 驱动所有训练参数，核心配置项：

```yaml
mode: pretrain                    # 运行模式

name: experiment_name             # 实验名称（用于 checkpoint 目录）

model:
  name: gpt2                      # 模型名称
  config:
    vocab_size: 6400              # 词表大小
    n_layer: 8                    # Transformer 层数
    n_head: 8                     # 注意力头数
    n_embd: 768                   # 嵌入维度

training:
  batch_size: 16                  # 批大小
  seq_len: 1024                   # 序列长度
  epoch_num: 1                    # 训练轮数
  learning_rate: 3e-4             # 学习率
  warmup_steps: 100               # 预热步数
  grad_clip: 1.0                  # 梯度裁剪阈值
  accumulation_steps: 4           # 梯度累积步数
  amp: true                       # 是否启用混合精度
  amp_dtype: bf16                 # 混合精度类型: fp16 | bf16

data:
  data_strategy: 'padding'        # 数据策略: padding | megatron
  dataset_config:
    dataset_path: 'your/dataset'  # 数据集路径
    tokenizer_path: 'your/tokenizer'  # 分词器路径
  dataloader_config:
    num_workers: 4                # 数据加载进程数
    shuffle: true                 # 是否打乱数据

checkpoint:
  save_steps: 1000                # 每 N 步保存一次
  checkpoint_dir: checkpoints/pretrain  # 检查点目录

optimizer:
  name: adamw                     # 优化器: adamw | adam
  weight_decay: 0.01              # 权重衰减

swanlab:
  enabled: false                  # 是否启用 SwanLab
  project: llm-pretrain           # 项目名称
  experiment_name: experiment     # 实验名称
```

> 💡 **提示**：配置文件支持环境变量替换，例如 `dataset_path: ${DATA_PATH:-default/path}`

## 🧪 测试

```bash
# 运行全部测试
uv run python3 -m pytest tests/ -v

# 运行指定测试文件
uv run python3 -m pytest tests/test_models.py -v
```

## 📝 开发规范

```bash
# 代码格式化 (ruff + isort)
ruff format .
ruff check --fix .

# 运行测试
uv run python3 -m pytest tests/ -v
```

## 📄 License

MIT License
