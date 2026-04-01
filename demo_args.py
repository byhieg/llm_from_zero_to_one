#!/usr/bin/env python3
"""演示 train_args.py 的 YAML 功能"""

import os
from pathlib import Path
from trainer.train_args import (
    PretrainArgs, 
    load_args_from_yaml, 
    list_modes,
    generate_default_config,
    _substitute_env_vars,
)

print("=" * 60)
print("train_args.py YAML 功能演示")
print("=" * 60)

# 1. 列出所有可用的模式
print("\n1. 可用的训练模式:")
modes = list_modes()
for mode in modes:
    print(f"   - {mode}")

# 2. 测试环境变量替换
print("\n2. 测试环境变量替换:")
os.environ["MY_DATA_PATH"] = "/custom/data/path"
os.environ["MY_LR"] = "0.001"

test_yaml = """
mode: pretrain
data_path: ${MY_DATA_PATH}/train
learning_rate: ${MY_LR}
checkpoint_dir: ${CHECKPOINT_DIR:-default_checkpoints}
"""
print(f"   原始 YAML:\n{test_yaml}")

import yaml
data = yaml.safe_load(test_yaml)
substituted = _substitute_env_vars(data)
print(f"   替换后:")
print(f"     mode: {substituted.get('mode')}")
print(f"     data_path: {substituted['data_path']}")
print(f"     learning_rate: {substituted['learning_rate']}")
print(f"     checkpoint_dir: {substituted['checkpoint_dir']} (使用默认值)")

# 3. 测试从 YAML 中自动识别模式
print("\n3. 测试从 YAML 自动识别模式:")
yaml_with_mode = Path("/tmp/auto_mode_config.yaml")
yaml_with_mode.write_text("""
mode: pretrain
batch_size: 64
data_path: /data/train
epoch_num: 5
""")
args, detected_mode = load_args_from_yaml(config_path=yaml_with_mode)
print(f"   ✅ 自动识别模式: {detected_mode}")
print(f"   batch_size: {args.batch_size}")
print(f"   data_path: {args.data_path}")

# 4. 测试生成默认配置
print("\n4. 测试生成默认配置:")
gen_path = Path("/tmp/generated_pretrain.yaml")
generated = generate_default_config("pretrain", gen_path)
print(f"   ✅ 已生成: {generated}")
print(f"   内容预览:")
print("-" * 40)
content = gen_path.read_text()
for line in content.split('\n')[:12]:
    print(f"   {line}")
print("   ...")
print("-" * 40)

# 5. 测试配置验证
print("\n5. 测试配置验证:")
try:
    invalid_yaml = Path("/tmp/invalid_config.yaml")
    invalid_yaml.write_text("""
mode: pretrain
batch_size: 32
epoch_num: 10
data_path: ""
""")
    args, mode = load_args_from_yaml(config_path=invalid_yaml)
except ValueError as e:
    print(f"   ❌ 验证失败 (预期行为):")
    print(f"      {e}")

# 6. 使用方式说明
print("\n" + "=" * 60)
print("使用方式:")
print("=" * 60)
print("""
# 生成默认配置（包含 mode 字段）
python -m trainer.train --generate-config --mode pretrain

# 使用配置文件（自动识别 mode）
python -m trainer.train --config my_config.yaml

# 使用默认配置文件 configs/pretrain.yaml
python -m trainer.train --mode pretrain

# 跳过验证
python -m trainer.train --config config.yaml --no-validate

# 使用环境变量
export DATA_PATH=/my/data
# 在 config.yaml 中: data_path: ${DATA_PATH}/train
python -m trainer.train --config config.yaml
""")

print("所有功能演示完成！")
