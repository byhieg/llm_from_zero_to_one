#!/usr/bin/env python3
"""演示 train_args.py 的 YAML 功能"""

from pathlib import Path
from trainer.train_args import PretrainArgs, load_args_from_yaml, list_modes

print("=" * 60)
print("train_args.py YAML 功能演示")
print("=" * 60)

# 1. 列出所有可用的模式
print("\n1. 可用的训练模式:")
modes = list_modes()
for mode in modes:
    print(f"   - {mode}")

# 2. 测试创建默认参数并保存为 YAML
print("\n2. 创建默认参数并保存为 YAML:")
args = PretrainArgs(
    batch_size=32,
    learning_rate=0.001,
    epoch_num=10,
    data_path="/path/to/data",
)
print(f"   batch_size = {args.batch_size}")
print(f"   learning_rate = {args.learning_rate}")
print(f"   epoch_num = {args.epoch_num}")

# 保存到临时文件
temp_yaml = Path("/tmp/test_pretrain_config.yaml")
args.to_yaml(temp_yaml)
print(f"   ✅ 已保存到: {temp_yaml}")

# 3. 从 YAML 文件加载
print("\n3. 从 YAML 文件加载:")
loaded_args = PretrainArgs.from_yaml(temp_yaml)
print(f"   batch_size = {loaded_args.batch_size}")
print(f"   learning_rate = {loaded_args.learning_rate}")
print(f"   epoch_num = {loaded_args.epoch_num}")
print(f"   data_path = {loaded_args.data_path}")

# 4. 使用 load_args_from_yaml 工厂函数
print("\n4. 使用 load_args_from_yaml 工厂函数:")
factory_args = load_args_from_yaml("pretrain", temp_yaml)
print(f"   type = {type(factory_args).__name__}")
print(f"   batch_size = {factory_args.batch_size}")

# 5. 展示 YAML 文件内容
print("\n5. YAML 文件内容:")
print("-" * 40)
print(temp_yaml.read_text())
print("-" * 40)

# 6. 测试部分参数 (使用默认值填充)
print("\n6. 测试部分参数 YAML:")
partial_yaml = Path("/tmp/partial_config.yaml")
partial_yaml.write_text("""
batch_size: 64
learning_rate: 0.0001
""")
partial_args = PretrainArgs.from_yaml(partial_yaml)
print(f"   batch_size = {partial_args.batch_size} (来自 YAML)")
print(f"   learning_rate = {partial_args.learning_rate} (来自 YAML)")
print(f"   seq_len = {partial_args.seq_len} (默认值)")
print(f"   epoch_num = {partial_args.epoch_num} (默认值)")

# 7. 转换为字典
print("\n7. 转换为字典:")
args_dict = args.to_dict()
for key, value in list(args_dict.items())[:5]:
    print(f"   {key}: {value}")
print("   ...")

print("\n" + "=" * 60)
print("所有功能测试完成！")
print("=" * 60)
print("\n使用方式:")
print("  python -m trainer.train --config configs/pretrain.yaml --mode pretrain")
