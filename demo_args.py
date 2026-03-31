#!/usr/bin/env python3
"""演示 args.py 的功能"""

from trainer.args import parse_args, list_modes, get_args_class

print("=" * 60)
print("args.py 功能演示")
print("=" * 60)

# 1. 列出所有可用的模式
print("\n1. 可用的训练模式:")
modes = list_modes()
for mode in modes:
    print(f"   - {mode}")

# 2. 测试基本参数解析
print("\n2. 测试基本参数解析:")
print("   命令: --batch-size 32 --learning-rate 0.001")
args = parse_args("pretrain", ["--batch-size", "32", "--learning-rate", "0.001"])
print(f"   batch_size = {args.batch_size}")
print(f"   learning_rate = {args.learning_rate}")
print(f"   seq_len = {args.seq_len} (默认值)")

# 3. 测试下划线格式
print("\n3. 测试下划线格式参数:")
print("   命令: --batch_size 64 --epoch_num 10")
args2 = parse_args("pretrain", ["--batch_size", "64", "--epoch_num", "10"])
print(f"   batch_size = {args2.batch_size}")
print(f"   epoch_num = {args2.epoch_num}")

# 4. 测试默认值
print("\n4. 测试默认值 (无参数):")
args3 = parse_args("pretrain", [])
print(f"   batch_size = {args3.batch_size} (默认)")
print(f"   seq_len = {args3.seq_len} (默认)")
print(f"   epoch_num = {args3.epoch_num} (默认)")
print(f"   learning_rate = {args3.learning_rate} (默认)")

# 5. 测试 PretrainArgs 特有参数
print("\n5. 测试 PretrainArgs 特有参数:")
args4 = parse_args("pretrain", [
    "--inference-steps", "1000",
    "--inference-tokens", "50",
    "--inference-temperature", "0.8",
    "--inference-prompt", "Hello world"
])
print(f"   inference_steps = {args4.inference_steps}")
print(f"   inference_tokens = {args4.inference_tokens}")
print(f"   inference_temperature = {args4.inference_temperature}")
print(f"   inference_prompt = {args4.inference_prompt}")

# 6. 测试完整参数
print("\n6. 测试完整参数:")
args5 = parse_args("pretrain", [
    "--batch-size", "32",
    "--seq-len", "512",
    "--epoch-num", "10",
    "--data-path", "/path/to/data",
    "--learning-rate", "0.001",
    "--warmup-steps", "100",
    "--grad-clip", "0.5",
    "--accumulation-steps", "2",
    "--log-steps", "50",
    "--save-steps", "500",
    "--checkpoint-dir", "checkpoints/test",
])
print(f"   batch_size = {args5.batch_size}")
print(f"   seq_len = {args5.seq_len}")
print(f"   epoch_num = {args5.epoch_num}")
print(f"   data_path = {args5.data_path}")
print(f"   learning_rate = {args5.learning_rate}")
print(f"   warmup_steps = {args5.warmup_steps}")
print(f"   grad_clip = {args5.grad_clip}")
print(f"   accumulation_steps = {args5.accumulation_steps}")
print(f"   log_steps = {args5.log_steps}")
print(f"   save_steps = {args5.save_steps}")
print(f"   checkpoint_dir = {args5.checkpoint_dir}")

# 7. 测试 Optional 参数
print("\n7. 测试 Optional 参数:")
args6 = parse_args("pretrain", ["--resume-from-checkpoint", "path/to/checkpoint"])
print(f"   resume_from_checkpoint = {args6.resume_from_checkpoint}")

args7 = parse_args("pretrain", [])
print(f"   resume_from_checkpoint (无参数) = {args7.resume_from_checkpoint}")

print("\n" + "=" * 60)
print("所有功能测试完成！")
print("=" * 60)
