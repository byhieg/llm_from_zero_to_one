#!/bin/bash
source /etc/network_turbo

# 设置 PYTHONPATH，将 src 目录添加到模块搜索路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 设置分布式训练参数
# 如果你有多张 GPU，可以修改 --nproc_per_node 参数
# 如果是单机单卡，设置为 1
NUM_GPUS=2  # 根据你的 GPU 数量修改

# 配置文件路径
CONFIG_PATH="/root/llm_from_zero_to_one/configs/autodl_pretrain_minimind_deepspeed.yaml" 

# 使用 torchrun 启动训练
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    main.py \
    --config=${CONFIG_PATH} \
    "$@"  # 允许传递额外的命令行参数