import argparse
from pathlib import Path

from logger import get_logger
from trainer.train_args import load_args_from_yaml, list_modes, generate_default_config

logger = get_logger(__name__)


def run():
    parser = argparse.ArgumentParser(description="LLM training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (YAML can contain 'mode' field)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=list_modes(),
        help="Training mode (optional if YAML contains 'mode' field)",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate default config file and exit",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip config validation",
    )
    parsed = parser.parse_args()
    
    # 生成默认配置模式
    if parsed.generate_config:
        mode = parsed.mode or "pretrain"
        output_path = generate_default_config(mode, parsed.config)
        logger.info(f"✅ Generated default config for mode '{mode}': {output_path}")
        return
    
    # 必须指定 config 或 mode
    if parsed.config is None and parsed.mode is None:
        logger.error("❌ Please specify --config or --mode")
        logger.info("Usage examples:")
        logger.info("  python -m trainer.train --config my_config.yaml")
        logger.info("  python -m trainer.train --mode pretrain")
        return
    
    # 加载配置
    try:
        config_path = Path(parsed.config) if parsed.config else None
        if config_path:
            logger.info(f"📄 Loading config from: {config_path}")
        else:
            logger.info(f"📄 Loading default config for mode: {parsed.mode}")
        
        args, mode = load_args_from_yaml(
            mode=parsed.mode, 
            config_path=config_path, 
            validate=not parsed.no_validate
        )
        logger.info(f"🎯 Running in mode: {mode}")
        logger.info(f"📋 Loaded args: {args}")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        return
    except ValueError as e:
        logger.error(f"❌ {e}")
        return
    
    logger.info("🚀 train start")
    
    if mode == "pretrain":
        from .pretrain import PreTrainTrainer
        PreTrainTrainer(args).run()

if __name__ == "__main__":
    run()
