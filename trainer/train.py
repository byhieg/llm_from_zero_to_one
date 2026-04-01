import argparse
from pathlib import Path

from logger import get_logger
from trainer.train_args import load_args_from_yaml, list_modes

logger = get_logger(__name__)


def run():
    parser = argparse.ArgumentParser(description="LLM training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=list_modes(),
        help="Training mode (default: pretrain)",
    )
    parsed = parser.parse_args()
    mode = parsed.mode
    config_path = Path(parsed.config)

    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return

    logger.info(f"🎯 Running in mode: {mode}")
    logger.info(f"📄 Loading config from: {config_path}")
    args = load_args_from_yaml(mode, config_path)
    logger.info(f"📋 Loaded args: {args}")
    logger.info("🚀 train start")
    
    if mode == "pretrain":
        from .pretrain import PreTrainTrainer
        PreTrainTrainer(args).run()

if __name__ == "__main__":
    run()