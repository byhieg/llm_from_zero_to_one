import argparse
from pathlib import Path

from logger import get_logger
from trainer.train_args import load_args_from_yaml, generate_default_config

logger = get_logger(__name__)


def run():
    parser = argparse.ArgumentParser(description="LLM training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (YAML must contain 'mode' field)",
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

    if parsed.generate_config:
        from trainer.train_args import get_args_class
        output_path = generate_default_config("pretrain", parsed.config)
        logger.info(f"✅ Generated default config: {output_path}")
        return

    config_path = Path(parsed.config)
    if not config_path.exists():
        config_path = Path("configs") / parsed.config
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {parsed.config} (tried {config_path})")

    logger.info(f"📄 Loading config from: {config_path}")

    args, mode = load_args_from_yaml(
        config_path=config_path,
        validate=not parsed.no_validate
    )
    logger.info(f"🎯 Running in mode: {mode}")
    logger.info(f"📋 Loaded args: {args}")

    logger.info("🚀 train start")

    if mode == "pretrain":
        from .pretrain import PreTrainTrainer
        PreTrainTrainer(args).run()

if __name__ == "__main__":
    run()
