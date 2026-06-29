import argparse
from pathlib import Path

from logger import get_logger
from trainer.train_args import (
    detect_mode_from_yaml,
    load_pretrain_args_from_yaml,
)

logger = get_logger(__name__)


def run():
    parser = argparse.ArgumentParser(description="LLM training entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (mode can be in 'mode' or 'experiment.mode')",
    )
    parsed = parser.parse_args()

    config_path = Path(parsed.config)
    if not config_path.exists():
        config_path = Path("configs") / parsed.config
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {parsed.config} (tried {config_path})"
            )

    logger.info(f"📄 Loading config from: {config_path}")

    mode = detect_mode_from_yaml(config_path)
    logger.info(f"🎯 Detected mode: {mode}")

    if mode == "pretrain":
        from .pretrain import PreTrainTrainer

        args = load_pretrain_args_from_yaml(config_path)
        logger.info(f"📋 Loaded pretrain args: {args}")
        PreTrainTrainer(args).run()
        return
    raise ValueError(
        f"Unsupported mode for trainer entrypoint: {mode}. Only 'pretrain' is supported."
    )


if __name__ == "__main__":
    run()
