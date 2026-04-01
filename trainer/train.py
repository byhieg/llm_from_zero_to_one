import argparse

from logger import get_logger
from trainer.train_args import parse_args, list_modes

logger = get_logger(__name__)


def run():
    parser = argparse.ArgumentParser(description="LLM training entrypoint")
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=list_modes(),
        help="Training mode (default: pretrain)",
    )
    parsed, argv = parser.parse_known_args()
    mode = parsed.mode

    logger.info(f"🎯 Running in mode: {mode}")
    args = parse_args(mode, argv)
    logger.info(f"📋 Parsed args: {args}")
    logger.info("🚀 train start")
    if mode == "pretrain":
        from .pretrain import PreTrainTrainer
        PreTrainTrainer(args).run()

if __name__ == "__main__":
    run()