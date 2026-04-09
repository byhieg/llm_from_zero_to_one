import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    from logger import init_logger
    from trainer import run

    init_logger("INFO")
    run()


if __name__ == "__main__":
    main()
