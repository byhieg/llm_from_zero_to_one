def main():
    from logger import init_logger
    from trainer.train import run

    init_logger("INFO", rank=0)
    run()


if __name__ == "__main__":
    main()
