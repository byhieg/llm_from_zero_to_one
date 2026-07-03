def main():
    from logger import init_logger
    from trainer.train import run

    init_logger()
    run()


if __name__ == "__main__":
    main()
