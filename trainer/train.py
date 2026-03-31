import argparse

from logger import get_logger
logger = get_logger(__name__)
def run():
    
    logger.info("hello world")
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

if __name__ == "__main__":
    run()