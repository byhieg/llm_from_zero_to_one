from ..train_args import PretrainArgs

from logger import get_logger

logger = get_logger(__name__)


class PreTrainTrainer:
    
    def __init__(self, args: PretrainArgs):
        self.args = args
        
    def run(self):
        logger.info("PretrainTrainer run")