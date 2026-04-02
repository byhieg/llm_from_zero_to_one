from ..train_args import PretrainArgs

from logger import get_logger
from models import create_model,BaseModel
logger = get_logger(__name__)


class PreTrainTrainer:
    
    def __init__(self, args: PretrainArgs):
        self.args = args
        
    def run(self)->None:
        logger.info("PretrainTrainer run")
        model = create_model(self.args.model.name,self.args.model.config)
        logger.info(model)