from ..train_args import PretrainArgs

from logger import get_logger
from models import create_model,BaseModel
from dataset import create_dataset
logger = get_logger(__name__)


class PreTrainTrainer:
    
    def __init__(self, args: PretrainArgs):
        self.args = args
        
    def run(self)->None:
        logger.info("PretrainTrainer run")
        model = create_model(self.args.model.name,self.args.model.config)
        logger.info(model)
        
        tokenizer = None
        if self.args.data.data_strategy == 'padding':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
            logger.info("使用边训练边分词策略，已加载 tokenizer")
        
        dataset = create_dataset(
            data_strategy=self.args.data.data_strategy,
            tokenizer=tokenizer,
            args=self.args
        )
        logger.info(f"已创建数据集: {type(dataset).__name__}")
        logger.info(f"数据集大小: {len(dataset)} 样本")