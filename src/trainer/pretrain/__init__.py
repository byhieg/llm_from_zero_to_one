from .pretrain import PreTrainTrainer
from .pretrain_args import PretrainArgs
from trainer.train_args import register_args

register_args("pretrain", PretrainArgs)

__all__ = ["PreTrainTrainer", "PretrainArgs"]