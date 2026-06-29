from .pretrain import PreTrainTrainer
from .pretrain_args import PreTrainArgs
from trainer.train_args import register_args

register_args("pretrain", PreTrainArgs)

__all__ = ["PreTrainTrainer", "PreTrainArgs"]
