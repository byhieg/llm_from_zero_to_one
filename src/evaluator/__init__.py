from .checkpoint_evaluator import PretrainEvaluator
from .eval_args import EvalArgs
from trainer.train_args import register_args

register_args("eval", EvalArgs)

__all__ = ["PretrainEvaluator", "EvalArgs"]