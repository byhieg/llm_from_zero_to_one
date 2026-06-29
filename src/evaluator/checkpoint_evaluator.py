import math
from typing import Any

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from checkpoint_manager import CheckpointManager
from dataset import PretrainPaddingDataset
from logger import get_logger
from models import create_model
from .eval_args import EvalArgs

logger = get_logger(__name__)


class PretrainEvaluator:
    def __init__(self, args: EvalArgs | Any):
        self.args = args
        self.checkpoint_manager = CheckpointManager(
            args.checkpoint, self._get_checkpoint_model_name()
        )
        self._tokenizer = None
        self._eval_dataloader = None

    def run(self) -> dict[str, float]:
        return self.evaluate_checkpoint()

    def evaluate_checkpoint(self, step: int | None = None) -> dict[str, float]:
        device = self._get_device()
        model = create_model(self.args.model.name, self._get_model_config())
        checkpoint = self.checkpoint_manager.get_checkpoint(
            step=step if step is not None else self.args.eval.checkpoint_step
        )
        if checkpoint is None:
            raise ValueError(
                f"No checkpoint found under {self.checkpoint_manager.checkpoint_dir}"
            )

        model.load_state_dict(checkpoint.model_state_dict)
        model = model.to(device)
        checkpoint_step = (
            checkpoint.metadata.get("global_step")
            if checkpoint.metadata is not None
            else None
        )
        metrics = self.evaluate_model(
            model=model,
            device=device,
            checkpoint_step=checkpoint_step,
        )

        return metrics

    def evaluate_model(
        self,
        model: torch.nn.Module,
        device: torch.device | None = None,
        checkpoint_step: int | None = None,
    ) -> dict[str, float]:
        if device is None:
            device = self._get_model_device(model)
        dataloader = self._get_eval_dataloader()
        metrics = self._evaluate(model, dataloader, device)

        logger.info(f"eval device: {device}")
        if checkpoint_step is not None:
            logger.info(f"eval checkpoint step: {checkpoint_step}")
        logger.info(
            f"eval score - loss: {metrics['loss']:.6f}, perplexity: {metrics['perplexity']:.6f}, "
            f"tokens: {int(metrics['token_count'])}, samples: {int(metrics['sample_count'])}"
        )
        return metrics

    def _get_model_config(self) -> dict:
        model_config = dict(self.args.model.config)
        model_config.setdefault("block_size", self._get_seq_len())
        return model_config

    def _get_checkpoint_model_name(self) -> str:
        if hasattr(self.args, "name") and self.args.name:
            return self.args.name
        experiment = getattr(self.args, "experiment", None)
        if experiment is not None and experiment.name:
            return experiment.name
        return self.args.model.name

    def _get_seq_len(self) -> int:
        if hasattr(self.args, "training"):
            return self.args.training.seq_len
        return self.args.train.seq_len

    def _get_tokenizer_path(self) -> str:
        return self._get_eval_dataset_config().get("tokenizer_path", "")

    def _get_eval_dataset_config(self) -> dict[str, Any]:
        if hasattr(self.args, "data") and hasattr(self.args.data, "eval"):
            return dict(self.args.data.eval.dataset_config)
        return {
            "dataset_path": self.args.eval.dataset_path,
            "dataset_name": self.args.eval.dataset_name,
            "data_files": self.args.eval.data_files,
            "split": self.args.eval.split,
            "text_column": self.args.eval.text_column,
            "tokenizer_path": self.args.eval.tokenizer_path,
            "add_bos_id": self.args.eval.add_bos_id,
            "add_eos_id": self.args.eval.add_eos_id,
        }

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._get_tokenizer_path())
        return self._tokenizer

    def _get_eval_dataloader(self) -> DataLoader:
        if self._eval_dataloader is None:
            self._eval_dataloader = self._build_eval_dataloader(self._get_tokenizer())
        return self._eval_dataloader

    def _build_eval_dataloader(self, tokenizer) -> DataLoader:
        eval_dataset_config = self._get_eval_dataset_config()
        dataset = load_dataset(
            eval_dataset_config.get("dataset_path"),
            eval_dataset_config.get("dataset_name") or None,
            split=eval_dataset_config.get("split", "test"),
            data_files=eval_dataset_config.get("data_files") or None,
        )
        max_samples = min(self.args.eval.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
        eval_dataset = PretrainPaddingDataset(
            tokenizer=tokenizer,
            max_seq=self._get_seq_len(),
            dataset=dataset,
            dataset_config={
                "col_name": eval_dataset_config.get(
                    "col_name", eval_dataset_config.get("text_column", "text")
                ),
                "add_bos_id": eval_dataset_config.get("add_bos_id", False),
                "add_eos_id": eval_dataset_config.get("add_eos_id", True),
            },
        )
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval.batch_size,
            shuffle=False,
        )

    @torch.no_grad()
    def _evaluate(
        self,
        model,
        dataloader: DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        was_training = getattr(model, "training", None)
        model.eval()
        total_loss = 0.0
        total_token_count = 0
        total_sample_count = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits, _ = model(x, y)
            loss_sum = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1).long(),
                ignore_index=-100,
                reduction="sum",
            )
            token_count = (y != -100).sum().item()
            total_loss += loss_sum.item()
            total_token_count += token_count
            total_sample_count += x.size(0)
        if total_token_count == 0:
            raise ValueError("No valid target tokens found in evaluation dataset")
        if was_training:
            model.train()
        avg_loss = total_loss / total_token_count
        perplexity = math.exp(avg_loss)
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "token_count": float(total_token_count),
            "sample_count": float(total_sample_count),
        }

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _get_model_device(self, model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return self._get_device()
