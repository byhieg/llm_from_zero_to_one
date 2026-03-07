import math

from datasets import load_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from checkpoint_manager import CheckpointManager
from dataset import PretrainPaddingDataset
from logger import get_logger
from models import create_model
from trainer.train_args import EvalArgs

logger = get_logger(__name__)


class PretrainEvaluator:
    def __init__(self, args: EvalArgs):
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
        model_config.setdefault("block_size", self.args.training.seq_len)
        return model_config

    def _get_checkpoint_model_name(self) -> str:
        return self.args.name or self.args.model.name

    def _get_tokenizer_path(self) -> str:
        return self.args.eval.tokenizer_path

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._get_tokenizer_path())
        return self._tokenizer

    def _get_eval_dataloader(self) -> DataLoader:
        if self._eval_dataloader is None:
            self._eval_dataloader = self._build_eval_dataloader(self._get_tokenizer())
        return self._eval_dataloader

    def _build_eval_dataloader(self, tokenizer) -> DataLoader:
        dataset = load_dataset(
            self.args.eval.dataset_path,
            self.args.eval.dataset_name or None,
            split=self.args.eval.split,
            data_files=self.args.eval.data_files or None,
        )
        max_samples = min(self.args.eval.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
        eval_dataset = PretrainPaddingDataset(
            tokenizer=tokenizer,
            max_seq=self.args.training.seq_len,
            dataset=dataset,
            dataset_config={
                "col_name": self.args.eval.text_column,
                "add_bos_id": self.args.eval.add_bos_id,
                "add_eos_id": self.args.eval.add_eos_id,
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
