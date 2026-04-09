import random
from dataclasses import asdict
from functools import partial
from importlib import import_module

import math
import time

from checkpoint_manager import CheckpointManager, Checkpoint
import torch
import torch.distributed as dist
from evaluator import PretrainEvaluator
from dataset import create_dataset
from logger import get_logger
from models import create_model
import os
from torch.utils.data import Sampler, DistributedSampler

from ..train_args import PretrainArgs

logger = get_logger(__name__)


def _set_process_seed(seed: int, init_cuda: bool = False) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    if init_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def _seed_dataloader_worker(worker_id: int, base_seed: int) -> None:
    _set_process_seed(base_seed + worker_id)


class ResumableDistributedSampler(DistributedSampler):
    """支持 checkpoint 断点续训的分布式采样器

    在 DistributedSampler 基础上，增加根据 micro_step 跳过已训练数据的能力。
    """

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.sample_offset = 0

    def set_micro_step_offset(self, micro_step_offset: int, batch_size: int) -> None:
        self.sample_offset = max(0, micro_step_offset * batch_size)

    def __iter__(self):
        # 直接复用父类的分片逻辑
        indices = list(super().__iter__())
        yield from indices[self.sample_offset :]


class PreTrainTrainer:
    def __init__(self, args: PretrainArgs):
        self.args = args
        self._swanlab = None
        self._swanlab_run_id: str | None = None
        self._evaluator = None
        self.checkpoint_manager = CheckpointManager(
            args.checkpoint, self._get_checkpoint_model_name()
        )

    def run(self) -> None:
        self._init_seed()

        self.rank_info = self._build_distributed()
        device = self._get_device()
        model = create_model(self.args.model.name, self._get_model_config())
        dataset = create_dataset(
            data_strategy=self.args.data.data_strategy,
            dataset_config=self._get_dataset_config(),
        )

        dataloader = self._build_dataloader(dataset)

        steps_per_epoch = len(dataloader) // self.args.training.accumulation_steps
        max_steps = self.args.training.epoch_num * steps_per_epoch

        logger.info(model)
        logger.info(f"dataset size: {len(dataset)} samples")
        logger.info(f"train device: {device}")
        logger.info(f"dataloader batch num: {len(dataloader)}")
        logger.info(
            f"total steps num: {max_steps} (epoch_num: {self.args.training.epoch_num}, "
            f"perepoch steps: {steps_per_epoch}, accumulation_steps: {self.args.training.accumulation_steps}, eval_steps: {self.args.eval.steps})"
        )

        checkpoint: Checkpoint | None = self.checkpoint_manager.get_checkpoint()
        global_step = 0
        start_epoch = 0
        start_micro_step_in_epoch = 0
        if checkpoint and self._is_checkpoint_compatible(checkpoint):
            metadata = checkpoint.metadata or {}
            global_step = metadata.get("global_step", 0)
            start_epoch = metadata.get("epoch", 0)
            start_micro_step_in_epoch = metadata.get("micro_step_in_epoch", 0)
            self._swanlab_run_id = metadata.get("swanlab_run_id") or None
            logger.info(
                f"Loading checkpoint from step {global_step}, "
                f"epoch {start_epoch}, micro_step {start_micro_step_in_epoch}"
            )
            model.load_state_dict(checkpoint.model_state_dict)
            if self._is_distributed():
                dist.barrier()
        else:
            logger.info("No checkpoint found, starting from scratch")

        if self._is_main_process():
            self._init_swanlab(device, dataset, dataloader, run_id=self._swanlab_run_id)

        model = model.to(device)
        optimizer = self._build_optimizer(model)
        grad_scaler = self._build_grad_scaler(device)
        if checkpoint and checkpoint.optimizer_state_dict:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        model = self._maybe_compile_model(model, device)
        if self._is_distributed():
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.rank_info["local_rank"]]
            )
        logger.info(f"optimizer: {type(optimizer).__name__}")
        accumulated_loss = torch.tensor(0.0, device=device)
        tokens = (
            self.args.training.batch_size
            * self.args.training.seq_len
            * self.args.training.accumulation_steps
            * self.args.training.log_steps
            * self.rank_info["world_size"]
        )
        try:
            eval_elapsed_since_log = 0.0
            for epoch in range(start_epoch, self.args.training.epoch_num):
                self._set_dataloader_epoch(dataloader, epoch)
                model.train()
                optimizer.zero_grad()
                logger.info(f"🚀 Epoch {epoch} start to train")
                self._log_swanlab({"train/epoch": epoch})
                start_time = time.perf_counter()
                micro_step_offset = (
                    start_micro_step_in_epoch if epoch == start_epoch else 0
                )
                epoch_iterator = self._build_epoch_iterator(
                    dataloader, micro_step_offset
                )
                for step, (x, y) in enumerate(epoch_iterator, start=micro_step_offset):
                    lr = self._get_lr(
                        global_step,
                        max_steps,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    x, y = (
                        x.to(device, non_blocking=True),
                        y.to(device, non_blocking=True),
                    )
                    should_skip_optimizer_step = (
                        step + 1
                    ) % self.args.training.accumulation_steps != 0
                    if self._is_amp_enabled(device):
                        with torch.autocast(
                            device_type="cuda",
                            dtype=self._get_amp_dtype(),
                        ):
                            _, loss = model(x, y)
                    else:
                        _, loss = model(x, y)
                    loss = loss / self.args.training.accumulation_steps
                    if grad_scaler is not None:
                        grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    accumulated_loss += loss.detach()

                    if should_skip_optimizer_step:
                        continue
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.training.grad_clip
                    )
                    if grad_scaler is not None:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    self._save_checkpoint_if_needed(
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        epoch=epoch,
                        micro_step_in_epoch=step + 1,
                        dataloader_length=len(dataloader),
                    )
                    if global_step % self.args.training.log_steps == 0:
                        elapsed_ms = (
                            time.perf_counter() - start_time - eval_elapsed_since_log
                        ) * 1000
                        self._log_swanlab(
                            {
                                "train/step": global_step,
                                "train/loss": accumulated_loss.item(),
                                "train/grad_norm": grad_norm.item(),
                                "train/lr": lr,
                                "train/throughput": int(tokens / (elapsed_ms / 1000)),
                            }
                        )
                        start_time = time.perf_counter()
                        eval_elapsed_since_log = 0.0

                    accumulated_loss = torch.tensor(0.0, device=device)
                    eval_elapsed_since_log += self._run_eval_if_needed(
                        model=model,
                        device=device,
                        global_step=global_step,
                    )

            self._save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                global_step=global_step,
                epoch=self.args.training.epoch_num,
                micro_step_in_epoch=0,
                dataloader_length=len(dataloader),
            )

        finally:
            self._finish_swanlab()

    def _build_distributed(self):
        if not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                if torch.cuda.is_available():
                    dist.init_process_group(backend="nccl")
                else:
                    raise ValueError(
                        "CUDA is not available, but distributed training is enabled."
                    )
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            node_rank = int(os.environ.get("NODE_RANK", 0))
        else:
            world_size = 1
            rank = 0
            local_rank = 0
            local_world_size = 1
            node_rank = 0

        return {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "local_world_size": local_world_size,
            "node_rank": node_rank,
            "is_distributed": world_size > 1,
            "is_multi_node": world_size > local_world_size,
        }

    def _is_distributed(self) -> bool:
        return self.rank_info["is_distributed"]

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            if self.rank_info["is_distributed"]:
                local_rank = self.rank_info["local_rank"]
                torch.cuda.set_device(local_rank)
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _is_main_process(self) -> bool:
        return self.rank_info["rank"] == 0

    def _get_model_config(self) -> dict:
        model_config = dict(self.args.model.config)
        model_config.setdefault("block_size", self.args.training.seq_len)
        return model_config

    def _get_checkpoint_model_name(self) -> str:
        return self.args.name or self.args.model.name

    def _get_dataset_config(self) -> dict:
        dataset_config = dict(self.args.data.dataset_config)
        dataset_config.setdefault("seq_len", self.args.training.seq_len)
        return dataset_config

    def _build_dataloader(self, dataset):
        dataloader_config = self.args.data.dataloader_config
        dataloader_seed = self._get_dataloader_seed()

        num_workers = dataloader_config.get("num_workers", 0)
        persistent_workers = dataloader_config.get("persistent_workers", False)
        if num_workers <= 0:
            persistent_workers = False
        worker_init_fn = None
        if num_workers > 0:
            worker_init_fn = partial(_seed_dataloader_worker, base_seed=dataloader_seed)
        shuffle = dataloader_config.get("shuffle", True)
        sampler = ResumableDistributedSampler(
            dataset,
            seed=dataloader_seed,
            shuffle=shuffle,
            num_replicas=self.rank_info["world_size"],
            rank=self.rank_info["rank"],
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=dataloader_config.get("pin_memory", False),
            drop_last=dataloader_config.get("drop_last", False),
            persistent_workers=persistent_workers,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
        )

    def _set_dataloader_epoch(self, dataloader, epoch: int) -> None:
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        optimizer_name = self.args.optimizer.name.lower()
        optimizer_kwargs = {
            "lr": self.args.training.learning_rate,
            "weight_decay": self.args.optimizer.weight_decay,
            "betas": tuple(self.args.optimizer.betas),
            "eps": self.args.optimizer.eps,
        }
        if optimizer_name == "adamw":
            return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        if optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        raise ValueError(f"Unsupported optimizer: {self.args.optimizer.name}")

    def _build_epoch_iterator(self, dataloader, micro_step_offset: int):
        sampler = getattr(dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_micro_step_offset"):
            batch_size = getattr(dataloader, "batch_size", None)
            if batch_size is None:
                raise ValueError("dataloader.batch_size must be set for resume skip")
            sampler.set_micro_step_offset(micro_step_offset, batch_size)
            return iter(dataloader)
        iterator = iter(dataloader)
        for _ in range(micro_step_offset):
            try:
                next(iterator)
            except StopIteration:
                return iter(())
        return iterator

    def _maybe_compile_model(
        self, model: torch.nn.Module, device: torch.device
    ) -> torch.nn.Module:
        if not hasattr(torch, "compile"):
            return model
        if device.type not in ("cuda", "mps"):
            return model
        try:
            if device.type == "cuda":
                torch.set_float32_matmul_precision("high")
            compiled_model = torch.compile(model)
            logger.info(f"torch.compile enabled on {device.type}")
            return compiled_model
        except Exception as exc:
            logger.warning(
                f"torch.compile failed on {device.type}, skip compile: {exc}"
            )
            return model

    def _init_swanlab(
        self,
        device: torch.device,
        dataset,
        dataloader,
        run_id: str | None = None,
    ) -> None:
        if not self.args.swanlab.enabled:
            return
        try:
            self._swanlab = import_module("swanlab")
        except ImportError as exc:
            raise ImportError(
                "swanlab.enabled=true but swanlab is not installed, please install it first."
            ) from exc
        run = self._swanlab.init(
            project=self.args.swanlab.project,
            experiment_name=self.args.swanlab.experiment_name,
            config=self._build_swanlab_config(device, dataset, dataloader),
            tags=self.args.swanlab.tags,
            id=run_id if run_id else None,
            resume="allow" if run_id else None,
        )
        try:
            self._swanlab_run_id = (
                getattr(getattr(run, "public", None), "run_id", None) or run_id
            )
        except Exception:
            self._swanlab_run_id = run_id

    def _log_swanlab(self, data: dict) -> None:
        if self._swanlab is not None:
            if self._is_main_process():
                self._swanlab.log(data)

        logger.info(data)

    def _finish_swanlab(self) -> None:
        if self._swanlab is not None:
            if self._is_main_process():
                self._swanlab.finish()
                self._swanlab = None

    def _build_swanlab_config(self, device: torch.device, dataset, dataloader) -> dict:
        config = asdict(self.args)
        config.pop("swanlab", None)
        config["data"]["dataset_config"] = self._get_dataset_config()
        config["runtime"] = {
            "dataset_size": len(dataset),
            "dataloader_batches": len(dataloader),
            "device": str(device),
        }
        return config

    def _get_lr(self, step: int, max_steps: int) -> float:
        warmup_steps = self.args.training.warmup_steps
        learning_rate = self.args.training.learning_rate
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        if step > max_steps:
            return learning_rate * 0.1
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * coeff

    def _get_dataloader_seed(self) -> int:
        return self.args.data.dataloader_config.get(
            "seed", 42 if not self.args.training.seed else self.args.training.seed
        )

    def _is_amp_enabled(self, device: torch.device) -> bool:
        return device.type == "cuda" and self.args.training.amp

    def _get_amp_dtype(self) -> torch.dtype:
        if self.args.training.amp_dtype == "bf16":
            return torch.bfloat16
        if self.args.training.amp_dtype == "fp16":
            return torch.float16
        raise ValueError(f"Unsupported amp dtype: {self.args.training.amp_dtype}")

    def _build_grad_scaler(self, device: torch.device) -> torch.amp.GradScaler | None:
        if not self._is_amp_enabled(device):
            return None
        if self.args.training.amp_dtype != "fp16":
            return None
        return torch.amp.GradScaler("cuda")

    def _is_eval_enabled(self) -> bool:
        return self.args.eval.steps > 0

    def _get_pretrain_evaluator(self) -> PretrainEvaluator:
        if self._evaluator is None:
            self._evaluator = PretrainEvaluator(self.args)
        return self._evaluator

    def _run_eval_if_needed(
        self,
        model: torch.nn.Module,
        device: torch.device,
        global_step: int,
    ) -> float:
        if not self._is_eval_enabled():
            return 0.0
        if global_step <= 0 or global_step % self.args.eval.steps != 0:
            return 0.0
        if self._is_main_process():
            eval_start_time = time.perf_counter()
            metrics = self._get_pretrain_evaluator().evaluate_model(
                model=model,
                device=device,
                checkpoint_step=global_step,
            )
            self._log_swanlab(
                {
                    "eval/step": global_step,
                    "eval/loss": metrics["loss"],
                    "eval/perplexity": metrics["perplexity"],
                    "eval/token_count": metrics["token_count"],
                    "eval/sample_count": metrics["sample_count"],
                }
            )
            
        if self._is_distributed():
            dist.barrier()

        return time.perf_counter() - eval_start_time if self._is_main_process() else 0.0

    def _get_checkpoint_model_state(self, model: torch.nn.Module) -> dict:
        # for ddp 
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            return model._orig_mod.state_dict()
        return model.state_dict()

    def _get_checkpoint_resume_config(self) -> dict:
        return {
            "checkpoint_model_name": self._get_checkpoint_model_name(),
            "model_arch": self.args.model.name,
            "data_strategy": self.args.data.data_strategy,
            "world_size": self.rank_info["world_size"],
            "training": {
                "batch_size": self.args.training.batch_size,
                "seq_len": self.args.training.seq_len,
                "accumulation_steps": self.args.training.accumulation_steps,
                "seed": self.args.training.seed,
            },
            "optimizer": {
                "name": self.args.optimizer.name,
            },
            "dataloader": {
                "seed": self._get_dataloader_seed(),
                "shuffle": self.args.data.dataloader_config.get("shuffle", True),
                "drop_last": self.args.data.dataloader_config.get("drop_last", False),
            },
        }

    def _is_checkpoint_compatible(self, checkpoint: Checkpoint | None) -> bool:
        if checkpoint is None:
            return False
        metadata = checkpoint.metadata or {}
        checkpoint_resume_config = metadata.get("resume_config")
        if checkpoint_resume_config is None:
            logger.warning("checkpoint is missing resume_config, skip resume")
            return False
        current_resume_config = self._get_checkpoint_resume_config()
        if checkpoint_resume_config == current_resume_config:
            return True
        mismatch_keys = []
        for key, current_value in current_resume_config.items():
            checkpoint_value = checkpoint_resume_config.get(key)
            if checkpoint_value != current_value:
                mismatch_keys.append(key)
        logger.warning(f"checkpoint config mismatch: {', '.join(mismatch_keys)}")
        return False

    def _normalize_resume_position(
        self, epoch: int, micro_step_in_epoch: int, dataloader_length: int
    ) -> tuple[int, int]:
        if dataloader_length <= 0:
            return epoch, micro_step_in_epoch
        normalized_epoch = epoch + micro_step_in_epoch // dataloader_length
        normalized_micro_step = micro_step_in_epoch % dataloader_length
        return normalized_epoch, normalized_micro_step

    def _save_checkpoint_if_needed(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        epoch: int,
        micro_step_in_epoch: int,
        dataloader_length: int,
    ) -> None:
        save_steps = self.args.checkpoint.save_steps
        if save_steps <= 0 or global_step % save_steps != 0:
            return
        self._save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            global_step=global_step,
            epoch=epoch,
            micro_step_in_epoch=micro_step_in_epoch,
            dataloader_length=dataloader_length,
        )

    def _save_training_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        epoch: int,
        micro_step_in_epoch: int,
        dataloader_length: int,
    ) -> None:
        if not self._is_main_process():
            return
        checkpoint_epoch, checkpoint_micro_step_in_epoch = (
            self._normalize_resume_position(
                epoch, micro_step_in_epoch, dataloader_length
            )
        )
        self.checkpoint_manager.save_checkpoint(
            checkpoint=Checkpoint(
                model_state_dict=self._get_checkpoint_model_state(model),
                optimizer_state_dict=optimizer.state_dict(),
                metadata={
                    "global_step": global_step,
                    "epoch": checkpoint_epoch,
                    "micro_step_in_epoch": checkpoint_micro_step_in_epoch,
                    "resume_config": self._get_checkpoint_resume_config(),
                    "swanlab_run_id": self._swanlab_run_id,
                },
            ),
            step=global_step,
        )

    def _set_seed(self, seed: int, init_cuda: bool = False) -> None:
        _set_process_seed(seed, init_cuda=init_cuda)

    def _init_seed(self):
        seed = self.args.training.seed
        self._set_seed(seed, init_cuda=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
