import random
from functools import partial
from dataclasses import asdict
from importlib import import_module

import math

from ..checkpoint_manager import CheckpointManager, Checkpoint
from ..train_args import PretrainArgs

from logger import get_logger
from models import create_model
from dataset import create_dataset
import torch
import time

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


class EpochSeededRandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, data_source, base_seed: int, shuffle: bool = True):
        self.data_source = data_source
        self.base_seed = base_seed
        self.shuffle = shuffle
        self.epoch = 0
        self.sample_offset = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_micro_step_offset(self, micro_step_offset: int, batch_size: int) -> None:
        self.sample_offset = max(0, micro_step_offset * batch_size)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + self.epoch)
            indices = torch.randperm(
                len(self.data_source), generator=generator
            ).tolist()
        else:
            indices = range(len(self.data_source))
        yield from indices[self.sample_offset :]

    def __len__(self) -> int:
        return len(self.data_source)


class PreTrainTrainer:
    def __init__(self, args: PretrainArgs):
        self.args = args
        self._swanlab = None
        self.checkpoint_manager = CheckpointManager(
            args.checkpoint, self.args.model.name
        )

    def run(self) -> None:
        self._init_seed()
        model = create_model(self.args.model.name, self._get_model_config())
        dataset = create_dataset(
            data_strategy=self.args.data.data_strategy,
            dataset_config=self._get_dataset_config(),
        )

        dataloader = self._build_dataloader(dataset)
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        steps_per_epoch = len(dataloader) // self.args.training.accumulation_steps
        max_steps = self.args.training.epoch_num * steps_per_epoch
        logger.info(model)
        logger.info(f"数据集大小: {len(dataset)} 样本")
        logger.info(f"设备: {device}")
        logger.info(f"dataloader batch 数: {len(dataloader)}")
        logger.info(
            f"总训练步数: {max_steps} (epoch_num: {self.args.training.epoch_num}, "
            f"每个epoch步数: {steps_per_epoch}, 梯度累加步数: {self.args.training.accumulation_steps})"
        )

        self._init_swanlab(device, dataset, dataloader)
        checkpoint: Checkpoint | None = self.checkpoint_manager.get_checkpoint()
        global_step = 0
        start_epoch = 0
        start_micro_step_in_epoch = 0
        if checkpoint and self._is_checkpoint_compatible(checkpoint):
            metadata = checkpoint.metadata or {}
            global_step = metadata.get("global_step", metadata.get("step", 0))
            start_epoch = metadata.get("epoch", 0)
            start_micro_step_in_epoch = metadata.get("micro_step_in_epoch", 0)
            logger.info(
                f"Loading checkpoint from step {global_step}, "
                f"epoch {start_epoch}, micro_step {start_micro_step_in_epoch}"
            )
            model.load_state_dict(checkpoint.model_state_dict)
        else:
            logger.info("No checkpoint found, starting from scratch")
        model = model.to(device)
        optimizer = self._build_optimizer(model)
        if checkpoint and checkpoint.optimizer_state_dict:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        model = self._maybe_compile_model(model, device)
        logger.info(f"优化器: {type(optimizer).__name__}")
        accumulated_loss = torch.tensor(0.0, device=device)
        tokens = (
            self.args.training.batch_size
            * self.args.training.seq_len
            * self.args.training.accumulation_steps
            * self.args.training.log_steps
        )
        try:
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
                    if device == torch.device("cuda") and self.args.training.amp:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _, loss = model(x, y)
                    else:
                        _, loss = model(x, y)
                    loss = loss / self.args.training.accumulation_steps
                    loss.backward()
                    accumulated_loss += loss.detach()

                    if should_skip_optimizer_step:
                        continue
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.training.grad_clip
                    )
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
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
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

                    accumulated_loss = torch.tensor(0.0, device=device)

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

    def _get_model_config(self) -> dict:
        model_config = dict(self.args.model.config)
        model_config.setdefault("block_size", self.args.training.seq_len)
        return model_config

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
        sampler = EpochSeededRandomSampler(
            dataset,
            base_seed=dataloader_seed,
            shuffle=shuffle,
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
            logger.info(f"torch.compile 已启用: {device.type}")
            return compiled_model
        except Exception as exc:
            logger.warning(f"torch.compile 在 {device.type} 上不可用，已跳过: {exc}")
            return model

    def _init_swanlab(self, device: torch.device, dataset, dataloader) -> None:
        if not self.args.swanlab.enabled:
            return
        try:
            self._swanlab = import_module("swanlab")
        except ImportError as exc:
            raise ImportError(
                "swanlab.enabled=true 但当前环境未安装 swanlab，请先安装 swanlab。"
            ) from exc
        self._swanlab.init(
            project=self.args.swanlab.project,
            experiment_name=self.args.swanlab.experiment_name,
            config=self._build_swanlab_config(device, dataset, dataloader),
            tags=self.args.swanlab.tags,
        )

    def _log_swanlab(self, data: dict) -> None:
        if self._swanlab is not None:
            self._swanlab.log(data)
        else:
            logger.info(data)

    def _finish_swanlab(self) -> None:
        if self._swanlab is not None:
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
        return self.args.data.dataloader_config.get("seed", self.args.training.seed)

    def _get_checkpoint_model_state(self, model: torch.nn.Module) -> dict:
        return model.state_dict()

    def _get_checkpoint_resume_config(self) -> dict:
        return {
            "model_name": self.args.model.name,
            "data_strategy": self.args.data.data_strategy,
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
            logger.warning("checkpoint 缺少 resume_config，已跳过恢复")
            return False
        current_resume_config = self._get_checkpoint_resume_config()
        if checkpoint_resume_config == current_resume_config:
            return True
        mismatch_keys = []
        for key, current_value in current_resume_config.items():
            checkpoint_value = checkpoint_resume_config.get(key)
            if checkpoint_value != current_value:
                mismatch_keys.append(key)
        logger.warning(
            f"checkpoint 配置与当前配置不一致，已跳过恢复: {', '.join(mismatch_keys)}"
        )
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
                    "step": global_step,
                    "global_step": global_step,
                    "epoch": checkpoint_epoch,
                    "micro_step_in_epoch": checkpoint_micro_step_in_epoch,
                    "resume_config": self._get_checkpoint_resume_config(),
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