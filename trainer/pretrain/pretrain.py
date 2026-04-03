import random
from functools import partial
from dataclasses import asdict
from importlib import import_module

import math
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


class PreTrainTrainer:
    def __init__(self, args: PretrainArgs):
        self.args = args
        self._swanlab = None
        
    def _get_model_config(self) -> dict:
        model_config = dict(self.args.model.config)
        model_config.setdefault("block_size", self.args.training.seq_len)
        return model_config

    def _get_dataset_config(self) -> dict:
        dataset_config = dict(self.args.data.dataset_config)
        dataset_config.setdefault("seq_len", self.args.training.seq_len)
        return dataset_config

    def _get_dataloader_seed(self) -> int:
        return self.args.data.dataloader_config.get("seed", self.args.training.seed)

    def _set_seed(self, seed: int, init_cuda: bool = False) -> None:
        _set_process_seed(seed, init_cuda=init_cuda)

    def _init_seed(self):
        seed = self.args.training.seed
        self._set_seed(seed, init_cuda=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_dataloader(self, dataset):
        dataloader_config = self.args.data.dataloader_config
        dataloader_seed = self._get_dataloader_seed()
        generator = torch.Generator()
        generator.manual_seed(dataloader_seed)

        num_workers = dataloader_config.get("num_workers", 0)
        persistent_workers = dataloader_config.get("persistent_workers", False)
        if num_workers <= 0:
            persistent_workers = False
        worker_init_fn = None
        if num_workers > 0:
            worker_init_fn = partial(_seed_dataloader_worker, base_seed=dataloader_seed)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.training.batch_size,
            shuffle=dataloader_config.get("shuffle", True),
            num_workers=num_workers,
            pin_memory=dataloader_config.get("pin_memory", False),
            drop_last=dataloader_config.get("drop_last", False),
            persistent_workers=persistent_workers,
            generator=generator,
            worker_init_fn=worker_init_fn,
        )

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

    def _count_effective_tokens(self, y: torch.Tensor) -> int:
        if y.numel() == 0:
            return 0
        return int((y != -100).sum().item())

    def _synchronize_device(self, device: torch.device) -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif (
            device.type == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "synchronize")
            and torch.backends.mps.is_available()
        ):
            torch.mps.synchronize()

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
            
    def _get_lr(self, step: int, max_steps: int) -> float:
        warmup_steps = self.args.training.warmup_steps
        learning_rate = self.args.training.learning_rate
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        if step > max_steps:
            return learning_rate * 0.1
        # 余弦退火
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return learning_rate * coeff
    
    
    def run(self) -> None:
        logger.info("PretrainTrainer run")
        self._init_seed()
        model = create_model(self.args.model.name, self._get_model_config())
        dataset_config = self._get_dataset_config()
        dataset = create_dataset(
            data_strategy=self.args.data.data_strategy, dataset_config=dataset_config
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
        logger.info(f"总训练步数: {max_steps} (epoch_num: {self.args.training.epoch_num}, "
                    f"每个epoch步数: {steps_per_epoch}, 梯度累加步数: {self.args.training.accumulation_steps})")
        
        self._init_swanlab(device, dataset, dataloader)
        model = model.to(device)
        model = self._maybe_compile_model(model, device)
        optimizer = self._build_optimizer(model)
        logger.info(f"优化器: {type(optimizer).__name__}")
        global_step = 0
        accumulated_loss = torch.tensor(0.0, device=device)
        try:
            for epoch in range(self.args.training.epoch_num):
                model.train()
                optimizer.zero_grad()
                logger.info(f"🚀 Epoch {epoch} start to train")
                self._log_swanlab({"train/epoch": epoch})
                window_data_time = 0.0
                window_compute_time = 0.0
                window_total_tokens = 0
                window_effective_tokens = 0
                window_micro_steps = 0
                window_start_time = time.perf_counter()
                epoch_iterator = iter(dataloader)
                step = 0
                while True:
                    data_start_time = time.perf_counter()
                    try:
                        x, y = next(epoch_iterator)
                    except StopIteration:
                        break
                    window_data_time += time.perf_counter() - data_start_time
                    lr = self._get_lr(
                        step,
                        max_steps,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    compute_start_time = time.perf_counter()
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    window_total_tokens += x.numel()
                    window_effective_tokens += self._count_effective_tokens(y)
                    window_micro_steps += 1
                    is_accumulation_step = (step + 1) % self.args.training.accumulation_steps != 0
                    if device == torch.device("cuda") and self.args.training.amp:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                    else:
                        logits, loss = model(x, y)
                    loss = loss / self.args.training.accumulation_steps
                    loss.backward()
                    accumulated_loss += loss.detach()
                    
                    if is_accumulation_step:
                        self._synchronize_device(device)
                        window_compute_time += time.perf_counter() - compute_start_time
                        step += 1
                        continue
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.training.grad_clip
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    self._synchronize_device(device)
                    window_compute_time += time.perf_counter() - compute_start_time
                    
                    if global_step % self.args.training.log_steps == 0:
                        elapsed_ms = (time.perf_counter() - window_start_time) * 1000
                        self._log_swanlab(
                            {
                                "train/step": global_step,
                                "train/loss": accumulated_loss.item(),
                                "train/grad_norm": grad_norm.item(),
                                "train/lr": lr,
                                "train/throughput": int(window_total_tokens / (elapsed_ms / 1000)),
                                "train/effective_throughput": int(
                                    window_effective_tokens / (elapsed_ms / 1000)
                                ),
                                "train/data_time_ms": int(window_data_time * 1000),
                                "train/compute_time_ms": int(window_compute_time * 1000),
                                "train/step_time_ms": int(elapsed_ms),
                                "train/micro_steps": window_micro_steps,
                            }
                        )
                        window_data_time = 0.0
                        window_compute_time = 0.0
                        window_total_tokens = 0
                        window_effective_tokens = 0
                        window_micro_steps = 0
                        window_start_time = time.perf_counter()

                    accumulated_loss = torch.tensor(0.0, device=device)
                    step += 1
                    
        finally:
            self._finish_swanlab()
