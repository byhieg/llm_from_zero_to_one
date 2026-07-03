import os
from pathlib import Path
import random
import time
from dataclasses import asdict
from functools import partial
from importlib import import_module

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler

from dataset import create_dataset
from logger import get_logger
from models import create_model

from . import deepspeed_train, naive_train
from .pretrain_args import PreTrainArgs
from itertools import islice
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
        return islice(super().__iter__(),self.sample_offset,None)


class PreTrainTrainer:
    """预训练 Trainer。"""

    def __init__(self, args: PreTrainArgs):
        self.args = args
        self.rank_info = {
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "local_world_size": 1,
            "node_rank": 0,
            "is_distributed": False,
            "is_multi_node": False,
        }
        self._swanlab = None
        self._swanlab_run_id: str | None = None

    def run(self) -> None:
        self._init_seed()
        self.rank_info = self._build_distributed()

        dataset = create_dataset(
            data_strategy=self.args.data.train.data_strategy,
            dataset_config=self._get_train_dataset_config(),
        )
        dataloader = self._build_dataloader(dataset)

        ################################### 获取全局信息，进行打印 ###############################################
        world_size = self.rank_info["world_size"]
        per_gpu_batch_size = self._get_batch_size_per_gpu()
        total_batch_size = per_gpu_batch_size * world_size

        self.device = self._get_device()
        model = create_model(self.args.model.name, self._get_model_config())
        trainable_params = (
            model.count_parameters()
            if hasattr(model, "count_parameters")
            else sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
        if self._is_main_process():
            logger.info(
                f"model: {model.__class__.__name__}, trainable_params={trainable_params:,}"
            )
            logger.info(f"dataset size: {len(dataset)} samples")
            logger.info(
                f"world_size={world_size}, "
                f"rank={self.rank_info['rank']}, "
                f"local_rank={self.rank_info['local_rank']}"
            )
            logger.info(f"train device: {self.device}")
            logger.info(
                f"batch size: {per_gpu_batch_size} per GPU × {world_size} GPU = {total_batch_size} total"
            )
            logger.info(
                f"dataloader batch num: {len(dataloader)} per GPU, "
                f"total batch num: {len(dataloader) * world_size}"
            )

        ################################### 初始化训练引擎，checkpoint 恢复 ###############################################

        backend_state = self._prepare_backend(model, self.device)

        if self.args.checkpoint.resume_checkpoint_dir:
            if self._is_deepspeed_backend():
                load_path, states = self._deepspeed_runtime.load_checkpoint(
                    self.args.checkpoint.resume_checkpoint_dir,
                    self.args.checkpoint.resume_tag,
                )
                global_step = states["global_step"]
                start_epoch = states["start_epoch"]
                start_micro_step_in_epoch = states["start_micro_step_in_epoch"]
                self._swanlab_run_id = states["swanlab_run_id"]
                logger.info(f"Checkpoint prepare to resume checkpoint path:{load_path},states:{states}")
            else:
                raise NotImplementedError
        else:
            logger.info("current experiment will train without checkpoint")
            global_step = 0
            start_epoch = 0
            start_micro_step_in_epoch = 0
        if self.args.checkpoint.save_checkpoint_dir:
            self.args.checkpoint.save_checkpoint_dir = str(
                Path(self.args.checkpoint.save_checkpoint_dir)
                / self.args.experiment.name
            )
            os.makedirs(self.args.checkpoint.save_checkpoint_dir, exist_ok=True)
        if self._is_main_process():
            self._init_swanlab(
                self.device, dataset, dataloader, run_id=self._swanlab_run_id
            )

        accumulation_steps = self._get_backend_accumulation_steps(backend_state)
        steps_per_epoch = len(dataloader) // accumulation_steps
        max_steps = self.args.train.epoch_num * steps_per_epoch
        if self._is_main_process():
            logger.info(
                f"total steps num: {max_steps} (epoch_num: {self.args.train.epoch_num}, "
                f"perepoch steps: {steps_per_epoch}, accumulation_steps: {accumulation_steps}, eval_steps: {self.args.eval.steps}, save checkpoint step:{self.args.checkpoint.save_step})"
            )
        tokens = (
            self.args.train.batch_size
            * self.args.train.seq_len
            * accumulation_steps
            * self.args.train.log_steps
            * self.rank_info["world_size"]
        )
        try:
            for epoch in range(start_epoch, self.args.train.epoch_num):
                self._set_dataloader_epoch(dataloader, epoch)
                self._set_backend_train_mode(backend_state)
                logger.info(f"🚀 Epoch {epoch} start to train")
                self._log_swanlab({"train/epoch": epoch})
                start_time = time.perf_counter()
                micro_step_offset = (
                    start_micro_step_in_epoch if epoch == start_epoch else 0
                )
                epoch_iterator = self._build_epoch_iterator(
                    dataloader, micro_step_offset
                )
                logger.info(f'dataloader len:{len(dataloader)},micro_step_offset:{micro_step_offset}')
                for step, (x, y) in enumerate(epoch_iterator, start=micro_step_offset):
                    x, y = (
                        x.to(self.device, non_blocking=True),
                        y.to(self.device, non_blocking=True),
                    )
                    step_result = self._train_backend_batch(x, y)
                    if not step_result:
                        continue
                    grad_norm = (
                        step_result["grad_norm"]
                        if step_result["grad_norm"] is not None
                        else torch.tensor(0.0, device=self.device)
                    )
                    
                    if (
                        self.args.checkpoint.save_step > 0
                        and global_step % self.args.checkpoint.save_step == 0
                    ):
                        self._save_checkpoint(
                            checkpoint_dir=self.args.checkpoint.save_checkpoint_dir,
                            global_step=global_step + 1,
                            epoch=epoch,
                            micro_step_in_epoch=step + 1,
                        )
                    if global_step % self.args.train.log_steps == 0:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        self._log_swanlab(
                            {
                                "train/step": global_step,
                                "train/loss": step_result["log_loss"].item(),
                                "train/grad_norm": grad_norm.item(),
                                "train/lr": step_result["lr"] or 0.0,
                                "train/throughput": int(tokens / (elapsed_ms / 1000)),
                            }
                        )
                        start_time = time.perf_counter()

                    global_step += 1
                    
            self._save_checkpoint(
                checkpoint_dir=self.args.checkpoint.save_checkpoint_dir,
                global_step=global_step,
                epoch=epoch,
                micro_step_in_epoch=0,
            )

        finally:
            self._finish_swanlab()

    def _build_distributed(self) -> None:
        if not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                if torch.cuda.is_available():
                    if self._is_deepspeed_backend():
                        self._init_deepspeed()
                    else:
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

    def _is_deepspeed_backend(self) -> bool:
        """判断当前训练后端是否为 DeepSpeed。"""

        return self.args.train.backend == "deepspeed"

    def _init_deepspeed(self) -> deepspeed_train.DeepSpeedPretrainRuntime:
        if getattr(self, "_deepspeed_runtime", None) is None:
            self._deepspeed_runtime: deepspeed_train.DeepSpeedPretrainRuntime = (
                deepspeed_train.DeepSpeedPretrainRuntime(self.args)
            )
            import logging
            logging.getLogger("deepspeed").setLevel(logging.INFO)
        return self._deepspeed_runtime

    def _prepare_backend(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ):
        if self._is_deepspeed_backend():
            return self._deepspeed_runtime.prepare(model)
        return naive_train.prepare_backend(self, model, device, None)

    def _get_backend_accumulation_steps(self, backend_state) -> int:
        if self._is_deepspeed_backend():
            return self._deepspeed_runtime.get_accumulation_steps()
        return naive_train.get_accumulation_steps(backend_state)

    def _get_batch_size_per_gpu(self) -> int:
        if self._is_deepspeed_backend():
            return self._deepspeed_runtime.__get_batch_size_per_gpu()
        return self.args.train.batch_size

    def _set_backend_train_mode(self, backend_state) -> None:
        if self._is_deepspeed_backend():
            self._deepspeed_runtime.set_train_mode()
            return
        naive_train.set_train_mode(backend_state)

    def _train_backend_batch(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> dict:
        if self._is_deepspeed_backend():
            return self._deepspeed_runtime.train(x=x, y=y, device=self.device)
        return naive_train.train_batch(self, x=x, y=y, device=self.device, **kwargs)

    def _get_model_config(self) -> dict:
        model_config = dict(self.args.model.config)
        model_config.setdefault("block_size", self.args.train.seq_len)
        return model_config

    def _get_train_dataset_config(self) -> dict:
        dataset_config = dict(self.args.data.train.dataset_config)
        dataset_config.setdefault("seq_len", self.args.train.seq_len)
        return dataset_config

    def _build_dataloader(self, dataset):
        dataloader_config = self.args.data.train.dataloader_config
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
            batch_size=self.args.train.batch_size,
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
        return torch.optim.AdamW(
            model.parameters(),
            lr=self.args.train.naive_config.get("learning_rate", 3e-4),
            weight_decay=0.0,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    def _get_grad_clip(self) -> float:
        return float(self.args.train.naive_config.get("grad_clip", 1.0))

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

    def _init_swanlab(
        self,
        device: torch.device,
        dataset,
        dataloader,
        run_id: str | None = None,
    ) -> None:
        if not self.args.experiment.swanlab.enabled:
            return
        try:
            self._swanlab = import_module("swanlab")
        except ImportError as exc:
            raise ImportError(
                "experiment.swanlab.enabled=true but swanlab is not installed, please install it first."
            ) from exc
        run = self._swanlab.init(
            project=self.args.experiment.swanlab.project,
            experiment_name=self.args.experiment.name,
            config=self._build_swanlab_config(device, dataset, dataloader),
            tags=self.args.experiment.swanlab.tags,
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
        config.pop("experiment", None)
        config["data"]["train"]["dataset_config"] = self._get_train_dataset_config()
        config["runtime"] = {
            "dataset_size": len(dataset),
            "dataloader_batches": len(dataloader),
            "device": str(device),
        }
        return config

    def _get_dataloader_seed(self) -> int:
        return self.args.data.train.dataloader_config.get(
            "seed", 42 if not self.args.train.seed else self.args.train.seed
        )

    def _save_checkpoint(
        self,
        checkpoint_dir,
        global_step: int,
        epoch: int,
        micro_step_in_epoch: int,
        tag: str | None = None,
    ) -> None:
        if self._is_deepspeed_backend():
            state = {
                "global_step": global_step,
                "start_epoch": epoch,
                "start_micro_step_in_epoch": micro_step_in_epoch,
                "swanlab_run_id": self._swanlab_run_id,
            }
            self._deepspeed_runtime.save_checkpoint(checkpoint_dir, state, tag)
            logger.info(f"save checkpoint global_step: {global_step} epoch: {epoch} micro_step_in_epoch:{micro_step_in_epoch} checkpoint_dir:{checkpoint_dir}")

        # naive_train.save_checkpoint_if_needed(
        #     self,
        #     backend_state=backend_state,
        #     global_step=global_step,
        #     epoch=epoch,
        #     micro_step_in_epoch=micro_step_in_epoch,
        #     dataloader_length=dataloader_length,
        # )

    def _set_seed(self, seed: int, init_cuda: bool = False) -> None:
        _set_process_seed(seed, init_cuda=init_cuda)

    def _init_seed(self):
        seed = self.args.train.seed
        self._set_seed(seed, init_cuda=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
