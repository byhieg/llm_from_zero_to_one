import os
import random
import time
from dataclasses import asdict
from functools import partial
from importlib import import_module

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler

from checkpoint_manager import Checkpoint
from dataset import create_dataset
from evaluator.checkpoint_evaluator import PretrainEvaluator
from logger import get_logger
from models import create_model

from . import deepspeed_train, naive_train
from .pretrain_args import PreTrainArgs

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
        self._evaluator = None
        self._deepspeed_runtime: deepspeed_train.DeepSpeedPretrainRuntime | None = None
        self.checkpoint_manager = None

    def run(self) -> None:
        self._init_seed()
        self.rank_info = self._build_distributed()
        device = self._get_device()
        model = create_model(self.args.model.name, self._get_model_config())
        dataset = create_dataset(
            data_strategy=self.args.data.train.data_strategy,
            dataset_config=self._get_train_dataset_config(),
        )
        dataloader = self._build_dataloader(dataset)

        world_size = self.rank_info["world_size"]
        per_gpu_batch_size = self.args.train.batch_size
        total_batch_size = per_gpu_batch_size * world_size
        trainable_params = (
            model.count_parameters()
            if hasattr(model, "count_parameters")
            else sum(p.numel() for p in model.parameters() if p.requires_grad)
        )

        logger.info(
            f"model: {model.__class__.__name__}, trainable_params={trainable_params:,}"
        )
        logger.info(f"dataset size: {len(dataset)} samples")
        logger.info(
            f"world_size={world_size}, "
            f"rank={self.rank_info['rank']}, "
            f"local_rank={self.rank_info['local_rank']}"
        )
        logger.info(f"train device: {device}")
        logger.info(
            f"batch size: {per_gpu_batch_size} per GPU × {world_size} GPU = {total_batch_size} total"
        )
        logger.info(
            f"dataloader batch num: {len(dataloader)} per GPU, "
            f"total batch num: {len(dataloader) * world_size}"
        )

        checkpoint: Checkpoint | None = None
        global_step = 0
        start_epoch = 0
        start_micro_step_in_epoch = 0
        logger.info("Checkpoint loading is disabled for pretrain, starting from scratch")

        if self._is_main_process():
            self._init_swanlab(device, dataset, dataloader, run_id=self._swanlab_run_id)

        backend_state = self._prepare_backend(model, device, checkpoint)
        accumulation_steps = self._get_backend_accumulation_steps(backend_state)
        steps_per_epoch = len(dataloader) // accumulation_steps
        max_steps = self.args.train.epoch_num * steps_per_epoch
        logger.info(
            f"total steps num: {max_steps} (epoch_num: {self.args.train.epoch_num}, "
            f"perepoch steps: {steps_per_epoch}, accumulation_steps: {accumulation_steps}, eval_steps: {self.args.eval.steps})"
        )
        tokens = (
            self.args.train.batch_size
            * self.args.train.seq_len
            * accumulation_steps
            * self.args.train.log_steps
            * self.rank_info["world_size"]
        )
        try:
            eval_elapsed_since_log = 0.0
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
                for step, (x, y) in enumerate(epoch_iterator, start=micro_step_offset):
                    x, y = (
                        x.to(device, non_blocking=True),
                        y.to(device, non_blocking=True),
                    )
                    step_result = self._train_backend_batch(
                        backend_state=backend_state,
                        x=x,
                        y=y,
                        global_step=global_step,
                        max_steps=max_steps,
                        micro_step=step + 1,
                        device=device,
                    )
                    if not step_result:
                        continue
                    grad_norm = (
                        step_result["grad_norm"]
                        if step_result["grad_norm"] is not None
                        else torch.tensor(0.0, device=device)
                    )
                    global_step += 1
                    self._save_checkpoint_if_needed(
                        backend_state=backend_state,
                        global_step=global_step,
                        epoch=epoch,
                        micro_step_in_epoch=step + 1,
                        dataloader_length=len(dataloader),
                    )
                    if global_step % self.args.train.log_steps == 0:
                        elapsed_ms = (
                            time.perf_counter() - start_time - eval_elapsed_since_log
                        ) * 1000
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
                        eval_elapsed_since_log = 0.0
                    # eval_elapsed_since_log += self._run_eval_if_needed(
                    #     model=model,
                    #     device=device,
                    #     global_step=global_step,
                    # )

            # self._save_training_checkpoint(
            #     backend_state=backend_state,
            #     global_step=global_step,
            #     epoch=self.args.train.epoch_num,
            #     micro_step_in_epoch=0,
            #     dataloader_length=len(dataloader),
            # )

        finally:
            self._finish_swanlab()

    def _build_distributed(self):
        if not dist.is_initialized():
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                if torch.cuda.is_available():
                    if self.args.train.backend == "deepspeed":
                        self._get_deepspeed_runtime()
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

    def _get_deepspeed_runtime(self) -> deepspeed_train.DeepSpeedPretrainRuntime:
        if self.args.train.backend != "deepspeed":
            raise ValueError(
                "DeepSpeed runtime is only available when train.backend=deepspeed"
            )
        if self._deepspeed_runtime is None:
            self._deepspeed_runtime = deepspeed_train.DeepSpeedPretrainRuntime(self.args)
        return self._deepspeed_runtime

    def _prepare_backend(
        self,
        model: torch.nn.Module,
        device: torch.device,
        checkpoint: Checkpoint | None,
    ):
        if self.args.train.backend == "deepspeed":
            return self._get_deepspeed_runtime().prepare(model)
        return naive_train.prepare_backend(self, model, device, checkpoint)

    def _get_backend_accumulation_steps(self, backend_state) -> int:
        if self.args.train.backend == "deepspeed":
            return self._get_deepspeed_runtime().get_accumulation_steps()
        return naive_train.get_accumulation_steps(backend_state)

    def _set_backend_train_mode(self, backend_state) -> None:
        if self.args.train.backend == "deepspeed":
            self._get_deepspeed_runtime().set_train_mode()
            return
        naive_train.set_train_mode(backend_state)

    def _train_backend_batch(
        self,
        backend_state,
        x: torch.Tensor,
        y: torch.Tensor,
        global_step: int,
        max_steps: int,
        micro_step: int,
        device: torch.device,
    ) -> dict:
        if self.args.train.backend == "deepspeed":
            return self._get_deepspeed_runtime().train(
                x=x,
                y=y,
                device=device,
            )
        return naive_train.train_batch(
            self,
            backend_state=backend_state,
            x=x,
            y=y,
            global_step=global_step,
            max_steps=max_steps,
            micro_step=micro_step,
            device=device,
        )

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

    def _save_checkpoint_if_needed(
        self,
        backend_state,
        global_step: int,
        epoch: int,
        micro_step_in_epoch: int,
        dataloader_length: int,
    ) -> None:
        if self.args.train.backend == "deepspeed":
            return
        naive_train.save_checkpoint_if_needed(
            self,
            backend_state=backend_state,
            global_step=global_step,
            epoch=epoch,
            micro_step_in_epoch=micro_step_in_epoch,
            dataloader_length=dataloader_length,
        )

    def _save_training_checkpoint(
        self,
        backend_state,
        global_step: int,
        epoch: int,
        micro_step_in_epoch: int,
        dataloader_length: int,
    ) -> None:
        if self.args.train.backend == "deepspeed":
            return
        naive_train.save_training_checkpoint(
            self,
            backend_state=backend_state,
            global_step=global_step,
            epoch=epoch,
            micro_step_in_epoch=micro_step_in_epoch,
            dataloader_length=dataloader_length,
        )

    def _set_seed(self, seed: int, init_cuda: bool = False) -> None:
        _set_process_seed(seed, init_cuda=init_cuda)

    def _init_seed(self):
        seed = self.args.train.seed
        self._set_seed(seed, init_cuda=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
