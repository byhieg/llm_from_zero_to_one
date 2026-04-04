from ..train_args import CheckpointConfig
from pathlib import Path
import torch
import json
from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    model_state_dict: dict

    optimizer_state_dict: dict

    metadata: dict = None


class CheckpointManager:
    METADATA_FILE = "metadata.json"

    def __init__(self, checkpoint_config: CheckpointConfig, model_name: str):
        self.checkpoint_config = checkpoint_config
        self.checkpoint_dir = checkpoint_config.checkpoint_dir
        if not self.checkpoint_dir:
            raise ValueError("checkpoint_dir must be set")
        self.checkpoint_dir = Path(self.checkpoint_dir) / model_name
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

    def save_checkpoint(self, checkpoint: Checkpoint, step: int):
        checkpoint_step_path = self.checkpoint_dir / f"{step:06d}"
        checkpoint_step_path.mkdir(parents=True, exist_ok=True)
        if checkpoint.metadata:
            meta_path = checkpoint_step_path / type(self).METADATA_FILE
            with open(meta_path, "w") as f:
                json.dump(checkpoint.metadata, f, indent=4)
        torch.save(checkpoint.model_state_dict, checkpoint_step_path / "model.pt")
        torch.save(
            checkpoint.optimizer_state_dict, checkpoint_step_path / "optimizer.pt"
        )

    def get_checkpoint(self, step: int = None) -> Checkpoint:
        checkpoint_step_path = None
        if step is not None:
            checkpoint_step_path = self.checkpoint_dir / f"{step:06d}"
            if not checkpoint_step_path.exists():
                logger.error(
                    f"checkpoint step {step} not found,using latest checkpoint"
                )
                checkpoint_step_path = None
        if not checkpoint_step_path:
            checkpoint_dirs = []
            for path in self.checkpoint_dir.iterdir():
                if not path.is_dir():
                    continue
                if not path.name.isdigit():
                    continue
                model_path = path / "model.pt"
                if not model_path.exists():
                    continue
                checkpoint_dirs.append(path)
            if not checkpoint_dirs:
                return None
            checkpoint_step_path = max(checkpoint_dirs, key=lambda path: int(path.name))
        model_state_dict = torch.load(
            checkpoint_step_path / "model.pt", weights_only=False
        )
        optimizer_state_dict = torch.load(
            checkpoint_step_path / "optimizer.pt", weights_only=False
        )
        metadata = {}
        if (checkpoint_step_path / type(self).METADATA_FILE).exists():
            with open(checkpoint_step_path / type(self).METADATA_FILE, "r") as f:
                metadata = json.load(f)
        return Checkpoint(
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            metadata=metadata,
        )
