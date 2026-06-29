from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Protocol

import torch

logger = logging.getLogger(__name__)
COMPILED_MODEL_PREFIX = "_orig_mod."


@dataclass
class Checkpoint:
    model_state_dict: dict
    optimizer_state_dict: dict
    metadata: dict | None = None


class CheckpointConfigProtocol(Protocol):
    """Checkpoint 配置协议。"""

    checkpoint_dir: str


class CheckpointManager:
    METADATA_FILE = "metadata.json"

    def __init__(self, checkpoint_config: CheckpointConfigProtocol, model_name: str):
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

    def get_checkpoint(
        self,
        step: int = None,
        map_location: torch.device | str = "cpu",
    ) -> Checkpoint | None:
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
            checkpoint_step_path / "model.pt",
            weights_only=False,
            map_location=map_location,
        )
        model_state_dict = self._normalize_model_state_dict(model_state_dict)
        optimizer_state_dict = torch.load(
            checkpoint_step_path / "optimizer.pt",
            weights_only=False,
            map_location=map_location,
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

    def _normalize_model_state_dict(self, model_state_dict: dict) -> dict:
        if not model_state_dict:
            return model_state_dict
        if not all(
            isinstance(key, str) and key.startswith(COMPILED_MODEL_PREFIX)
            for key in model_state_dict
        ):
            return model_state_dict
        logger.info("detected compiled checkpoint, stripping _orig_mod prefix")
        return {
            key[len(COMPILED_MODEL_PREFIX) :]: value
            for key, value in model_state_dict.items()
        }
