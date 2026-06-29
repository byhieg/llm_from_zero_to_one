import torch

from checkpoint_manager import Checkpoint, CheckpointManager
from evaluator.eval_args import EvalCheckpointConfig


def test_get_checkpoint_returns_latest_numeric_checkpoint(tmp_path):
    manager = CheckpointManager(
        EvalCheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
        "demo-model",
    )

    manager.save_checkpoint(
        Checkpoint(
            model_state_dict={"step": 1},
            optimizer_state_dict={"optimizer": 1},
            metadata={"step": 1},
        ),
        1,
    )
    manager.save_checkpoint(
        Checkpoint(
            model_state_dict={"step": 10},
            optimizer_state_dict={"optimizer": 10},
            metadata={"step": 10},
        ),
        10,
    )
    (manager.checkpoint_dir / "latest").mkdir()
    (manager.checkpoint_dir / "000020").mkdir()

    checkpoint = manager.get_checkpoint()

    assert checkpoint.model_state_dict == {"step": 10}
    assert checkpoint.optimizer_state_dict == {"optimizer": 10}
    assert checkpoint.metadata == {"step": 10}


def test_get_checkpoint_falls_back_to_latest_when_step_missing(tmp_path):
    manager = CheckpointManager(
        EvalCheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
        "demo-model",
    )

    manager.save_checkpoint(
        Checkpoint(
            model_state_dict={"step": 2},
            optimizer_state_dict={"optimizer": 2},
            metadata={"step": 2},
        ),
        2,
    )
    manager.save_checkpoint(
        Checkpoint(
            model_state_dict={"step": 8},
            optimizer_state_dict={"optimizer": 8},
            metadata={"step": 8},
        ),
        8,
    )

    checkpoint = manager.get_checkpoint(step=5)

    assert checkpoint.model_state_dict == {"step": 8}
    assert checkpoint.optimizer_state_dict == {"optimizer": 8}
    assert checkpoint.metadata == {"step": 8}


def test_get_checkpoint_loads_to_cpu_by_default(tmp_path, monkeypatch):
    manager = CheckpointManager(
        EvalCheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
        "demo-model",
    )
    checkpoint_dir = manager.checkpoint_dir / "000001"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(b"model")
    (checkpoint_dir / "optimizer.pt").write_bytes(b"optimizer")

    calls = []

    def fake_torch_load(path, *, weights_only, map_location):
        calls.append((path.name, weights_only, map_location))
        if path.name == "model.pt":
            return {"step": 1}
        return {"optimizer": 1}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    checkpoint = manager.get_checkpoint()

    assert checkpoint.model_state_dict == {"step": 1}
    assert checkpoint.optimizer_state_dict == {"optimizer": 1}
    assert calls == [
        ("model.pt", False, "cpu"),
        ("optimizer.pt", False, "cpu"),
    ]


def test_get_checkpoint_strips_compiled_model_prefix(tmp_path, monkeypatch):
    manager = CheckpointManager(
        EvalCheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
        "demo-model",
    )
    checkpoint_dir = manager.checkpoint_dir / "000001"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(b"model")
    (checkpoint_dir / "optimizer.pt").write_bytes(b"optimizer")

    def fake_torch_load(path, *, weights_only, map_location):
        if path.name == "model.pt":
            return {
                "_orig_mod.linear.weight": 1,
                "_orig_mod.linear.bias": 2,
            }
        return {"optimizer": 1}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    checkpoint = manager.get_checkpoint()

    assert checkpoint.model_state_dict == {
        "linear.weight": 1,
        "linear.bias": 2,
    }
