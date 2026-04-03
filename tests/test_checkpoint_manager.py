from trainer.checkpoint_manager import CheckpointManager
from trainer.train_args import CheckpointConfig


def test_get_checkpoint_returns_latest_numeric_checkpoint(tmp_path):
    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
        "demo-model",
    )

    manager.save_checkpoint({"step": 1}, 1, {"step": 1})
    manager.save_checkpoint({"step": 10}, 10, {"step": 10})
    (manager.checkpoint_dir / "latest").mkdir()
    (manager.checkpoint_dir / "000020").mkdir()

    checkpoint, metadata = manager.get_checkpoint()

    assert checkpoint == {"step": 10}
    assert metadata == {"step": 10}


def test_get_checkpoint_falls_back_to_latest_when_step_missing(tmp_path):
    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
        "demo-model",
    )

    manager.save_checkpoint({"step": 2}, 2, {"step": 2})
    manager.save_checkpoint({"step": 8}, 8, {"step": 8})

    checkpoint, metadata = manager.get_checkpoint(step=5)

    assert checkpoint == {"step": 8}
    assert metadata == {"step": 8}
