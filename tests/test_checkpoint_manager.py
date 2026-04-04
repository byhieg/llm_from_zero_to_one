from trainer.checkpoint_manager import Checkpoint, CheckpointManager
from trainer.train_args import CheckpointConfig


def test_get_checkpoint_returns_latest_numeric_checkpoint(tmp_path):
    manager = CheckpointManager(
        CheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
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
        CheckpointConfig(checkpoint_dir=str(tmp_path / "checkpoints")),
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
