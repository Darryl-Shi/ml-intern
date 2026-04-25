from pathlib import Path

from agent.core.local_checkpoint import LocalSnapshotManager


def test_local_snapshot_restore_restores_files_and_deletes_new_files(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "tracked.txt").write_text("before")
    nested = root / "nested"
    nested.mkdir()
    (nested / "keep.txt").write_text("nested before")

    manager = LocalSnapshotManager(
        root,
        "session-test",
        base_dir=tmp_path / "snapshots",
    )
    checkpoint = manager.create(label="initial", message_count=1, turn_count=0)

    (root / "tracked.txt").write_text("after")
    (root / "new.txt").write_text("new")
    (nested / "keep.txt").unlink()
    (nested / "created.txt").write_text("created")

    manager.restore(checkpoint)

    assert (root / "tracked.txt").read_text() == "before"
    assert (nested / "keep.txt").read_text() == "nested before"
    assert not (root / "new.txt").exists()
    assert not (nested / "created.txt").exists()


def test_local_snapshot_restore_leaves_excluded_directories_alone(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    session_logs = root / "session_logs"
    session_logs.mkdir()
    (session_logs / "live.json").write_text("do not touch")

    manager = LocalSnapshotManager(
        root,
        "session-test",
        base_dir=tmp_path / "snapshots",
    )
    checkpoint = manager.create(label="initial", message_count=1, turn_count=0)
    (session_logs / "live.json").write_text("still live")

    manager.restore(checkpoint)

    assert (session_logs / "live.json").read_text() == "still live"
