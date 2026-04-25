import json
import os
from pathlib import Path

from agent.main import _session_log_candidates


def _write_session(path: Path, *, session_id: str, preview: str, checkpoints: int):
    data = {
        "session_id": session_id,
        "model_name": "test-model",
        "last_save_time": f"2026-01-01T00:00:0{checkpoints}",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": preview},
        ],
        "local_state": {
            "checkpoints": [
                {"checkpoint_id": str(index)} for index in range(checkpoints)
            ]
        },
    }
    path.write_text(json.dumps(data))


def test_session_log_candidates_are_sorted_and_summarized(tmp_path: Path):
    older = tmp_path / "session_old.json"
    newer = tmp_path / "session_new.json"
    ignored = tmp_path / "notes.json"

    _write_session(older, session_id="old", preview="older message", checkpoints=1)
    _write_session(newer, session_id="new", preview="newer message", checkpoints=2)
    ignored.write_text("{}")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    candidates = _session_log_candidates(tmp_path)

    assert [item["session_id"] for item in candidates] == ["new", "old"]
    assert candidates[0]["preview"] == "newer message"
    assert candidates[0]["checkpoint_count"] == 2
