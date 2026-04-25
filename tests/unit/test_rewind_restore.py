import asyncio
from pathlib import Path

from agent.config import Config
from agent.core.message import Message
from agent.core.session import Session
from agent.main import _rewind_to_user_turn


def test_rewind_to_user_turn_restores_before_message_and_returns_prompt(
    tmp_path: Path,
):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    target = workdir / "notes.txt"
    target.write_text("initial")

    session = Session(
        asyncio.Queue(),
        config=Config(model_name="test-model", save_sessions=False),
        local_mode=True,
        stream=False,
    )
    session.attach_local_snapshots(workdir)
    session.checkpoint_local_state(label="initial")

    session.context_manager.add_message(Message(role="user", content="change notes"))
    session.context_manager.add_message(Message(role="assistant", content="done"))
    session.turn_count = 1
    target.write_text("changed")
    session.checkpoint_local_state(label="turn 1")

    restored_prompt = _rewind_to_user_turn(session, 1)

    assert restored_prompt == "change notes"
    assert target.read_text() == "initial"
    assert session.turn_count == 0
    assert [msg.role for msg in session.context_manager.items] == ["system"]
