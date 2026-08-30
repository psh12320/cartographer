from __future__ import annotations

from pathlib import Path

from cartographer.config import AgentConfig
from cartographer.engine import CartographerEngine


class Agent:
    """Official TechJam entry point for the Cartographer shopping copilot."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
    ) -> None:
        self.engine = CartographerEngine(catalog_path, config=config)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.engine.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self.engine.respond(session_id, user_message, turn, top_k)

    def get_trace(self, session_id: str) -> list[dict]:
        """Diagnostic helper; never included in the official response payload."""

        return self.engine.get_trace(session_id)
