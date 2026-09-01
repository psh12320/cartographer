"""Long-term personalisation: distil session evidence into a durable profile.

Pillar III (Self-Evolution) asks for "Personalized Context Distillation,
continuously updating short-term session states and long-term user profiles".
The short-term half lives in `SessionState`. This module is the long-term half.

At the end of a session the agent distils what the shopper actually revealed --
the attributes they were willing to specify, and the categories they explored --
into a compact, human-readable record keyed by a stable user key. On the next
`reset` for that user the record is merged into the incoming profile, so a
returning shopper starts from what they previously cared about instead of from
nothing.

Two deliberate design constraints:

* It is opt-in (`enable_profile_memory`). The competition evaluator gives every
  session a fresh user and never repeats one, so this can only be exercised in
  a real deployment or the dashboard demo. Leaving it off by default keeps the
  benchmarked agent byte-for-byte identical.
* It stores only attribute names, coarse categories and counts -- never product
  identifiers, never evaluator labels, never raw customer text. That keeps the
  runtime free of anything resembling memorised ground truth.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .models import SessionState
from .text import canonical

FORMAT_VERSION = 1


def user_key(user_profile: dict) -> str:
    """Stable identity for a shopper, derived only from profile content.

    The competition profile carries no user id, so the aggregate description is
    the identity. Sessions describing the same shopper therefore share a record.
    """

    tags = sorted(str(tag).lower() for tag in (user_profile or {}).get("preference_tags") or [])
    frequency = str((user_profile or {}).get("purchase_frequency") or "")
    style = str((user_profile or {}).get("rating_style") or "")
    return canonical(" | ".join([*tags, frequency, style])) or "anonymous"


class ProfileMemory:
    """A small JSON-backed store of distilled, cross-session preferences."""

    def __init__(self, path: str | Path | None, max_tags: int = 8) -> None:
        self.path = Path(path) if path else None
        self.max_tags = max(1, int(max_tags))
        self.records: dict[str, dict[str, Any]] = {}
        self.failure_reason: str | None = None
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("format_version") != FORMAT_VERSION:
                raise ValueError("unsupported profile memory format")
            records = payload.get("records")
            if isinstance(records, dict):
                self.records = {str(k): dict(v) for k, v in records.items() if isinstance(v, dict)}
        except Exception as error:  # personalisation must never break inference
            self.failure_reason = f"Profile memory unavailable: {error}"
            self.records = {}

    def recall(self, user_profile: dict) -> dict:
        """Merge a remembered profile into the incoming one, without overwriting it."""

        record = self.records.get(user_key(user_profile))
        if not record:
            return dict(user_profile or {})
        merged = dict(user_profile or {})
        observed = [tag for tag, _ in sorted(
            dict(record.get("attribute_counts") or {}).items(),
            key=lambda item: (-item[1], item[0]),
        )][: self.max_tags]
        if observed:
            existing = [str(tag) for tag in merged.get("preference_tags") or []]
            merged["preference_tags"] = list(dict.fromkeys([*existing, *observed]))
        merged["remembered_sessions"] = int(record.get("sessions", 0))
        return merged

    def distil(self, state: SessionState) -> dict[str, Any]:
        """Fold one finished session's evidence into the long-term record."""

        key = state.profile_key or user_key(state.user_profile)
        record = self.records.setdefault(
            key, {"sessions": 0, "attribute_counts": {}, "categories": {}}
        )
        record["sessions"] = int(record.get("sessions", 0)) + 1
        counts = record.setdefault("attribute_counts", {})
        for constraint in state.constraints:
            # What the shopper was willing to specify is the durable signal;
            # the specific value belongs to this session only.
            counts[constraint.attribute] = int(counts.get(constraint.attribute, 0)) + 1
        if state.category:
            categories = record.setdefault("categories", {})
            key_category = canonical(state.category)
            categories[key_category] = int(categories.get(key_category, 0)) + 1
        return record

    def save(self) -> bool:
        """Persist atomically; a failure degrades personalisation, never inference."""

        if not self.path:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"format_version": FORMAT_VERSION, "records": self.records}
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                temporary = Path(handle.name)
            temporary.replace(self.path)
            return True
        except Exception as error:
            self.failure_reason = f"Profile memory not saved: {error}"
            return False
