from __future__ import annotations

import re

from .models import Constraint, IntentMessage, SessionState
from .text import COLORS, MATERIALS, canonical, classify_constraint


CATEGORY_PATTERNS = (
    re.compile(r"\blooking for\s+(.+?)(?:\.\s|,\s|$)", re.I),
    re.compile(r"\b(?:searching for|shopping for|interested in)\s+(.+?)(?:\.\s|,\s|$)", re.I),
    re.compile(
        r"^\s*(?:please\s+)?(?:i need|i want|show me)\s+(?:an?\s+|some\s+)?(.+?)(?:\.\s|,\s|$)",
        re.I,
    ),
)
HARD_RE = re.compile(
    r"(?:key requirement is|what i need(?:\s+instead)? is)\s*:\s*(.+?)(?:\.$|$)",
    re.I,
)
MATTERS_RE = re.compile(r"what matters is\s*:\s*(.+?)(?:\.$|$)", re.I)
OVERRIDE_RE = re.compile(
    r"\b(?:actually|ignore|instead|changed my mind|rather than|forget that|switch(?:ing)? to|not anymore)\b",
    re.I,
)
NO_PREFERENCE_RE = re.compile(
    r"\b(?:(?:do not|don't|dont|no)\s+(?:have\s+)?(?:an?\s+)?(?:additional\s+)?preference|"
    r"no preference|(?:does not|doesn't|doesnt|do not|don't|dont) matter|any\s+\w+\s+is fine)\b",
    re.I,
)


def profile_attributes(profile: dict) -> set[str]:
    attributes: set[str] = set()
    for value in profile.get("preference_tags") or []:
        key = canonical(value)
        if key in {"material", "color", "size", "style", "brand", "budget", "feature", "use case"}:
            attributes.add(key.replace(" ", "_"))
        elif key in {"weather", "warmth", "performance", "durability", "comfort"}:
            attributes.add("feature")
        elif key == "fit":
            attributes.add("style")
    return attributes


class DialogManager:
    """Compile customer messages into replaceable session state."""

    def __init__(self, config=None) -> None:
        from .config import AgentConfig

        self.config = config or AgentConfig()

    """Compile free text and history into a compact, replaceable intent frame."""

    def update(self, state: SessionState, user_message: str, turn: int, preserve_state: bool = True) -> None:
        message = user_message.strip()
        if not preserve_state:
            state.constraints.clear()
            state.replaceable_constraint = None
            state.intent_messages.clear()
            state.asked_attributes.clear()
            state.declined_attributes.clear()
            state.last_asked = None
            state.category = ""

        if NO_PREFERENCE_RE.search(message) or "no additional preference" in message.lower():
            if state.last_asked:
                state.declined_attributes.add(state.last_asked)
                # "please use your judgment" is a deflection, not an exhausted
                # attribute: the requirement is still held and can still be
                # disclosed if the question is asked again.
                deflection = "use your judgment" in message.lower()
                if deflection and not self.config.boundary_deflection_retires_attribute:
                    state.deflected_attributes.add(state.last_asked)

        override = bool(OVERRIDE_RE.search(message))
        if override:
            state.intent_epoch += 1
            superseded = state.replaceable_constraint
            if superseded is None or not superseded.active:
                # Compatibility fallback for states created before explicit
                # replaceable-preference tracking was introduced.
                active = [constraint for constraint in state.constraints if constraint.active]
                if active:
                    earliest_turn = min(constraint.source_turn for constraint in active)
                    superseded = next(
                        (
                            constraint
                            for constraint in active
                            if constraint.source_turn == earliest_turn
                            and constraint.strength == "soft"
                        ),
                        next(
                            constraint
                            for constraint in active
                            if constraint.source_turn == earliest_turn
                        ),
                    )
            if superseded is not None:
                superseded.active = False
            state.replaceable_constraint = None
            state.route = "buying"
            state.override_shortlist = set(state.seen_products)
            state.seen_products.clear()
            state.last_asked = None
            state.cached_hits.clear()
            state.last_query_signature = ()

        category_match = next(
            (match for pattern in CATEGORY_PATTERNS if (match := pattern.search(message))),
            None,
        )
        if category_match:
            state.category = category_match.group(1).strip(" ,.;")

        extracted: list[tuple[str, str]] = []
        hard_match = HARD_RE.search(message)
        matters_match = MATTERS_RE.search(message)
        if hard_match:
            payload = hard_match.group(1)
            values = (
                self._split_values(payload)
                if self.config.split_hard_requirement_values
                else [payload.strip(" ,.;")]
            )
            extracted.extend((value, "hard") for value in values if value)
        elif matters_match:
            extracted.extend((value, "hard") for value in self._split_values(matters_match.group(1)))
        elif category_match:
            suffix = message[category_match.end():].strip(" ,.;")
            if suffix and "still exploring" not in suffix.lower() and "key requirement" not in suffix.lower():
                extracted.append((suffix, "soft"))
        elif message and not NO_PREFERENCE_RE.search(message) and "no additional preference" not in message.lower():
            extracted.extend((value, "soft") for value in self._extract_explicit_values(message))

        if override:
            extracted = [(value, "hard") for value, _strength in extracted]

        for value, strength in extracted:
            constraint = self._add_constraint(state, value, strength, turn)
            if constraint is None:
                continue
            if state.replaceable_constraint is None and (
                override or (turn == 1 and state.intent_epoch == 0 and strength == "soft")
            ):
                state.replaceable_constraint = constraint

        if message and not NO_PREFERENCE_RE.search(message) and "no additional preference" not in message.lower():
            # Keep the user's full phrasing for lexical/semantic ranking. Epoch filtering
            # guarantees that superseded language disappears after an intent override.
            state.intent_messages.append(
                IntentMessage(text=message, source_turn=turn, epoch=state.intent_epoch)
            )

        lowered = message.lower()
        buying_cue = bool(
            re.search(r"\b(?:must|need|require|only|under|below|exactly|specific)\b", lowered)
        )
        if override or hard_match or buying_cue or (extracted and "still exploring" not in lowered):
            state.route = "buying"
        elif "explor" in lowered or not state.active_constraints:
            state.route = "browsing"

    @staticmethod
    def _split_values(value: str) -> list[str]:
        return [item.strip(" ,.;") for item in value.split(";") if item.strip(" ,.;")]

    @staticmethod
    def _extract_explicit_values(message: str) -> list[str]:
        values: list[str] = []
        lowered = message.lower()
        for material in MATERIALS:
            if re.search(rf"\b{re.escape(material)}\b", lowered):
                values.append(material)
                break
        for color in COLORS:
            if re.search(rf"\b{re.escape(color)}\b", lowered):
                values.append(f"color: {color}")
                break
        budget = re.search(r"(?:budget|under|below|around)\s*(?:is|of|:)?\s*\$?\s*(\d+(?:\.\d+)?)", lowered)
        if budget:
            values.append(f"budget around ${budget.group(1)}")
        return values

    @staticmethod
    def _add_constraint(
        state: SessionState,
        value: str,
        strength: str,
        turn: int,
    ) -> Constraint | None:
        value = value.strip(" ,.;")
        key = canonical(value)
        if not key:
            return None
        for existing in state.constraints:
            if existing.active and canonical(existing.value) == key:
                if strength == "hard":
                    existing.strength = "hard"
                return existing
        constraint = Constraint(
            attribute=classify_constraint(value),
            value=value,
            strength=strength,
            source_turn=turn,
            epoch=state.intent_epoch,
        )
        state.constraints.append(constraint)
        return constraint
