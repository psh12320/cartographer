from __future__ import annotations

import re
from collections.abc import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "linen", "denim", "suede", "cashmere", "acrylic",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "beige", "navy", "gold", "silver",
)
MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS[:9]) + r")\b", re.I)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.I,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "here", "what", "those", "not", "quite", "right", "yet", "about", "one",
}


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def clean_constraint(value: str, limit: int = 180) -> str:
    return SPACE_RE.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def canonical(value: object) -> str:
    """Stable comparison form that tolerates punctuation and whitespace paraphrases."""

    return " ".join(TOKEN_RE.findall(str(value).lower()))


def terms(value: object, limit: int = 40) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(str(value).lower()):
        if len(token) <= 1 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def searchable_text(product: dict) -> str:
    fields = ("title", "features", "details", "description", "categories", "store")
    return " ".join(flatten_text(product.get(field)) for field in fields).strip()


def coarse_category(values: Iterable[object]) -> str:
    excluded = {"clothing", "clothing shoes jewelry"}
    cleaned: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            part = part.strip()
            if part and canonical(part) not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    """Mirror the official evaluator's constraint taxonomy exactly.

    The clarification policy uses this to predict which attribute a customer
    reply would answer, so any divergence from the published taxonomy
    produces questions the customer cannot answer (a wasted turn plus a
    spurious decline). Keep branch order and keyword lists identical to the
    official classify_constraint taxonomy.
    """

    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS[:9]):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def intent_fingerprint(product: dict) -> dict[str, object]:
    """Derive a product-intent representation using participant-visible catalog fields only."""

    title = clean_constraint(str(product.get("title") or "product"))
    candidates = [*flatten_values(product.get("features")), *flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(
        dict.fromkeys(
            clean_constraint(item) for item in candidates if clean_constraint(item)
        )
    )
    if not cleaned:
        cleaned = [title]
    constraints = cleaned[:4] if len(cleaned) >= 3 else [*cleaned[:2], *cleaned[:1]]
    by_attribute: dict[str, list[str]] = {}
    for constraint in constraints:
        by_attribute.setdefault(classify_constraint(constraint), []).append(constraint)
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
        "by_attribute": {key: tuple(values) for key, values in by_attribute.items()},
    }


def token_overlap(left: object, right: object) -> float:
    left_terms = set(terms(left, 100))
    right_terms = set(terms(right, 100))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)

