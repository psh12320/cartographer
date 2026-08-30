from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
DEFAULT_SEED = "cartographer-public-100-100-v1"
DEFAULT_MANIFEST_PATH = Path("docs/public_split_v1.json")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scenario_counts(samples: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(sample["scenario_type"]) for sample in samples).items()))


def build_manifest(
    samples: list[dict[str, Any]],
    dataset_path: str | Path,
    seed: str = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create a stable 50/50 split within every scenario without using labels."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["scenario_type"])].append(sample)
    development: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for scenario in sorted(groups):
        group = groups[scenario]
        if len(group) % 2:
            raise ValueError(f"scenario {scenario!r} has an odd sample count")
        ranked = sorted(
            group,
            key=lambda sample: (
                hashlib.sha256(
                    f"{seed}\0{scenario}\0{sample['sample_id']}".encode("utf-8")
                ).hexdigest(),
                str(sample["sample_id"]),
            ),
        )
        midpoint = len(ranked) // 2
        development.extend(ranked[:midpoint])
        holdout.extend(ranked[midpoint:])

    development.sort(key=lambda sample: str(sample["sample_id"]))
    holdout.sort(key=lambda sample: str(sample["sample_id"]))
    source_path = Path(dataset_path)
    manifest = {
        "format_version": FORMAT_VERSION,
        "name": "public-stratified-100-100-v1",
        "source": {
            "path": source_path.as_posix(),
            "sha256": file_sha256(source_path),
            "sample_count": len(samples),
        },
        "method": {
            "type": "scenario-stratified deterministic SHA-256 ordering",
            "seed": seed,
            "development_fraction": 0.5,
            "uses_ground_truth": False,
        },
        "development": {
            "sample_count": len(development),
            "scenario_counts": _scenario_counts(development),
            "sample_ids": [str(sample["sample_id"]) for sample in development],
        },
        "holdout": {
            "sample_count": len(holdout),
            "scenario_counts": _scenario_counts(holdout),
            "sample_ids": [str(sample["sample_id"]) for sample in holdout],
        },
    }
    validate_manifest(samples, manifest, dataset_path)
    return manifest


def validate_manifest(
    samples: list[dict[str, Any]],
    manifest: dict[str, Any],
    dataset_path: str | Path,
) -> None:
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported split manifest format")
    source = manifest.get("source") or {}
    if source.get("sha256") != file_sha256(dataset_path):
        raise ValueError("split manifest dataset checksum does not match")
    all_ids = [str(sample["sample_id"]) for sample in samples]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("dataset contains duplicate sample IDs")
    sample_by_id = {str(sample["sample_id"]): sample for sample in samples}
    partitions: dict[str, set[str]] = {}
    for name in ("development", "holdout"):
        payload = manifest.get(name) or {}
        identifiers = [str(value) for value in payload.get("sample_ids") or []]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{name} split contains duplicate sample IDs")
        unknown = set(identifiers) - set(sample_by_id)
        if unknown:
            raise ValueError(f"{name} split contains unknown sample IDs")
        selected = [sample_by_id[identifier] for identifier in identifiers]
        if int(payload.get("sample_count", -1)) != len(selected):
            raise ValueError(f"{name} split sample count is inconsistent")
        if dict(payload.get("scenario_counts") or {}) != _scenario_counts(selected):
            raise ValueError(f"{name} split scenario counts are inconsistent")
        partitions[name] = set(identifiers)
    if partitions["development"] & partitions["holdout"]:
        raise ValueError("development and holdout splits overlap")
    if partitions["development"] | partitions["holdout"] != set(all_ids):
        raise ValueError("development and holdout splits are not exhaustive")


def load_manifest(
    path: str | Path,
    samples: list[dict[str, Any]],
    dataset_path: str | Path,
) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(samples, manifest, dataset_path)
    return manifest


def select_split(
    samples: list[dict[str, Any]],
    manifest: dict[str, Any],
    partition: str,
) -> list[dict[str, Any]]:
    if partition == "all":
        return list(samples)
    if partition not in {"development", "holdout"}:
        raise ValueError(f"unknown split partition: {partition}")
    identifiers = set(str(value) for value in manifest[partition]["sample_ids"])
    return [sample for sample in samples if str(sample["sample_id"]) in identifiers]


def main() -> None:
    from evaluator.local_evaluator import load_jsonl

    parser = argparse.ArgumentParser(description="Build the locked public 100/100 split manifest")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    samples = load_jsonl(args.dataset)
    manifest = build_manifest(samples, args.dataset, args.seed)
    output = Path(args.output)
    serialized = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"refusing to replace a different locked split manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
