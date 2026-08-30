from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .catalog import CatalogIndex
    from .config import AgentConfig


class SemanticRetriever:
    """Optional offline BGE route backed by precomputed normalized embeddings."""

    def __init__(self, catalog: "CatalogIndex", config: "AgentConfig") -> None:
        self.catalog = catalog
        self.config = config
        self.enabled = False
        self.failure_reason: str | None = None
        self._numpy = None
        self._embeddings = None
        self._model = None
        if config.enable_dense:
            self._load()

    def _load(self) -> None:
        embeddings_path = self.config.index_dir / "embeddings.npy"
        model_path = self.config.index_dir / "bge-small-en-v1.5"
        manifest_path = self.config.index_dir / "embeddings_manifest.json"
        if not embeddings_path.exists() or not model_path.exists() or not manifest_path.exists():
            self.failure_reason = (
                "Run `python -m cartographer.build_embeddings --device cuda` during setup "
                "or import a verified embedding bundle."
            )
            return
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != 1:
                self.failure_reason = "Unsupported embedding artifact format."
                return
            if manifest.get("embedding_rows") != len(self.catalog.products):
                self.failure_reason = "Embedding artifact does not match the current catalog."
                return
            if manifest.get("catalog_sha256") != self.catalog.catalog_sha256():
                self.failure_reason = "Embedding artifact was built from a different catalog file."
                return
            if manifest.get("asin_order_sha256") != self.catalog.asin_order_sha256():
                self.failure_reason = "Embedding rows do not match the current ASIN ordering."
                return
            embeddings = np.load(embeddings_path, mmap_mode="r")
            if embeddings.shape[0] != len(self.catalog.products):
                self.failure_reason = "Embedding matrix row count does not match the catalog."
                return
            if embeddings.ndim != 2 or embeddings.shape[1] != manifest.get("embedding_dimensions"):
                self.failure_reason = "Embedding matrix dimensions do not match its manifest."
                return
            self._numpy = np
            self._embeddings = embeddings
            self._model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
            self.enabled = True
        except Exception as error:  # optional route must never invalidate the agent
            self.failure_reason = f"Dense route unavailable: {error}"

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        if not self.enabled or self._numpy is None or self._model is None or self._embeddings is None:
            return []
        vector = self._model.encode(
            [self.config.bge_query_instruction + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = self._embeddings @ vector
        limit = min(limit, len(scores))
        if limit <= 0:
            return []
        candidates = self._numpy.argpartition(scores, -limit)[-limit:]
        candidates = candidates[self._numpy.argsort(scores[candidates])[::-1]]
        return [(int(index), float(scores[index])) for index in candidates]


class CrossEncoderReranker:
    """Optional local MiniLM feature, gated by config and local model availability."""

    def __init__(self, index_dir: Path, enabled: bool) -> None:
        self.enabled = False
        self.failure_reason: str | None = None
        self._model = None
        model_path = index_dir / "cross-encoder-ms-marco-MiniLM-L6-v2"
        if not enabled:
            return
        if not model_path.exists():
            self.failure_reason = "Cross-encoder model is not cached."
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(str(model_path), device="cpu")
            self.enabled = True
        except Exception as error:
            self.failure_reason = f"Cross-encoder unavailable: {error}"

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not self.enabled or self._model is None or not documents:
            return [0.0] * len(documents)
        raw = self._model.predict([(query, document) for document in documents], show_progress_bar=False)
        return [1.0 / (1.0 + math.exp(-float(value))) for value in raw]


def build_bge_embeddings(
    catalog: "CatalogIndex",
    index_dir: Path,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    device: str = "cpu",
    storage_dtype: str = "float32",
    model_path: str | Path | None = None,
) -> Path:
    """Build a portable, checksummed BGE matrix aligned to catalog row order."""

    import numpy as np
    from sentence_transformers import SentenceTransformer

    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    index_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = Path(model_path) if model_path else index_dir / "bge-small-en-v1.5"
    model_source = str(local_model_path) if (local_model_path / "config.json").exists() else model_name
    model = SentenceTransformer(
        model_source,
        device=device,
        local_files_only=Path(model_source).exists(),
    )
    documents = [product.search_text for product in catalog.products]
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float16 if storage_dtype == "float16" else np.float32)
    embeddings_path = index_dir / "embeddings.npy"
    temporary_path = index_dir / "embeddings.tmp.npy"
    np.save(temporary_path, embeddings)
    temporary_path.replace(embeddings_path)
    model.save(str(index_dir / "bge-small-en-v1.5"))
    manifest = {
        "format_version": 1,
        "embedding_model": model_name,
        "embedding_rows": int(embeddings.shape[0]),
        "embedding_dimensions": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "normalized": True,
        "catalog_sha256": catalog.catalog_sha256(),
        "asin_order_sha256": catalog.asin_order_sha256(),
        "matrix_sha256": file_sha256(embeddings_path),
    }
    (index_dir / "embeddings_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return embeddings_path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bge_embeddings(catalog: "CatalogIndex", index_dir: str | Path) -> dict[str, object]:
    """Fail loudly when a transferred semantic artifact is incomplete or misaligned."""

    import numpy as np

    root = Path(index_dir)
    embeddings_path = root / "embeddings.npy"
    manifest_path = root / "embeddings_manifest.json"
    model_path = root / "bge-small-en-v1.5"
    missing = [str(path) for path in (embeddings_path, manifest_path, model_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing semantic artifacts: {', '.join(missing)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, mmap_mode="r")
    checks = {
        "format_version": manifest.get("format_version") == 1,
        "catalog_sha256": manifest.get("catalog_sha256") == catalog.catalog_sha256(),
        "asin_order_sha256": manifest.get("asin_order_sha256") == catalog.asin_order_sha256(),
        "embedding_rows": manifest.get("embedding_rows") == len(catalog.products) == embeddings.shape[0],
        "embedding_dimensions": embeddings.ndim == 2
        and manifest.get("embedding_dimensions") == embeddings.shape[1],
        "embedding_dtype": manifest.get("embedding_dtype") == str(embeddings.dtype),
        "matrix_sha256": manifest.get("matrix_sha256") == file_sha256(embeddings_path),
        "model_config": (model_path / "config.json").exists(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Embedding verification failed: {', '.join(failed)}")
    return {
        "verified": True,
        "rows": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "matrix_mib": round(embeddings_path.stat().st_size / (1024 * 1024), 3),
        "matrix_sha256": manifest["matrix_sha256"],
        "catalog_sha256": manifest["catalog_sha256"],
        "asin_order_sha256": manifest["asin_order_sha256"],
    }


def cache_cross_encoder(
    index_dir: Path,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
) -> Path:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cpu")
    model_path = index_dir / "cross-encoder-ms-marco-MiniLM-L6-v2"
    model.save(str(model_path))
    return model_path
