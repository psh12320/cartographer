from __future__ import annotations

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
        manifest_path = self.config.index_dir / "manifest.json"
        if not embeddings_path.exists() or not model_path.exists() or not manifest_path.exists():
            self.failure_reason = "Run `python -m cartographer.build_index --with-embeddings` to enable BGE."
            return
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("embedding_rows") != len(self.catalog.products):
                self.failure_reason = "Embedding artifact does not match the current catalog."
                return
            embeddings = np.load(embeddings_path, mmap_mode="r")
            if embeddings.shape[0] != len(self.catalog.products):
                self.failure_reason = "Embedding matrix row count does not match the catalog."
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
) -> Path:
    """Download BGE during setup, save it locally, and write catalog embeddings."""

    import numpy as np
    from sentence_transformers import SentenceTransformer

    index_dir.mkdir(parents=True, exist_ok=True)
    model_path = index_dir / "bge-small-en-v1.5"
    model_source = str(model_path) if (model_path / "config.json").exists() else model_name
    model = SentenceTransformer(model_source, device="cpu")
    documents = [product.search_text for product in catalog.products]
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    embeddings_path = index_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    model.save(str(model_path))
    catalog.write_manifest(
        {
            "embedding_model": model_name,
            "embedding_rows": int(embeddings.shape[0]),
            "embedding_dimensions": int(embeddings.shape[1]),
            "embedding_dtype": str(embeddings.dtype),
        }
    )
    return embeddings_path


def cache_cross_encoder(
    index_dir: Path,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
) -> Path:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device="cpu")
    model_path = index_dir / "cross-encoder-ms-marco-MiniLM-L6-v2"
    model.save(str(model_path))
    return model_path
