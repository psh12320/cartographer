from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .catalog import CatalogIndex
from .semantic import build_bge_embeddings, cache_cross_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Cartographer's offline catalog indexes")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--index-dir", default="data/cartographer_index")
    parser.add_argument("--with-embeddings", action="store_true")
    parser.add_argument("--with-cross-encoder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float32")
    args = parser.parse_args()

    started = time.perf_counter()
    index_dir = Path(args.index_dir)
    catalog = CatalogIndex(args.catalog, index_dir)
    database_path = catalog.build_persistent_fts()
    result: dict[str, object] = {
        "products": len(catalog.products),
        "fts_database": str(database_path),
    }
    if args.with_embeddings:
        result["embeddings"] = str(
            build_bge_embeddings(
                catalog,
                index_dir,
                batch_size=args.batch_size,
                device=args.device,
                storage_dtype=args.embedding_dtype,
            )
        )
    if args.with_cross_encoder:
        result["cross_encoder"] = str(cache_cross_encoder(index_dir))
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
