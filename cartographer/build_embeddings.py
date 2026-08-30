from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .catalog import CatalogIndex
from .semantic import build_bge_embeddings, verify_bge_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify Cartographer's portable BGE embedding artifacts"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--index-dir", default="data/cartographer_index")
    parser.add_argument("--model-path", default="data/cartographer_index/bge-small-en-v1.5")
    parser.add_argument("--model-name", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    index_dir = Path(args.index_dir)
    catalog = CatalogIndex(args.catalog, index_dir)
    if not args.verify_only:
        build_bge_embeddings(
            catalog,
            index_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
            storage_dtype=args.dtype,
            model_path=args.model_path,
        )
    result = verify_bge_embeddings(catalog, index_dir)
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
