from __future__ import annotations

import json
import hashlib
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from .models import ProductRecord
from .text import canonical, coarse_category, flatten_text, intent_fingerprint, searchable_text, terms


class CatalogIndex:
    """Participant-visible catalog metadata plus a SQLite FTS5 retrieval route."""

    def __init__(self, catalog_path: str | Path, index_dir: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.index_dir = Path(index_dir)
        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found at {self.catalog_path}. See data/README.md for download instructions."
            )
        self.products: list[ProductRecord] = []
        self.asin_to_index: dict[str, int] = {}
        self.constraint_lookup: dict[str, list[int]] = defaultdict(list)
        self.category_lookup: dict[str, list[int]] = defaultdict(list)
        self._catalog_sha256_cache: str | None = None
        self._asin_order_sha256_cache: str | None = None
        self._load_metadata()
        self.connection = self._open_or_build_fts()

    def catalog_sha256(self) -> str:
        """Return a stable content digest for cross-machine artifact validation."""

        if self._catalog_sha256_cache is None:
            digest = hashlib.sha256()
            with self.catalog_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._catalog_sha256_cache = digest.hexdigest()
        return self._catalog_sha256_cache

    def asin_order_sha256(self) -> str:
        """Identify the exact product-to-embedding row alignment."""

        if self._asin_order_sha256_cache is None:
            digest = hashlib.sha256()
            for product in self.products:
                digest.update(product.parent_asin.encode("utf-8"))
                digest.update(b"\n")
            self._asin_order_sha256_cache = digest.hexdigest()
        return self._asin_order_sha256_cache

    def _catalog_signature(self) -> dict[str, int | str]:
        stat = self.catalog_path.stat()
        return {
            "catalog": self.catalog_path.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "products": len(self.products),
        }

    def _load_metadata(self) -> None:
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                index = len(self.products)
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                title = str(product.get("title") or "")
                category = coarse_category(product.get("categories") or [])
                corpus = searchable_text(product)
                fingerprint = intent_fingerprint(product)
                raw_constraints = [
                    *fingerprint["hard_constraints"],
                    *fingerprint["soft_preferences"],
                ]
                constraints = tuple(dict.fromkeys(str(value) for value in raw_constraints))
                by_attribute = {
                    str(key): tuple(dict.fromkeys(str(item) for item in value))
                    for key, value in dict(fingerprint["by_attribute"]).items()
                }
                try:
                    price = None if product.get("price") in (None, "") else float(product["price"])
                except (TypeError, ValueError):
                    price = None
                try:
                    average_rating = float(product.get("average_rating") or 0.0)
                except (TypeError, ValueError):
                    average_rating = 0.0
                try:
                    rating_number = int(product.get("rating_number") or 0)
                except (TypeError, ValueError):
                    rating_number = 0
                record = ProductRecord(
                    index=index,
                    parent_asin=parent_asin,
                    title=title,
                    category=category,
                    category_key=canonical(category),
                    search_text=corpus,
                    search_key=canonical(corpus),
                    price=price,
                    average_rating=average_rating,
                    rating_number=rating_number,
                    constraints=constraints,
                    by_attribute=by_attribute,
                )
                self.products.append(record)
                self.asin_to_index[parent_asin] = index
                self.category_lookup[record.category_key].append(index)
                for constraint in constraints:
                    key = canonical(constraint)
                    if key:
                        self.constraint_lookup[key].append(index)
        for indices in self.category_lookup.values():
            indices.sort(key=self._popularity_key, reverse=True)

    def _popularity_key(self, product_index: int) -> tuple[float, float, str]:
        product = self.products[product_index]
        return (
            product.average_rating * math.log1p(product.rating_number),
            product.average_rating,
            product.parent_asin,
        )

    def _manifest_is_current(self, manifest_path: Path) -> bool:
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return manifest.get("catalog_signature") == self._catalog_signature()

    def _open_or_build_fts(self) -> sqlite3.Connection:
        database_path = self.index_dir / "catalog.sqlite3"
        manifest_path = self.index_dir / "manifest.json"
        if database_path.exists() and self._manifest_is_current(manifest_path):
            return sqlite3.connect(
                f"file:{database_path.resolve().as_posix()}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._populate_fts(connection)
        return connection

    def _populate_fts(self, connection: sqlite3.Connection) -> None:
        cursor = connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "product_index UNINDEXED, parent_asin UNINDEXED, title, categories, features, "
            "details, store, description, tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[int, str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            product_index = 0
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                batch.append(
                    (
                        product_index,
                        str(product["parent_asin"]),
                        flatten_text(product.get("title")),
                        flatten_text(product.get("categories")),
                        flatten_text(product.get("features")),
                        flatten_text(product.get("details")),
                        flatten_text(product.get("store")),
                        flatten_text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
                product_index += 1
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        connection.commit()

    def build_persistent_fts(self) -> Path:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        database_path = self.index_dir / "catalog.sqlite3"
        temporary_path = self.index_dir / "catalog.sqlite3.tmp"
        if temporary_path.exists():
            temporary_path.unlink()
        connection = sqlite3.connect(temporary_path)
        try:
            self._populate_fts(connection)
        finally:
            connection.close()
        self.connection.close()
        if database_path.exists():
            database_path.unlink()
        temporary_path.replace(database_path)
        self.write_manifest({})
        self.connection = sqlite3.connect(
            f"file:{database_path.resolve().as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        return database_path

    def write_manifest(self, updates: dict) -> Path:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.index_dir / "manifest.json"
        manifest: dict = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {}
        manifest.update(updates)
        manifest["catalog_signature"] = self._catalog_signature()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest_path

    def lexical_search(self, query: str, limit: int) -> list[int]:
        # Long OR expressions dominate tail latency and add little once a full fingerprint matches.
        query_terms = terms(query, 8)
        if not query_terms:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in query_terms)
        try:
            rows = self.connection.execute(
                "SELECT product_index FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [int(row[0]) for row in rows]

    def exact_constraint_indices(self, value: str) -> list[int]:
        return self.constraint_lookup.get(canonical(value), [])

    def category_indices(self, category: str, limit: int | None = None) -> list[int]:
        key = canonical(category)
        exact = self.category_lookup.get(key)
        if exact is not None:
            return exact[:] if limit is None else exact[:limit]
        category_terms = set(terms(key, 20))
        if not category_terms:
            return []
        scored: list[tuple[float, int]] = []
        for category_key, indices in self.category_lookup.items():
            product_terms = set(terms(category_key, 20))
            overlap = len(category_terms & product_terms) / len(category_terms)
            if overlap > 0:
                scored.extend((overlap, index) for index in indices[: max(20, (limit or 200) // 4)])
        scored.sort(key=lambda pair: (pair[0], self._popularity_key(pair[1])), reverse=True)
        output: list[int] = []
        seen: set[int] = set()
        for _, product_index in scored:
            if product_index in seen:
                continue
            seen.add(product_index)
            output.append(product_index)
            if limit is not None and len(output) >= limit:
                break
        return output
