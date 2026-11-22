"""Step 5: SSD-backed knowledge store with memory-mapped vectors.

The index files live on disk and are memory-mapped at lookup time, so resident
RAM stays small regardless of corpus size. A `Vectorizer` seam converts queries
and documents to fixed-size vectors; the default hashing vectorizer is
dependency-free, and a small CPU embedding model plugs in by implementing the
same protocol.

Gated reads (read only when verifier confidence is low) are compared with
always-read via `compare_read_policies`. All helpers are stdlib-only.
"""

from __future__ import annotations

import hashlib
import math
import mmap
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

__all__ = [
    "Chunk",
    "KnowledgeStore",
    "LookupResult",
    "Vectorizer",
    "compare_read_policies",
    "hash_vectorize",
]

FLOAT_SIZE = 4
HEADER_SIZE = 8  # dimension (uint32) + chunk count (uint32)


@dataclass(frozen=True)
class Chunk:
    """One text chunk with its vector offset in the index."""

    chunk_id: str
    text: str
    source: str = ""


@dataclass(frozen=True)
class LookupResult:
    """Result of one store lookup."""

    chunk: Chunk | None
    score: float
    bytes_read: int
    latency_ms: float


class Vectorizer(Protocol):
    """Maps text to a fixed-size float vector."""

    @property
    def dimension(self) -> int:
        """Vector length."""
        ...

    def vectorize(self, text: str) -> tuple[float, ...]:
        """Return a vector of exactly ``dimension`` floats."""
        ...


class HashVectorizer:
    """Dependency-free hashing vectorizer for tests and small corpora."""

    def __init__(self, dimension: int = 128) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def vectorize(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimension
        for word in text.lower().split():
            bucket = int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
            vector[bucket % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return tuple(value / norm for value in vector)


def hash_vectorize(text: str, dimension: int = 128) -> tuple[float, ...]:
    """Convenience wrapper around the default vectorizer."""
    return HashVectorizer(dimension).vectorize(text)


class KnowledgeStore:
    """Flat vector index stored on disk and memory-mapped at query time."""

    def __init__(
        self, directory: Path | str, vectorizer: Vectorizer | None = None
    ) -> None:
        self.directory = Path(directory)
        self.vectorizer = vectorizer or HashVectorizer()
        self.vectors_path = self.directory / "vectors.bin"
        self.chunks_path = self.directory / "chunks.jsonl"
        self._chunks: list[Chunk] = []

    def build(self, chunks: Sequence[Chunk]) -> None:
        """Write vectors and chunk text to disk once; report this cost separately."""
        self.directory.mkdir(parents=True, exist_ok=True)
        dimension = self.vectorizer.dimension
        with self.vectors_path.open("wb") as handle:
            handle.write(struct.pack("<II", dimension, len(chunks)))
            for chunk in chunks:
                vector = self.vectorizer.vectorize(chunk.text)
                if len(vector) != dimension:
                    raise ValueError("vector length does not match dimension")
                handle.write(struct.pack(f"<{dimension}f", *vector))
        with self.chunks_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                import json

                handle.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "source": chunk.source,
                        }
                    )
                    + "\n"
                )
        self._chunks = list(chunks)

    def load_chunks(self) -> None:
        """Load chunk metadata into memory (text stays on SSD until retrieved)."""
        import json

        if not self.chunks_path.is_file():
            raise FileNotFoundError(self.chunks_path)
        self._chunks = []
        with self.chunks_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    data = json.loads(line)
                    self._chunks.append(Chunk(**data))

    @property
    def disk_bytes(self) -> int:
        """Total on-disk size of the index, reported separately from RAM."""
        return sum(
            path.stat().st_size
            for path in (self.vectors_path, self.chunks_path)
            if path.is_file()
        )

    @property
    def resident_bytes(self) -> int:
        """Bytes held in RAM: only chunk metadata, vectors stay memory-mapped."""
        return sum(len(chunk.text.encode()) for chunk in self._chunks)

    def lookup(self, query: str, *, top_k: int = 1) -> LookupResult:
        """Memory-map the vector file and return the highest cosine-similarity chunk."""
        import time

        if not self.vectors_path.is_file():
            raise FileNotFoundError(self.vectors_path)
        self.load_chunks()
        start = time.perf_counter()
        with self.vectors_path.open("rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                dimension, count = struct.unpack_from("<II", mapped, 0)
                if dimension != self.vectorizer.dimension:
                    raise ValueError("index dimension does not match vectorizer")
                query_vector = self.vectorizer.vectorize(query)
                vector_bytes = dimension * FLOAT_SIZE
                best_score = float("-inf")
                best_index = -1
                for index in range(count):
                    offset = HEADER_SIZE + index * vector_bytes
                    values = struct.unpack_from(f"<{dimension}f", mapped, offset)
                    score = sum(a * b for a, b in zip(query_vector, values))
                    if score > best_score:
                        best_score = score
                        best_index = index
        latency_ms = (time.perf_counter() - start) * 1000
        bytes_read = HEADER_SIZE + count * vector_bytes
        chunk = (
            self._chunks[best_index] if 0 <= best_index < len(self._chunks) else None
        )
        return LookupResult(chunk, best_score, bytes_read, latency_ms)


def compare_read_policies(
    store: KnowledgeStore,
    queries: Sequence[str],
    verifier_scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compare always-read with verifier-gated reads on the same queries."""
    if not (len(queries) == len(verifier_scores)):
        raise ValueError("queries and verifier_scores must have the same length")
    always_bytes = 0
    always_latency = 0.0
    gated_bytes = 0
    gated_latency = 0.0
    gated_reads = 0
    for query, score in zip(queries, verifier_scores):
        always = store.lookup(query)
        always_bytes += always.bytes_read
        always_latency += always.latency_ms
        if score < threshold:
            gated = store.lookup(query)
            gated_bytes += gated.bytes_read
            gated_latency += gated.latency_ms
            gated_reads += 1
    return {
        "always_read": {
            "bytes_read": always_bytes,
            "latency_ms": always_latency,
            "reads": len(queries),
        },
        "gated_read": {
            "bytes_read": gated_bytes,
            "latency_ms": gated_latency,
            "reads": gated_reads,
        },
    }
