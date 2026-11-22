"""Step 5 tests: SSD store build/lookup, gated reads and metrics with fake data."""

import pytest

from kestrel.knowledge_store import (
    Chunk,
    HashVectorizer,
    KnowledgeStore,
    compare_read_policies,
    hash_vectorize,
)


def test_hash_vectorizer_is_normalized_and_stable() -> None:
    first = hash_vectorize("the capital of France", dimension=64)
    second = hash_vectorize("the capital of France", dimension=64)
    assert first == second
    assert len(first) == 64
    assert 0.99 <= sum(value * value for value in first) <= 1.01
    with pytest.raises(ValueError):
        HashVectorizer(0)


def _build_store(tmp_path):
    store = KnowledgeStore(tmp_path, HashVectorizer(64))
    chunks = [
        Chunk("a", "Paris is the capital of France", "wiki"),
        Chunk("b", "Berlin is the capital of Germany", "wiki"),
    ]
    store.build(chunks)
    return store


def test_build_and_lookup(tmp_path) -> None:
    store = _build_store(tmp_path)
    result = store.lookup("capital of France")
    assert result.chunk is not None
    assert result.chunk.chunk_id == "a"
    assert result.bytes_read > 0
    assert result.latency_ms >= 0
    assert store.disk_bytes > 0
    assert store.resident_bytes < store.disk_bytes


def test_lookup_rejects_missing_index(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        KnowledgeStore(tmp_path).lookup("anything")


def test_compare_read_policies_counts_gated_reads(tmp_path) -> None:
    store = _build_store(tmp_path)
    result = compare_read_policies(
        store, ["capital of France", "capital of Germany"], [0.9, 0.2], threshold=0.5
    )
    assert result["always_read"]["reads"] == 2
    assert result["gated_read"]["reads"] == 1
    assert result["gated_read"]["bytes_read"] <= result["always_read"]["bytes_read"]


def test_compare_rejects_mismatched_lengths(tmp_path) -> None:
    store = _build_store(tmp_path)
    with pytest.raises(ValueError):
        compare_read_policies(store, ["a"], [0.1, 0.2])
