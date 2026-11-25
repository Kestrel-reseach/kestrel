"""Step 7 tests: router decisions with fake verifier, store and memory."""

import pytest
from typing import Any

from kestrel.router import (
    CalibratedVerifierAdapter,
    GatedMemoryAdapter,
    KnowledgeStoreRetriever,
    Router,
)


class FakeVerifier:
    def __init__(self, score: float) -> None:
        self.score_value = score

    def score(self, query: str, answer: str) -> float:
        return self.score_value


class FakeRetriever:
    def __init__(self, context: str = "context") -> None:
        self.context = context
        self.calls = 0

    def retrieve(self, query: str) -> tuple[str, int]:
        self.calls += 1
        return self.context, 100


class FakeMemory:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def recall(self, query: str) -> str:
        return self.text


def test_abstains_below_floor() -> None:
    router = Router(FakeVerifier(0.05), FakeRetriever(), FakeMemory("fact"))
    decision = router.decide("q", "a")
    assert decision.route == "abstain"
    assert decision.used_store is False
    assert decision.used_memory is False


def test_low_score_retrieves_without_memory() -> None:
    retriever = FakeRetriever()
    router = Router(FakeVerifier(0.3), retriever)
    decision = router.decide("q", "a")
    assert decision.route == "retrieve"
    assert decision.used_store is True
    assert retriever.calls == 1


def test_high_score_answers_directly() -> None:
    retriever = FakeRetriever()
    router = Router(FakeVerifier(0.9), retriever)
    decision = router.decide("q", "a")
    assert decision.route == "answer"
    assert decision.used_store is False
    assert retriever.calls == 0


def test_memory_wins_when_available() -> None:
    router = Router(FakeVerifier(0.8), FakeRetriever(), FakeMemory("verified fact"))
    decision = router.decide("q", "a")
    assert decision.route == "memory"
    assert decision.used_memory is True


def test_no_retriever_still_answers() -> None:
    router = Router(FakeVerifier(0.2))
    assert router.decide("q", "a").route == "answer"


def test_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError):
        Router(FakeVerifier(0.5), threshold=0.4, abstain_below=0.6)


def test_calibrated_verifier_adapter_score_fn() -> None:
    adapter = CalibratedVerifierAdapter(score_fn=lambda q, a: 0.75)
    assert adapter.score("q", "a") == 0.75


def test_calibrated_verifier_adapter_head_and_temperature() -> None:
    class DummyHead:
        def predict(self, features: Any) -> float:
            return 0.8

    adapter = CalibratedVerifierAdapter(
        head=DummyHead(),
        feature_fn=lambda q, a: [1.0],
        temperature=2.0,
    )
    score = adapter.score("query", "answer")
    assert 0.0 < score < 0.8  # Temperature > 1 softens towards 0.5


def test_knowledge_store_retriever_adapter() -> None:
    from kestrel.knowledge_store import Chunk, LookupResult

    class DummyStore:
        def lookup(self, query: str) -> LookupResult:
            return LookupResult(
                chunk=Chunk("c1", "retrieved content", "source"),
                score=0.95,
                bytes_read=256,
                latency_ms=12.5,
            )

    retriever = KnowledgeStoreRetriever(DummyStore())
    text, bytes_read = retriever.retrieve("query")
    assert text == "retrieved content"
    assert bytes_read == 256


def test_gated_memory_adapter() -> None:
    class DummyMemory:
        def recall(self, query: str) -> str:
            return "user preference: coffee"

    adapter = GatedMemoryAdapter(DummyMemory())
    assert adapter.recall("drink?") == "user preference: coffee"


def test_router_with_real_adapters_integration() -> None:
    from kestrel.knowledge_store import Chunk, LookupResult

    class DummyStore:
        def lookup(self, query: str) -> LookupResult:
            return LookupResult(
                chunk=Chunk("c1", "wiki doc", "wiki"),
                score=0.9,
                bytes_read=128,
                latency_ms=10.0,
            )

    class DummyMemory:
        def recall(self, query: str) -> str:
            return "user fact" if "user" in query else ""

    verifier = CalibratedVerifierAdapter(score_fn=lambda q, a: 0.3)
    retriever = KnowledgeStoreRetriever(DummyStore())
    memory = GatedMemoryAdapter(DummyMemory())

    router = Router(
        verifier, retriever=retriever, memory=memory, threshold=0.5, abstain_below=0.1
    )

    # Low confidence query -> retrieve
    decision = router.decide("what is gravity?", "mass attracts")
    assert decision.route == "retrieve"
    assert decision.used_store is True
    assert decision.used_memory is False

    # Query matching personal memory -> memory wins
    decision_mem = router.decide("user email", "test@test.com")
    assert decision_mem.route == "memory"
    assert decision_mem.used_memory is True
