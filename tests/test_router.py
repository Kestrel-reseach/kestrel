"""Step 7 tests: router decisions with fake verifier, store and memory."""

import pytest

from kestrel.router import Router


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
