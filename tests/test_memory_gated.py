"""Unit tests for Step 6: write-gated memory store and gate decision logic."""

import pytest

from kestrel.memory_gated import (
    MemoryEntry,
    PlainMemory,
    WriteGatedMemory,
    paired_bootstrap_accuracy_difference,
    step6_gate_decision,
)


def test_entry_creation_and_defaults() -> None:
    entry = MemoryEntry(
        entry_id="e1",
        text="Moved to Minneapolis.",
        source="address_update_note",
        timestamp="2026-11-09 17:45",
        confidence=0.95,
        attribute="current city of residence",
    )
    assert entry.entry_id == "e1"
    assert entry.text == "Moved to Minneapolis."
    assert entry.confidence == 0.95
    assert entry.decay_rule == "none"
    assert entry.recheck_rule == "none"
    assert entry.verified is True
    assert entry.flagged_stale is False
    assert entry.deleted is False


def test_write_gating_rejects_unverified() -> None:
    store = WriteGatedMemory(write_threshold=0.6)
    unverified = MemoryEntry(
        entry_id="u1",
        text="Unverified rumor.",
        source="unverified_chat",
        timestamp="2026-05-01 12:00",
        confidence=0.3,
        verified=False,
    )
    # Rejected without verifier score or owner approval
    assert store.write(unverified) is False
    assert len(store.get_active_entries()) == 0

    # Rejected if verifier score below threshold
    assert store.write(unverified, verifier_score=0.4) is False
    assert len(store.get_active_entries()) == 0

    # Accepted if verifier score above threshold
    assert store.write(unverified, verifier_score=0.8) is True
    assert len(store.get_active_entries()) == 1


def test_write_gating_owner_approved() -> None:
    store = WriteGatedMemory(write_threshold=0.8)
    entry = MemoryEntry(
        entry_id="o1",
        text="Special user setting.",
        source="manual_entry",
        timestamp="2026-06-01 10:00",
        confidence=0.2,
        verified=False,
    )
    # Even with low confidence, owner approval forces write
    assert store.write(entry, owner_approved=True) is True
    assert entry.verified is True
    assert len(store.get_active_entries()) == 1


def test_conflict_resolution_flags_old_entry_stale() -> None:
    store = WriteGatedMemory(write_threshold=0.5)

    e1 = MemoryEntry(
        entry_id="e1",
        text="I moved to Atlanta.",
        source="lease_email",
        timestamp="2026-02-21 09:15",
        confidence=0.9,
        attribute="current city of residence",
    )
    e2 = MemoryEntry(
        entry_id="e2",
        text="Relocated officially to Chicago.",
        source="slack_msg",
        timestamp="2026-05-17 14:30",
        confidence=0.9,
        attribute="current city of residence",
    )
    e3 = MemoryEntry(
        entry_id="e3",
        text="My new address is Minneapolis.",
        source="address_note",
        timestamp="2026-11-09 17:45",
        confidence=0.95,
        attribute="current city of residence",
    )

    assert store.write(e1) is True
    assert len(store.get_active_entries()) == 1
    assert e1.flagged_stale is False

    # Write e2: newer than e1 -> e1 flagged stale, e2 active
    assert store.write(e2) is True
    assert e1.flagged_stale is True
    assert e1.deleted is False  # Old entry NOT deleted per PLAN.md
    assert e2.flagged_stale is False
    assert len(store.get_active_entries()) == 1
    assert store.get_active_entries()[0].entry_id == "e2"

    # Write e3: newer than e2 -> e2 flagged stale, e3 active
    assert store.write(e3) is True
    assert e2.flagged_stale is True
    assert e2.deleted is False
    assert e3.flagged_stale is False

    active = store.get_active_entries()
    assert len(active) == 1
    assert active[0].entry_id == "e3"

    stale = store.get_stale_entries()
    assert len(stale) == 2
    assert {s.entry_id for s in stale} == {"e1", "e2"}


def test_recall_filters_stale_entries() -> None:
    store = WriteGatedMemory(write_threshold=0.5)
    e1 = MemoryEntry(
        entry_id="e1",
        text="I moved to Atlanta.",
        source="lease_email",
        timestamp="2026-02-21 09:15",
        confidence=0.9,
        attribute="current city of residence",
    )
    e2 = MemoryEntry(
        entry_id="e2",
        text="My new address is Minneapolis.",
        source="address_note",
        timestamp="2026-11-09 17:45",
        confidence=0.95,
        attribute="current city of residence",
    )
    store.write(e1)
    store.write(e2)

    # Recall returns only the active unflagged entry
    recalled = store.recall("What is my current city of residence?")
    assert "Minneapolis" in recalled
    assert "Atlanta" not in recalled


def test_edit_and_delete() -> None:
    store = WriteGatedMemory(write_threshold=0.5)
    entry = MemoryEntry(
        entry_id="e1",
        text="Favorite coffee is Sumatra.",
        source="notes",
        timestamp="2026-01-01 10:00",
        confidence=0.8,
        attribute="coffee",
    )
    store.write(entry)

    # Edit text
    assert store.edit("e1", text="Favorite coffee is Ethiopian Yirgacheffe.") is True
    assert (
        store.get_active_entries()[0].text
        == "Favorite coffee is Ethiopian Yirgacheffe."
    )

    # Soft delete
    assert store.delete("e1") is True
    assert len(store.get_active_entries()) == 0
    assert store.recall("What coffee do I like?") == ""


def test_plain_memory_keeps_all_entries() -> None:
    plain = PlainMemory()
    plain.write(
        MemoryEntry("p1", "Moved to Atlanta.", "email", "2026-02-21 09:15", 0.9)
    )
    plain.write(
        MemoryEntry("p2", "Moved to Minneapolis.", "note", "2026-11-09 17:45", 0.9)
    )

    recalled = plain.recall("What is my city?")
    assert "Atlanta" in recalled
    assert "Minneapolis" in recalled
    assert len(plain.get_all_entries()) == 2


def test_paired_bootstrap_accuracy_difference() -> None:
    a = [True] * 90 + [False] * 10
    b = [True] * 80 + [False] * 20
    mean_diff, ci_l, ci_u = paired_bootstrap_accuracy_difference(a, b, n_resamples=1000)
    assert pytest.approx(mean_diff, 0.01) == 0.10
    assert ci_l > 0.0
    assert ci_u >= ci_l


def test_step6_gate_decision_pass() -> None:
    decision = step6_gate_decision(
        no_memory_acc=0.02,
        plain_acc=0.8267,
        gated_acc=0.9200,
        plain_stale_errors=22,
        gated_stale_errors=5,
        plain_static_acc=0.92,
        gated_static_acc=0.94,
        paired_ci=(0.045, 0.141),
    )
    assert decision.gate_decision == "PASS"
    assert decision.recommended_component == "gated_memory"
    assert decision.passes_accuracy_gain is True
    assert decision.passes_ci_excludes_zero is True
    assert decision.passes_stale_error_reduction is True
    assert decision.passes_forgetting_cap is True


def test_step6_gate_decision_fail_kills_to_plain() -> None:
    decision = step6_gate_decision(
        no_memory_acc=0.02,
        plain_acc=0.8267,
        gated_acc=0.8400,  # Only +1.33 pp gain (fails >= 5 pp)
        plain_stale_errors=22,
        gated_stale_errors=20,  # Fails halving rule
        plain_static_acc=0.92,
        gated_static_acc=0.92,
        paired_ci=(-0.02, 0.05),
    )
    assert decision.gate_decision == "FAIL"
    assert decision.recommended_component == "plain_memory"
    assert decision.passes_accuracy_gain is False
    assert decision.passes_stale_error_reduction is False
