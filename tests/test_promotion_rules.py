# Tests for the promotion decision rules (no MLflow or DagsHub needed)
from scripts.promotion_rules import passes_quality_gate, should_promote


# The gate needs BOTH accuracy and F1 to meet the minimums
def test_gate_passes_when_both_metrics_meet_minimums():
    assert passes_quality_gate(0.75, 0.60, min_accuracy=0.70, min_f1=0.55) is True


def test_gate_fails_when_accuracy_is_low():
    assert passes_quality_gate(0.65, 0.60, min_accuracy=0.70, min_f1=0.55) is False


def test_gate_fails_when_f1_is_low():
    assert passes_quality_gate(0.75, 0.50, min_accuracy=0.70, min_f1=0.55) is False


# Exactly at the threshold counts as passing
def test_gate_boundary_is_inclusive():
    assert passes_quality_gate(0.70, 0.55, min_accuracy=0.70, min_f1=0.55) is True


# Rule 1: a failed gate is never promoted, even with no Production model
def test_failed_gate_never_promotes():
    assert should_promote(0.90, None, gate_passed=False) is False


def test_failed_gate_keeps_current_production():
    assert should_promote(0.90, 0.70, gate_passed=False) is False


# Rule 2: first passing candidate goes to Production when none exists
def test_first_passing_candidate_is_promoted():
    assert should_promote(0.74, None, gate_passed=True) is True


# Rule 3: promote only when strictly better
def test_better_candidate_is_promoted():
    assert should_promote(0.75, 0.74, gate_passed=True) is True


def test_worse_candidate_keeps_production():
    assert should_promote(0.70, 0.74, gate_passed=True) is False


def test_equal_candidate_keeps_production():
    assert should_promote(0.74, 0.74, gate_passed=True) is False
