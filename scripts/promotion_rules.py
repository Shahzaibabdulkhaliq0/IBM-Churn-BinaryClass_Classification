# Pure decision rules for promotion. No MLflow calls here, so they are easy to test.


def passes_quality_gate(test_accuracy, test_f1, min_accuracy, min_f1):
    # A candidate must meet BOTH absolute minimums on the test set (inclusive)
    return test_accuracy >= min_accuracy and test_f1 >= min_f1


def should_promote(candidate_cv_f1, production_cv_f1, gate_passed):
    # Rule 1: a candidate that fails the quality gate is never promoted
    if not gate_passed:
        return False

    # Rule 2: if there is no Production model yet, the first passing candidate is promoted
    if production_cv_f1 is None:
        return True

    # Rule 3: promote only if strictly better than the current Production model.
    # Equal scores keep the current Production model (no pointless swap).
    return candidate_cv_f1 > production_cv_f1
