import numpy as np
import pytest

from src.ab_test import (cuped_adjust, proportion_test, required_sample_size,
                         srm_check)


def test_smaller_effect_needs_bigger_sample():
    assert required_sample_size(0.05, 0.01) > required_sample_size(0.05, 0.10)


def test_srm_flags_a_broken_split():
    assert srm_check(10_000, 10_050)["srm_detected"] is False
    assert srm_check(10_000, 12_000)["srm_detected"] is True


def test_identical_groups_are_not_significant():
    r = proportion_test(500, 10_000, 500, 10_000)
    assert r["p_value"] > 0.9
    assert r["significant"] is False


def test_large_true_effect_is_detected():
    r = proportion_test(400, 10_000, 600, 10_000)
    assert r["significant"] is True
    assert r["relative_lift_%"] == pytest.approx(50.0, rel=1e-6)


def test_cuped_reduces_variance():
    rng = np.random.default_rng(0)
    pre = rng.normal(100, 20, 5000)
    post = pre * 0.7 + rng.normal(0, 10, 5000)
    assert cuped_adjust(post, pre).var() < post.var()
