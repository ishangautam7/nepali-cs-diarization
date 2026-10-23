"""Tests for the statistics in csdiar.analysis.pilot (review finding m8)."""
import numpy as np

from csdiar.analysis.pilot import peak_strength, signflip_p, stratified_auc


def test_stratified_auc_weights_cells_by_positives_and_skips_empty_cells():
    pos = {"a": [3.0, 3.0], "b": [0.0], "c": [5.0]}
    neg = {"a": [1.0, 2.0], "b": [1.0]}  # cell c has no controls: skipped
    auc, n = stratified_auc(pos, neg)
    assert n == 3 and abs(auc - (2 * 1.0 + 0.0) / 3) < 1e-12
    assert stratified_auc({"a": [1.0]}, {"a": [1.0]})[0] == 0.5  # ties count one half


def test_peak_strength_respects_nms_and_window():
    d = np.full(30, np.nan, np.float32)
    d[10:] = 0.1
    d[15], d[18] = 0.9, 0.5  # 0.5 at 18 lies within 0.5 s (5 steps) of the larger peak, so it is suppressed
    assert peak_strength(d, 15) == np.float32(0.9)
    assert peak_strength(d, 19) == np.float32(0.9)  # the 0.9 peak is within +-0.5 s of index 19
    assert peak_strength(d, 26) < 0.9  # beyond +-0.5 s: only the flat 0.1 plateau maxima remain


def test_signflip_p_exact_and_one_sided():
    rng = np.random.default_rng(0)
    x = np.array([1.0, 1.0, 1.0, 1.0])
    assert abs(signflip_p(x, 2 ** 17, rng) - 1 / 16) < 1e-9  # only the all-positive pattern is as large
    assert signflip_p(-x, 2 ** 17, rng) == 1.0
