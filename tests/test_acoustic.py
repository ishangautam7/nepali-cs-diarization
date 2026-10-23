"""Tests for the acoustic covariate helpers (analytic inputs only; no audio is generated)."""
import numpy as np

from csdiar.data.acoustic import band_slope, semitones, syllable_nuclei


def test_semitones():
    assert np.allclose(semitones([100.0, 200.0, 50.0]), [0.0, 12.0, -12.0])


def test_band_slope_of_power_laws():
    f = np.linspace(1, 8000, 4097)
    assert abs(band_slope(f, f ** -2.0) - (-20 * np.log10(2))) < 0.05  # -6.02 dB per octave
    assert abs(band_slope(f, np.ones_like(f))) < 1e-9


def test_syllable_nuclei_dip_threshold_and_voicing():
    t = np.arange(12) * 0.01
    v = np.array([0, 5, 10, 7, 10.5, 4, 9, 3, 8, 2, -40, -40], float)
    vt = np.arange(0, 0.12, 0.005)
    voiced = np.ones(len(vt), bool)
    # peaks at 2, 4, 6, 8; the dip after peak 2 is 3 dB (kept), after 4 is 6.5 (kept), after 6 is 6 (kept);
    # the last peak has no following peak (never counted, as in the original script)
    assert np.allclose(syllable_nuclei(v, t, vt, voiced), t[[2, 4, 6]])
    voiced[(vt > 0.035) & (vt < 0.045)] = False  # peak at 0.04 s unvoiced
    assert np.allclose(syllable_nuclei(v, t, vt, voiced), t[[2, 6]])
    # a dip of 2 dB or less does not separate peaks
    v2 = np.array([0, 10, 8.5, 10, 0, 9, 0], float)
    assert np.allclose(syllable_nuclei(v2, np.arange(7) * 0.01, vt, np.ones(len(vt), bool)), [0.03])
