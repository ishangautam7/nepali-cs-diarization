"""Unit tests for M1 SubspaceProjector (csdiar.mitigations.m1_subspace)."""
import numpy as np

from csdiar.mitigations.m1_subspace import project_out, unit


def test_project_out_removes_component_and_normalises():
    rng = np.random.default_rng(0)
    v = unit(rng.standard_normal(32)[None])[0]
    e = rng.standard_normal((100, 32)).astype(np.float32)
    ep = project_out(e, v)
    # no residual component along v, and unit norm
    assert np.allclose(ep @ v, 0.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(ep, axis=1), 1.0, atol=1e-5)


def test_project_out_keeps_orthogonal_directions():
    rng = np.random.default_rng(1)
    v = np.zeros(8, np.float32); v[0] = 1.0           # language axis = first coord
    e = rng.standard_normal((50, 8)).astype(np.float32)
    e[:, 0] += 5.0                                      # inject a big language component
    ep = project_out(e, v)
    base = unit(e.copy()); base_no0 = e.copy(); base_no0[:, 0] = 0.0; base_no0 = unit(base_no0)
    # after projection the first coord is ~0 and the rest match the normalised zeroed-first-coord vector
    assert np.allclose(ep[:, 0], 0.0, atol=1e-5)
    assert np.allclose(ep[:, 1:], base_no0[:, 1:], atol=1e-5)


def test_v_lang_recovers_planted_direction():
    # build two (speaker,video) groups whose en-mean minus ne-mean points along a known axis
    rng = np.random.default_rng(2)
    axis = unit(rng.standard_normal(16)[None])[0]
    from csdiar.mitigations.m1_subspace import estimate_v_lang
    pts, pe = {}, {}

    class FakeNpz:
        def __init__(self, ids, left, right):
            self._d = {"point_id": np.array(ids), "left": left, "right": right}
        def __contains__(self, k): return k in self._d
        def __getitem__(self, k): return self._d[k]

    ids, left, right = [], [], []
    for g in range(4):
        base = rng.standard_normal(16).astype(np.float32)
        ne = base.copy(); en = base + 2.0 * axis
        pid = f"p{g}"
        pts[pid] = {"kind": "ssdl", "speaker_id": f"s{g}", "video_id": f"v{g}",
                    "lang_left": "ne", "lang_right": "en"}
        ids.append(pid); left.append(ne); right.append(en)
    npz = FakeNpz(ids, np.array(left), np.array(right))

    import csdiar.mitigations.m1_subspace as m
    orig = np.load
    try:
        np.load = lambda *a, **k: npz           # estimate_v_lang loads one npz per split
        v = m.estimate_v_lang(pts, __import__("pathlib").Path("."), "e1", 1.0, ("train",))
    finally:
        np.load = orig
    assert v is not None
    assert abs(float(v @ axis)) > 0.9    # recovered axis (up to sign); planted 2*axis over unit noise
