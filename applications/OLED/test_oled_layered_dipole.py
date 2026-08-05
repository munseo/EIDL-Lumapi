"""Regression tests for OLED_layered_dipole.py (analytic CPS multilayer solver).

The heavy lifting lives in the module's own `--validate` suite; these tests wrap
it plus a few fast unit-level invariants so a plain `pytest` run catches
regressions.

    python -m pytest test_oled_layered_dipole.py -q
    python test_oled_layered_dipole.py              # no pytest needed
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import OLED_layered_dipole as M       # noqa: E402


def test_branch_cut():
    """Im(l) >= 0 (decaying) and Re(l) >= 0 (outgoing) everywhere."""
    u = np.linspace(0.0, 20.0, 501)
    for n in (1.0, 1.8, complex(0.76, 5.9), complex(0.25, 3.1)):
        l = M._ell(n, u)
        assert np.all(l.imag >= -1e-15)
        assert np.all(l.real >= -1e-15)


def test_fresnel_conventions():
    """r must be antisymmetric under swapping the media and t t' = 1 - r^2,
    which is what makes the compact reflection recursions valid."""
    u = np.linspace(0.0, 4.0, 257)
    n1, n2 = 1.8, complex(0.25, 3.1)
    l1, l2 = M._ell(n1, u), M._ell(n2, u)
    for pol in ("s", "p"):
        r12, t12 = M._fresnel(n1, n2, l1, l2, pol)
        r21, t21 = M._fresnel(n2, n1, l2, l1, pol)
        assert np.allclose(r12, -r21, atol=1e-12)
        assert np.allclose(t12 * t21, 1.0 - r12 ** 2, atol=1e-12)
    # perfect-mirror limit: r_s -> -1, r_p -> +1
    l_pec = M._ell(complex(0.0, 1e7), u)
    rs, _ = M._fresnel(n1, complex(0.0, 1e7), l1, l_pec, "s")
    rp, _ = M._fresnel(n1, complex(0.0, 1e7), l1, l_pec, "p")
    assert np.allclose(rs, -1.0, atol=1e-5)
    assert np.allclose(rp, +1.0, atol=1e-5)


def test_homogeneous_purcell_is_one():
    st = M.Stack(["bot", "a", "b", "top"], [1.77] * 4, [None, 0.21, 0.33, None],
                 0.55, 2, 0.17, label="homogeneous")
    res = M.analyse(st)
    for key in ("h", "v", "iso"):
        assert abs(res["orientations"][key]["purcell"] - 1.0) < 1e-6


def test_energy_balance_and_cps_identity():
    st = M.build_legacy_stack()
    res = M.analyse(st)
    for key in ("h", "v"):
        rec = res["orientations"][key]
        tot = rec["total"][M.IDX_TOTAL]
        recon = (rec["total"][M.IDX_OUT] + rec["total"][M.IDX_BOTTOM]
                 + sum(rec["total"][M.idx_abs(st, j)] for j in range(1, st.N - 1)))
        assert abs(1.0 - recon / tot) < 1e-10
        assert abs(rec["total"][M.idx_cps(st)] / tot - 1.0) < 1e-9


def test_escape_cone_against_independent_integral():
    n1, lam = 1.8, 0.55
    st = M.Stack(["semi", "slab", "air"], [n1, n1, 1.0],
                 [None, 20.0 * lam / n1, None], lam, 1, 0.0, label="half space")
    res = M.analyse(st)
    for key in ("h", "v"):
        rec = res["orientations"][key]
        got = rec["total"][M.IDX_OUT] / rec["total"][M.IDX_TOTAL]
        want = M.independent_escape_cone(n1, 1.0, key)
        assert abs(got / want - 1.0) < 0.01


def test_recursion_matches_matrix_tmm():
    st = M.build_microcavity_stack("green")
    u = np.concatenate([np.linspace(1e-4, 0.999, 200), np.linspace(1.001, 6.0, 200)])
    for pol in ("s", "p"):
        Rum, Rdm, _ = M.matrix_cross_check(st, u, pol)
        Rur, Rdr = M._recursion_reference(st, u, pol)
        assert np.max(np.abs(Rum - Rur)) < 1e-10
        assert np.max(np.abs(Rdm - Rdr)) < 1e-10


def test_near_field_quenching():
    """A dipole approaching a metal must be quenched (F grows without bound)."""
    prev = 0.0
    for dnm in (50.0, 20.0, 10.0, 5.0):
        st = M.Stack(["Ag", "org", "air"], [complex(0.09, 3.32), 1.79, 1.0],
                     [None, 1.0, None], 0.53, 1, dnm * 1e-3, label="drexhage")
        f = M.analyse(st, rtol_seq=(1e-6, 1e-8))["orientations"]["v"]["purcell"]
        assert f > prev
        prev = f
    assert prev > 20.0


def test_full_validation_suite():
    assert M.run_validation()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            bad += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"{len(fns) - bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
