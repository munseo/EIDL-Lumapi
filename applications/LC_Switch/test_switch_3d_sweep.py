"""Checks for Switch_3D_sweep: the Stage A metric must be LC_valid_2D's, the LC
tensors must be Switch.py's, the resolved geometry must be self-consistent, and the
tilted-port flux integrator must reproduce an analytically known plane-wave power."""
import math
import os

import numpy as np
import pytest

import LC_valid_2D as v2
import Switch_3D_sweep as s3


# -----------------------------------------------------------------------------
# Stage A model
# -----------------------------------------------------------------------------
def test_score_matches_LC_valid_2D_design_score_grid():
    """switch_score must reproduce v2.design_score_grid cell for cell."""
    os.environ["LC2D_MAX_WIDTH_UM"] = "4.0"
    n_v = v2.vertical_slab_neff(0.38)
    alphas, widths, grid = v2.design_score_grid(n_core=n_v)
    checked = 0
    for i in range(0, widths.size, 7):
        neff = v2.slab_neff(float(widths[i]), n_core=n_v)
        if neff is None:
            continue
        for j in range(0, alphas.size, 11):
            mine = s3.switch_score(neff, float(widths[i]), float(alphas[j]))[0]
            theirs = grid[i, j]
            if np.isnan(theirs):
                assert np.isnan(mine)
            else:
                assert mine == pytest.approx(theirs, rel=1e-12)
            checked += 1
    assert checked > 50


def test_lc_tensor_mapping_matches_Switch_py_literals():
    """Switch.py hard-codes LC_0_n (transmit) and LC_1_n (reflect) in device axes."""
    by_name = {n: t for n, _p, t in s3.LC_STATES_3D}
    assert by_name["transmit"] == [1.5, 1.685, 1.5]   # Switch.py LC_0_n, n_yy = n_e -> passes
    assert by_name["reflect"] == [1.5, 1.5, 1.685]    # Switch.py LC_1_n, n_yy = n_o -> blocks


def test_vertical_multimode_thicknesses_are_rejected():
    rows, info = s3.prescreen()
    assert info["multimode_from"] is not None
    for r in rows:
        assert v2.vertical_slab_neff(r["t"], mode_m=1) is None, r["t"]
        assert s3.N_O < r["x_tan"] < s3.N_E


def test_fp_orders_are_half_wave_multiples():
    x_tan = 1.6
    t_wall, m, _t_min = v2.wall_thickness(x_tan)
    assert s3.fp_wall_thickness(x_tan, m) == pytest.approx(t_wall, rel=1e-12)
    assert s3.fp_wall_thickness(x_tan, m + 1) == pytest.approx(t_wall * (m + 1) / m, rel=1e-12)
    assert s3.fp_wall_thickness(x_tan, 0) is None


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
def _alpha_window(t, w):
    """(alpha_lo, alpha_hi) for which n_eff_inplane*cos(alpha) lies in (n_o, n_e)."""
    n_v = v2.vertical_slab_neff(t)
    neff = v2.slab_neff(w, n_core=n_v)
    th_lo = math.asin(min(s3.N_O / neff, 1.0))
    th_hi = math.asin(min(s3.N_E / neff, 1.0))
    return 90.0 - math.degrees(th_hi), 90.0 - math.degrees(th_lo)


def _candidate(t=0.38, w=4.0, alpha=None, frac=0.5):
    if alpha is None:
        lo, hi = _alpha_window(t, w)
        alpha = lo + frac * (hi - lo)
    n_v = v2.vertical_slab_neff(t)
    neff = v2.slab_neff(w, n_core=n_v)
    x_tan = neff * math.cos(math.radians(alpha))
    t_wall, m, t_min = v2.wall_thickness(x_tan)
    score, margin, sigma = s3.switch_score(neff, w, alpha)
    return dict(label="T1", t=t, w=w, alpha=alpha, n_v=n_v, neff=neff, x_tan=x_tan,
                margin_deg=math.degrees(margin), spread_deg=math.degrees(sigma),
                score=score, t_wall=t_wall, fp_order=m, t_tunnel_min=t_min,
                ip_multimode=False, variant="test")


@pytest.mark.parametrize("frac", [0.15, 0.5, 0.85])
@pytest.mark.parametrize("t", [0.24, 0.38, 0.50])
@pytest.mark.parametrize("w", [1.5, 4.0, 6.0])
def test_geometry_is_self_consistent(t, w, frac):
    geom = s3.resolve_geometry(_candidate(t=t, w=w, frac=frac))
    assert s3.check_geometry(geom) == []

    # the stack must add up
    # The stack is BOX | core | TOX | air. Switch_2.py stops at the TOX surface with
    # an oxide background, so the air cap is the piece this sweep adds; the LC wall is
    # the etched trench and therefore stops at that same surface, air above it.
    assert geom["Sx"] == pytest.approx(s3.BOX_H + geom["core_h"] + s3.TOX_H + s3.AIR_H)
    assert geom["box_x_range"][1] == pytest.approx(geom["x_core_bot"])
    assert geom["tox_x_range"] == pytest.approx([geom["x_core_top"], geom["x_tox_top"]])
    if s3.AIR_H > 0.0:
        assert geom["air_x_range"] == pytest.approx([geom["x_tox_top"], geom["X_max"]])
        assert geom["x_tox_top"] < geom["X_max"]
    assert geom["wall"]["x_range"][1] <= geom["x_tox_top"] + 1e-9
    assert geom["x_core_top"] - geom["x_core_bot"] == pytest.approx(geom["core_h"])

    # rotation convention: Lumerical rotation_1 = -alpha, guide axis = (0, sin a, cos a)
    for g in geom["guides"]:
        assert g["rotation_1"] == pytest.approx(-g["alpha_deg"])
        a = math.radians(g["alpha_deg"])
        assert g["axis_unit"] == pytest.approx([0.0, math.sin(a), math.cos(a)])

    # source propagates along the input guide axis (theta=-alpha, phi=-90)
    src = geom["source"]
    assert src["theta_deg"] == pytest.approx(-geom["alpha_deg"])
    assert src["k_hat"] == pytest.approx(geom["guides"][0]["axis_unit"])

    # the through port sits on guide 0 downstream, the mirror port on guide 1
    thr, mir = geom["ports"]["through"], geom["ports"]["mirror"]
    assert thr["d_hat"] == pytest.approx(geom["guides"][0]["axis_unit"])
    assert mir["d_hat"] == pytest.approx(geom["guides"][1]["axis_unit"])
    assert thr["center"][2] > 0 and mir["center"][2] > 0
    assert thr["center"][1] > 0 and mir["center"][1] < 0     # opposite sides of the wall

    # specular reflection off the y=0 wall maps guide 0 onto guide 1
    d0 = np.asarray(thr["d_hat"])
    assert np.asarray(mir["d_hat"]) == pytest.approx([d0[0], -d0[1], d0[2]])

    # the monitor bounding box must contain the whole tilted plane
    for p in geom["ports"].values():
        for axis, lo, hi in (("y", *p["plane_y_range"]), ("z", *p["plane_z_range"])):
            i = "xyz".index(axis)
            c = p["monitor_center"][i]
            half = 0.5 * p["monitor_span"][i]
            assert c - half <= lo + 1e-9 and hi - 1e-9 <= c + half


def test_geometry_clears_the_wall_and_scales_the_domain():
    """At shallow angles the guide runs alongside the wall for a long distance, so
    the domain has to grow; that is what s_place encodes."""
    lo, hi = _alpha_window(0.38, 6.0)
    shallow = s3.resolve_geometry(_candidate(w=6.0, alpha=lo + 0.1 * (hi - lo)))
    steep = s3.resolve_geometry(_candidate(w=6.0, alpha=hi - 0.1 * (hi - lo)))
    assert shallow["alpha_deg"] < steep["alpha_deg"]
    assert shallow["s_place"] > steep["s_place"]
    assert shallow["Sz"] > steep["Sz"]
    for geom in (shallow, steep):
        # guide edge really is WALL_GAP clear of the wall face at the placement arm
        a = math.radians(geom["alpha_deg"])
        y_edge = geom["s_place"] * math.sin(a) - 0.5 * geom["w_top"] / math.cos(a)
        assert y_edge == pytest.approx(0.5 * geom["t_wall"] + s3.WALL_GAP)


def test_infeasible_candidate_is_rejected_not_silently_resolved():
    """Outside the switching window v2.wall_thickness returns a ~1e14 um wall; that
    must never reach the CAD builder."""
    bad = _candidate(w=6.0, alpha=45.0)          # neff*cos(45) < n_o
    assert not (s3.N_O < bad["x_tan"] < s3.N_E)
    with pytest.raises(ValueError, match="outside the switching window"):
        s3.resolve_geometry(bad)


def test_over_budget_candidate_is_flagged(monkeypatch):
    monkeypatch.setattr(s3, "MAX_CELLS", 1.0e3)
    geom = s3.resolve_geometry(_candidate())
    assert geom["over_budget"]


# -----------------------------------------------------------------------------
# Tilted-port flux integrator
# -----------------------------------------------------------------------------
def test_port_flux_recovers_a_tilted_plane_wave():
    """Synthetic plane wave along d_hat: flux must equal 0.5*|E|^2/eta * area, and
    must be sign-correct and invariant to the tilt."""
    eta0 = 376.730313668
    n_med = 1.6
    eta = eta0 / n_med
    lam = s3.WL
    E0 = 3.0
    dx = 0.04

    for alpha in (0.0, 20.0, 36.75, 55.0):
        a = math.radians(alpha)
        d = np.array([0.0, math.sin(a), math.cos(a)])
        nh = np.array([0.0, d[2], -d[1]])
        e_pol = nh                      # transverse, in-plane (TE-like)
        h_pol = np.cross(d, e_pol) / eta

        x = np.arange(-1.0, 1.0 + 1e-9, dx)
        y = np.arange(-4.0, 4.0 + 1e-9, dx)
        z = np.arange(-4.0, 4.0 + 1e-9, dx)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        phase = np.exp(1j * 2.0 * np.pi * n_med / lam * (d[1] * Y + d[2] * Z))
        E = E0 * phase[..., None] * e_pol[None, None, None, :]
        H = E0 * phase[..., None] * h_pol[None, None, None, :]

        half_len, half_x = 2.0, 0.8
        flux, pol = s3.port_flux_3d((x, y, z, E, H), [0.0, 0.0, 0.0], d,
                                    half_len, half_x, dx)
        area = (2 * half_len * 1e-6) * (2 * half_x * 1e-6)
        expected = 0.5 * E0 ** 2 / eta * area
        assert flux == pytest.approx(expected, rel=2e-3)
        assert pol[0] < 1e-12                       # no layer-normal E
        assert sum(pol) == pytest.approx(1.0)

        back = s3.port_flux_3d((x, y, z, E, H), [0.0, 0.0, 0.0], -d,
                               half_len, half_x, dx)[0]
        assert back == pytest.approx(-expected, rel=2e-3)


def test_port_flux_is_zero_outside_the_monitor():
    dx = 0.04
    x = np.arange(-0.2, 0.2 + 1e-9, dx)
    y = np.arange(-0.2, 0.2 + 1e-9, dx)
    z = np.arange(-0.2, 0.2 + 1e-9, dx)
    E = np.ones(x.shape + y.shape + z.shape + (3,), dtype=complex)
    H = np.ones_like(E)
    flux, _ = s3.port_flux_3d((x, y, z, E, H), [0.0, 0.0, 50.0], [0.0, 0.0, 1.0],
                              1.0, 0.5, dx)
    assert flux == 0.0


def test_wall_mode_window_is_below_the_bulk_lc_window():
    """The wall is a finite LC slab on oxide capped by air, so its mode indices sit
    below the bulk (n_o, n_e). Screening on the bulk window is what predicted
    transmission the 3D run did not deliver."""
    for t in (0.30, 0.55, 0.90):
        lo, hi = s3.wall_mode_window(t)
        assert lo < s3.N_O, f"t={t}: reflect wall mode {lo} should sit below n_o"
        assert hi < s3.N_E, f"t={t}: transmit wall mode {hi} should sit below n_e"
        assert lo < hi
