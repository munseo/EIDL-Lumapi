"""Switch_3D_sweep: two-stage (thickness, angle, LC-wall thickness, width) sweep
for the 3D LC TIR switch of Switch.py.

GOAL
    Find parameter combinations that already reach, AT THE INITIAL (un-optimized,
    CAD) STRUCTURE, transmission T >= 0.70 in the transmit LC state and
    reflection R >= 0.80 in the reflect LC state.  Nothing here optimizes: Stage B
    is a pure forward evaluation of the CAD (two rotated SiN guides + LC wall).

WHY TWO STAGES
    A blind 4-D sweep in 3D FDTD is unaffordable (each 3D case is ~40 Mcells and
    needs 3 runs).  Stage A is the analytic effective-index pre-screen of
    LC_valid_2D (seconds, no FDTD) and Stage B confirms only its top candidates.

STAGE A -- analytic pre-screen (imports LC_valid_2D as v2, no FDTD)
    EIM step 1   v2.vertical_slab_neff(t)              -> vertical SiN/SiO2 slab n_eff
    EIM step 2   v2.slab_neff(w, n_core=<step 1>)      -> in-plane n_eff
    switching    N_O < n_eff_inplane * cos(alpha) < N_E
    ranking      score = angular margin / beam angular spread   (v2.design_score_grid)
    LC thickness v2.wall_thickness(n_eff*cos alpha) -> (t_wall, FP order m, tunneling
                 minimum).  The LC thickness is CHOSEN from this resonance condition
                 rather than swept blindly; FP orders m-1 / m+1 are added as extra
                 candidates so the LC thickness is still genuinely explored in 3D.
    rejection    thicknesses whose vertical slab is MULTIMODE
                 (v2.vertical_slab_neff(t, mode_m=1) is not None) are dropped: the
                 EIM single-mode reduction is invalid there.  The boundary is reported.

STAGE B -- 3D FDTD confirmation (msopt, same usage as Switch.py)
    Geometry, materials, source and rotation convention are Switch.py's:
        x = layer normal (BOX / SiN core / TOX), y = wall normal, z = wall axis
        waveguide k = rectangle rotated about x by -alpha_k, alpha = [+a, -a]
        guide axis  d_hat_k = (0, sin alpha_k, cos alpha_k)
        LC wall     = un-rotated y-thin band on the z axis, mesh order 1
        source      = tilted mode source, injection axis z, theta=-alpha, phi=-90
    Per candidate 3 simulations: NORM (input guide only) + transmit + reflect.
    Ports are PLANES perpendicular to the guide axis.  DFT monitors cannot rotate,
    so E and H are interpolated from an axis-aligned volume monitor that bounds the
    tilted plane -- the 3D version of LC_valid_2D.port_flux.  Powers are normalized
    by the NORM run measured on the IDENTICAL through-port plane, so the port
    geometry cancels:  T = P_through/P_ref,  R = P_mirror/P_ref.

USAGE
    run Switch_3D_sweep.py -th 8 -GPU <idle gpu>      # Stage A + Stage B
    python Switch_3D_sweep.py --dry-run              # geometry only, no Lumerical
    python Switch_3D_sweep.py --stage-a-only         # analytic pre-screen only

OUTPUT ($EIDL_RUN_DIR/A/)
    Switch3D_prescreen.txt / .png        ranked Stage A table + design maps
    Switch3D_candidates.txt              resolved geometry of the chosen candidates
    Switch3D_results.txt / .json         measured T/R per candidate with PASS/FAIL
    Switch3D_results.png                 T/R summary with the 0.70 / 0.80 targets
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import LC_valid_2D as v2  # analytic design model + LC state tensors (Stage A)

RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
os.makedirs(design_dir, exist_ok=True)


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_int(name, default):
    return int(float(os.environ.get(name, str(default))))


def env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


# -----------------------------------------------------------------------------
# Physical constants -- taken from LC_valid_2D so the two stages cannot drift
# -----------------------------------------------------------------------------
WL = v2.WL            # 1.55 um
N_SIN = v2.N_SIN      # 2.0
N_SIO2 = v2.N_SIO2    # 1.44
N_O = v2.N_O          # 1.5   LC ordinary
N_E = v2.N_E          # 1.685 LC extraordinary
K0 = v2.K0

SIN_N = [N_SIN, N_SIN, N_SIN]
SIO2_N = [N_SIO2, N_SIO2, N_SIO2]


def lc_tensor_3d(lc_n_2d):
    """Map an LC_valid_2D diagonal tensor to the 3D device axes of Switch.py.

    LC_valid_2D works in the Lumerical 2D x-y plane with the wall ON the x axis:
        2D sim x  (wall axis)      -> 3D z
        2D sim y  (wall normal)    -> 3D y
        2D out-of-plane z          -> 3D x  (layer normal / thickness)
    so [n_x2D, n_y2D, n_out] becomes [n_out, n_y2D, n_x2D].
    """
    nx2, ny2, nout = (float(v) for v in lc_n_2d)
    return [nout, ny2, nx2]


# ("reflect", phi, tensor) / ("transmit", phi, tensor) in 3D axes.
LC_STATES_3D = [(name, phi, lc_tensor_3d(n2d)) for name, phi, n2d in v2.LC_STATES]
_LC_BY_NAME = {name: tensor for name, _phi, tensor in LC_STATES_3D}

# Switch.py hard-codes the same two tensors; keep the mapping honest.
assert _LC_BY_NAME["transmit"] == [1.5, 1.685, 1.5], _LC_BY_NAME["transmit"]
assert _LC_BY_NAME["reflect"] == [1.5, 1.5, 1.685], _LC_BY_NAME["reflect"]


# -----------------------------------------------------------------------------
# Sweep / cost / geometry knobs
# -----------------------------------------------------------------------------
RESOLUTION = env_int("SWITCH3D_RESOLUTION", 25)          # cells per um (Switch.py)
BOX_H = env_float("SWITCH3D_BOX_UM", 2.0)
TOX_H = env_float("SWITCH3D_TOX_UM", 2.0)      # oxide ABOVE the core (Switch_2.py convention)
# Air on top of the upper oxide. Switch_2.py sets Sx = BOX_h + Core_h + TOX_h with an
# oxide background, i.e. the domain stops at the TOX surface and the PML sits directly
# on oxide -- there is no air in the model at all. The real device is open to air, and
# the LC wall (which is the etched trench, so it reaches the TOX surface) has its own
# vertical modes whose count depends on what caps it. Set 0 to reproduce the old
# no-air behaviour.
AIR_H = env_float("SWITCH3D_AIR_UM", 1.5)
N_AIR = env_float("SWITCH3D_AIR_INDEX", 1.0)

N_CANDIDATES = env_int("SWITCH3D_N_CANDIDATES", 8)
N_FP_VARIANTS = env_int("SWITCH3D_FP_VARIANTS", 2)       # +-1 FP order around the best
MAX_CELLS = env_float("SWITCH3D_MAX_CELLS", 120e6)
BYTES_PER_CELL = env_float("SWITCH3D_BYTES_PER_CELL", 120.0)  # measured: 4.35 GiB / 40.1 Mcells
KEEP_FSP = env_flag("SWITCH3D_KEEP_FSP", "0")
ALL_PORTS = env_flag("SWITCH3D_ALL_PORTS", "0")          # also measure back/idle

T_MIN = env_float("SWITCH3D_T_MIN_UM", 0.20)
T_MAX = env_float("SWITCH3D_T_MAX_UM", 1.00)
T_STEP = env_float("SWITCH3D_T_STEP_UM", 0.02)
W_MIN = env_float("SWITCH3D_W_MIN_UM", 1.0)
W_MAX = env_float("SWITCH3D_W_MAX_UM", 6.0)
W_STEP = env_float("SWITCH3D_W_STEP_UM", 0.10)
A_MIN = env_float("SWITCH3D_A_MIN_DEG", 20.0)
A_MAX = env_float("SWITCH3D_A_MAX_DEG", 60.0)
A_STEP = env_float("SWITCH3D_A_STEP_DEG", 0.25)

# Candidate diversity: a new pick must differ from every earlier pick by at least
# one of these amounts, otherwise the whole shortlist collapses onto one ridge point.
DIV_T = env_float("SWITCH3D_DIV_T_UM", 0.06)
DIV_W = env_float("SWITCH3D_DIV_W_UM", 0.80)
DIV_A = env_float("SWITCH3D_DIV_A_DEG", 4.0)

WALL_GAP = env_float("SWITCH3D_WALL_GAP_UM", 1.2)        # guide edge <-> wall face at source/port
PORT_MARGIN = env_float("SWITCH3D_PORT_MARGIN_UM", 1.5)  # port half-length beyond w/2
PORT_X_MARGIN = env_float("SWITCH3D_PORT_X_MARGIN_UM", 1.2)
SRC_Y_MARGIN = env_float("SWITCH3D_SRC_Y_MARGIN_UM", 1.0)
SRC_X_MARGIN = env_float("SWITCH3D_SRC_X_MARGIN_UM", 1.5)
EDGE_MARGIN = env_float("SWITCH3D_EDGE_MARGIN_UM", 1.0)  # feature <-> domain boundary (PML)
WALL_KEEPOUT = env_float("SWITCH3D_WALL_KEEPOUT_UM", 0.3)
WALL_MAX = env_float("SWITCH3D_WALL_MAX_UM", 20.0)       # sanity limit on t_wall
# Vertical extent of the LC wall above the core bottom. 0 = Switch.py's core+TOX.
WALL_HEIGHT = env_float("SWITCH3D_WALL_HEIGHT_UM", 0.0)
S_MIN = env_float("SWITCH3D_S_MIN_UM", 10.0)

# Top-view |H| map on the plane through the core, one per LC state, with T/R
# annotated. Off by default: it adds a full-area 2D monitor to every run.
FIELD_MAP = env_flag("SWITCH3D_FIELD_MAP", "0")

CORE_REFINE = env_flag("SWITCH3D_CORE_REFINE", "1")
CORE_CELLS_MIN = env_float("SWITCH3D_CORE_CELLS", 10.0)  # min cells across the core thickness
CORE_REFINE_PAD = env_float("SWITCH3D_CORE_REFINE_PAD_UM", 0.25)

T_TARGET = env_float("SWITCH3D_T_TARGET", 0.70)
R_TARGET = env_float("SWITCH3D_R_TARGET", 0.80)

# dataviz reference categorical palette, slots 1-4 in fixed order (light surface).
# (The bundled validator needs `??=`; node here is v10, so the pre-validated
# reference values are used unchanged rather than eyeballed substitutes.)
C_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C_INK = "#0b0b0b"
C_MUTED = "#52514e"


# =============================================================================
# STAGE A -- analytic pre-screen
# =============================================================================
def switch_score(neff, w, alpha_deg):
    """The LC_valid_2D ranking metric: angular switch margin / beam angular spread.

    Identical formula to v2.design_score_grid (asserted in test_switch_3d_sweep.py);
    written out here so the pre-screen can use its own (t, w, alpha) grid instead of
    v2's fixed one.  Returns (score, margin_rad, spread_rad); score is nan when the
    operating point is outside the switching window.
    """
    th_lo = math.asin(min(N_O / neff, 1.0))
    th_hi = math.asin(min(N_E / neff, 1.0))
    sigma = WL / (math.pi * neff * w)
    th = math.radians(90.0 - alpha_deg)
    margin = min(th - th_lo, th_hi - th)
    return (margin / sigma if margin > 0.0 else float("nan")), margin, sigma


def prescreen():
    """Grid the (thickness, width, alpha) design space with the EIM model."""
    thicknesses = np.arange(T_MIN, T_MAX + 1e-9, T_STEP)
    widths = np.arange(W_MIN, W_MAX + 1e-9, W_STEP)
    alphas = np.arange(A_MIN, A_MAX + 1e-9, A_STEP)
    th_all = np.deg2rad(90.0 - alphas)
    cos_all = np.cos(np.deg2rad(alphas))

    rows = []
    t_notes = []
    multimode_ts = []
    cutoff_ts = []
    for t in thicknesses:
        t = float(t)
        n_v = v2.vertical_slab_neff(t)
        if n_v is None:
            cutoff_ts.append(t)
            t_notes.append((t, None, False, "below vertical cutoff"))
            continue
        vmulti = v2.vertical_slab_neff(t, mode_m=1) is not None
        if vmulti:
            multimode_ts.append(t)
            t_notes.append((t, n_v, True, "vertically multimode - EIM invalid, rejected"))
            continue
        t_notes.append((t, n_v, False, "ok"))

        for w in widths:
            w = float(w)
            neff = v2.slab_neff(w, n_core=n_v)
            if neff is None:
                continue
            ip_multi = v2.slab_neff(w, mode_m=1, n_core=n_v) is not None
            th_lo = math.asin(min(N_O / neff, 1.0))
            th_hi = math.asin(min(N_E / neff, 1.0))
            sigma = WL / (math.pi * neff * w)
            margin = np.minimum(th_all - th_lo, th_hi - th_all)
            ok = margin > 0.0
            if not np.any(ok):
                continue
            x_tan_all = neff * cos_all
            for j in np.flatnonzero(ok):
                x_tan = float(x_tan_all[j])
                if not (N_O < x_tan < N_E):
                    # margin>0 already implies this; keep the guard explicit because
                    # v2.wall_thickness is only defined inside the switching window.
                    continue
                t_wall, m_fp, t_min = v2.wall_thickness(x_tan)
                rows.append(dict(
                    t=t, w=w, alpha=float(alphas[j]),
                    n_v=float(n_v), neff=float(neff), x_tan=x_tan,
                    margin_deg=float(np.degrees(margin[j])),
                    spread_deg=float(np.degrees(sigma)),
                    score=float(margin[j] / sigma),
                    t_wall=float(t_wall), fp_order=int(m_fp), t_tunnel_min=float(t_min),
                    ip_multimode=bool(ip_multi),
                ))

    rows.sort(key=lambda r: -r["score"])
    info = dict(
        n_rows=len(rows),
        multimode_from=min(multimode_ts) if multimode_ts else None,
        cutoff_below=max(cutoff_ts) if cutoff_ts else None,
        thicknesses=thicknesses, widths=widths, alphas=alphas, t_notes=t_notes,
    )
    return rows, info


def write_prescreen_table(rows, info, path, n_show=200):
    lines = [
        "# Stage A analytic pre-screen (effective-index method, LC_valid_2D model)",
        f"# lambda={WL} um  SiN={N_SIN}  SiO2={N_SIO2}  LC n_o={N_O} n_e={N_E}",
        f"# grid: t {T_MIN}:{T_STEP}:{T_MAX} um, w {W_MIN}:{W_STEP}:{W_MAX} um, "
        f"alpha {A_MIN}:{A_STEP}:{A_MAX} deg",
        "# switching window: n_o < neff_inplane*cos(alpha) < n_e",
        "# score = angular switch margin / beam angular spread (higher = harder switching contrast)",
        "# t_wall = tunneling-isolation minimum snapped up to the transmit-state FP resonance (order m)",
        f"# feasible points: {len(rows)}",
    ]
    if info["multimode_from"] is not None:
        lines.append(f"# vertical slab MULTIMODE (rejected) from t = {info['multimode_from']:.3f} um upward")
    if info["cutoff_below"] is not None:
        lines.append(f"# vertical slab BELOW CUTOFF up to t = {info['cutoff_below']:.3f} um")
    lines.append(
        f"{'rank':>5}{'t_um':>8}{'w_um':>7}{'alpha':>8}{'n_vert':>8}{'neff_ip':>9}"
        f"{'n*cos':>8}{'margin':>8}{'spread':>8}{'score':>8}{'t_wall':>8}{'m':>3}"
        f"{'t_tun':>8}{'ipMM':>6}"
    )
    for i, r in enumerate(rows[:n_show]):
        lines.append(
            f"{i + 1:>5}{r['t']:>8.3f}{r['w']:>7.2f}{r['alpha']:>8.2f}{r['n_v']:>8.4f}"
            f"{r['neff']:>9.4f}{r['x_tan']:>8.4f}{r['margin_deg']:>8.2f}{r['spread_deg']:>8.2f}"
            f"{r['score']:>8.3f}{r['t_wall']:>8.3f}{r['fp_order']:>3d}"
            f"{r['t_tunnel_min']:>8.3f}{int(r['ip_multimode']):>6}"
        )
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(text + "\n")
    return text


def plot_prescreen(rows, info, candidates, path):
    thicknesses, widths, alphas = info["thicknesses"], info["widths"], info["alphas"]
    best_ta = np.full((thicknesses.size, alphas.size), np.nan)
    best_t = np.full(thicknesses.size, np.nan)
    t_index = {round(float(t), 6): i for i, t in enumerate(thicknesses)}
    a_index = {round(float(a), 6): j for j, a in enumerate(alphas)}
    for r in rows:
        i = t_index[round(r["t"], 6)]
        j = a_index[round(r["alpha"], 6)]
        if not (best_ta[i, j] >= r["score"]):
            best_ta[i, j] = r["score"]
        if not (best_t[i] >= r["score"]):
            best_t[i] = r["score"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True)

    ax = axes[0, 0]
    mesh = ax.pcolormesh(alphas, thicknesses, best_ta, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="best score over width")
    seen = {}
    for k, c in enumerate(candidates):
        ax.plot(c["alpha"], c["t"], marker="o", ms=7, mfc="none", mew=1.6,
                color="#ffffff" if k else "#eb6834")
        # several candidates can share an (alpha, t) cell (they differ in w or in
        # LC thickness); fan their labels out so none is hidden
        key = (round(c["alpha"], 3), round(c["t"], 4))
        rank = seen.get(key, 0)
        seen[key] = rank + 1
        ax.annotate(c["label"], (c["alpha"], c["t"]), color="#ffffff", fontsize=7.5,
                    xytext=(5, 3 - 9 * rank), textcoords="offset points")
    ax.set_xlabel("crossing angle alpha from the wall axis (deg)")
    ax.set_ylabel("waveguide thickness (um)")
    ax.set_title("switch margin / beam spread (circles = 3D candidates)", color=C_INK)

    ax = axes[0, 1]
    ax.plot(thicknesses, best_t, lw=2, color=C_SERIES[0])
    if info["multimode_from"] is not None:
        ax.axvline(info["multimode_from"], color=C_MUTED, ls="--", lw=1.2)
        ax.annotate("vertical slab multimode ->\n(EIM invalid, rejected)",
                    (info["multimode_from"], np.nanmax(best_t) * 0.92),
                    color=C_MUTED, fontsize=8, xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("waveguide thickness (um)")
    ax.set_ylabel("best score")
    ax.set_title("Best achievable score vs thickness", color=C_INK)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    if rows:
        t_best, a_best = rows[0]["t"], rows[0]["alpha"]
        sel = [r for r in rows if r["t"] == t_best and r["alpha"] == a_best]
        sel.sort(key=lambda r: r["w"])
        ax.plot([r["w"] for r in sel], [r["score"] for r in sel], lw=2, color=C_SERIES[1])
        ax.set_title(f"score vs width at t={t_best:.2f} um, alpha={a_best:.2f} deg", color=C_INK)
    ax.set_xlabel("waveguide width (um)")
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    for k, c in enumerate(candidates):
        ax.plot([k + 1], [c["t_wall"]], marker="s", ms=8, color=C_SERIES[0], ls="none")
        ax.plot([k + 1], [c["t_tunnel_min"]], marker="_", ms=12, mew=2,
                color=C_SERIES[3], ls="none")
        ax.annotate(f"m={c['fp_order']}", (k + 1, c["t_wall"]), fontsize=7.5,
                    color=C_MUTED, xytext=(3, 4), textcoords="offset points")
    ax.plot([], [], marker="s", ls="none", color=C_SERIES[0], label="LC wall thickness (FP resonant)")
    ax.plot([], [], marker="_", ls="none", mew=2, color=C_SERIES[3], label="tunneling-isolation minimum")
    ax.set_xlabel("candidate")
    ax.set_ylabel("LC wall thickness (um)")
    ax.set_title("Chosen LC thickness per candidate", color=C_INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Stage A: analytic pre-screen of the 3D LC TIR switch", fontsize=12.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fp_wall_thickness(x_tan, order):
    """LC wall thickness at an explicit Fabry-Perot order (v2.wall_thickness picks
    the lowest order above the tunneling minimum; +-1 around it probes whether the
    FP resonance really is what sets the transmit-state wrong-port bounce)."""
    th_t = math.asin(min(x_tan / N_E, 1.0))
    kz = K0 * N_E * math.cos(th_t)
    if kz <= 0.0 or order < 1:
        return None
    return float(order * math.pi / kz)


def fp_wall_thickness_uniaxial(x_tan, order):
    """Transmit-state FP resonance with the wall treated as the UNIAXIAL medium it is.

    v2.wall_thickness (and fp_wall_thickness above) put the isotropic value n_e into
    the internal wavevector.  That is not what the transmitted wave sees: in the
    transmit state the LC director lies along the WALL NORMAL (n_yy = n_e) and the
    guided mode is p-polarised (E in the plane of incidence), so the wave inside the
    wall is EXTRAORDINARY and its index depends on its own refraction angle,

        n(theta) = n_o n_e / sqrt(n_e^2 cos^2 theta + n_o^2 sin^2 theta),

    with theta measured from the optic axis = from the wall normal.  Snell
    (n(theta) sin theta = x_tan) then closes on

        sin^2 theta = x_tan^2 n_e^2 / (n_o^2 n_e^2 + x_tan^2 (n_e^2 - n_o^2)).

    At the near-grazing internal angles this device runs at, the difference is large
    enough to move the half-wave resonance by ~12%, which detunes the etalon and
    bounces most of the "transmitted" power back to the mirror port.
    """
    if order < 1 or not (0.0 < x_tan):
        return None
    s2 = x_tan ** 2 * N_E ** 2 / (N_O ** 2 * N_E ** 2 + x_tan ** 2 * (N_E ** 2 - N_O ** 2))
    if not (0.0 < s2 < 1.0):
        return None
    s = math.sqrt(s2)
    kz = K0 * (x_tan / s) * math.sqrt(1.0 - s2)
    if kz <= 0.0:
        return None
    return float(order * math.pi / kz)


FP_MODEL = os.environ.get("SWITCH3D_FP_MODEL", "v2").strip().lower()


def fp_thickness(x_tan, order):
    """FP resonance under the configured model (SWITCH3D_FP_MODEL=v2|uniaxial)."""
    if FP_MODEL.startswith("uni"):
        return fp_wall_thickness_uniaxial(x_tan, order)
    return fp_wall_thickness(x_tan, order)


def candidates_from_spec(spec):
    """Explicit candidate list: 'SWITCH3D_CANDIDATES=t,w,alpha[,t_wall]; ...'.

    The FP resonance is the least reliable part of the analytic model, so this lets a
    follow-up run probe chosen LC wall thicknesses at a fixed (t, w, alpha) without
    disturbing the Stage A ranking.
    """
    cands = []
    for item in spec.replace(";", "\n").splitlines():
        item = item.strip()
        if not item:
            continue
        parts = [float(v) for v in item.split(",") if v.strip()]
        if len(parts) < 3:
            raise ValueError(f"candidate spec needs t,w,alpha[,t_wall]: {item!r}")
        t, w, alpha = parts[:3]
        n_v = v2.vertical_slab_neff(t)
        if n_v is None:
            raise ValueError(f"t={t} um is below the vertical slab cutoff")
        neff = v2.slab_neff(w, n_core=n_v)
        if neff is None:
            raise ValueError(f"w={w} um is below the in-plane cutoff at t={t} um")
        x_tan = neff * math.cos(math.radians(alpha))
        # The switching limits are the WALL's own vertical mode indices, not the bulk
        # (n_o, n_e): the wall is a finite LC slab standing on oxide and capped by air,
        # so its fundamental sits below the bulk index in both states. Outside the bulk
        # window is a warning, outside the mode window is fatal.
        x_lo, x_hi = wall_mode_window(t)
        if not (x_lo < x_tan < x_hi):
            raise ValueError(f"t={t}, w={w}, alpha={alpha}: neff*cos(alpha)={x_tan:.4f} "
                             f"is outside the wall-mode switching window "
                             f"({x_lo:.4f}, {x_hi:.4f})")
        if not (N_O < x_tan < N_E):
            print(f"[candidates] NOTE t={t}, w={w}, alpha={alpha}: x_tan={x_tan:.4f} is "
                  f"outside the bulk window ({N_O}, {N_E}) but inside the wall-mode "
                  f"window ({x_lo:.4f}, {x_hi:.4f}) -- the wall modes are what matters.")
        t_wall_auto, m_auto, t_min = v2.wall_thickness(x_tan)
        t_wall = parts[3] if len(parts) > 3 else t_wall_auto
        score, margin, sigma = switch_score(neff, w, alpha)
        kz_order = t_wall / max(t_wall_auto / max(m_auto, 1), 1e-12)
        cands.append(dict(
            label=f"C{len(cands) + 1}", t=t, w=w, alpha=alpha,
            n_v=float(n_v), neff=float(neff), x_tan=float(x_tan),
            margin_deg=math.degrees(margin), spread_deg=math.degrees(sigma),
            score=float(score), t_wall=float(t_wall),
            fp_order=int(round(kz_order)), t_tunnel_min=float(t_min),
            ip_multimode=bool(v2.slab_neff(w, mode_m=1, n_core=n_v) is not None),
            below_tunnel_min=bool(t_wall < t_min),
            variant="explicit SWITCH3D_CANDIDATES entry",
        ))
    return cands


def select_candidates(rows, n_candidates=None, n_fp=None):
    """Top-scoring, mutually distinct (t, w, alpha) picks + FP-order variants."""
    n_candidates = N_CANDIDATES if n_candidates is None else n_candidates
    n_fp = N_FP_VARIANTS if n_fp is None else n_fp
    n_base = max(1, n_candidates - max(0, n_fp))

    picks = []
    for r in rows:
        if len(picks) >= n_base:
            break
        far_enough = all(
            (abs(r["t"] - p["t"]) >= DIV_T
             or abs(r["w"] - p["w"]) >= DIV_W
             or abs(r["alpha"] - p["alpha"]) >= DIV_A)
            for p in picks
        )
        if far_enough:
            picks.append(dict(r))

    cands = []
    for i, p in enumerate(picks):
        c = dict(p)
        c["label"] = f"C{i + 1}"
        c["variant"] = f"FP m={p['fp_order']} (v2.wall_thickness)"
        cands.append(c)

    if picks and n_fp > 0:
        best = picks[0]
        # v2.wall_thickness returns the LOWEST order above the tunneling minimum, so
        # m-1 usually does not exist (m=1). Walk outward and keep the first n_fp
        # orders that are physically defined, so the LC thickness is always probed.
        n_added = 0
        for off in (-1, +1, +2, -2, +3, +4):
            if n_added >= n_fp:
                break
            order = best["fp_order"] + off
            t_wall = fp_thickness(best["x_tan"], order)
            if t_wall is None or t_wall > WALL_MAX:
                continue
            n_added += 1
            c = dict(best)
            c["t_wall"] = t_wall
            c["fp_order"] = order
            c["label"] = f"C{len(cands) + 1}"
            below = t_wall < best["t_tunnel_min"]
            c["variant"] = (f"FP m={order} of C1"
                            + (" (BELOW tunneling minimum)" if below else ""))
            c["below_tunnel_min"] = below
            cands.append(c)

    for c in cands:
        c.setdefault("below_tunnel_min", c["t_wall"] < c["t_tunnel_min"])
    return cands[:n_candidates]


# =============================================================================
# Geometry resolution -- pure python, no Lumerical.  Used by --dry-run.
# =============================================================================
def resolve_geometry(cand):
    """Resolve one candidate into the full 3D layout (Switch.py axes/conventions)."""
    t = float(cand["t"])
    w = float(cand["w"])
    alpha = float(cand["alpha"])
    t_wall = float(cand["t_wall"])

    # v2.wall_thickness is only defined inside the switching window: outside it the
    # tunneling exponent collapses and it returns a ~1e14 um "thickness", which would
    # silently produce a nonsense domain.  Fail loudly instead.
    x_tan = float(cand["x_tan"])
    if not (N_O < x_tan < N_E):
        raise ValueError(f"candidate is outside the switching window: "
                         f"neff*cos(alpha)={x_tan:.4f} not in ({N_O}, {N_E})")
    if not math.isfinite(t_wall) or not (0.0 < t_wall <= WALL_MAX):
        raise ValueError(f"implausible LC wall thickness {t_wall:g} um "
                         f"(limit SWITCH3D_WALL_MAX_UM={WALL_MAX:g})")
    if not (0.0 < w) or not (0.0 < t) or not (0.0 < alpha < 90.0):
        raise ValueError(f"implausible candidate geometry t={t} w={w} alpha={alpha}")

    dx = 1.0 / RESOLUTION
    ca = math.cos(math.radians(alpha))
    sa = math.sin(math.radians(alpha))

    # --- vertical stack (x) ---------------------------------------------------
    core_h = t
    Sx = BOX_H + core_h + TOX_H + AIR_H
    X_min, X_max = -0.5 * Sx, 0.5 * Sx
    x_core_bot = X_min + BOX_H
    x_core_top = x_core_bot + core_h
    x_core = x_core_bot + 0.5 * core_h
    x_tox_top = x_core_top + TOX_H          # oxide surface = top of the etched trench

    # --- in-plane placement ---------------------------------------------------
    # Arm distance at which the guide EDGE clears the wall face by WALL_GAP.
    # At arm distance s the guide centre sits at y = s*sin(alpha) and, on a plane of
    # constant z, the guide half-width is w/(2 cos alpha).
    s_clear = (0.5 * t_wall + 0.5 * w / ca) / sa
    s_place = (0.5 * t_wall + WALL_GAP + 0.5 * w / ca) / sa

    y_port = s_place * sa
    z_port = s_place * ca

    # Port half-length along n_hat, clipped so the plane never reaches the LC wall
    # (same rule as LC_valid_2D.port_geometry.half()).
    h_req = 0.5 * w + PORT_MARGIN
    h_wall = (y_port - 0.5 * t_wall - WALL_KEEPOUT) / ca
    h_port = max(min(h_req, h_wall), 0.6)
    port_x_span = min(Sx - 2.0 * EDGE_MARGIN, core_h + 2.0 * PORT_X_MARGIN)
    port_x_half = 0.5 * port_x_span

    # --- source ---------------------------------------------------------------
    y_src = -s_place * sa
    z_src = -s_place * ca
    src_y_half_req = 0.5 * w / ca + SRC_Y_MARGIN
    src_y_half_lim = abs(y_src) - 0.5 * t_wall - WALL_KEEPOUT
    src_y_half = max(min(src_y_half_req, src_y_half_lim), 0.5 * w / ca)
    src_x_span = min(Sx - 2.0 * EDGE_MARGIN, core_h + 2.0 * SRC_X_MARGIN)

    # --- domain ---------------------------------------------------------------
    y_need = max(y_port + h_port * ca, abs(y_src) + src_y_half) + EDGE_MARGIN
    z_need = max(z_port + h_port * sa, abs(z_src)) + EDGE_MARGIN
    Sy = max(S_MIN, math.ceil(2.0 * y_need / 0.2) * 0.2)
    Sz = max(S_MIN, math.ceil(2.0 * z_need / 0.2) * 0.2)
    Y_min, Y_max = -0.5 * Sy, 0.5 * Sy
    Z_min, Z_max = -0.5 * Sz, 0.5 * Sz

    # --- mesh -----------------------------------------------------------------
    dx_core = dx
    if CORE_REFINE:
        dx_core = max(min(dx, core_h / CORE_CELLS_MIN), 0.008)
    refine_span = core_h + 2.0 * CORE_REFINE_PAD
    n_x = Sx / dx
    if CORE_REFINE and dx_core < dx:
        n_x += refine_span * (1.0 / dx_core - 1.0 / dx)
    n_x = int(round(n_x)) + 1
    n_y = int(round(Sy / dx)) + 1
    n_z = int(round(Sz / dx)) + 1
    cells = n_x * n_y * n_z

    # --- waveguides (Switch.py: rect rotated about x by -alpha_k) -------------
    guide_len = 2.0 * max(Sy, Sz)
    guides = []
    for k, a_k in enumerate((alpha, -alpha)):
        guides.append(dict(
            name=f"wg{k}",
            alpha_deg=float(a_k),
            rotation_1=float(-a_k),
            first_axis="x",
            center=[x_core, 0.0, 0.0],
            span=[core_h, w, guide_len],
            axis_unit=[0.0, math.sin(math.radians(a_k)), math.cos(math.radians(a_k))],
            index=list(SIN_N),
        ))

    # --- LC wall (Switch.py add_lc_wall) --------------------------------------
    # Switch.py spans the core AND the whole top cladding, so inside the wall the SiN
    # core is REPLACED by a tall LC block.  The barrier is then a thick LC slab whose
    # vertical mode is far taller than the core mode -- the dominant transmit-state
    # loss (see the barrier diagnostic below).  SWITCH3D_WALL_HEIGHT_UM shortens it.
    wall_height = WALL_HEIGHT if WALL_HEIGHT > 0.0 else core_h + TOX_H
    wall_height = min(wall_height, x_tox_top - x_core_bot)   # LC fills the trench, air above
    wall = dict(
        name="LC_wall",
        center=[x_core_bot + 0.5 * wall_height, 0.0, 0.0],
        span=[wall_height, t_wall, math.sqrt(2.0) * max(Sy, Sz)],
        x_range=[x_core_bot, x_core_bot + wall_height],
        height=float(wall_height),
        rotation_1=0.0,
        mesh_order=1,
    )

    # --- ports ----------------------------------------------------------------
    def port(name, guide_idx, out_side):
        a_k = alpha if guide_idx == 0 else -alpha
        ck, sk = math.cos(math.radians(a_k)), math.sin(math.radians(a_k))
        d_hat = np.array([0.0, sk, ck])
        n_hat = np.array([0.0, ck, -sk])
        sgn = 1.0 if out_side else -1.0
        centre = sgn * s_place * d_hat
        centre[0] = x_core
        y_lo = centre[1] - h_port * abs(ck)
        y_hi = centre[1] + h_port * abs(ck)
        z_lo = centre[2] - h_port * abs(sk)
        z_hi = centre[2] + h_port * abs(sk)
        # Monitor = axis-aligned bounding box of the tilted plane + 2 cells of margin,
        # with a floor so every axis keeps >= 7 samples for the interpolation.
        mon_y = max(2.0 * h_port * abs(ck) + 4.0 * dx, 6.0 * dx)
        mon_z = max(2.0 * h_port * abs(sk) + 4.0 * dx, 6.0 * dx)
        mon_x = min(port_x_span + 4.0 * dx, Sx - 2.0 * dx)
        mon_cells = ((int(round(mon_x / dx)) + 1) * (int(round(mon_y / dx)) + 1)
                     * (int(round(mon_z / dx)) + 1))
        return dict(
            name=name, guide=guide_idx, alpha_deg=float(a_k),
            outward=bool(out_side), sign=(1.0 if out_side else -1.0),
            center=[float(v) for v in centre],
            d_hat=[float(v) for v in d_hat],
            n_hat=[float(v) for v in n_hat],
            half_len=float(h_port), half_x=float(port_x_half),
            plane_y_range=[float(y_lo), float(y_hi)],
            plane_z_range=[float(z_lo), float(z_hi)],
            monitor_center=[x_core, float(centre[1]), float(centre[2])],
            monitor_span=[float(mon_x), float(mon_y), float(mon_z)],
            monitor_cells=int(mon_cells),
        )

    ports = {
        "through": port("port_through", 0, True),
        "mirror": port("port_mirror", 1, True),
    }
    if ALL_PORTS:
        ports["back"] = port("port_back", 0, False)
        ports["idle"] = port("port_idle", 1, False)

    monitor_cells = sum(p["monitor_cells"] for p in ports.values())
    geom = dict(
        candidate=dict(cand),
        alpha_deg=alpha, theta=[alpha, -alpha], w_top=w, core_h=core_h, t_wall=t_wall,
        BOX_h=BOX_H, TOX_h=TOX_H, Sx=Sx, Sy=Sy, Sz=Sz,
        X_min=X_min, X_max=X_max, Y_min=Y_min, Y_max=Y_max, Z_min=Z_min, Z_max=Z_max,
        x_core_bot=x_core_bot, x_core_top=x_core_top, x_core=x_core,
        box_x_range=[X_min, x_core_bot], tox_x_range=[x_core_top, x_tox_top],
        air_x_range=([x_tox_top, X_max] if AIR_H > 0.0 else None),
        x_tox_top=x_tox_top, AIR_h=AIR_H,
        resolution=RESOLUTION, dx=dx, dx_core=dx_core,
        core_refine=bool(CORE_REFINE and dx_core < dx),
        refine_x_span=refine_span,
        n_x=n_x, n_y=n_y, n_z=n_z, cells=int(cells),
        fdtd_mem_GiB=cells * BYTES_PER_CELL / 1024 ** 3,
        monitor_cells=int(monitor_cells),
        monitor_mem_GiB=monitor_cells * 6 * 16 / 1024 ** 3,
        guides=guides, wall=wall, ports=ports,
        s_clear=float(s_clear), s_place=float(s_place),
        source=dict(
            name="source",
            injection_axis="z-axis", direction="forward",
            theta_deg=float(-alpha), phi_deg=-90.0,
            center=[x_core, float(y_src), float(z_src)],
            span=[float(src_x_span), float(2.0 * src_y_half), 0.0],
            k_hat=[0.0, float(sa), float(ca)],
        ),
        over_budget=bool(cells > MAX_CELLS),
    )
    return geom


def format_geometry(geom):
    c = geom["candidate"]
    L = []
    L.append(f"=== {c.get('label', '?')}  t={geom['core_h']:.3f} um  w={geom['w_top']:.2f} um  "
             f"alpha=+-{geom['alpha_deg']:.2f} deg  t_wall={geom['t_wall']:.3f} um "
             f"[{c.get('variant', '')}] ===")
    L.append(f"  Stage A     : n_vert={c['n_v']:.4f}  neff_inplane={c['neff']:.4f}  "
             f"neff*cos(alpha)={c['x_tan']:.4f} in ({N_O}, {N_E})")
    L.append(f"                margin={c['margin_deg']:.2f} deg  spread={c['spread_deg']:.2f} deg  "
             f"score={c['score']:.3f}  FP order m={c['fp_order']}  "
             f"tunneling min={c['t_tunnel_min']:.3f} um"
             + ("  [wall BELOW tunneling minimum]" if c.get("below_tunnel_min") else ""))
    t_uni = fp_wall_thickness_uniaxial(c["x_tan"], max(c["fp_order"], 1))
    if t_uni:
        L.append(f"  FP check    : isotropic-n_e resonance {fp_wall_thickness(c['x_tan'], max(c['fp_order'],1)):.3f} um "
                 f"vs UNIAXIAL (extraordinary) resonance {t_uni:.3f} um "
                 f"-> this wall is detuned by {abs(geom['t_wall'] - t_uni):.3f} um "
                 f"({100 * abs(geom['t_wall'] - t_uni) / (t_uni / max(c['fp_order'], 1)):.0f}% of a half-period)")
    L.append(f"  domain      : Sx={geom['Sx']:.3f}  Sy={geom['Sy']:.3f}  Sz={geom['Sz']:.3f} um   "
             f"x[{geom['X_min']:.3f},{geom['X_max']:.3f}] "
             f"y[{geom['Y_min']:.3f},{geom['Y_max']:.3f}] "
             f"z[{geom['Z_min']:.3f},{geom['Z_max']:.3f}]")
    L.append(f"  layers (x)  : BOX  [{geom['box_x_range'][0]:.4f}, {geom['box_x_range'][1]:.4f}] n={N_SIO2}")
    L.append(f"                core [{geom['x_core_bot']:.4f}, {geom['x_core_top']:.4f}] n={N_SIN} "
             f"(centre {geom['x_core']:.4f})")
    L.append(f"                TOX  [{geom['tox_x_range'][0]:.4f}, {geom['tox_x_range'][1]:.4f}] n={N_SIO2}")
    for g in geom["guides"]:
        L.append(f"  {g['name']:<11}: rect centre={_v(g['center'])} span={_v(g['span'])} "
                 f"first axis=x rotation_1={g['rotation_1']:+.2f} deg  axis d_hat={_v(g['axis_unit'])}")
    wall = geom["wall"]
    L.append(f"  LC wall     : rect centre={_v(wall['center'])} span={_v(wall['span'])} "
             f"rotation_1={wall['rotation_1']:+.2f} deg mesh order={wall['mesh_order']} "
             f"x[{wall['x_range'][0]:.4f},{wall['x_range'][1]:.4f}] height={wall['height']:.3f} um")
    b = barrier_indices(geom["core_h"], wall["height"])
    x_tan = c["x_tan"]
    nr, sr = b["reflect"]
    nt, st = b["transmit"]
    _nc, sc = b["core"]
    L.append(f"  barrier     : vertical n_eff  reflect={_f(nr)}  transmit={_f(nt)}   "
             f"vs x_tan={x_tan:.4f}")
    L.append(f"                -> reflect {'TIR (good)' if (nr and nr < x_tan) else 'NOT TIR (bad)'}"
             f", transmit {'propagating (good)' if (nt and nt > x_tan) else 'TIR TOO (cannot transmit)'}")
    if sc and st:
        L.append(f"                mode height: core {sc:.2f} um vs transmit barrier {st:.2f} um "
                 f"-> vertical mismatch factor {st / sc:.2f}"
                 + ("  [severe: dominates the transmit loss]" if st / sc > 1.8 else ""))
    src = geom["source"]
    L.append(f"  source      : mode, injection {src['injection_axis']}, theta={src['theta_deg']:+.2f} "
             f"phi={src['phi_deg']:+.1f} -> k_hat={_v(src['k_hat'])}")
    L.append(f"                centre={_v(src['center'])} span={_v(src['span'])}")
    L.append(f"  arm         : guide edge touches wall at s={geom['s_clear']:.3f} um, "
             f"source/ports placed at s={geom['s_place']:.3f} um "
             f"(lateral clearance {WALL_GAP:g} um)")
    for name, p in geom["ports"].items():
        L.append(f"  port {name:<7}: centre={_v(p['center'])} normal d_hat={_v(p['d_hat'])} "
                 f"in-plane n_hat={_v(p['n_hat'])} sign={p['sign']:+.0f}")
        L.append(f"                half_len={p['half_len']:.3f} um  half_x={p['half_x']:.3f} um  "
                 f"y[{p['plane_y_range'][0]:.3f},{p['plane_y_range'][1]:.3f}] "
                 f"z[{p['plane_z_range'][0]:.3f},{p['plane_z_range'][1]:.3f}]")
        L.append(f"                monitor bbox centre={_v(p['monitor_center'])} "
                 f"span={_v(p['monitor_span'])} -> {p['monitor_cells'] / 1e6:.2f} Mpts")
    L.append(f"  mesh        : dx=dy=dz={geom['dx']:.4f} um"
             + (f"; core-band dx={geom['dx_core']:.4f} um over "
                f"{geom['refine_x_span']:.3f} um" if geom["core_refine"] else "; no core refinement"))
    L.append(f"  cells       : {geom['n_x']} x {geom['n_y']} x {geom['n_z']} = "
             f"{geom['cells'] / 1e6:.2f} Mcells   est. FDTD mem "
             f"{geom['fdtd_mem_GiB']:.2f} GiB   port monitors "
             f"{geom['monitor_mem_GiB']:.2f} GiB")
    if geom["over_budget"]:
        L.append(f"  ** OVER BUDGET: {geom['cells'] / 1e6:.1f} Mcells > "
                 f"SWITCH3D_MAX_CELLS={MAX_CELLS / 1e6:.0f}M -> will be SKIPPED")
    return "\n".join(L)


def wall_mode_window(core_h):
    """(lo, hi) tangential-index window set by the WALL's OWN vertical modes.

    The wall is the etched trench filled with LC: a slab of height core_h + TOX_H
    standing on the buried oxide and capped by air (or by oxide when AIR_H = 0).
    Being finite and asymmetric, its mode indices sit BELOW the bulk LC indices, so
    the usable window is not (n_o, n_e):

        lo = highest mode index of the REFLECT-state wall (n_o fill)
             -> above it every wall mode is evanescent, which is what TIR needs
        hi = fundamental of the TRANSMIT-state wall (n_e fill)
             -> below it at least the fundamental propagates, which is what
                transmission needs

    Ignoring this is why a screen written on (n_o, n_e) predicted transmission the
    3D run did not deliver. Implementation is LC_design_search's, so there is one
    slab solver in the package, not two.
    """
    import LC_design_search as lds
    h = float(core_h) + TOX_H
    cover = N_AIR if AIR_H > 0.0 else N_SIO2
    refl = lds.slab_modes(N_O, h, N_SIO2, cover, tm=False, n_max_modes=16)
    trans = lds.slab_modes(N_E, h, N_SIO2, cover, tm=False, n_max_modes=16)
    if not refl or not trans:
        return (N_O, N_E)
    return (max(refl), max(trans))


def barrier_indices(core_h, wall_height):
    """Vertical effective index of the LC BARRIER region, per LC state.

    This is the quantity the switching condition actually depends on in 3D, and it is
    the one LC_valid_2D's model cannot see.  Inside the wall the SiN core is replaced
    by an LC slab of height `wall_height` on SiO2, so the barrier is not a bulk medium
    of index n_o / n_e: it is a slab whose own effective index is well below the bulk
    value.  The switching window is therefore

        n_eff_barrier(reflect) < n_eff_inplane * cos(alpha) < n_eff_barrier(transmit)

    not (n_o, n_e).  The guided E is parallel to the horizontal interfaces, so this is
    the TE slab.  Also returns the vertical scale of each mode: transmission needs the
    barrier mode to MATCH the core mode, which the (n_o, n_e) criterion never checks.
    """
    n_ref = _LC_BY_NAME["reflect"][1]     # n_yy in the reflect state
    n_tra = _LC_BY_NAME["transmit"][1]    # n_yy in the transmit state
    out = {}
    for name, n_slab in (("reflect", n_ref), ("transmit", n_tra)):
        neff = v2.slab_neff(wall_height, tm=False, n_core=n_slab, n_clad=N_SIO2)
        scale = None
        if neff is not None and neff > N_SIO2:
            scale = wall_height + 2.0 / (K0 * math.sqrt(max(neff ** 2 - N_SIO2 ** 2, 1e-30)))
        out[name] = (neff, scale)
    core_neff = v2.vertical_slab_neff(core_h)
    core_scale = None
    if core_neff is not None:
        core_scale = core_h + 2.0 / (K0 * math.sqrt(max(core_neff ** 2 - N_SIO2 ** 2, 1e-30)))
    out["core"] = (core_neff, core_scale)
    return out


def _v(vec):
    return "[" + ", ".join(f"{float(x):8.4f}" for x in vec) + "]"


def _f(val, fmt="{:.4f}"):
    return "cutoff" if val is None else fmt.format(float(val))


def check_geometry(geom):
    """Assertions a reviewer would otherwise have to make by hand."""
    problems = []
    for p in geom["ports"].values():
        d = np.asarray(p["d_hat"])
        n = np.asarray(p["n_hat"])
        if abs(float(np.dot(d, n))) > 1e-12:
            problems.append(f"{p['name']}: port normal and in-plane axis not orthogonal")
        if abs(np.linalg.norm(d) - 1.0) > 1e-12 or abs(np.linalg.norm(n) - 1.0) > 1e-12:
            problems.append(f"{p['name']}: non-unit port vectors")
        # the port plane must sit on the guide axis
        s = float(np.dot(np.asarray(p["center"]) * np.array([0.0, 1.0, 1.0]), d))
        pred = s * d
        tol = 1e-9 * (1.0 + abs(s))
        if abs(pred[1] - p["center"][1]) > tol or abs(pred[2] - p["center"][2]) > tol:
            problems.append(f"{p['name']}: centre is off the guide axis")
        # the plane must clear the LC wall and stay inside the domain
        half_w = 0.5 * geom["t_wall"]
        ylo, yhi = p["plane_y_range"]
        if ylo < half_w + 1e-9 and yhi > -half_w - 1e-9:
            problems.append(f"{p['name']}: port plane intersects the LC wall band")
        if min(abs(ylo), abs(yhi)) < half_w:
            problems.append(f"{p['name']}: port plane edge inside the LC wall")
        if (yhi > geom["Y_max"] - 1e-9 or ylo < geom["Y_min"] + 1e-9
                or p["plane_z_range"][1] > geom["Z_max"] - 1e-9
                or p["plane_z_range"][0] < geom["Z_min"] + 1e-9):
            problems.append(f"{p['name']}: port plane leaves the simulation domain")
        if p["half_x"] * 2.0 > geom["Sx"]:
            problems.append(f"{p['name']}: port plane taller than the stack")
    src = geom["source"]
    s_axis = np.asarray(geom["guides"][0]["axis_unit"])
    proj = float(np.dot(np.asarray(src["center"]) * np.array([0.0, 1.0, 1.0]), s_axis))
    if abs(proj * s_axis[1] - src["center"][1]) > 1e-9:
        problems.append("source centre is off the input guide axis")
    if src["center"][2] >= 0.0:
        problems.append("source is not on the input (z<0) side")
    if abs(src["center"][1]) - 0.5 * src["span"][1] < 0.5 * geom["t_wall"]:
        problems.append("source plane overlaps the LC wall band")
    return problems


# =============================================================================
# STAGE B -- 3D FDTD forward evaluation
# =============================================================================
def _msopt():
    import msopt as ms  # imported lazily so --dry-run never touches Lumerical
    return ms


def _drop_last_fsp(sim):
    """Delete the just-analyzed run's .fsp, its sibling directory and _p0.log.

    Same pattern as oled_common._drop_last_fsp: Lumerical writes ~200 MB of
    by-products per run and this filesystem is at 98%.  Results are already
    extracted by the time this is called.
    """
    if KEEP_FSP:
        return
    fsp = getattr(sim, "_last_run_fsp_path", None)
    if not fsp:
        return
    stem = fsp[:-4] if fsp.endswith(".fsp") else fsp
    for path in (fsp, f"{stem}_p0.log"):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    try:
        if os.path.isdir(stem):
            shutil.rmtree(stem, ignore_errors=True)
    except Exception:
        pass


def add_tilted_waveguide(sim, name, center, size, index, alpha):
    """Switch.py's add_tilted_waveguide: rectangle rotated about x by -alpha."""
    fdtd = sim.fdtd
    fdtd.addrect()
    fdtd.set("name", name)
    for dim, c, s in zip("xyz", center, size):
        fdtd.set(dim, float(c) * 1e-6)
        if s != 0:
            fdtd.set(f"{dim} span", float(s) * 1e-6)
    fdtd.set("first axis", "x")
    fdtd.set("rotation 1", -float(alpha))
    sim._set_object_index(index, object_name=name, material_name=f"SiN_{name}",
                          wavelength=WL)


def add_tilted_mode_source(sim, name, center, size, alpha):
    """Switch.py's add_tilted_mode_source (injection axis z, theta=-alpha, phi=-90)."""
    fdtd = sim.fdtd
    fdtd.addmode()
    fdtd.set("name", name)
    fdtd.set("injection axis", "z-axis")
    fdtd.set("direction", "forward")
    for dim, c, s in zip("xyz", center, size):
        fdtd.set(dim, float(c) * 1e-6)
        if s != 0:
            fdtd.set(f"{dim} span", float(s) * 1e-6)
    fdtd.set("theta", -float(alpha))
    fdtd.set("phi", -90)
    fdtd.set("wavelength start", WL * 1e-6)
    fdtd.set("wavelength stop", WL * 1e-6)
    try:
        fdtd.set("mode selection", "user select")
        fdtd.set("selected mode number", env_int("SWITCH3D_MODE_NUM", 0))
    except Exception as exc:
        print(f"[source] user-select mode failed ({exc}); falling back to fundamental")
        fdtd.set("mode selection", "fundamental mode")


def add_lc_wall(sim, geom, lc_index, name="LC_wall"):
    """Switch.py's add_lc_wall: un-rotated y-thin band, mesh order 1."""
    fdtd = sim.fdtd
    wall = geom["wall"]
    fdtd.addrect()
    fdtd.set("name", name)
    for dim, c, s in zip("xyz", wall["center"], wall["span"]):
        fdtd.set(dim, float(c) * 1e-6)
        fdtd.set(f"{dim} span", float(s) * 1e-6)
    fdtd.set("first axis", "x")
    fdtd.set("rotation 1", wall["rotation_1"])
    sim._set_object_index(lc_index, object_name=name, material_name=f"{name}_mat",
                          wavelength=WL)
    try:
        fdtd.set("override mesh order from material database", True)
        fdtd.set("mesh order", wall["mesh_order"])
    except Exception as exc:
        print(f"[LC_wall] could not set mesh order: {exc}")


def add_core_mesh_refinement(sim, geom):
    if not geom["core_refine"]:
        return
    fdtd = sim.fdtd
    fdtd.addmesh()
    fdtd.set("name", "core_x_mesh")
    fdtd.set("x", geom["x_core"] * 1e-6)
    fdtd.set("x span", geom["refine_x_span"] * 1e-6)
    fdtd.set("y", 0.0)
    fdtd.set("y span", geom["Sy"] * 1e-6)
    fdtd.set("z", 0.0)
    fdtd.set("z span", geom["Sz"] * 1e-6)
    fdtd.set("override x mesh", 1)
    fdtd.set("override y mesh", 0)
    fdtd.set("override z mesh", 0)
    fdtd.set("dx", geom["dx_core"] * 1e-6)


def add_port_monitors(sim, geom):
    fdtd = sim.fdtd
    try:
        fdtd.setglobalmonitor("frequency points", 1)
    except Exception:
        pass
    for p in geom["ports"].values():
        sim.add_monitor(name=p["name"], center=p["monitor_center"],
                        size=p["monitor_span"], N_f=1)
        for comp in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            try:
                fdtd.setnamed(p["name"], f"output {comp}", True)
            except Exception:
                pass


def add_field_map_monitor(sim, geom):
    """2D x-normal monitor on the plane through the middle of the core: the
    top view in which both arms, the LC wall and the crossing are all visible."""
    sim.add_monitor(name="field_map",
                    center=[geom["x_core"], 0.0, 0.0],
                    size=[0.0, geom["Sy"], geom["Sz"]], N_f=1)
    for comp in ("Hx", "Hy", "Hz"):
        try:
            sim.fdtd.setnamed("field_map", f"output {comp}", True)
        except Exception:
            pass


def grab_field_map(fdtd):
    r = fdtd.getresult("field_map", "H")
    y = np.ravel(np.asarray(r["y"], dtype=float)) * 1e6
    z = np.ravel(np.asarray(r["z"], dtype=float)) * 1e6
    H = np.asarray(r["H"], dtype=np.complex128)
    while H.ndim > 4:
        H = H[:, :, :, 0, :] if H.shape[3] == 1 else H[..., 0, :]
    H = np.squeeze(H)                      # -> (Ny, Nz, 3)
    return y, z, np.sum(np.abs(H) ** 2, axis=-1)


def build_sim(geom, lc_index=None, with_second_guide=True):
    """Assemble the initial CAD structure. lc_index=None -> no LC wall (NORM run)."""
    ms = _msopt()
    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[geom["Sx"], geom["Sy"], geom["Sz"]],
        resolution=geom["resolution"],
        unit=1e-6,
        background_index=N_SIO2,
        center_wl=WL,
        N_f=1,
    )
    add_core_mesh_refinement(sim, geom)

    # The background is oxide, so the air cap has to be an explicit block; without
    # it the PML would terminate an infinite oxide and the wall would be capped by
    # oxide instead of air.
    air = geom.get("air_x_range")
    if air is not None:
        sim.add_geo(
            center=[0.5 * (air[0] + air[1]), 0.0, 0.0],
            size=[air[1] - air[0], 2.0 * geom["Sy"], 2.0 * geom["Sz"]],
            index=[N_AIR, N_AIR, N_AIR],
            name="air_cap",
            wavelength=WL,
        )

    guides = geom["guides"] if with_second_guide else geom["guides"][:1]
    for g in guides:
        add_tilted_waveguide(sim, g["name"], g["center"], g["span"], g["index"],
                             g["alpha_deg"])
    if lc_index is not None:
        add_lc_wall(sim, geom, lc_index)

    src = geom["source"]
    add_tilted_mode_source(sim, src["name"], src["center"], src["span"],
                           geom["alpha_deg"])
    add_port_monitors(sim, geom)
    if FIELD_MAP:
        add_field_map_monitor(sim, geom)
    return sim


def grab_monitor(fdtd, name):
    """Return (x, y, z, E, H) in um / SI for a 3D DFT monitor."""
    resE = fdtd.getresult(name, "E")
    resH = fdtd.getresult(name, "H")
    x = np.ravel(np.asarray(resE["x"], dtype=float)) * 1e6
    y = np.ravel(np.asarray(resE["y"], dtype=float)) * 1e6
    z = np.ravel(np.asarray(resE["z"], dtype=float)) * 1e6
    E = np.asarray(resE["E"], dtype=np.complex128)
    H = np.asarray(resH["H"], dtype=np.complex128)
    # (Nx, Ny, Nz, Nf, 3) -> (Nx, Ny, Nz, 3) at the single frequency point
    if E.ndim == 5:
        E = E[:, :, :, 0, :]
        H = H[:, :, :, 0, :]
    for axis, arr in (("x", x), ("y", y), ("z", z)):
        if arr.size < 2:
            raise RuntimeError(f"monitor {name}: only {arr.size} sample(s) along {axis}")
    return x, y, z, E, H


def port_flux_3d(fields, center, d_hat, half_len, half_x, dx):
    """Poynting flux through the TILTED port plane, from an axis-aligned monitor.

    The plane is spanned by n_hat (in-plane, perpendicular to the guide axis) and
    x_hat (the layer normal); its surface normal is the guide axis d_hat.  DFT
    monitors cannot rotate, so E and H are interpolated onto the plane exactly as
    LC_valid_2D.port_flux does in 2D, one dimension higher.
    """
    x, y, z, E, H = fields
    d_hat = np.asarray(d_hat, dtype=float)
    n_hat = np.array([0.0, d_hat[2], -d_hat[1]])
    x_hat = np.array([1.0, 0.0, 0.0])

    n_s = min(int(np.ceil(4.0 * half_len / dx)) + 1, 801)
    n_u = min(int(np.ceil(4.0 * half_x / dx)) + 1, 401)
    s = np.linspace(-half_len, half_len, n_s)
    u = np.linspace(-half_x, half_x, n_u)
    S, U = np.meshgrid(s, u, indexing="ij")
    pts = (np.asarray(center, dtype=float)[None, None, :]
           + S[..., None] * n_hat[None, None, :]
           + U[..., None] * x_hat[None, None, :])
    flat = pts.reshape(-1, 3)

    vals = {}
    for tag, arr in (("E", E), ("H", H)):
        for k, comp in enumerate("xyz"):
            itp = RegularGridInterpolator((x, y, z), arr[..., k],
                                          bounds_error=False, fill_value=0.0)
            vals[tag + comp] = itp(flat).reshape(S.shape)

    Sx = 0.5 * np.real(vals["Ey"] * np.conj(vals["Hz"]) - vals["Ez"] * np.conj(vals["Hy"]))
    Sy = 0.5 * np.real(vals["Ez"] * np.conj(vals["Hx"]) - vals["Ex"] * np.conj(vals["Hz"]))
    Sz = 0.5 * np.real(vals["Ex"] * np.conj(vals["Hy"]) - vals["Ey"] * np.conj(vals["Hx"]))
    ds = float(s[1] - s[0]) * 1e-6
    du = float(u[1] - u[0]) * 1e-6
    flux = float(np.sum(Sx * d_hat[0] + Sy * d_hat[1] + Sz * d_hat[2]) * ds * du)

    e2 = np.array([float(np.sum(np.abs(vals[f"E{c}"]) ** 2)) for c in "xyz"])
    pol = e2 / max(float(e2.sum()), 1e-300)
    return flux, pol


def measure_ports(sim, geom):
    out = {}
    for key, p in geom["ports"].items():
        fields = grab_monitor(sim.fdtd, p["name"])
        flux, pol = port_flux_3d(fields, p["center"], p["d_hat"], p["half_len"],
                                 p["half_x"], geom["dx"])
        out[key] = dict(flux=p["sign"] * flux, pol=[float(v) for v in pol])
    return out


def run_one(geom, tag, lc_index, with_second_guide):
    sim = build_sim(geom, lc_index=lc_index, with_second_guide=with_second_guide)
    try:
        sim.run(name=os.path.join(design_dir, f"S3D_{tag}"), save=True)
        result = measure_ports(sim, geom)
        if FIELD_MAP:
            try:
                result["_field"] = grab_field_map(sim.fdtd)
            except Exception as exc:
                print(f"[field_map] {tag}: {exc}")
    finally:
        _drop_last_fsp(sim)
        try:
            sim.fdtd.close()
        except Exception:
            pass
    return result


def plot_field_maps(cand, geom, fields, res, path):
    """|H|^2 on the plane through the core, reflect state next to transmit state.

    Same structure the design-region plots use: physical um axes, the CAD outline
    of both arms and the LC wall drawn on top, and the measured numbers written on
    the panel they belong to. This is the CAD structure only -- no design region --
    so it shows what the bare crossing does before any optimization.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    have = [(n, f) for n, f in fields.items() if f is not None]
    if not have:
        return None
    fig, axes = plt.subplots(1, len(have), figsize=(7.4 * len(have), 6.4))
    axes = np.atleast_1d(axes)
    vmax = max(float(np.nanmax(f[2])) for _n, f in have)

    def arm_outline(alpha_deg, length, width):
        a = math.radians(alpha_deg)
        d = np.array([math.sin(a), math.cos(a)])       # (y, z) axis direction
        n = np.array([math.cos(a), -math.sin(a)])      # in-plane normal
        c = [d * length + n * width / 2, d * length - n * width / 2,
             -d * length - n * width / 2, -d * length + n * width / 2]
        return np.array(c)

    for ax, (state, (y, z, P)) in zip(axes, have):
        im = ax.pcolormesh(z, y, P, cmap="inferno", vmin=0.0, vmax=vmax, shading="auto")
        fig.colorbar(im, ax=ax, label="|H|^2  (a.u., common scale)")
        for al in geom["theta"]:
            ax.add_patch(Polygon(arm_outline(al, geom["Sz"], geom["w_top"])[:, ::-1],
                                 closed=True, fill=False, ec="cyan", lw=1.0, alpha=0.7))
        hw = 0.5 * geom["t_wall"]
        ax.add_patch(Polygon(np.array([[-geom["Sz"], -hw], [geom["Sz"], -hw],
                                       [geom["Sz"], hw], [-geom["Sz"], hw]]),
                             closed=True, fill=False, ec="lime", lw=1.4))
        for key, lab, col in (("through", "T", "white"), ("mirror", "R", "white")):
            p = geom["ports"][key]
            ax.plot(p["center"][2], p["center"][1], "o", ms=7, mfc="none", mec=col, mew=1.6)
            ax.annotate(lab, (p["center"][2], p["center"][1]), color=col,
                        fontsize=13, weight="bold", xytext=(6, 6),
                        textcoords="offset points")
        T = res.get(f"through_{state}", float("nan"))
        R = res.get(f"mirror_{state}", float("nan"))
        ax.text(0.02, 0.98,
                f"{state.upper()} state\n"
                f"T (through) = {T:.3f}\n"
                f"R (mirror)  = {R:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=11,
                family="monospace", color="white",
                bbox=dict(boxstyle="round", fc="black", alpha=0.55, ec="0.6"))
        ax.set_xlabel("z (um)   <- input   output ->")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal")
        ax.set_title(f"LC {state}")

    c = cand
    fig.suptitle(f"Bare crossing (CAD waveguides + LC wall, NO design region)   "
                 f"t={c['t']:g} um  w={c['w']:g} um  crossing={2*c['alpha']:g} deg  "
                 f"LC wall={c['t_wall']:.3f} um   |H|^2 on the plane through the core")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def evaluate_candidate(cand, geom):
    """3 forward simulations: NORM (input guide only) + transmit + reflect."""
    label = cand["label"]
    t0 = time.time()

    norm = run_one(geom, f"{label}_norm", lc_index=None, with_second_guide=False)
    p_ref = norm["through"]["flux"]
    print(f"[{label}] NORM P_ref={p_ref:.6e} W, mirror-port leak in NORM="
          f"{norm['mirror']['flux'] / max(abs(p_ref), 1e-300):+.4f}")
    pol = norm["through"]["pol"]
    print(f"[{label}] NORM through-port polarisation |Ex|^2:|Ey|^2:|Ez|^2 = "
          f"{pol[0]:.3f}:{pol[1]:.3f}:{pol[2]:.3f}")
    if pol[0] > 0.5:
        print(f"[{label}] WARNING: layer-normal E dominates -- this is the TM-like mode, "
              "which does not see the LC switching (n_yy). Check the source mode number.")
    if p_ref <= 0.0:
        raise RuntimeError(f"{label}: non-positive reference power {p_ref}")

    res = {"label": label, "P_ref": p_ref,
           "norm_pol": pol,
           "norm_mirror_leak": norm["mirror"]["flux"] / p_ref}
    field_maps = {}
    for state_name, _phi, lc_n in LC_STATES_3D:
        meas = run_one(geom, f"{label}_{state_name}", lc_index=lc_n,
                       with_second_guide=True)
        field_maps[state_name] = meas.pop("_field", None)
        for key, m in meas.items():
            res[f"{key}_{state_name}"] = m["flux"] / p_ref
        print(f"[{label}] {state_name:<8} T={res[f'through_{state_name}']:.4f}  "
              f"R={res[f'mirror_{state_name}']:.4f}")

    res["T"] = res["through_transmit"]
    res["R"] = res["mirror_reflect"]
    res["R_leak"] = res["mirror_transmit"]      # wrong port while transmitting
    res["T_leak"] = res["through_reflect"]      # wrong port while reflecting
    res["pass_T"] = bool(res["T"] >= T_TARGET)
    res["pass_R"] = bool(res["R"] >= R_TARGET)
    res["passed"] = bool(res["pass_T"] and res["pass_R"])
    res["seconds"] = time.time() - t0
    if FIELD_MAP and any(v is not None for v in field_maps.values()):
        fp = plot_field_maps(cand, geom, field_maps, res,
                             os.path.join(design_dir, f"S3D_{label}_field_map.png"))
        if fp:
            print(f"[{label}] wrote field map: {fp}")
            res["field_map_png"] = fp
    return res


# =============================================================================
# Reporting
# =============================================================================
def write_results(cands, geoms, results, skipped):
    rows = []
    header = (f"{'id':>4} {'t_um':>6} {'w_um':>6} {'alpha':>7} {'t_w':>6} {'m':>2} "
              f"{'T(tran)':>8} {'R(refl)':>8} {'R@tran':>8} {'T@refl':>8} "
              f"{'Mcells':>7} {'PASS':>5}")
    lines = [
        "# Stage B: 3D FDTD forward evaluation of the INITIAL CAD structure",
        "#   (two rotated SiN guides + LC wall; no inverse design, no optimizer)",
        f"# targets: T(transmit) >= {T_TARGET:.2f} AND R(reflect) >= {R_TARGET:.2f}",
        "# T = P_through/P_ref, R = P_mirror/P_ref; P_ref from a NORM run with only the",
        "# input waveguide, measured on the identical tilted through-port plane.",
        header,
    ]
    for c, g in zip(cands, geoms):
        r = results.get(c["label"])
        if r is None:
            reason = skipped.get(c["label"], "not run")
            lines.append(f"{c['label']:>4} {c['t']:>6.3f} {c['w']:>6.2f} {c['alpha']:>7.2f} "
                         f"{c['t_wall']:>6.3f} {c['fp_order']:>2d} "
                         f"{'-':>8} {'-':>8} {'-':>8} {'-':>8} "
                         f"{g['cells'] / 1e6:>7.1f} {'SKIP':>5}   ({reason})")
            continue
        lines.append(f"{c['label']:>4} {c['t']:>6.3f} {c['w']:>6.2f} {c['alpha']:>7.2f} "
                     f"{c['t_wall']:>6.3f} {c['fp_order']:>2d} "
                     f"{r['T']:>8.4f} {r['R']:>8.4f} {r['R_leak']:>8.4f} {r['T_leak']:>8.4f} "
                     f"{g['cells'] / 1e6:>7.1f} "
                     f"{'PASS' if r['passed'] else 'FAIL':>5}")
        rows.append((c, r))
    passing = [c["label"] for c, r in rows if r["passed"]]
    lines.append("")
    lines.append("PASS (T>=%.2f and R>=%.2f): %s" % (
        T_TARGET, R_TARGET, ", ".join(passing) if passing else "none"))
    text = "\n".join(lines)
    with open(os.path.join(design_dir, "Switch3D_results.txt"), "w", encoding="utf-8") as fp:
        fp.write(text + "\n")
    payload = dict(
        targets=dict(T=T_TARGET, R=R_TARGET),
        resolution=RESOLUTION, wavelength_um=WL,
        candidates=[
            dict(candidate=c,
                 geometry={k: v for k, v in g.items()
                           if k not in ("candidate", "guides", "wall", "ports", "source")},
                 ports={k: p for k, p in g["ports"].items()},
                 source=g["source"],
                 result=results.get(c["label"]),
                 skipped=skipped.get(c["label"]))
            for c, g in zip(cands, geoms)
        ],
    )
    with open(os.path.join(design_dir, "Switch3D_results.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=float)
    return text


def plot_results(cands, results, path):
    done = [(c, results[c["label"]]) for c in cands if c["label"] in results]
    if not done:
        return
    labels = [c["label"] for c, _ in done]
    idx = np.arange(len(done), dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(max(7.5, 1.5 * len(done)), 7.6),
                             constrained_layout=True, sharex=True)

    ax = axes[0]
    bw = 0.36
    ax.bar(idx - bw / 2, [r["T"] for _, r in done], bw, color=C_SERIES[0],
           label="T (transmit state)", linewidth=0)
    ax.bar(idx + bw / 2, [r["R"] for _, r in done], bw, color=C_SERIES[1],
           label="R (reflect state)", linewidth=0)
    ax.axhline(T_TARGET, color=C_SERIES[0], ls="--", lw=1.2)
    ax.axhline(R_TARGET, color=C_SERIES[1], ls="--", lw=1.2)
    ax.annotate(f"T target {T_TARGET:.2f}", (len(done) - 0.45, T_TARGET), color=C_MUTED,
                fontsize=8, va="bottom", ha="right")
    ax.annotate(f"R target {R_TARGET:.2f}", (len(done) - 0.45, R_TARGET), color=C_MUTED,
                fontsize=8, va="bottom", ha="right")
    for k, (_c, r) in enumerate(done):
        ax.annotate("PASS" if r["passed"] else "FAIL", (k, 1.03), ha="center",
                    fontsize=8, color=C_INK if r["passed"] else C_MUTED,
                    fontweight="bold" if r["passed"] else "normal")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("power fraction")
    ax.set_title("Initial (un-optimized) CAD structure: measured switching performance",
                 color=C_INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")

    ax = axes[1]
    ax.bar(idx - bw / 2, [r["R_leak"] for _, r in done], bw, color=C_SERIES[2],
           label="R while transmitting (wrong port)", linewidth=0)
    ax.bar(idx + bw / 2, [r["T_leak"] for _, r in done], bw, color=C_SERIES[3],
           label="T while reflecting (wrong port)", linewidth=0)
    ax.set_ylabel("power fraction")
    ax.set_xlabel("candidate")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{lab}\nt={c['t']:.2f} w={c['w']:.1f}\n"
                        f"a={c['alpha']:.1f} tw={c['t_wall']:.2f}"
                        for lab, (c, _r) in zip(labels, done)], fontsize=7.5)
    ax.set_title("Wrong-port leakage", color=C_INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=8.5, frameon=False)

    fig.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved geometry of every candidate and exit "
                         "without opening a Lumerical session")
    ap.add_argument("--stage-a-only", action="store_true",
                    help="run the analytic pre-screen only")
    ap.add_argument("-n", "--n-candidates", type=int, default=None)
    args = ap.parse_args(argv)
    dry_run = args.dry_run or env_flag("SWITCH3D_DRY_RUN", "0")

    print("=" * 78)
    print("Stage A: analytic pre-screen (effective-index method, no FDTD)")
    print("=" * 78)
    t0 = time.time()
    rows, info = prescreen()
    print(f"[stage A] {len(rows)} feasible (t, w, alpha) points in {time.time() - t0:.1f} s")
    for t, n_v, vmulti, note in info["t_notes"]:
        if note != "ok":
            print(f"[stage A] t={t:.3f} um: {note}"
                  + (f" (n_vert={n_v:.4f})" if n_v is not None else ""))
            break
    if info["multimode_from"] is not None:
        print(f"[stage A] vertical slab is MULTIMODE from t={info['multimode_from']:.3f} um "
              "upward; those thicknesses are rejected because the EIM single-mode "
              "reduction is invalid there")
    if not rows:
        print("[stage A] no switchable operating point in the swept grid; nothing to confirm")
        return 0

    table = write_prescreen_table(rows, info, os.path.join(design_dir, "Switch3D_prescreen.txt"))
    print("\n".join(table.splitlines()[:8 + 25]))

    spec = os.environ.get("SWITCH3D_CANDIDATES", "").strip()
    if spec:
        cands = candidates_from_spec(spec)
        print(f"\n[stage A] SWITCH3D_CANDIDATES overrides the ranked selection: "
              f"{len(cands)} explicit candidate(s)")
    else:
        cands = select_candidates(rows, n_candidates=args.n_candidates)
    plot_prescreen(rows, info, cands, os.path.join(design_dir, "Switch3D_prescreen.png"))
    print(f"\n[stage A] wrote {design_dir}Switch3D_prescreen.txt/.png")

    geoms = [resolve_geometry(c) for c in cands]
    geo_lines = []
    for c, g in zip(cands, geoms):
        block = format_geometry(g)
        problems = check_geometry(g)
        if problems:
            block += "\n  ** GEOMETRY PROBLEMS: " + "; ".join(problems)
        geo_lines.append(block)
    geo_text = "\n\n".join(geo_lines)
    with open(os.path.join(design_dir, "Switch3D_candidates.txt"), "w", encoding="utf-8") as fp:
        fp.write(geo_text + "\n")

    print()
    print("=" * 78)
    print("Resolved 3D geometry of the candidates")
    print("=" * 78)
    print(geo_text)
    total_cells = sum(g["cells"] for g in geoms if not g["over_budget"])
    print(f"\n[geometry] {len(geoms)} candidates, "
          f"{sum(1 for g in geoms if g['over_budget'])} over the "
          f"{MAX_CELLS / 1e6:.0f} Mcell budget; total {3 * total_cells / 1e6:.0f} Mcells "
          f"of FDTD work (3 runs per candidate)")

    if args.stage_a_only:
        return 0
    if dry_run:
        print("\n[dry run] geometry only -- no Lumerical session was opened.")
        return 0

    print()
    print("=" * 78)
    print("Stage B: 3D FDTD forward evaluation (norm + transmit + reflect per candidate)")
    print("=" * 78)
    results, skipped = {}, {}
    for c, g in zip(cands, geoms):
        if g["over_budget"]:
            skipped[c["label"]] = (f"{g['cells'] / 1e6:.1f} Mcells > SWITCH3D_MAX_CELLS="
                                   f"{MAX_CELLS / 1e6:.0f}M")
            print(f"[{c['label']}] SKIPPED: {skipped[c['label']]}")
            continue
        problems = check_geometry(g)
        if problems:
            skipped[c["label"]] = "geometry check failed: " + "; ".join(problems)
            print(f"[{c['label']}] SKIPPED: {skipped[c['label']]}")
            continue
        print(f"\n[{c['label']}] t={c['t']:.3f} w={c['w']:.2f} alpha={c['alpha']:.2f} "
              f"t_wall={c['t_wall']:.3f} -> {g['cells'] / 1e6:.1f} Mcells, "
              f"est. {g['fdtd_mem_GiB']:.2f} GiB")
        try:
            results[c["label"]] = evaluate_candidate(c, g)
            print(f"[{c['label']}] done in {results[c['label']]['seconds'] / 60:.1f} min "
                  f"-> {'PASS' if results[c['label']]['passed'] else 'FAIL'}")
        except Exception as exc:
            skipped[c["label"]] = f"FDTD failure: {exc}"
            print(f"[{c['label']}] FAILED: {exc}")

    text = write_results(cands, geoms, results, skipped)
    plot_results(cands, results, os.path.join(design_dir, "Switch3D_results.png"))
    print()
    print(text)
    print(f"\n[stage B] wrote {design_dir}Switch3D_results.txt/.json/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
