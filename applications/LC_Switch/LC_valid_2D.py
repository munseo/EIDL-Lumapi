"""LC_valid_2D: 2D validation of the symmetric X-crossing LC TIR switch.

Everything (code AND figures) lives in the Lumerical 2D simulation plane x-y
(Lumerical's 2D solver only supports the x-y plane; in the 3D Switch_2.py axes
this x corresponds to the physical propagation axis z, and the out-of-plane
axis here is the physical thickness direction).

Geometry:
    - the LC wall is a straight band ON the x axis (normal = y, never rotated);
    - two SiN waveguides cross at the origin at +-alpha from the x axis, so the
      whole structure is mirror-symmetric about both axes;
    - light enters on the (x<0, y>0) arm. At the wall it arrives with incidence
      theta = 90 - alpha from the wall normal:
          reflect  : n_eff * cos(alpha) > n_wall  ->  mirror arm  (x>0, y>0)
          transmit : n_eff * cos(alpha) < n_wall  ->  straight arm (x>0, y<0)

LC alignment angle phi is defined with the n_e (director) axis ALONG THE WALL
AXIS (sim x, physical z) at phi = 0. The wall blocks/passes the wave according
to the index seen by E || y (the p-polarized wave inside the wall propagates
nearly along the wall, E nearly along y):

    n_wall(phi) = n_e*n_o / sqrt(n_e^2 cos^2(phi) + n_o^2 sin^2(phi))

LOWEST (n_o = 1.5 -> reflect) at phi = 0, HIGHEST (n_e = 1.685 -> transmit) at
phi = 90 deg. The two extreme states are modelled with true diagonal
anisotropic tensors (same convention as Switch_2.py), not an isotropic average.
Only the in-plane-E polarization (Lumerical-2D "TE", textbook slab TM) sees the
LC rotation.

Port powers (reference-normalized, per spec):
    1) a NORM run places only the input waveguide in SiO2;
    2) E/H are extracted on the tilted port cross-section line (parallel to the
       waveguide cross-section -- DFT monitors cannot rotate, so the fields come
       from the full-area monitor by interpolation);
    3) the fields are projected onto the guide axis (rotation) and the Poynting
       flux through the line gives the reference power P_ref;
    4) the full X-crossing runs repeat the same extraction on the T/R port lines
       and report T = P_through/P_ref, R = P_mirror/P_ref (back/idle likewise).

Stages: A) analytic design map over (alpha, width) -> pick alpha;
B) FDTD width sweep at that alpha, both LC states each -> T*R vs width plot,
material-index section of the best combination, field maps with the port
cross-section lines overlaid.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import msopt as ms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
os.makedirs(design_dir, exist_ok=True)

C0 = 299792458.0


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_flag(name, default="1"):
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
WL = env_float("LC2D_WAVELENGTH_UM", 1.55)
N_SIN = 2.0
N_SIO2 = 1.44
N_O = 1.5      # LC ordinary
N_E = 1.685    # LC extraordinary
K0 = 2.0 * np.pi / WL

RESOLUTION = int(env_float("LC2D_RESOLUTION", 25))          # cells / um
S_DOM = env_float("LC2D_S_UM", 26.0)                        # square domain span
ISOLATION_TARGET = env_float("LC2D_TUNNEL_ISOLATION", 0.01)  # tolerated TIR leakage
SKIP_FDTD = env_flag("LC2D_SKIP_FDTD", "0")
FORCE_W = os.environ.get("LC2D_FORCE_WIDTH_UM")
FORCE_ALPHA = os.environ.get("LC2D_FORCE_ANGLE_DEG")        # waveguide angle from x axis
WIDTH_LIST = os.environ.get("LC2D_WIDTH_LIST")              # comma list, um

# LC states, diagonal tensors in the SIM axes [n_x, n_y, n_z(out-of-plane)]
LC_STATES = [
    ("reflect", 0.0, [N_E, N_O, N_O]),      # director || wall axis -> n_yy = n_o (blocks)
    ("transmit", 90.0, [N_O, N_E, N_O]),    # director || y         -> n_yy = n_e (passes)
]


# -----------------------------------------------------------------------------
# Stage A1: wall blocking index vs LC alignment (phi=0 -> n_e axis || wall axis)
# -----------------------------------------------------------------------------
def lc_wall_index(phi_deg):
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    return N_E * N_O / np.sqrt((N_E * np.cos(phi)) ** 2 + (N_O * np.sin(phi)) ** 2)


# -----------------------------------------------------------------------------
# Stage A2: symmetric-slab effective index (in-plane E = Lumerical-2D TE = slab TM)
# -----------------------------------------------------------------------------
def slab_neff(width_um, mode_m=0, tm=True, n_core=None, n_clad=None):
    """Symmetric-slab effective index.

    n_core defaults to CORE_N, the index the IN-PLANE (2D) problem should use:
    that is N_SIN for a slab of infinite vertical extent, or -- when a finite
    waveguide thickness is specified -- the vertical slab's effective index
    (effective-index method, see vertical_slab_neff).
    """
    n_core = CORE_N if n_core is None else float(n_core)
    n_clad = N_SIO2 if n_clad is None else float(n_clad)
    if n_core <= n_clad:
        return None
    lo, hi = n_clad + 1e-9, n_core - 1e-9
    ratio = (n_core / n_clad) ** 2 if tm else 1.0

    def resid(n):
        kap = K0 * np.sqrt(max(n_core ** 2 - n ** 2, 1e-30))
        gam = K0 * np.sqrt(max(n ** 2 - n_clad ** 2, 1e-30))
        return kap * width_um - 2.0 * np.arctan(ratio * gam / kap) - mode_m * np.pi

    if resid(lo) < 0.0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def vertical_slab_neff(thickness_um, mode_m=0):
    """Effective-index-method step 1: the SiN slab confined VERTICALLY.

    A 2D simulation has no vertical dimension, so waveguide thickness can only
    enter through this reduction: solve the vertical SiN/SiO2 slab, then run the
    in-plane problem with its effective index as the core index.  The guided E
    field lies in the device plane, i.e. PARALLEL to the top/bottom interfaces,
    so this slab is TE (no index-ratio factor) even though the in-plane problem
    is solved as TM.  Returns None below cutoff.
    """
    return slab_neff(thickness_um, mode_m, tm=False, n_core=N_SIN, n_clad=N_SIO2)


# In-plane core index. Without LC2D_THICKNESS_UM this is the bulk SiN index and
# every result is the original infinite-thickness 2D idealization.
CORE_N = N_SIN
_THICKNESS_UM = os.environ.get("LC2D_THICKNESS_UM")
if _THICKNESS_UM:
    _t = float(_THICKNESS_UM)
    _n_v = vertical_slab_neff(_t)
    if _n_v is None:
        raise ValueError(
            f"LC2D_THICKNESS_UM={_t} um is below the vertical slab cutoff "
            f"(SiN {N_SIN} / SiO2 {N_SIO2} at {WL} um): no guided mode exists."
        )
    CORE_N = _n_v
    print(f"[EIM] waveguide thickness {_t:.3f} um -> vertical n_eff = {CORE_N:.4f} "
          f"(used as the in-plane core index instead of N_SIN={N_SIN})")


def wall_thickness(x_tangential):
    """Tunneling-isolation minimum, snapped up to the transmit-state half-wave
    Fabry-Perot resonance (off-resonance walls bounce 15-20% of the transmitted
    power to the wrong port)."""
    gam = K0 * np.sqrt(max(x_tangential ** 2 - N_O ** 2, 1e-30))
    t_min = max(float(np.log(1.0 / ISOLATION_TARGET) / (2.0 * gam)), 0.8)
    th_t = np.arcsin(min(x_tangential / N_E, 1.0))
    kz = K0 * N_E * np.cos(th_t)
    m = max(1, int(np.ceil(t_min * kz / np.pi)))
    return float(m * np.pi / kz), m, t_min


# -----------------------------------------------------------------------------
# Stage A3: (alpha, width) design map. alpha = waveguide angle from the x axis,
# swept 20-60 deg per spec; incidence from the wall normal is theta = 90-alpha.
# Score = angular switch margin / beam angular spread (validated metric: the raw
# index margin misses that narrow guides have a wide angular spectrum).
# -----------------------------------------------------------------------------
def design_score_grid(n_core=None):
    """(alphas, widths, score) for a given in-plane core index.

    Factored out of design_map so the thickness sweep can re-score the whole
    design space for each candidate thickness without duplicating the metric.
    """
    alphas = np.arange(20.0, 60.01, 0.25)
    w_max = env_float("LC2D_MAX_WIDTH_UM", 4.0)
    widths = np.arange(0.40, w_max + 1e-9, 0.05)
    score = np.full((widths.size, alphas.size), np.nan)
    for i, w in enumerate(widths):
        neff = slab_neff(w, n_core=n_core)
        if neff is None:
            continue
        th_lo = np.arcsin(min(N_O / neff, 1.0))
        th_hi = np.arcsin(min(N_E / neff, 1.0))
        sigma = WL / (np.pi * neff * w)
        for j, a in enumerate(alphas):
            th = np.deg2rad(90.0 - a)
            margin = min(th - th_lo, th_hi - th)
            if margin > 0.0:
                score[i, j] = margin / sigma
    return alphas, widths, score


def design_map():
    alphas, widths, score = design_score_grid()

    if FORCE_ALPHA and FORCE_W:
        a_pick, w_pick = float(FORCE_ALPHA), float(FORCE_W)
    else:
        i, j = np.unravel_index(np.nanargmax(score), score.shape)
        w_pick, a_pick = float(widths[i]), float(alphas[j])

    neff = slab_neff(w_pick)
    x = neff * np.cos(np.deg2rad(a_pick))            # tangential index n_eff*sin(theta)
    t_wall, m_fp, t_min = wall_thickness(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(alphas, widths, score, shading="auto", cmap="viridis")
    ax.plot(a_pick, w_pick, "r*", ms=16, label=f"pick alpha={a_pick:.1f}deg, w={w_pick:.2f}um")
    ax.set_xlabel("waveguide angle alpha from x axis (deg); incidence theta = 90-alpha")
    ax.set_ylabel("waveguide width (um)")
    ax.set_title("normalized switch margin (angular margin / beam spread)")
    fig.colorbar(im, ax=ax, label="margin / spread")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(design_dir, "LC2D_design_map.png"), dpi=160)
    plt.close(fig)

    print(f"[design] alpha={a_pick:.1f} deg (incidence {90 - a_pick:.1f} deg), "
          f"w={w_pick:.2f} um -> neff={neff:.4f}, n_eff*cos(alpha)={x:.4f} in (1.5, 1.685)")
    print(f"[design] wall: FP order m={m_fp}, tunneling min {t_min:.2f} -> t={t_wall:.2f} um")
    return dict(alpha=a_pick, w=w_pick, neff=neff, x=x, t_wall=t_wall)


def thickness_sweep():
    """Sweep the waveguide THICKNESS through the effective-index method.

    A 2D simulation resolves only the in-plane width; thickness enters by
    reducing the vertical SiN/SiO2 slab to an effective core index, which then
    replaces N_SIN in the in-plane problem.  For each thickness the whole
    (alpha, width) design space is re-scored, so the sweep answers the design
    question directly: how thick must the guide be before a switchable operating
    point exists at all, and where does the margin stop improving?

    The switch needs n_eff_inplane * cos(alpha) to land between the LC ordinary
    and extraordinary indices, and n_eff_inplane < n_vertical_eff < N_SIN, so a
    thin guide simply cannot reach n_o = 1.5 -- that cutoff is the sweep's main
    result.
    """
    listed = os.environ.get("LC2D_THICKNESS_LIST")
    if listed:
        thicknesses = np.asarray([float(v) for v in listed.replace(";", ",").split(",") if v.strip()])
    else:
        t_lo = env_float("LC2D_THICKNESS_MIN_UM", 0.15)
        t_hi = env_float("LC2D_THICKNESS_MAX_UM", 0.90)
        t_step = env_float("LC2D_THICKNESS_STEP_UM", 0.025)
        thicknesses = np.arange(t_lo, t_hi + 1e-9, t_step)

    rows = []
    for t in thicknesses:
        n_v = vertical_slab_neff(float(t))
        row = dict(t=float(t), n_v=n_v, score=np.nan, alpha=np.nan, w=np.nan,
                   neff=np.nan, x=np.nan, t_wall=np.nan, n_v_multimode=False)
        if n_v is not None:
            row["n_v_multimode"] = vertical_slab_neff(float(t), mode_m=1) is not None
            _alphas, _widths, score = design_score_grid(n_core=n_v)
            if np.any(np.isfinite(score)):
                i, j = np.unravel_index(np.nanargmax(score), score.shape)
                w_b, a_b = float(_widths[i]), float(_alphas[j])
                neff_b = slab_neff(w_b, n_core=n_v)
                x_b = neff_b * np.cos(np.deg2rad(a_b))
                t_wall_b = wall_thickness(x_b)[0] if x_b > N_O else np.nan
                row.update(score=float(score[i, j]), alpha=a_b, w=w_b,
                           neff=float(neff_b), x=float(x_b), t_wall=float(t_wall_b))
        rows.append(row)

    t_arr = np.array([r["t"] for r in rows])
    nv_arr = np.array([r["n_v"] if r["n_v"] is not None else np.nan for r in rows])
    sc_arr = np.array([r["score"] for r in rows])
    feasible = np.isfinite(sc_arr)

    path_txt = os.path.join(design_dir, "LC2D_thickness_sweep.txt")
    with open(path_txt, "w", encoding="utf-8") as fp:
        fp.write(f"# effective-index-method thickness sweep, lambda={WL} um, "
                 f"SiN={N_SIN}, SiO2={N_SIO2}, LC n_o={N_O} n_e={N_E}\n")
        fp.write("# score = angular switch margin / beam angular spread (higher is better)\n")
        fp.write("# nan score = no (alpha, width) gives a switchable operating point\n")
        fp.write(f"{'t_um':>7}{'n_vert':>9}{'vmulti':>8}{'score':>9}{'alpha':>8}"
                 f"{'w_um':>8}{'neff_ip':>9}{'n*cos':>8}{'t_wall':>8}\n")
        for r in rows:
            nv = r["n_v"] if r["n_v"] is not None else float("nan")
            fp.write(f"{r['t']:7.3f}{nv:9.4f}{int(r['n_v_multimode']):8d}"
                     f"{r['score']:9.3f}{r['alpha']:8.2f}{r['w']:8.2f}"
                     f"{r['neff']:9.4f}{r['x']:8.4f}{r['t_wall']:8.3f}\n")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t_arr, nv_arr, lw=2, color="tab:blue")
    ax.axhline(N_O, color="tab:red", ls="--", lw=1.2, label=f"LC n_o={N_O} (switch needs above)")
    ax.axhline(N_E, color="tab:green", ls="--", lw=1.2, label=f"LC n_e={N_E}")
    ax.axhline(N_SIO2, color="0.5", ls=":", lw=1.0, label=f"SiO2={N_SIO2}")
    ax.set_xlabel("waveguide thickness (um)")
    ax.set_ylabel("vertical slab n_eff")
    ax.set_title("EIM step 1: vertical confinement")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(t_arr, sc_arr, lw=2, color="tab:purple")
    if np.any(feasible):
        t_min_ok = float(t_arr[feasible][0])
        ax.axvline(t_min_ok, color="tab:red", ls="--", lw=1.2,
                   label=f"thinnest switchable t={t_min_ok:.3f} um")
        i_best = int(np.nanargmax(sc_arr))
        ax.plot(t_arr[i_best], sc_arr[i_best], "r*", ms=15,
                label=f"best t={t_arr[i_best]:.3f} um (score {sc_arr[i_best]:.2f})")
        ax.legend(fontsize=7)
    ax.set_xlabel("waveguide thickness (um)")
    ax.set_ylabel("best margin / spread")
    ax.set_title("Best achievable switch margin vs thickness")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_arr, [r["alpha"] for r in rows], lw=2, color="tab:orange", label="best alpha (deg)")
    ax.set_xlabel("waveguide thickness (um)")
    ax.set_ylabel("best alpha (deg)", color="tab:orange")
    ax.tick_params(axis="y", labelcolor="tab:orange")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(t_arr, [r["w"] for r in rows], lw=2, ls="--", color="tab:blue", label="best width (um)")
    ax2.set_ylabel("best in-plane width (um)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_title("Operating point vs thickness")

    ax = axes[1, 1]
    ax.plot(t_arr, [r["t_wall"] for r in rows], lw=2, color="tab:brown")
    ax.set_xlabel("waveguide thickness (um)")
    ax.set_ylabel("LC wall thickness (um)")
    ax.set_title("Required wall thickness (tunneling + FP resonance)")
    ax.grid(alpha=0.3)

    fig.suptitle("Waveguide thickness sweep via the effective-index method "
                 "(2D approximation)", fontsize=12)
    path_png = os.path.join(design_dir, "LC2D_thickness_sweep.png")
    fig.savefig(path_png, dpi=160)
    plt.close(fig)

    if np.any(feasible):
        i_best = int(np.nanargmax(sc_arr))
        b = rows[i_best]
        print(f"[thickness] switchable from t={float(t_arr[feasible][0]):.3f} um; "
              f"best t={b['t']:.3f} um -> vertical n_eff={b['n_v']:.4f}, "
              f"alpha={b['alpha']:.1f} deg, w={b['w']:.2f} um, score={b['score']:.2f}")
        multi = [r["t"] for r in rows if r["n_v_multimode"]]
        if multi:
            print(f"[thickness] vertically MULTIMODE above t={min(multi):.3f} um "
                  "(the EIM single-mode reduction stops being valid there)")
    else:
        print("[thickness] no thickness in the swept range gives a switchable operating point")
    print(f"[thickness] wrote {path_png} and {path_txt}")
    return rows


def save_analytic_plots(pick):
    phi = np.linspace(0.0, 90.0, 181)
    n_phi = lc_wall_index(phi)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(phi, n_phi, lw=2)
    ax.axhline(pick["x"], color="tab:red", ls="--", lw=1.2,
               label=f"n_eff*cos(alpha)={pick['x']:.3f} (wall index below -> reflect)")
    phi_c = float(phi[np.argmin(np.abs(n_phi - pick["x"]))])
    ax.axvline(phi_c, color="tab:gray", ls=":", lw=1.0, label=f"switch at phi ~{phi_c:.0f} deg")
    ax.set_xlabel("LC alignment phi (deg; 0 = n_e axis along the wall axis)")
    ax.set_ylabel("wall blocking index (seen by E || y)")
    ax.set_title(f"LC wall index vs alignment (n_o={N_O}, n_e={N_E}): reflect@0, transmit@90")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(design_dir, "LC2D_lc_index_vs_angle.png"), dpi=160)
    plt.close(fig)

    widths = np.arange(0.30, max(1.51, pick["w"] + 0.5), 0.02)
    tm0 = [slab_neff(w, 0) for w in widths]
    tm1 = [slab_neff(w, 1) for w in widths]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(widths, [v if v else np.nan for v in tm0], lw=2, label="TM0 (Lumerical-2D TE)")
    ax.plot(widths, [v if v else np.nan for v in tm1], lw=1.2, ls="--", label="TM1")
    ax.axvline(pick["w"], color="tab:red", ls=":", label=f"picked w={pick['w']:.2f} um")
    ax.set_xlabel("SiN width (um)")
    ax.set_ylabel("effective index")
    ax.set_title(f"core n={CORE_N:.4f} (SiN {N_SIN})/SiO2({N_SIO2}) slab, lambda={WL} um")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(design_dir, "LC2D_neff_vs_width.png"), dpi=160)
    plt.close(fig)
    return phi_c


# -----------------------------------------------------------------------------
# Tilted-port flux: integrate the Poynting component along the guide axis on a
# line PARALLEL TO THE GUIDE CROSS-SECTION (perpendicular to the axis). DFT
# monitors cannot rotate, so E and H come from the full-area field monitor.
# -----------------------------------------------------------------------------
def port_flux(fields, center, d_hat, half_len):
    x, y, E, H = fields
    n_hat = np.array([-d_hat[1], d_hat[0]])
    s = np.linspace(-half_len, half_len, 501)
    pts = center[None, :] + s[:, None] * n_hat[None, :]
    vals = {}
    for tag, arr in (("E", E), ("H", H)):
        for k, comp in enumerate("xyz"):
            itp = RegularGridInterpolator((x, y), arr[:, :, k],
                                          bounds_error=False, fill_value=0.0)
            vals[tag + comp] = itp(pts)
    Sx = 0.5 * np.real(vals["Ey"] * np.conj(vals["Hz"]) - vals["Ez"] * np.conj(vals["Hy"]))
    Sy = 0.5 * np.real(vals["Ez"] * np.conj(vals["Hx"]) - vals["Ex"] * np.conj(vals["Hz"]))
    dl = float(s[1] - s[0]) * 1e-6
    return float(np.sum(Sx * d_hat[0] + Sy * d_hat[1]) * dl)


def read_sourcepower(fdtd):
    f = C0 / (WL * 1e-6)
    try:
        return float(np.real(np.asarray(fdtd.sourcepower(f)).reshape(-1)[0]))
    except Exception:
        fdtd.putv("lc2d_f", np.asarray([f]))
        fdtd.eval("lc2d_sp = sourcepower(lc2d_f);")
        return float(np.real(np.asarray(fdtd.getv("lc2d_sp")).reshape(-1)[0]))


def port_geometry(alpha, w, t_wall):
    """Shared port-line layout for the norm and full runs (identical lines so the
    normalization cancels line-geometry effects). Each entry: (center, guide
    axis direction, outflow sign, half-length of the cross-section line)."""
    a_r = np.deg2rad(alpha)
    ca, sa = float(np.cos(a_r)), float(np.sin(a_r))
    d1 = np.array([ca, -sa])       # guide 1 axis (input -> through)
    d2 = np.array([ca, +sa])       # guide 2 axis (idle -> mirror)
    ds = 0.5 * S_DOM - 3.5         # source |x| offset; source ARM distance = ds/ca
    r_out = 0.5 * S_DOM - 1.2
    r_in = ds / ca + 0.9           # behind the source along the arm

    def half(center):
        # keep line endpoints clear of the wall and of the PML band (~0.3 um
        # inside the span edge; fields there are damped and fake ~1% fluxes)
        lim_wall = (abs(center[1]) - 0.5 * t_wall - 0.4) / ca
        lim_dom = (0.5 * S_DOM - 0.6 - abs(center[0])) / sa
        return max(min(2.0 * w + 1.0, lim_wall, lim_dom), 0.8)

    ports = {}
    for pname, cen, dh, sign in (
        ("through", np.array([r_out * ca, -r_out * sa]), d1, +1.0),
        ("mirror", np.array([r_out * ca, +r_out * sa]), d2, +1.0),
        ("back", np.array([-r_in * ca, +r_in * sa]), d1, -1.0),
        ("idle", np.array([-r_in * ca, -r_in * sa]), d2, -1.0),
    ):
        ports[pname] = (cen, dh, sign, half(cen))
    return ports, ds


def add_tilted_mode_source(fdtd, alpha, w, t_wall, ds):
    a_r = np.deg2rad(alpha)
    um = 1e-6
    x_src, y_src = -ds, ds * np.tan(a_r)
    span_max = 2.0 * (y_src - 0.5 * t_wall - 0.6)
    src_span = min((4.0 * w + 2.0) / np.cos(a_r), span_max)
    fdtd.addmode()
    fdtd.set("name", "source")
    fdtd.set("injection axis", "x-axis")
    fdtd.set("direction", "forward")
    fdtd.set("theta", -alpha)      # mode-source rotation property is "theta" in 2D
    fdtd.set("x", x_src * um)
    fdtd.set("y", y_src * um)
    fdtd.set("y span", src_span * um)
    fdtd.set("wavelength start", WL * um)
    fdtd.set("wavelength stop", WL * um)
    try:
        fdtd.set("mode selection", "fundamental TE mode")
    except Exception:
        fdtd.set("mode selection", "user select")
        fdtd.set("selected mode number", int(env_float("LC2D_MODE_NUM", 1)))


def grab_fields(fdtd):
    resE = fdtd.getresult("mon_field", "E")
    resH = fdtd.getresult("mon_field", "H")
    x_ax = np.ravel(np.asarray(resE["x"], dtype=float)) * 1e6
    y_ax = np.ravel(np.asarray(resE["y"], dtype=float)) * 1e6
    E = np.squeeze(np.asarray(resE["E"], dtype=np.complex128))
    H = np.squeeze(np.asarray(resH["H"], dtype=np.complex128))
    return (x_ax, y_ax, E, H)


def run_norm(alpha, w, t_wall, tag):
    """Reference run per spec: ONLY the input waveguide in the SiO2 background.
    P_ref = axis-projected E/H flux on the same cross-section line the through
    port uses in the full structure."""
    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[S_DOM, S_DOM, 0.0],
        resolution=RESOLUTION,
        unit=1e-6,
        background_index=N_SIO2,
        center_wl=WL,
        N_f=1,
    )
    fdtd = sim.fdtd
    um = 1e-6
    fdtd.addrect()
    fdtd.set("name", "wg_in_only")
    fdtd.set("x", 0.0)
    fdtd.set("y", 0.0)
    fdtd.set("x span", 1.6 * S_DOM * um)
    fdtd.set("y span", w * um)
    fdtd.set("first axis", "z")
    fdtd.set("rotation 1", -alpha)
    sim._set_object_index([CORE_N] * 3, object_name="wg_in_only",
                          material_name="SiN_norm", wavelength=WL)

    ports, ds = port_geometry(alpha, w, t_wall)
    add_tilted_mode_source(fdtd, alpha, w, t_wall, ds)
    fdtd.setglobalmonitor("frequency points", 1)
    sim.add_monitor(name="mon_field", center=[0, 0, 0], size=[S_DOM, S_DOM, 0], N_f=1)
    for comp in ("Hx", "Hy", "Hz"):
        try:
            fdtd.setnamed("mon_field", f"output {comp}", True)
        except Exception:
            pass
    sim.run(name=os.path.join(design_dir, f"LC2D_norm_{tag}"), save=True)

    fields = grab_fields(fdtd)
    cen, dh, _, h = ports["through"]
    p_ref = port_flux(fields, cen, dh, h)
    sp = read_sourcepower(fdtd)
    print(f"[norm {tag}] P_ref={p_ref:.6e} W/m, injection efficiency P_ref/sourcepower="
          f"{p_ref / max(sp, 1e-300):.4f}")
    try:
        fdtd.close()
    except Exception:
        pass
    if p_ref <= 0:
        raise RuntimeError(f"norm run {tag}: non-positive reference power {p_ref}")
    return p_ref


# -----------------------------------------------------------------------------
# Stage B: one FDTD case = (alpha, width, LC state)
# -----------------------------------------------------------------------------
def run_case(alpha, w, state, tag, p_ref, save_index_fig=True):
    state_name, phi_deg, lc_n = state
    a_r = np.deg2rad(alpha)
    ca, sa = float(np.cos(a_r)), float(np.sin(a_r))
    neff = slab_neff(w)
    x_tan = neff * ca
    t_wall, _, _ = wall_thickness(x_tan)

    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[S_DOM, S_DOM, 0.0],
        resolution=RESOLUTION,
        unit=1e-6,
        background_index=N_SIO2,
        center_wl=WL,
        N_f=1,
    )
    fdtd = sim.fdtd
    um = 1e-6
    guide_len = 1.6 * S_DOM

    # two crossing guides at +-alpha from the x axis (mirror pair)
    for name, rot in (("wg_in_thru", -alpha), ("wg_mirror", +alpha)):
        fdtd.addrect()
        fdtd.set("name", name)
        fdtd.set("x", 0.0)
        fdtd.set("y", 0.0)
        fdtd.set("x span", guide_len * um)
        fdtd.set("y span", w * um)
        fdtd.set("first axis", "z")
        fdtd.set("rotation 1", rot)
        sim._set_object_index([CORE_N] * 3, object_name=name,
                              material_name=f"SiN_{name}", wavelength=WL)

    # LC wall fixed on the x axis (no rotation), anisotropic diagonal tensor
    fdtd.addrect()
    fdtd.set("name", "LC_wall")
    fdtd.set("x", 0.0)
    fdtd.set("y", 0.0)
    fdtd.set("x span", (S_DOM + 4.0) * um)
    fdtd.set("y span", t_wall * um)
    sim._set_object_index(lc_n, object_name="LC_wall",
                          material_name=f"LC_{state_name}", wavelength=WL)
    fdtd.set("override mesh order from material database", True)
    fdtd.set("mesh order", 1)

    # mode source on the input arm (x<0, y>0), identical to the norm run
    ports, ds = port_geometry(alpha, w, t_wall)
    add_tilted_mode_source(fdtd, alpha, w, t_wall, ds)

    fdtd.setglobalmonitor("frequency points", 1)

    # full-area field monitor: source of both the images and the port fluxes
    sim.add_monitor(name="mon_field", center=[0, 0, 0], size=[S_DOM, S_DOM, 0], N_f=1)
    for comp in ("Hx", "Hy", "Hz"):
        try:
            fdtd.setnamed("mon_field", f"output {comp}", True)
        except Exception:
            pass

    fdtd.addindex()
    fdtd.set("name", "mon_index")
    fdtd.set("x", 0.0)
    fdtd.set("y", 0.0)
    fdtd.set("x span", S_DOM * um)
    fdtd.set("y span", S_DOM * um)

    sim.run(name=os.path.join(design_dir, f"LC2D_{tag}"), save=True)

    # --- tilted-port powers, normalized by the norm run's P_ref --------------
    fields = grab_fields(fdtd)
    E = fields[2]
    x_ax, y_ax = fields[0], fields[1]
    powers, lines = {}, []
    for pname, (cen, dh, sign, h) in ports.items():
        powers[pname] = sign * port_flux(fields, cen, dh, h) / max(p_ref, 1e-300)
        n_hat = np.array([-dh[1], dh[0]])
        lines.append((cen - h * n_hat, cen + h * n_hat, pname))

    out = dict(state=state_name, phi=phi_deg, alpha=alpha, w=w, t_wall=t_wall,
               neff=neff, x_tan=x_tan, T=powers["through"], R=powers["mirror"],
               back=powers["back"], idle=powers["idle"])

    # field section with the port cross-section lines overlaid
    try:
        ez_frac = float(np.sum(np.abs(E[..., 2]) ** 2) / max(float(np.sum(np.abs(E) ** 2)), 1e-30))
        if ez_frac > 0.5:
            print(f"[WARNING] out-of-plane E dominates ({ez_frac:.1%}); wrong source polarization?")
        inten = np.sum(np.abs(E) ** 2, axis=-1)
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        ax.imshow(inten.T, origin="lower", extent=(x_ax[0], x_ax[-1], y_ax[0], y_ax[-1]),
                  cmap="hot", aspect="equal")
        for p0, p1, pname in lines:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "w--", lw=1.0, alpha=0.8)
            ax.annotate(pname, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2),
                        color="w", fontsize=8, ha="center")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title(f"|E|^2  {state_name} (phi={phi_deg:g}deg) alpha={alpha:g} w={w:g} | "
                     f"T={out['T']:.3f} R={out['R']:.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(design_dir, f"LC2D_field_{tag}.png"), dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"[field] plot skipped: {exc}")

    # material-index section, n seen by E||x and by E||y
    if save_index_fig:
        try:
            ires = fdtd.getresult("mon_index", "index")
            nx = np.real(np.squeeze(np.asarray(ires["index_x"])))
            ny = np.real(np.squeeze(np.asarray(ires["index_y"])))
            xi = np.ravel(np.asarray(ires["x"])) * 1e6
            yi = np.ravel(np.asarray(ires["y"])) * 1e6
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            for ax, dat, lab in ((axes[0], nx, "n for E||x"), (axes[1], ny, "n for E||y")):
                im = ax.imshow(dat.T, origin="lower", extent=(xi[0], xi[-1], yi[0], yi[-1]),
                               cmap="viridis", aspect="equal", vmin=1.4, vmax=2.0)
                ax.set_title(lab)
                ax.set_xlabel("x (um)")
                ax.set_ylabel("y (um)")
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"index section  {state_name} (phi={phi_deg:g}deg) "
                         f"alpha={alpha:g}deg w={w:g}um t_wall={t_wall:.2f}um")
            fig.tight_layout()
            fig.savefig(os.path.join(design_dir, f"LC2D_index_{tag}.png"), dpi=160)
            plt.close(fig)
        except Exception as exc:
            print(f"[index] plot skipped: {exc}")

    try:
        fdtd.close()
    except Exception:
        pass
    print(f"[case {tag}] {state_name}: T={out['T']:.4f} R={out['R']:.4f} "
          f"back={out['back']:.4f} idle={out['idle']:.4f}")
    return out


# -----------------------------------------------------------------------------
# Stage C: width sweep at the picked alpha -> T*R metric, best-case figures
# -----------------------------------------------------------------------------
def feasible_width_list(alpha):
    if WIDTH_LIST:
        return [float(v) for v in WIDTH_LIST.split(",")]
    cosa = np.cos(np.deg2rad(alpha))
    ws = []
    for w in np.arange(0.5, env_float("LC2D_MAX_WIDTH_UM", 4.0) + 1e-9, 0.05):
        neff = slab_neff(w)
        if neff and N_O < neff * cosa < N_E:
            ws.append(round(float(w), 2))
    if not ws:
        raise RuntimeError(f"no feasible width at alpha={alpha}")
    n_pts = min(6, len(ws))
    idx = np.unique(np.linspace(0, len(ws) - 1, n_pts).astype(int))
    return [ws[i] for i in idx]


def main():
    if env_flag("LC2D_THICKNESS_SWEEP", "1"):
        thickness_sweep()
    pick = design_map()
    phi_c = save_analytic_plots(pick)
    print(f"[design] predicted switch alignment angle phi ~{phi_c:.0f} deg "
          f"(reflect below, transmit above)")
    if SKIP_FDTD:
        print("[fdtd] skipped (LC2D_SKIP_FDTD=1)")
        return

    alpha = pick["alpha"]
    widths = feasible_width_list(alpha)
    print(f"[sweep] alpha={alpha:g} deg, widths={widths}")

    rows = []
    for w in widths:
        neff = slab_neff(w)
        t_wall, _, _ = wall_thickness(neff * np.cos(np.deg2rad(alpha)))
        p_ref = run_norm(alpha, w, t_wall, tag=f"a{alpha:g}_w{w:g}")
        res = {s[0]: run_case(alpha, w, s, tag=f"a{alpha:g}_w{w:g}_{s[0]}", p_ref=p_ref)
               for s in LC_STATES}
        rows.append(dict(w=w,
                         T_high=res["transmit"]["T"], R_when_high=res["transmit"]["R"],
                         R_low=res["reflect"]["R"], T_when_low=res["reflect"]["T"],
                         TR=res["transmit"]["T"] * res["reflect"]["R"]))

    best = max(rows, key=lambda r: r["TR"])
    lines = [f"alpha={alpha:g} deg (incidence {90 - alpha:g} deg from wall normal)",
             f"{'w_um':>6} {'T(transmit)':>12} {'R(reflect)':>11} {'T*R':>8} "
             f"{'R_leak@T':>9} {'T_leak@R':>9}"]
    for r in rows:
        mark = "  <-- best" if r is best else ""
        lines.append(f"{r['w']:>6.2f} {r['T_high']:>12.4f} {r['R_low']:>11.4f} "
                     f"{r['TR']:>8.4f} {r['R_when_high']:>9.4f} {r['T_when_low']:>9.4f}{mark}")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(design_dir, "LC2D_summary.txt"), "w", encoding="utf-8") as fp:
        fp.write(summary + "\n")

    ws = [r["w"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ws, [r["T_high"] for r in rows], "o-", label="T (transmit state, phi=90)")
    ax.plot(ws, [r["R_low"] for r in rows], "s-", label="R (reflect state, phi=0)")
    ax.plot(ws, [r["TR"] for r in rows], "^--", lw=1.4, label="T x R")
    ax.axvline(best["w"], color="tab:red", ls=":", lw=1.2,
               label=f"best w={best['w']:g} um (TxR={best['TR']:.3f})")
    ax.set_xlabel("waveguide width (um)")
    ax.set_ylabel("power fraction")
    ax.set_title(f"width sweep at alpha={alpha:g} deg (tilted-port flux)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(design_dir, "LC2D_width_sweep.png"), dpi=160)
    plt.close(fig)

    # keep a separate copy of the best combination's material section, per spec
    import shutil
    for s in LC_STATES:
        src = os.path.join(design_dir, f"LC2D_index_a{alpha:g}_w{best['w']:g}_{s[0]}.png")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(design_dir, f"LC2D_best_index_section_{s[0]}.png"))
    print(f"[best] w={best['w']:g} um: T={best['T_high']:.4f}, R={best['R_low']:.4f}, "
          f"TxR={best['TR']:.4f} -> LC2D_best_index_section_*.png")


if __name__ == "__main__":
    main()
