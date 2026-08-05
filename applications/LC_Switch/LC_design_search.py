"""LC_design_search: initial-condition search for the LC TIR switch in the
FIXED stack the device is actually built in.

Stack (fixed, not swept)
    z < 0            lower oxide, SiO2 n=1.44, 2 um
    0 < z < t        SiN core, n=2.0, thickness t          <- swept
    t < z < TOX      upper oxide, SiO2 n=1.44
    z > TOX          air
    TOX = 2.0 um measured from the top of the lower oxide.

The LC wall is made by etching the WHOLE upper oxide away and filling with LC,
so inside the wall there is no core either: the wall is a 2 um LC slab standing
on the lower oxide with air above it, and the guided mode has to cross it
UNGUIDED in the vertical sense.  That is the part the earlier screen missed:
a criterion written only on n_eff*cos(alpha) predicts high transmission and the
3D run then delivers 0.2, because the 0.9 um core mode does not re-form after
crossing a 2 um multimode LC slab.

Swept parameters
    t      SiN core thickness            (um)
    w      waveguide width               (um)
    alpha  waveguide angle from the wall (deg); crossing angle = 2*alpha
    d      LC wall thickness             (um)

Physics
    1. EIM.  Vertical SiN slab (TE, E parallel to the substrate) -> n_v(t).
       In-plane slab of width w with core n_v (TM in-plane) -> n_eff(t, w).
    2. The tangential index is conserved at the wall: x = n_eff*cos(alpha).
       Switching needs  n_o < x < n_e, so that the SAME geometry sees
           reflect  (wall index n_o = 1.50):  x > n_o  -> TIR
           transmit (wall index n_e = 1.685): x < n_e  -> propagates.
    3. The wall is its own vertical slab (SiO2 | LC 2 um | air) and is
       MULTIMODE.  The core mode is decomposed onto its modes,
           a_m = <E_m | E_core>,
       each mode crosses with its own normal wavevector
           kz_m = k0 * sqrt(n_m^2 - x^2)          (imaginary -> tunnelling)
       and a p-polarized 3-layer transfer matrix (n_eff | n_m | n_eff) supplies
       the Fresnel/Fabry-Perot amplitude t_m.  Recombination into the outgoing
       core mode gives
           T = |sum_m a_m^2 t_m|^2 ,    R = 1 - T - (radiation).
       Intermodal DEPHASING is what limits T; it is invisible to a Fresnel-only
       model and is the dominant term here.
    4. In-plane the wall region is unguided, so the beam also diffracts over
       the slant path and walks off along the wall.  Both are reported; the
       walk-off is a fixable lateral offset of the output arm, not a loss, so
       it is quoted as a required offset AND as the penalty if left
       uncompensated.

This is a screening model. It keeps the vertical multimode physics and the
tangential-index bookkeeping exactly, and approximates the in-plane crossing as
a Gaussian gap.

WHAT 3D ACTUALLY SHOWED (Switch_3D_sweep, same stack, 12 confirmed points)

    This model is DIRECTIONALLY right and QUANTITATIVELY not usable. It puts R
    too high (0.75 vs 0.58 measured) and T too low (0.67 vs 0.85 measured). Use it
    to choose what to try; take the numbers from 3D. The rules 3D established:

    1. R is governed by the TIR ANGULAR MARGIN = (operating angle - critical
       angle), where the critical angle comes from the WALL's own vertical mode
       index, not bulk n_o. At t=0.9 um that index is 1.4856.
           margin 0.1-1.3 deg -> R 0.58-0.71   (fails)
           margin 4 deg       -> R 0.82
           margin 5 deg       -> R 0.85
           margin 6 deg       -> R 0.88
           margin 8 deg       -> R 0.92 but T 0.61 (fails the other way)
       The usable window is margin 4-6 deg. Raising the margin trades T for R.
    2. WIDTH IS NOT A TRADE. Going w 8 -> 12 um at fixed margin raised BOTH
       (margin 4 deg: 0.776/0.818 -> 0.808/0.830; margin 5 deg: 0.748/0.852 ->
       0.778/0.863), because a wider guide has a narrower angular spectrum and so
       a smaller tail below the critical angle. Spend width before margin.
    3. A THICKER WALL IS A NET LOSS. At margin 4 deg, d 0.575 -> 0.8 -> 1.1 um
       moved R 0.818 -> 0.918 -> 0.959 but T 0.776 -> 0.668 -> 0.561. Keep the
       wall thin; the old "tunnelling minimum" of several um is the wrong target
       because it is written for 1 % leakage, far stricter than R >= 0.80.

    Best confirmed point: t=0.90, w=12.0, 2*alpha=67.4 deg, d=0.575 um
    -> T=0.778, R=0.863.

KNOWN OMISSION: the projection is scalar (overlap - propagate - overlap), so the
LONGITUDINAL impedance step between the guide (n_eff ~ 1.8-1.9) and the wall
modes (n ~ 1.66) is not charged.  A single-interface p-polarized estimate at the
recommended operating point puts it at ~2.5 % per interface, and a per-mode
Fresnel experiment moved T by 4-11 %.  So treat T as OPTIMISTIC BY ROUGHLY
5-10 %; R is not affected, because in the reflect state every wall mode is
evanescent and R is set by exp(-2|kz|d) tunnelling, which the model does carry
exactly.  Confirm the shortlist in 3D.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
OUT_DIR = os.path.join(RUN_DIR, "A") + os.sep
os.makedirs(OUT_DIR, exist_ok=True)


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


# -----------------------------------------------------------------------------
# Fixed stack
# -----------------------------------------------------------------------------
WL = env_float("LCDS_WAVELENGTH_UM", 1.55)
K0 = 2.0 * np.pi / WL
N_SIN = 2.0
N_SIO2 = 1.44
N_AIR = 1.0
N_O = 1.5        # LC ordinary       -> reflect state (wall blocks)
N_E = 1.685      # LC extraordinary  -> transmit state (wall passes)
# Upper oxide is measured ABOVE THE CORE (same convention as Switch_2.py, where
# Sx = BOX_h + Core_h + TOX_h). Etching it away for the wall also removes the
# core, so the wall stands (t + TOX_ABOVE) tall, not TOX_ABOVE tall.
TOX_ABOVE = env_float("LCDS_UPPER_OXIDE_UM", 2.0)
TOX_LO = env_float("LCDS_LOWER_OXIDE_UM", 2.0)
# What sits on top of the upper oxide. Switch_2.py currently ends the domain at
# the TOX surface with an oxide background, i.e. no air at all; the intended
# device has air there, and the wall's mode count is sensitive to it.
N_COVER = env_float("LCDS_COVER_INDEX", N_AIR)


def wall_height(t):
    return float(t) + TOX_ABOVE

# Acceptance thresholds
T_TARGET = env_float("LCDS_T_TARGET", 0.70)
R_TARGET = env_float("LCDS_R_TARGET", 0.80)

# Switching window on the tangential index. The physical limits are the wall's
# own vertical mode indices, not the bulk (n_o, n_e): the wall is a finite 2 um
# slab with air on top, so its fundamental sits below the bulk index.
X_MIN = env_float("LCDS_X_MIN", 1.47)
X_MAX = env_float("LCDS_X_MAX", 1.68)


# -----------------------------------------------------------------------------
# Slab modes
# -----------------------------------------------------------------------------
def slab_modes(n_film, h, n_sub, n_cov, tm=False, n_max_modes=12):
    """Guided modes of an asymmetric 3-layer slab.

    Returns a list of effective indices, lowest order first. TE (tm=False) is
    the polarization with E parallel to the interfaces, which is what the
    vertical problem sees here (the guided E lies in the device plane).
    """
    n_lo = max(n_sub, n_cov)
    if n_film <= n_lo:
        return []
    out = []
    for m in range(n_max_modes):
        def resid(n):
            kap = K0 * np.sqrt(max(n_film ** 2 - n ** 2, 1e-30))
            gs = K0 * np.sqrt(max(n ** 2 - n_sub ** 2, 1e-30))
            gc = K0 * np.sqrt(max(n ** 2 - n_cov ** 2, 1e-30))
            rs = (n_film / n_sub) ** 2 if tm else 1.0
            rc = (n_film / n_cov) ** 2 if tm else 1.0
            return kap * h - np.arctan(rs * gs / kap) - np.arctan(rc * gc / kap) - m * np.pi

        lo, hi = n_lo + 1e-9, n_film - 1e-9
        if resid(lo) < 0.0:
            break
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            if resid(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


def slab_profile(n_eff, n_film, h, n_sub, n_cov, z, tm=False):
    """Field profile of a slab mode. Film occupies 0 <= z <= h."""
    kap = K0 * np.sqrt(max(n_film ** 2 - n_eff ** 2, 1e-30))
    gs = K0 * np.sqrt(max(n_eff ** 2 - n_sub ** 2, 1e-30))
    gc = K0 * np.sqrt(max(n_eff ** 2 - n_cov ** 2, 1e-30))
    if tm:
        gs = gs * (n_film / n_sub) ** 2
        gc = gc * (n_film / n_cov) ** 2
    phi = np.arctan2(gs, kap)
    E = np.empty_like(z)
    below = z < 0.0
    above = z > h
    infilm = ~(below | above)
    E[infilm] = np.cos(kap * z[infilm] - phi)
    E[below] = np.cos(phi) * np.exp(gs * z[below])
    E[above] = np.cos(kap * h - phi) * np.exp(-gc * (z[above] - h))
    return E


def normalize(E, z):
    return E / np.sqrt(np.trapezoid(E * E, z))


def in_plane_neff(width, n_core, n_clad=N_SIO2, mode_m=0):
    """In-plane slab (TM: E perpendicular to the sidewalls)."""
    modes = slab_modes(n_core, width, n_clad, n_clad, tm=True, n_max_modes=mode_m + 1)
    return modes[mode_m] if len(modes) > mode_m else None


def beam_angular_spectrum(w, n_core, n_eff, n_pts=257, span=6.0):
    """(delta, weight) -- the IN-PLANE angular content of the guided mode.

    A guided mode is not a plane wave. Its lateral profile of width w carries a
    spread of transverse wavevectors, i.e. a spread of incidence angles at the
    wall.  That spread is what breaks TIR: the critical angle is a hard edge, so
    every component on the wrong side of it transmits in full even when the
    NOMINAL operating point sits above it.  Measured on the first 3D run, the TIR
    margin was 0.13-0.78 deg against a beam spread of 1.9-2.5 deg, and R came out
    at 0.58-0.63 where a plane-wave model had promised 0.81-0.86.

    Weights are |A(k_y)|^2 from the actual mode profile, not a Gaussian fit.
    """
    y = np.linspace(-span * w, span * w, 4096)
    E = slab_profile(n_eff, n_core, w, N_SIO2, N_SIO2, y + 0.5 * w, tm=True)
    A = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(E)))
    ky = np.fft.fftshift(np.fft.fftfreq(y.size, d=float(y[1] - y[0]))) * 2.0 * np.pi
    P = np.abs(A) ** 2
    sin_d = np.clip(ky / (K0 * n_eff), -0.999999, 0.999999)
    keep = np.abs(sin_d) < 0.9
    delta = np.arcsin(sin_d[keep])
    P = P[keep]
    # thin to n_pts so the crossing integral stays cheap
    idx = np.linspace(0, delta.size - 1, min(n_pts, delta.size)).astype(int)
    delta, P = delta[idx], P[idx]
    P = P / max(float(np.sum(P)), 1e-300)
    return delta, P


# -----------------------------------------------------------------------------
# Wall crossing
# -----------------------------------------------------------------------------
def kz(n, x_tan):
    """Normal wavevector inside a medium of index n at tangential index x_tan.
    Returns a complex value; positive-imaginary means evanescent (tunnelling)."""
    return K0 * np.sqrt(complex(n * n - x_tan * x_tan))


def slab_amplitude_t(n_in, n_mid, x_tan, d):
    """p-polarized amplitude transmission through n_in | n_mid (thickness d) |
    n_in at fixed tangential index. Covers Fresnel, Fabry-Perot and frustrated
    TIR in one expression."""
    k1, k2 = kz(n_in, x_tan), kz(n_mid, x_tan)
    e1, e2 = k1 / (n_in ** 2), k2 / (n_mid ** 2)      # p-polarization admittance
    if abs(e1 + e2) < 1e-30:
        return 0.0 + 0.0j
    r = (e1 - e2) / (e1 + e2)
    t12 = 2.0 * e1 / (e1 + e2)
    t21 = 2.0 * e2 / (e1 + e2)
    ph = np.exp(1j * k2 * d)
    denom = 1.0 - (r ** 2) * (ph ** 2)
    if abs(denom) < 1e-30:
        return 0.0 + 0.0j
    return t12 * t21 * ph / denom


def wall_basis(n_wall, t, z):
    """Guided vertical modes of the wall slab -- DIAGNOSTIC ONLY.

    The wall is SiO2 | LC(n_wall), height t + TOX_ABOVE | cover. How few modes
    it carries, and how far their indices sit below the bulk LC index, is the
    whole story of this device; the T/R numbers come from the complete basis.
    """
    h = wall_height(t)
    ns = slab_modes(n_wall, h, N_SIO2, N_COVER, tm=False, n_max_modes=16)
    prof = [normalize(slab_profile(n, n_wall, h, N_SIO2, N_COVER, z), z) for n in ns]
    return np.asarray(ns), prof


def wall_index_profile(n_wall, t, z):
    h = wall_height(t)
    n = np.full_like(z, N_SIO2)
    n[z > h] = N_COVER
    n[(z >= 0.0) & (z <= h)] = n_wall
    return n


def wall_complete_basis(n_wall, t, z):
    """COMPLETE orthonormal vertical basis of the wall cross-section.

    Finite-difference discretization of  d2E/dz2 + k0^2 n(z)^2 E = beta^2 E on a
    box. Unlike the guided-mode-only expansion this basis is complete, so
    sum |a_m|^2 = 1 exactly and the radiation continuum is represented as
    discretized box modes. That matters: with guided modes alone the model
    wrongly predicts a finite loss even for a vanishingly thin wall, because the
    part of the core mode that does not fit the two guided modes is counted as
    lost before it has had any distance to actually diverge.

    Returns (beta2, V) with V[:, m] the m-th normalized eigenvector.
    """
    from scipy.linalg import eigh_tridiagonal
    dz = float(z[1] - z[0])
    n = wall_index_profile(n_wall, t, z)
    diag = -2.0 / dz ** 2 + (K0 * n) ** 2
    off = np.full(z.size - 1, 1.0 / dz ** 2)
    beta2, V = eigh_tridiagonal(diag, off)
    V = V / np.sqrt(dz)                     # continuous normalization
    return beta2, V


def core_profile(t, z):
    """Vertical mode of the guiding section: SiO2 | SiN(t) | SiO2. The cover
    (air or PML) sits TOX_ABOVE = 2 um up, >7 evanescent decay lengths away, so
    the guiding section is the symmetric oxide-clad slab either way."""
    ns = slab_modes(N_SIN, t, N_SIO2, N_SIO2, tm=False, n_max_modes=1)
    if not ns:
        return None, None
    return ns[0], normalize(slab_profile(ns[0], N_SIN, t, N_SIO2, N_SIO2, z), z)


def coupling_weights(n_wall, t, z, basis_cache, core_cache, keep=0.9999):
    """|a_m|^2 of the core mode on the wall basis, truncated to the modes that
    carry `keep` of the power, plus their beta^2. Cached per (state, t)."""
    key = ("w", n_wall, t)
    if key in basis_cache:
        return basis_cache[key]
    bkey = ("b", n_wall, t)
    if bkey not in basis_cache:
        basis_cache[bkey] = wall_complete_basis(n_wall, t, z)
    beta2, V = basis_cache[bkey]
    _n_v, Ec = core_cache[t]
    dz = float(z[1] - z[0])
    a = (V.T @ Ec) * dz
    p = a * a
    order = np.argsort(p)[::-1]
    csum = np.cumsum(p[order])
    ncut = int(np.searchsorted(csum, keep * csum[-1]) + 1)
    sel = order[:ncut]
    basis_cache[key] = (beta2[sel], p[sel], float(csum[-1]))
    return basis_cache[key]


def crossing(t, w, alpha_deg, d_arr, z, basis_cache, core_cache):
    """T (transmit state) and R (reflect state) for one (t, w, alpha) over every
    wall thickness in d_arr. Returns a list of result dicts, one per d."""
    if t not in core_cache:
        core_cache[t] = core_profile(t, z)
    n_v, Ec = core_cache[t]
    if Ec is None:
        return []
    n_eff = in_plane_neff(w, n_v)
    if n_eff is None:
        return []
    x = n_eff * np.cos(np.deg2rad(alpha_deg))
    if not (X_MIN < x < X_MAX):
        return []

    d = np.asarray(d_arr, dtype=float)
    akey = (w, t)
    if akey not in basis_cache:
        basis_cache[akey] = beam_angular_spectrum(w, n_v, n_eff)
    delta, wgt = basis_cache[akey]
    # Every angular component of the beam meets the wall at its own incidence, so
    # it gets its own tangential index. Averaging over them is what a plane-wave
    # screen misses; the critical angle is a step, so the tail below it dominates R.
    x_all = n_eff * np.cos(np.deg2rad(alpha_deg) + delta)

    raw = {}
    nprop = {}
    for n_wall, key in ((N_E, "T"), (N_O, "R")):
        beta2, p, _tot = coupling_weights(n_wall, t, z, basis_cache, core_cache)
        acc = np.zeros(d.size)
        for xi, wi in zip(x_all, wgt):
            if wi <= 0.0:
                continue
            kzm = np.sqrt((beta2 - (K0 * xi) ** 2).astype(complex))
            amp = (p[None, :] * np.exp(1j * kzm[None, :] * d[:, None])).sum(axis=1)
            acc += wi * np.abs(amp) ** 2
        raw[key] = acc
        nprop[key] = int(np.sum(beta2 > (K0 * x) ** 2))
    spread_deg = float(np.rad2deg(
        np.sqrt(np.sum(wgt * delta ** 2) - np.sum(wgt * delta) ** 2)))

    # In-plane: the wall region is laterally unguided, so the beam diffracts over
    # the slant path and walks off along the wall. The walk-off is a fixed
    # lateral offset of the output arm, so it is reported, not charged as loss.
    th_i = np.deg2rad(90.0 - alpha_deg)
    th_t = np.arcsin(min(x / N_E, 1.0 - 1e-12))
    path = d / max(np.cos(th_t), 1e-6)
    w0 = 0.5 * w
    zR = np.pi * w0 * w0 * N_E / WL
    eta_diff = 1.0 / (1.0 + (path / (2.0 * zR)) ** 2)
    offset = d * (np.tan(th_t) - np.tan(th_i))
    eta_off = np.exp(-(offset / w0) ** 2)

    single_w = in_plane_neff(w, n_v, mode_m=1) is None
    single_t = len(slab_modes(N_SIN, t, N_SIO2, N_SIO2, n_max_modes=2)) == 1

    T = raw["T"] * eta_diff
    R = 1.0 - raw["R"]
    out = []
    for i in range(d.size):
        out.append({
            "t": t, "w": w, "alpha": alpha_deg, "d": float(d[i]),
            "n_v": n_v, "n_eff": n_eff, "x_tan": x,
            "theta_i_deg": float(np.rad2deg(th_i)), "theta_t_deg": float(np.rad2deg(th_t)),
            "T": float(T[i]), "R": float(R[i]),
            "T_uncorrected": float(T[i] * eta_off[i]),
            "offset_um": float(offset[i]), "eta_diffraction": float(eta_diff[i]),
            "nprop_T": nprop["T"], "nprop_R": nprop["R"],
            "beam_spread_deg": spread_deg,
            "single_w": single_w, "single_t": single_t,
            "score": min(float(T[i]) / T_TARGET, 1.0) * min(float(R[i]) / R_TARGET, 1.0),
        })
    return out


def main():
    z = np.linspace(-6.0, TOX_ABOVE + 7.0, 3001)
    basis_cache, core_cache = {}, {}

    t_list = np.round(np.arange(0.30, 0.901, 0.05), 3)
    w_list = np.round(np.arange(1.0, 8.01, 0.5), 3)
    a_list = np.round(np.arange(5.0, 50.1, 1.0), 3)
    d_list = np.round(np.r_[np.arange(0.15, 1.501, 0.025),
                            np.arange(1.6, 5.51, 0.10)], 4)

    print(f"[stack] SiN {N_SIN} / SiO2 {N_SIO2} / LC n_o {N_O} n_e {N_E} @ {WL} um")
    cover = "air" if abs(N_COVER - N_AIR) < 1e-9 else f"n={N_COVER}"
    print(f"[stack] lower oxide {TOX_LO} um; upper oxide {TOX_ABOVE} um ABOVE THE CORE, "
          f"fully etched inside the wall so the core is interrupted; cover = {cover}")
    print(f"[stack] wall height = t + {TOX_ABOVE} um")
    for tsh in (float(t_list[0]), float(t_list[-1])):
        for n_wall, lab in ((N_E, "transmit (n_e)"), (N_O, "reflect  (n_o)")):
            nm, _ = wall_basis(n_wall, tsh, z)
            print(f"[wall]  t={tsh:.2f} (h={wall_height(tsh):.2f} um) {lab}: "
                  f"n = {np.round(nm, 4)}")
    print(f"[sweep] t {t_list[0]}..{t_list[-1]} | w {w_list[0]}..{w_list[-1]} | "
          f"alpha {a_list[0]}..{a_list[-1]} deg | d {d_list[0]}..{d_list[-1]} um")

    rows = []
    for t in t_list:
        for w in w_list:
            for a in a_list:
                rows.extend(crossing(float(t), float(w), float(a), d_list,
                                     z, basis_cache, core_cache))
    print(f"[sweep] {len(rows)} points inside the switching window "
          f"({X_MIN} < n_eff*cos(alpha) < {X_MAX})")
    if not rows:
        print("[sweep] nothing in the window")
        return []

    ok = [r for r in rows if r["T"] >= T_TARGET and r["R"] >= R_TARGET]
    sm = [r for r in ok if r["single_w"] and r["single_t"]]
    print(f"[sweep] meeting T>={T_TARGET} and R>={R_TARGET}: {len(ok)}  "
          f"(of which single-mode in BOTH w and t: {len(sm)})")

    def show(rs, title, n=15):
        print(f"\n--- {title} ---")
        print(f"{'t':>5}{'w':>6}{'alpha':>7}{'2a':>6}{'d':>6}{'n_eff':>8}{'x_tan':>7}"
              f"{'T':>7}{'R':>7}{'off':>6}{'SMw':>5}{'SMt':>5}")
        for r in rs[:n]:
            print(f"{r['t']:>5.2f}{r['w']:>6.2f}{r['alpha']:>7.1f}{2*r['alpha']:>6.1f}"
                  f"{r['d']:>6.3f}{r['n_eff']:>8.4f}{r['x_tan']:>7.4f}"
                  f"{r['T']:>7.3f}{r['R']:>7.3f}{r['offset_um']:>6.2f}"
                  f"{'Y' if r['single_w'] else 'n':>5}{'Y' if r['single_t'] else 'n':>5}")

    rows.sort(key=lambda r: (-(min(r['T'] / T_TARGET, 1.2) + min(r['R'] / R_TARGET, 1.2)),))
    show(rows, "best overall (T and R balanced)")
    smr = [r for r in rows if r["single_w"] and r["single_t"]]
    show(smr, "best that is SINGLE-MODE in both width and thickness")

    path = os.path.join(OUT_DIR, "LC_design_search.txt")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(f"# lambda={WL} SiN={N_SIN} SiO2={N_SIO2} LC n_o={N_O} n_e={N_E}\n")
        fp.write(f"# upper oxide {TOX_ABOVE} um above the core; wall height = t + {TOX_ABOVE}; "
                 f"cover index {N_COVER}\n")
        fp.write(f"# targets T>={T_TARGET} R>={R_TARGET}; {len(ok)} meet both, {len(sm)} also single-mode\n")
        fp.write("# t_um w_um alpha_deg d_um n_v n_eff x_tan T R T_uncorrected offset_um "
                 "theta_i theta_t single_w single_t\n")
        for r in rows:
            fp.write(f"{r['t']:.3f} {r['w']:.3f} {r['alpha']:.2f} {r['d']:.4f} "
                     f"{r['n_v']:.5f} {r['n_eff']:.5f} {r['x_tan']:.5f} "
                     f"{r['T']:.5f} {r['R']:.5f} {r['T_uncorrected']:.5f} "
                     f"{r['offset_um']:.4f} {r['theta_i_deg']:.2f} {r['theta_t_deg']:.2f} "
                     f"{int(r['single_w'])} {int(r['single_t'])}\n")
    print(f"\n[sweep] wrote {path}")

    # Design map: T and R over (crossing angle, wall thickness) at the best t,
    # with the joint-acceptance region outlined. This is the picture that shows
    # WHY the window is where it is.
    best = rows[0]
    tb, wb = best["t"], best["w"]
    sub = [r for r in rows if r["t"] == tb and r["w"] == wb]
    if sub:
        aa = np.array(sorted({r["alpha"] for r in sub}))
        dd = np.array(sorted({r["d"] for r in sub}))
        Tm = np.full((dd.size, aa.size), np.nan)
        Rm = np.full((dd.size, aa.size), np.nan)
        ai = {v: i for i, v in enumerate(aa)}
        di = {v: i for i, v in enumerate(dd)}
        for r in sub:
            Tm[di[r["d"]], ai[r["alpha"]]] = r["T"]
            Rm[di[r["d"]], ai[r["alpha"]]] = r["R"]
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
        for ax, M, lab in ((axes[0], Tm, "T  (transmit state)"),
                           (axes[1], Rm, "R  (reflect state)")):
            im = ax.pcolormesh(2 * aa, dd, M, cmap="viridis", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax)
            ax.set_xlabel("crossing angle 2*alpha (deg)")
            ax.set_ylabel("LC wall thickness d (um)")
            ax.set_title(lab)
            ax.set_yscale("log")
        ok_m = ((Tm >= T_TARGET) & (Rm >= R_TARGET)).astype(float)
        axes[2].pcolormesh(2 * aa, dd, ok_m, cmap="Greens", vmin=0, vmax=1.4)
        axes[2].set_xlabel("crossing angle 2*alpha (deg)")
        axes[2].set_ylabel("LC wall thickness d (um)")
        axes[2].set_yscale("log")
        axes[2].set_title(f"BOTH  T>={T_TARGET} and R>={R_TARGET}")
        fig.suptitle(f"LC switch design map at core t={tb:g} um, width w={wb:g} um "
                     f"(LC wall = core + {TOX_ABOVE:g} um upper oxide, cover n={N_COVER:g})")
        fig.tight_layout()
        fpath = os.path.join(OUT_DIR, "LC_design_map.png")
        fig.savefig(fpath, dpi=170)
        plt.close(fig)
        print(f"[sweep] wrote {fpath}")
    return rows


if __name__ == "__main__":
    main()
