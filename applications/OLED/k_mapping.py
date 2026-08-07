"""k_mapping: which in-plane momentum inside the OLED feeds which emission angle.

A planar stack conserves the in-plane wavevector. Working in the normalized
variable

    u = k_par / k0            (dimensionless; u = n * sin(theta) in any layer)

every layer sees the same u, so u is the one coordinate that connects "a mode
trapped in the organic" to "a ray leaving into air". The bands are fixed by the
indices alone:

    u < 1                       escapes into air          (the escape cone)
    1 < u < n_sub               trapped in the substrate
    n_sub < u < n_org           guided in the organic/ITO waveguide
    u > n_org                   evanescent: surface plasmon at the metal

A PERIODIC out-coupler of pitch P moves power between u values in steps of
lambda/P: a mode at u_in leaves at

    sin(theta_air) = u_in - m * lambda / P                          (m = order)

That single relation is the whole point of this script. Given the angles the
device is supposed to emit into, it inverts to

    u_in = sin(theta_air) + m * lambda / P
    kx   = k0 * u_in ,   theta_org = asin(u_in / n_org)

so each target angle maps to ONE in-plane momentum per order -- the momentum the
grating has to find power at. The TMM solver supplies dP/du, i.e. how much power
the stack actually puts at each u, so a requested (angle, order) pair can be
labelled by how much there is to harvest and which band it comes from.

The TMM engine is OLED_layered_dipole (already validated, 22 self-checks); this
module only adds the momentum bookkeeping and the reporting on top of it.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import OLED_layered_dipole as ld


# ---------------------------------------------------------------------------
# Momentum bookkeeping
# ---------------------------------------------------------------------------
def u_for_angle(theta_deg, order, period_um, lam_um):
    """In-plane momentum inside the stack that the given order sends to
    theta_deg in air.  u = sin(theta) + m*lambda/P."""
    return float(np.sin(np.deg2rad(theta_deg))) + float(order) * lam_um / float(period_um)


def angle_for_u(u, order, period_um, lam_um):
    """Inverse: emission angle in air for a mode at u leaving via `order`.
    Returns None when the shifted momentum still exceeds the escape cone."""
    s = float(u) - float(order) * lam_um / float(period_um)
    if abs(s) > 1.0:
        return None
    return float(np.rad2deg(np.arcsin(s)))


def period_for(theta_deg, u_in, order, lam_um):
    """Pitch that makes `order` carry u_in to theta_deg.  None when the pair
    needs a non-positive pitch (i.e. the order points the wrong way)."""
    du = float(u_in) - float(np.sin(np.deg2rad(theta_deg)))
    if order == 0 or du * order <= 0.0:
        return None
    return float(order) * lam_um / du


def band_of(u, bands):
    for name, lo, hi in bands:
        if lo - 1e-12 <= u < hi:
            return name
    return bands[-1][0]


# ---------------------------------------------------------------------------
# Inverse design entry point: target angles -> minimum period + orders
# ---------------------------------------------------------------------------
def minimum_period(angles_deg, lam_um, m_max=12, tol=0.01, n_org=None):
    """Smallest pitch at which EVERY requested angle is a real diffraction order.

    With a normal-incidence probe the field inside a cell of pitch P only contains
    the Bloch momenta u = m*lambda/P. So an angle is reachable only when

        sin(theta_i) = m_i * lambda / P          for some integer m_i,

    i.e. P = m_i*lambda/sin(theta_i) must come out the SAME for every target. That
    also reproduces the textbook existence condition on its own: |m*lambda/P| <= 1
    gives P >= m*lambda, so a higher order simply needs a longer pitch.

    Asking for angles whose sines are not in rational ratio has no exact solution;
    `tol` is how far an order index may sit from an integer before the assignment
    is rejected. theta = 0 is free (m = 0) and never constrains the pitch.

    Returns (P, orders, report) with P None when nothing within m_max works.
    """
    ang = [float(a) for a in angles_deg]
    sines = [float(np.sin(np.deg2rad(a))) for a in ang]
    nz = [i for i, sv in enumerate(sines) if abs(sv) > 1e-9]
    if not nz:                                   # only normal emission requested
        P = float(lam_um)                        # m = 1 is the shortest that exists
        return P, [0 for _ in ang], ["theta=0 only: any pitch works; using lambda"]

    best = None
    lead = nz[0]
    for m_lead in range(1, m_max + 1):
        P = m_lead * lam_um / sines[lead]
        orders, ok, resid = [], True, 0.0
        for i, sv in enumerate(sines):
            if abs(sv) <= 1e-9:
                orders.append(0)
                continue
            m = P * sv / lam_um
            mr = round(m)
            if mr < 1 or mr > m_max or abs(m - mr) > tol:
                ok = False
                break
            resid = max(resid, abs(m - mr))
            orders.append(int(mr))
        if ok and (best is None or P < best[0]):
            best = (P, orders, resid)

    if best is None:
        return None, None, [
            f"no pitch up to order {m_max} makes every angle a Bloch order within "
            f"tol={tol}. sin(theta) values {['%.4f' % v for v in sines]} are not in "
            f"near-rational ratio; relax tol, drop an angle, or accept per-angle probes."]
    P, orders, resid = best
    rep = [f"minimum pitch P = {P:.5f} um  (lambda/P = {lam_um/P:.5f}), "
           f"worst order residual {resid:.2e}"]
    for a, sv, m in zip(ang, sines, orders):
        u = sv if m == 0 else m * lam_um / P
        line = f"  theta={a:6.2f} deg  m={m}  u={u:.5f}"
        if n_org is not None:
            line += ("  (inside the organic)" if u <= n_org
                     else f"  <- u > n_organic={n_org:.3f}: evanescent")
        rep.append(line)
    return P, orders, rep


# ---------------------------------------------------------------------------
# 2-D momentum: (m, n) orders of a square lattice, azimuth included
# ---------------------------------------------------------------------------
# Under a normal-incidence probe a square cell of pitch P carries the discrete
# in-plane momenta
#       (ux, uy) = (m, n) * lambda / P
# so a target direction (theta, phi) is reachable only when
#       ux = sin(theta) cos(phi) = m lambda/P
#       uy = sin(theta) sin(phi) = n lambda/P
# i.e. tan(phi) = n/m  and  sin(theta) = (lambda/P) sqrt(m^2 + n^2).
# Azimuth is therefore quantised too: only the directions of integer lattice
# vectors exist, which is why the two symmetry modes below pick their
# representative azimuths from (m, 0), (m, m) and (0, n).
SYMMETRY_MODES = ("radial", "fourfold")


def modes_2d(period_um, lam_um, m_max=8, u_max=1.0, symmetry="fourfold"):
    """Every (m, n) order of the square cell that still escapes into air.

    symmetry:
      "radial"    only (m, 0). The design is rotationally symmetric, so the FoM
                  is evaluated on ONE radial line at phi = 0 and every azimuth is
                  a copy of it -- 1-D diffraction covers the whole 2-D k-plane.
      "fourfold"  the representative azimuths of a 4-fold symmetric design:
                  phi = 0 from (m, 0), phi = 45 from (m, m), phi = 90 from (0, n).
                  The first quadrant repeats, so these three lines carry the
                  information of the full plane.
    """
    step = lam_um / float(period_um)
    out = []
    if symmetry == "radial":
        pairs = [(m, 0) for m in range(1, m_max + 1)]
    elif symmetry == "fourfold":
        pairs = ([(m, 0) for m in range(1, m_max + 1)]
                 + [(m, m) for m in range(1, m_max + 1)]
                 + [(0, n) for n in range(1, m_max + 1)])
    else:
        raise ValueError(f"symmetry must be one of {SYMMETRY_MODES}")
    for m, n in pairs:
        ux, uy = m * step, n * step
        u = float(np.hypot(ux, uy))
        if u > u_max + 1e-12:
            continue
        out.append({
            "m": int(m), "n": int(n), "ux": float(ux), "uy": float(uy), "u": u,
            "phi_deg": float(np.rad2deg(np.arctan2(uy, ux))),
            "theta_air_deg": float(np.rad2deg(np.arcsin(min(u, 1.0)))),
            "kx_per_um": float(2.0 * np.pi / lam_um * ux),
            "ky_per_um": float(2.0 * np.pi / lam_um * uy),
        })
    out.sort(key=lambda r: (r["phi_deg"], r["u"]))
    return out


def minimum_period_2d(targets, lam_um, m_max=12, tol=0.01, symmetry="fourfold"):
    """Smallest square pitch on which every requested (theta, phi) is a lattice
    order. targets is a list of (theta_deg, phi_deg).

    Radial mode ignores phi entirely (the design has no azimuthal structure), so
    it reduces to the 1-D condition sin(theta) = m lambda/P.
    """
    tg = [(float(t), float(p)) for t, p in targets]
    if symmetry == "radial":
        P, orders, rep = minimum_period([t for t, _p in tg], lam_um, m_max, tol)
        if P is None:
            return None, None, rep
        return P, [(m, 0) for m in orders], rep

    # fourfold: each target must land on an integer (m, n)
    ux = [np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(p)) for t, p in tg]
    uy = [np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(p)) for t, p in tg]
    comps = [v for v in (list(ux) + list(uy)) if abs(v) > 1e-9]
    if not comps:
        return float(lam_um), [(0, 0) for _ in tg], ["normal emission only"]
    best = None
    for m_lead in range(1, m_max + 1):
        P = m_lead * lam_um / abs(comps[0])
        step = lam_um / P
        orders, ok, resid = [], True, 0.0
        for a, b in zip(ux, uy):
            ma, mb = a / step, b / step
            ra, rb = round(ma), round(mb)
            if abs(ma - ra) > tol or abs(mb - rb) > tol or max(abs(ra), abs(rb)) > m_max:
                ok = False
                break
            resid = max(resid, abs(ma - ra), abs(mb - rb))
            orders.append((int(ra), int(rb)))
        if ok and (best is None or P < best[0]):
            best = (P, orders, resid)
    if best is None:
        return None, None, [
            f"no square pitch up to order {m_max} places every (theta, phi) on a "
            f"lattice point within tol={tol}. Both sin(theta)cos(phi) and "
            f"sin(theta)sin(phi) must be integer multiples of lambda/P."]
    P, orders, resid = best
    rep = [f"minimum square pitch P = {P:.5f} um (lambda/P = {lam_um/P:.5f}), "
           f"worst residual {resid:.2e}"]
    for (t, ph), (m, n) in zip(tg, orders):
        u = float(np.hypot(m, n)) * lam_um / P
        rep.append(f"  theta={t:6.2f} phi={ph:6.2f} -> (m,n)=({m},{n})  u={u:.5f}")
    return P, orders, rep


def angle_ladder(theta_max_deg, lam_um, n_orders, symmetry="radial", n_org=None,
                 snap_per_um=None):
    """Target angle as the HIGHEST order; the lower orders fill in the ramp.

    Demanding a specific set of angles (0/30/45, say) usually has no solution,
    because one pitch can only serve angles whose sines are in integer ratio.
    Turning the request around removes the problem entirely: fix the pitch so the
    TOP order lands on the target,

        P = M * lambda / sin(theta_max),

    and orders m = 1..M then sit at theta_m = asin(m*lambda/P) automatically --
    evenly spaced in sin(theta), which is exactly the ramp between 0 and the
    target. Every rung is a genuine Bloch order by construction, so nothing has
    to be approximated.

    Larger M gives a finer ramp at the cost of a longer pitch (and a design grid
    that grows with it): M=1 -> {0, 45}, M=3 -> {0, 13.6, 28.1, 45}.

    In fourfold mode each rung above 0 deg is emitted at every requested azimuth,
    so the returned list carries (theta, phi, m, n) and grows with the azimuth
    count.
    """
    M = max(1, int(n_orders))
    s_max = float(np.sin(np.deg2rad(theta_max_deg)))
    if s_max <= 1e-9:
        raise ValueError("theta_max must be > 0")
    P = M * lam_um / s_max
    # Snap the pitch onto the FDTD mesh. A Bloch cell has to hold a whole number of
    # cells, so Lumerical silently retunes dx to P/N -- and then the design grid,
    # built at dx = 1/resolution, no longer lines up with the monitor and msopt
    # rejects the run outright ("x-grid mismatch: monitor=(114,) design=(113,)").
    # Rounding the pitch to a mesh multiple keeps dx exactly 1/resolution; the
    # target angle shifts by a fraction of a degree, which is reported below.
    snap_note = None
    if snap_per_um:
        cells = max(1, int(round(P * float(snap_per_um))))
        P_snap = cells / float(snap_per_um)
        if abs(P_snap - P) > 1e-12:
            s_got = min(M * lam_um / P_snap, 1.0)
            snap_note = (f"pitch snapped to the mesh: {P:.6f} -> {P_snap:.6f} um "
                         f"({cells} cells at {float(snap_per_um):g}/um); "
                         f"top order now lands at "
                         f"{np.rad2deg(np.arcsin(s_got)):.3f} deg "
                         f"(asked {theta_max_deg:g})")
            P = P_snap
    rungs = []
    for m in range(0, M + 1):
        u = m * lam_um / P
        rungs.append({"m": m, "u": float(u),
                      "theta_air_deg": float(np.rad2deg(np.arcsin(min(u, 1.0))))})
    # Orders above the target that still escape. Order M+1 sits at u = s + s/M, so
    # it stays evanescent -- emission past the target is physically impossible --
    # only while M <= s/(1-s). Past that ceiling the ladder leaks into angles the
    # FoM never sees, and the purity denominator is the only thing discouraging it.
    m_ceiling = int(np.floor(s_max / (1.0 - s_max))) if s_max < 1.0 else 10 ** 6
    leaks = []
    m = M + 1
    while m * lam_um / P < 1.0:
        u = m * lam_um / P
        leaks.append({"m": m, "u": float(u),
                      "theta_air_deg": float(np.rad2deg(np.arcsin(u)))})
        m += 1
    rep = [f"top order M={M} carries theta_max={theta_max_deg:g} deg "
           f"-> pitch P = {P:.5f} um (lambda/P = {lam_um/P:.5f})"]
    if snap_note:
        rep.insert(0, snap_note)
    rep += [f"  m={r['m']}  u={r['u']:.5f}  theta={r['theta_air_deg']:6.2f} deg"
            for r in rungs]
    if leaks:
        rep.append(f"  LEAK: M={M} exceeds the no-leak ceiling M<=s/(1-s)={m_ceiling}; "
                   f"{len(leaks)} order(s) escape ABOVE the target and are not in the FoM:")
        rep += [f"    m={r['m']}  u={r['u']:.5f}  theta={r['theta_air_deg']:6.2f} deg"
                for r in leaks]
    else:
        rep.append(f"  no order above the target can escape "
                   f"(M={M} within the ceiling s/(1-s)={m_ceiling})")
    return {"period_um": float(P), "orders_max": M, "rungs": rungs,
            "lam_um": float(lam_um), "n_organic": n_org,
            "max_orders_no_leak": m_ceiling, "leaking_orders": leaks,
            "theta_max_achieved_deg": float(rungs[-1]["theta_air_deg"]),
            "snapped": snap_note is not None, "report": rep}


def ramp_weights(rungs, mode="linear", w_at_0=1.0, w_at_max=0.85, axis="theta"):
    """The TARGET EMISSION PROFILE across the ladder, as a set of relative shares.

    These are the numbers the run is asking for: `w_at_0` at normal incidence
    falling (or rising) linearly to `w_at_max` at the outermost angle, with every
    rung in between interpolated. Only the RATIOS carry meaning -- w=(1, .93, .85)
    and w=(100, 93, 85) request the same shape -- so the caller is free to rescale.

    `axis` picks what the interpolation is linear in. "theta" is linear in the
    emission angle, which is what "0 deg is 1 and 45 deg is 0.85, straight line in
    between" means literally. "u" is linear in sin(theta) = the order index, which
    spaces the profile evenly across the diffraction orders instead.

    Read with `combine="log"` in OLED_rec: a weighted SUM of these would collapse
    onto whichever rung scores highest, so the profile is realized as a weighted
    geometric mean, whose optimum is purity proportional to the weight.
    """
    th = np.array([r["theta_air_deg"] for r in rungs], dtype=float)
    if mode == "flat":
        return np.ones_like(th).tolist()
    if mode != "linear":
        raise ValueError(f"ramp mode must be 'linear' or 'flat', got {mode!r}")
    x = np.sin(np.deg2rad(th)) if axis == "u" else th
    span = float(x.max())
    t = np.zeros_like(x) if span <= 1e-12 else x / span
    w = float(w_at_0) + (float(w_at_max) - float(w_at_0)) * t
    if np.any(w <= 0.0):
        raise ValueError("ramp weights must stay positive; check w_at_0 / w_at_max")
    return w.tolist()


def nearest_achievable(angles_deg, lam_um, m_max=12, n_candidates=6, u_max=1.0):
    """When the requested set has no exact pitch, the closest sets that do.

    Only angle sets whose SINES are in integer ratio can share one pitch under a
    normal-incidence probe, so most hand-picked sets (0/30/45 among them) are
    unreachable. Rather than fail, sweep the order assignment and report the pitch
    whose Bloch angles land closest to what was asked for.
    """
    ang = [float(a) for a in angles_deg]
    sines = [float(np.sin(np.deg2rad(a))) for a in ang]
    nz = [i for i, v in enumerate(sines) if abs(v) > 1e-9]
    if not nz:
        return []
    out = []
    seen = set()
    for m_lead in range(1, m_max + 1):
        for i0 in nz:
            P = m_lead * lam_um / sines[i0]
            step = lam_um / P
            if step <= 0 or step > u_max:
                continue
            got, orders, err = [], [], 0.0
            for a, sv in zip(ang, sines):
                if abs(sv) <= 1e-9:
                    got.append(0.0); orders.append(0); continue
                m = max(1, int(round(sv / step)))
                u = m * step
                if u > u_max:
                    err = float("inf"); break
                th = float(np.rad2deg(np.arcsin(u)))
                got.append(th); orders.append(m)
                err += abs(th - a)
            if not np.isfinite(err):
                continue
            key = tuple(np.round(got, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append({"period_um": float(P), "orders": orders,
                        "angles_deg": [float(v) for v in got],
                        "total_abs_error_deg": float(err),
                        "max_abs_error_deg": float(max(abs(g - a) for g, a in zip(got, ang)))})
    out.sort(key=lambda r: r["total_abs_error_deg"])
    return out[:n_candidates]


def target_profile(st, angles_deg, period_um=None, orders=None, orientation="h",
                   m_max=12, tol=0.01):
    """Everything OLED_rec needs from one call: the pitch to build the cell at,
    the order per angle, the in-plane momentum of each target, and how much power
    the planar stack has sitting at that momentum."""
    n_org = st.organic_index_max()
    rep = []
    if period_um is None:
        period_um, orders, rep = minimum_period(angles_deg, st.lam, m_max, tol, n_org)
        if period_um is None:
            raise ValueError("\n".join(rep))
    if orders is None:
        orders = [max(1, int(round(period_um * np.sin(np.deg2rad(a)) / st.lam)))
                  for a in angles_deg]
    u, dpdu, out_dpdu, bands = spectrum(st, orientation)
    modes = []
    for a, m in zip(angles_deg, orders):
        ui = (m * st.lam / period_um) if m else float(np.sin(np.deg2rad(a)))
        modes.append({
            "theta_air_deg": float(a), "order": int(m), "u": float(ui),
            "kx_per_um": float(2.0 * np.pi / st.lam * ui),
            "theta_org_deg": (float(np.rad2deg(np.arcsin(min(ui / n_org, 1.0))))
                              if ui <= n_org else None),
            "band": band_of(ui, bands),
            "reachable": bool(ui <= n_org + 1e-9),
            "dPdu": float(np.interp(ui, u, dpdu)) if ui <= u[-1] else 0.0,
        })
    return {"period_um": float(period_um), "orders": [int(m) for m in orders],
            "modes": modes, "n_organic": float(n_org), "lam_um": float(st.lam),
            "bands": [[b[0], b[1], b[2]] for b in bands], "report": rep}


# ---------------------------------------------------------------------------
# TMM spectrum
# ---------------------------------------------------------------------------
def spectrum(st, orientation="h", u_max=None, n_u=4000):
    """dP/du on a dense u grid, plus the outcoupled part and the band edges.

    Returned power is per unit u and normalized to the total emitted power, so
    the numbers are comparable between stacks.
    """
    if u_max is None:
        u_max = max(3.0, 1.2 * st.organic_index_max() + 1.0)
    u = np.linspace(1e-4, float(u_max), int(n_u))
    rows = ld.components(st, u, orientation)
    total = np.asarray(rows[ld.IDX_TOTAL], dtype=float)
    out = np.asarray(rows[ld.IDX_OUT], dtype=float)
    norm = float(np.trapezoid(total, u)) or 1.0
    return u, total / norm, out / norm, ld.make_bands(st, float(u_max))


def find_modes(u, dpdu, bands, min_share=0.005):
    """Local maxima of dP/du outside the escape cone -- the trapped modes a
    grating would have to reach.  Each peak is integrated over its own valley-to-
    valley interval so the share is the power actually sitting in that mode, not
    a peak height that depends on the grid."""
    peaks = []
    for i in range(1, len(u) - 1):
        if u[i] <= 1.0:
            continue
        if dpdu[i] > dpdu[i - 1] and dpdu[i] >= dpdu[i + 1]:
            lo = i
            while lo > 1 and dpdu[lo - 1] < dpdu[lo]:
                lo -= 1
            hi = i
            while hi < len(u) - 2 and dpdu[hi + 1] < dpdu[hi]:
                hi += 1
            share = float(np.trapezoid(dpdu[lo:hi + 1], u[lo:hi + 1]))
            if share >= min_share:
                peaks.append({"u": float(u[i]), "share": share,
                              "band": band_of(float(u[i]), bands),
                              "u_lo": float(u[lo]), "u_hi": float(u[hi])})
    peaks.sort(key=lambda p: -p["share"])
    # merge peaks that share an interval (the same mode found twice)
    merged = []
    for p in peaks:
        if not any(abs(p["u"] - q["u"]) < 1e-6 for q in merged):
            merged.append(p)
    return merged


# ---------------------------------------------------------------------------
# The mapping table
# ---------------------------------------------------------------------------
def build_mapping(st, angles_deg, orders, period_um, u, dpdu, bands):
    """One row per (target angle, order): the momentum that has to be harvested,
    what band it falls in, and how much power the planar stack puts there."""
    lam = st.lam
    n_org = st.organic_index_max()
    rows = []
    for th in angles_deg:
        for m in orders:
            ui = u_for_angle(th, m, period_um, lam)
            row = {
                "theta_air_deg": float(th),
                "order": int(m),
                "period_um": float(period_um),
                "u": ui,
                "kx_per_um": float(2.0 * np.pi / lam * ui),
                "band": band_of(ui, bands),
                "reachable": bool(ui <= n_org + 1e-9),
                "theta_org_deg": (float(np.rad2deg(np.arcsin(min(ui / n_org, 1.0))))
                                  if ui <= n_org else None),
            }
            # local power density at that momentum, and the share within +-du/2
            if ui <= u[-1]:
                row["dPdu"] = float(np.interp(ui, u, dpdu))
                half = 0.5 * lam / period_um
                sel = (u >= ui - half) & (u <= ui + half)
                row["share_in_window"] = float(np.trapezoid(dpdu[sel], u[sel])) if sel.sum() > 1 else 0.0
            else:
                row["dPdu"] = 0.0
                row["share_in_window"] = 0.0
            rows.append(row)
    return rows


def suggest_periods(st, angles_deg, orders, modes):
    """For each trapped mode and each requested angle, the pitch that couples
    them through a given order.  This is the design question in reverse: 'what
    pitch do I need so that THIS mode lands at THAT angle'."""
    out = []
    for md in modes:
        for th in angles_deg:
            for m in orders:
                P = period_for(th, md["u"], m, st.lam)
                if P is None or not (0.05 <= P <= 20.0):
                    continue
                out.append({"mode_u": md["u"], "mode_band": md["band"],
                            "mode_share": md["share"], "theta_air_deg": float(th),
                            "order": int(m), "period_um": P})
    out.sort(key=lambda r: (-r["mode_share"], r["theta_air_deg"], r["order"]))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(st, angles, orders, period, u, dpdu, out_dpdu, bands, modes, mapping, suggest):
    L = []
    L.append(st.label or "stack")
    L.append(st.describe())
    L.append("")
    L.append(f"  lambda = {st.lam:.4f} um   k0 = {2*np.pi/st.lam:.4f} rad/um   "
             f"n_organic(max) = {st.organic_index_max():.4f}")
    L.append(f"  assumed out-coupler pitch P = {period:.4f} um  ->  lambda/P = "
             f"{st.lam/period:.4f} (one order step in u)")
    L.append("")
    L.append("  band shares of the planar stack (fraction of emitted power)")
    for name, lo, hi in bands:
        sel = (u >= lo) & (u < hi)
        s = float(np.trapezoid(dpdu[sel], u[sel])) if sel.sum() > 1 else 0.0
        L.append(f"    {name:<16s} u in [{lo:6.3f}, {hi:6.3f})   {100*s:6.2f} %")
    L.append("")
    L.append("  trapped modes (local maxima of dP/du outside the escape cone)")
    if modes:
        L.append(f"    {'u':>8s} {'kx (rad/um)':>13s} {'band':<16s} {'share':>8s}")
        for md in modes[:12]:
            L.append(f"    {md['u']:8.4f} {2*np.pi/st.lam*md['u']:13.4f} "
                     f"{md['band']:<16s} {100*md['share']:7.2f} %")
    else:
        L.append("    (none above the reporting threshold)")
    L.append("")
    L.append(f"  MAPPING: momentum each target angle needs, at P = {period:.4f} um")
    L.append(f"    {'theta_air':>10s} {'m':>3s} {'u':>8s} {'kx (rad/um)':>13s} "
             f"{'theta_org':>10s} {'band':<16s} {'dP/du':>10s} {'share':>8s}")
    for r in mapping:
        to = "  --  " if r["theta_org_deg"] is None else f"{r['theta_org_deg']:9.2f}"
        flag = "" if r["reachable"] else "   <- beyond n_organic: no propagating mode carries it"
        L.append(f"    {r['theta_air_deg']:10.1f} {r['order']:3d} {r['u']:8.4f} "
                 f"{r['kx_per_um']:13.4f} {to:>10s} {r['band']:<16s} "
                 f"{r['dPdu']:10.4f} {100*r['share_in_window']:7.2f} %{flag}")
    L.append("")
    L.append("  REVERSE: pitch that couples an existing trapped mode to a target angle")
    if suggest:
        L.append(f"    {'mode u':>8s} {'band':<16s} {'share':>8s} {'->':>3s} "
                 f"{'theta_air':>10s} {'m':>3s} {'period (um)':>12s}")
        for r in suggest[:20]:
            L.append(f"    {r['mode_u']:8.4f} {r['mode_band']:<16s} "
                     f"{100*r['mode_share']:7.2f} % {'->':>3s} {r['theta_air_deg']:10.1f} "
                     f"{r['order']:3d} {r['period_um']:12.4f}")
    else:
        L.append("    (no pitch in 0.05-20 um couples the found modes to the requested angles)")
    return "\n".join(L)


def plot_mapping(st, u, dpdu, out_dpdu, bands, modes, mapping, path):
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.semilogy(u, np.maximum(dpdu, 1e-8), lw=1.6, color="tab:blue", label="total dP/du")
    ax.semilogy(u, np.maximum(out_dpdu, 1e-8), lw=1.3, color="tab:green",
                label="outcoupled into air")
    colors = {"air": "#eaf4ea", "substrate": "#eef2fa", "waveguide": "#fdf3e3",
              "evanescent/SPP": "#fae9e9"}
    for name, lo, hi in bands:
        ax.axvspan(lo, min(hi, u[-1]), color=colors.get(name, "0.95"), zorder=0)
        ax.text(0.5 * (lo + min(hi, u[-1])), 0.90, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="0.35")
    # stagger the mode labels: neighbouring waveguide peaks sit within ~0.1 in u
    # and their annotations otherwise print on top of each other
    for k, md in enumerate(modes[:8]):
        ax.axvline(md["u"], color="0.55", ls=":", lw=1.0)
        ax.annotate(f"u={md['u']:.3f} ({100*md['share']:.1f}%)",
                    (md["u"], 1.005 + 0.045 * (k % 3)),
                    xycoords=("data", "axes fraction"), fontsize=7.5, color="0.35",
                    ha="center", va="bottom", rotation=0)
    seen = {}
    for r in mapping:
        if r["u"] > u[-1]:          # off-scale: the line would land outside the axes
            continue
        c = plt.cm.tab10(int(r["theta_air_deg"]) % 10)
        lab = None if r["theta_air_deg"] in seen else f"target {r['theta_air_deg']:.0f} deg"
        seen[r["theta_air_deg"]] = True
        ax.axvline(r["u"], color=c, ls="--", lw=1.5, label=lab)
        ax.annotate(f"m{r['order']}", (r["u"], 0.02), xycoords=("data", "axes fraction"),
                    fontsize=7.5, color=c, ha="center")
    ax.set_xlabel("in-plane momentum  u = k_par / k0   (= n sin(theta) in every layer)")
    ax.set_ylabel("dP/du   (normalized to total emitted power)")
    ax.set_xlim(0, u[-1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{st.label}\ndashed = momentum each target angle needs   |   "
                 f"dotted = trapped modes the stack actually has", pad=34)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stack", default="microcavity", choices=["microcavity", "legacy"])
    ap.add_argument("--color", default="green")
    ap.add_argument("--kind", default=None, help="microcavity variant: optimized | table")
    ap.add_argument("--angles", default="0,30,45,60", help="target emission angles in air, deg")
    ap.add_argument("--orders", default="1,2,3", help="diffraction orders to consider")
    ap.add_argument("--period", type=float, default=None,
                    help="out-coupler pitch in um (default lambda/(n_org-1), the pitch "
                         "whose first order just spans organic -> normal)")
    ap.add_argument("--orientation", default="h", choices=["h", "v"])
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)

    st = (ld.build_microcavity_stack(a.color, a.kind) if a.stack == "microcavity"
          else ld.build_legacy_stack())
    angles = [float(v) for v in a.angles.split(",") if v.strip()]
    orders = [int(v) for v in a.orders.split(",") if v.strip()]
    n_org = st.organic_index_max()
    period = a.period if a.period else st.lam / max(n_org - 1.0, 1e-3)

    u, dpdu, out_dpdu, bands = spectrum(st, a.orientation)
    modes = find_modes(u, dpdu, bands)
    mapping = build_mapping(st, angles, orders, period, u, dpdu, bands)
    suggest = suggest_periods(st, angles, orders, modes)

    outdir = a.outdir or os.path.join(os.environ.get("EIDL_RUN_DIR", os.getcwd()), "A")
    os.makedirs(outdir, exist_ok=True)
    report = format_report(st, angles, orders, period, u, dpdu, out_dpdu,
                           bands, modes, mapping, suggest)
    print(report)
    with open(os.path.join(outdir, "k_mapping.txt"), "w", encoding="utf-8") as fp:
        fp.write(report + "\n")
    with open(os.path.join(outdir, "k_mapping.json"), "w", encoding="utf-8") as fp:
        json.dump({"label": st.label, "lam_um": st.lam, "n_organic": n_org,
                   "period_um": period, "angles_deg": angles, "orders": orders,
                   "bands": [[b[0], b[1], b[2]] for b in bands],
                   "modes": modes, "mapping": mapping, "suggested_periods": suggest[:60]},
                  fp, indent=2)
    np.savetxt(os.path.join(outdir, "k_mapping_spectrum.txt"),
               np.column_stack([u, dpdu, out_dpdu]),
               header="u dPdu_total dPdu_outcoupled (normalized to total emitted power)")
    p = plot_mapping(st, u, dpdu, out_dpdu, bands, modes, mapping,
                     os.path.join(outdir, "k_mapping.png"))
    print(f"\n  wrote {outdir}/k_mapping.{{txt,json,png}} and k_mapping_spectrum.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
