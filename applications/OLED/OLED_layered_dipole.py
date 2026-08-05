#!/usr/bin/env python3
"""Analytic dipole-in-planar-multilayer solver (Chance-Prock-Silbey / Lukosz / Neyts).

Pure numpy/scipy.  No Lumerical, no FDTD, runs in seconds.

WHY THIS EXISTS
---------------
The FDTD postprocess measures light extraction by integrating flux through a top
monitor inside a finite lateral domain.  For a PLANAR stack that never converges:
a planar stack emits all the way to grazing incidence, so the escape cone always
leaks past the lateral PML and the measured LEE grows monotonically with the
domain width instead of saturating (we measured 0.835 / 0.904 / 1.044 % at
2.72 / 3.72 / 5.65 um lateral size).  The transfer-matrix / in-plane-wavevector
formulation below has no domain to truncate -- the integral over u = k_par/k0 runs
to infinity -- and it additionally separates the loss channels (metal absorption
per layer, waveguided, SPP), which a single FDTD flux monitor cannot do.

PHYSICS
-------
Layers j = 0 .. N-1 stacked bottom -> top.  j = 0 and j = N-1 are semi-infinite;
1 .. N-2 are finite with thickness d_j.  Indices n_j are arbitrary complex
(time convention exp(-i w t), so a lossy medium has Im(n) > 0).

For an in-plane wavevector k_par, u = k_par / k0 and

    l_j = k_zj / k0 = sqrt(n_j^2 - u^2),      branch: Im(l_j) >= 0, Re(l_j) >= 0

(the principal branch of numpy's complex sqrt already satisfies both for
Im(n) >= 0; the code enforces it anyway).

Fresnel amplitudes for an upward wave crossing interface j | j+1 (this is the
Born & Wolf "parallel" convention, in which r_p(0) = -r_s(0) = (n2-n1)/(n2+n1)):

    r_s = (l_j - l_{j+1}) / (l_j + l_{j+1})
    t_s = 2 l_j / (l_j + l_{j+1})
    r_p = (n_{j+1}^2 l_j - n_j^2 l_{j+1}) / (n_{j+1}^2 l_j + n_j^2 l_{j+1})
    t_p = 2 n_j n_{j+1} l_j / (n_{j+1}^2 l_j + n_j^2 l_{j+1})

Both r's are antisymmetric under j <-> j+1 and t t' = 1 - r^2, so the standard
compact recursions hold.  Looking UP from interface i and DOWN from interface i:

    Ru[N-2] = r_i(N-2)
    Ru[i]   = (r_i + Ru[i+1] P_{i+1}) / (1 + r_i Ru[i+1] P_{i+1})
    Rd[0]   = -r_i(0)
    Rd[i]   = (-r_i + Rd[i-1] P_i) / (1 - r_i Rd[i-1] P_i)         P_j = exp(2 i k0 l_j d_j)

Referenced to the dipole plane inside layer e, at height z_off above the bottom
interface of e (h_up = d_e - z_off above it):

    R_u = Ru[e]   exp(2 i k0 l_e h_up)
    R_d = Rd[e-1] exp(2 i k0 l_e z_off)
    D   = 1 - R_u R_d

DIPOLE SOURCE.  From the Weyl/angular-spectrum expansion of a dipole field, the
plane-wave E-amplitude emitted at the dipole plane carries a 1/k_z factor and the
projection of p onto the polarization unit vector (p_hat = s_hat x k_hat):

    vertical   (p || z), p-pol :  s_up = s_dn = +u / (n_e l_e)
    horizontal (p || x), s-pol :  s_up = s_dn = +1 / l_e         (x sin(phi))
    horizontal (p || x), p-pol :  s_up = -s_dn = +1 / n_e        (x cos(phi))

The azimuth average of sin^2 phi and cos^2 phi is 1/2, so each horizontal channel
carries a factor 1/2 and the two add.  The up/down symmetry of the source is what
produces the familiar (1 + R) for vertical-p and horizontal-s and (1 - R) for
horizontal-p.  Solving the two-way cavity,

    U = (s_up + R_d s_dn) / D        (up-going amplitude just above the dipole)
    W = (s_dn + R_u s_up) / D        (down-going amplitude just below the dipole)

POWER.  For amplitudes (a+, a-) referenced at a plane inside a layer of index n
and l, the net upward Poynting flux is  S_z = 0.5 Re[G (a+ + a-) conj(a+ - a-)]
with G = conj(l) for s and G = conj(l) n / conj(n) for p (the two share the same
overall constant 1/(2 eta0 k0), which is why the s/p and h/v weights below are
directly comparable).  Everything is then obtained from the field solution:

    dP_total/du  = u [ S_z(z_d+) - S_z(z_d-) ]
    dP_out/du    = u S_z(top semi-infinite medium)       (nonzero only for u < n_top)
    dP_bottom/du = u [-S_z(bottom semi-infinite medium)] (nonzero only for u < n_bot)
    dP_abs_j/du  = u [ S_z(bottom of j) - S_z(top of j) ]

so outcoupled + bottom + sum_j absorbed == total is an identity of the solution,
and checking it numerically checks the whole Fresnel/TMM chain (the fluxes are
computed independently on each side of every interface).

The closed CPS form is also implemented as a cross-check.  Carrying the algebra
through with a symmetric source (s_up = s_dn = s) gives exactly

    dP/du = u |s|^2 Re[ G (1 + R_u)(1 + R_d) / D ]

and with an antisymmetric source (s_up = -s_dn = s)

    dP/du = u |s|^2 Re[ conj(G) (1 - R_u)(1 - R_d) / D ]

(identical for real l and real n, where they reduce to the textbook
u^3/(n^2 l), u/l and u l/n^2 weights of Furno et al., PRB 85, 115205 (2012)).
Validation checks the two agree to 1e-9.

NORMALIZATION -- how the prefactors are fixed
---------------------------------------------
The prefactors above were derived, but they are then FIXED BY NORMALIZATION and
never trusted in absolute terms: the free-space reference P_free is computed with
exactly the same code path, on a stack whose every index equals the emitter index
(so R_u = R_d = 0 identically), separately for each orientation, and every
reported Purcell factor is F = P_total / P_free while every reported channel
fraction is P_channel / P_total.  Any overall constant -- 2 pi k0^2 from d^2k_par,
1/(2 eta0 k0) from the Poynting factor, the dipole moment, eps0 -- cancels
identically in both.  As a consequence F = 1 for a homogeneous medium by
construction; validation test 1 checks that a *multilayer* whose layers all share
the emitter index (which exercises the recursions, the interfaces and the
transfer matrices, all of which must return exactly zero reflection) still gives
F = 1, and additionally that the free-space integral equals its analytic value
2 n_e / 3 in the code's internal units.

NUMERICS
--------
The integrand has (a) integrable inverse-square-root branch points at every real
layer index u = Re(n_j) -- most importantly u = n_e, where the horizontal-s
weight u/l_e diverges -- and (b) narrow Lorentzian poles at the waveguide and
SPP resonances, whose width is set by the metal loss.  Both are handled:

 * The u axis is cut into base segments at u = 0, every Re(n_j), every channel
   boundary and a geometric ladder out to u_max; each segment is integrated in a
   mapped variable u(t) = a + (b-a)(3t^2 - 2t^3), whose derivative vanishes at
   both ends.  That turns any endpoint 1/sqrt into a bounded integrand and
   clusters nodes at exactly the branch points.
 * Inside each segment the quadrature is adaptive: a 12-point Gauss-Legendre rule
   (open, so the endpoints are never evaluated) is compared against the same rule
   on the two halves, and only the intervals that fail the local tolerance are
   bisected.  Resonances therefore get refined automatically, to whatever depth
   they need, without over-sampling the smooth regions.
 * u_max is chosen by doubling until the contribution of [u_max, 2 u_max] falls
   below 1e-9 of the total, so the evanescent tail is resolved rather than cut.
 * The whole calculation is repeated with successively tighter tolerances until
   every reported fraction moves by less than 1e-3; the number of quadrature
   nodes actually used is printed.
 * All exponentials are arranged to be of the decaying form exp(+i k0 l d) with
   Im(l) >= 0 (amplitudes are stored referenced to the bottom of a layer for the
   up-going wave and to the top for the down-going wave), so nothing overflows
   even at u ~ 10^3.

CHANNELS
--------
Two complementary decompositions are printed, each summing to 100 %:

  (1) SINKS (exact energy balance): outcoupled into the top medium, transmitted
      into the bottom medium, and absorption resolved per layer.
  (2) u-BANDS of the total dissipated power: air cone u < n_top, substrate band
      n_top < u < n_sub (if a substrate index is declared), waveguide band up to
      max Re(n) of the organic/dielectric layers, and SPP/evanescent above that.

A band x sink cross table is printed as well, because "waveguide" and "SPP" are
u-space labels for power that physically terminates in absorption -- the two
decompositions are views of the same total, not additive with each other.

Usage
-----
    python OLED_layered_dipole.py --validate
    python OLED_layered_dipole.py --stack microcavity
    python OLED_layered_dipole.py --stack legacy
    python OLED_layered_dipole.py                 # both stacks

Results (tables + PNG of dP/du with the channel bands shaded) go to
$EIDL_RUN_DIR/A/ when EIDL_RUN_DIR is set, else the current directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Stack container
# ---------------------------------------------------------------------------


class Stack:
    """Planar stack with a dipole plane inside one finite layer.

    names  : list of N layer names, bottom -> top
    n      : list of N complex refractive indices (Im(n) > 0 means loss)
    d      : list of N thicknesses in um; entries 0 and N-1 are ignored
             (semi-infinite media)
    lam    : vacuum wavelength in um
    e      : index of the emitter layer, must satisfy 1 <= e <= N-2
    z_off  : dipole height above the BOTTOM interface of layer e, in um
    n_sub  : optional substrate index used only to draw the "substrate" band
    """

    def __init__(self, names, n, d, lam, e, z_off, n_sub=None, label=""):
        self.names = list(names)
        self.n = np.asarray(n, dtype=complex)
        self.d = np.asarray([0.0 if x is None else float(x) for x in d], dtype=float)
        self.lam = float(lam)
        self.e = int(e)
        self.z_off = float(z_off)
        self.n_sub = None if n_sub is None else float(n_sub)
        self.label = label
        self.N = len(self.n)
        if not (1 <= self.e <= self.N - 2):
            raise ValueError("emitter must sit in a finite (inner) layer")
        if not (0.0 <= self.z_off <= self.d[self.e]):
            raise ValueError("z_off must lie inside the emitter layer")
        if abs(self.n[self.e].imag) > 1e-12:
            print(f"[warn] emitter layer {self.names[self.e]} is lossy "
                  f"(Im n = {self.n[self.e].imag:g}); the free-space Purcell "
                  f"reference for a lossy host is not well defined.")

    # -- convenience ---------------------------------------------------------
    @property
    def k0(self):
        return 2.0 * np.pi / self.lam

    @property
    def n_top(self):
        return float(self.n[-1].real)

    @property
    def n_bot(self):
        return float(self.n[0].real)

    def organic_index_max(self, k_tol=0.05):
        """Largest Re(n) among the finite, essentially lossless (dielectric)
        layers -- the top of the "waveguide" band."""
        vals = [self.n[j].real for j in range(1, self.N - 1)
                if abs(self.n[j].imag) < k_tol]
        return float(max(vals)) if vals else max(self.n_top, self.n_bot)

    def free_space_stack(self):
        """Same emitter index everywhere: R_u = R_d = 0 identically.  Used as the
        Purcell / normalization reference through the identical code path."""
        ne = self.n[self.e]
        return Stack(["free_bot", "free_mid", "free_top"], [ne, ne, ne],
                     [None, self.d[self.e], None], self.lam, 1, self.z_off,
                     label="homogeneous reference")

    def describe(self):
        out = [f"  {'#':>2s} {'layer':<20s}{'t (nm)':>9s}{'n':>8s}{'k':>8s}"]
        for j in range(self.N):
            t = "inf" if j in (0, self.N - 1) else f"{self.d[j] * 1e3:.1f}"
            mark = "  <- dipole" if j == self.e else ""
            out.append(f"  {j:2d} {self.names[j]:<20s}{t:>9s}"
                       f"{self.n[j].real:8.3f}{self.n[j].imag:8.3f}{mark}")
        out.append(f"  dipole at {self.z_off * 1e3:.1f} nm above the bottom of "
                   f"{self.names[self.e]}  (lambda = {self.lam:.3f} um)")
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Core electromagnetics
# ---------------------------------------------------------------------------


_ELL_FLOOR = 1e-14


def _ell(n, u):
    """l = sqrt(n^2 - u^2) in units of k0, with Im(l) >= 0 and Re(l) >= 0.

    A tiny evanescent floor is applied when l would be exactly zero (u sitting
    bit-exactly on a real layer index).  That point is a measure-zero grazing
    branch point; the floor keeps every ratio at its correct limit -- r -> 0,
    t -> 1 between identical media, r_s -> 1 at grazing on a real interface --
    instead of evaluating 0/0.  The open Gauss-Legendre nodes never land on a
    segment endpoint, so this only ever affects plotting / diagnostic grids.
    """
    val = np.sqrt(np.asarray(complex(n)) ** 2 - np.asarray(u, dtype=float) ** 2 + 0j)
    val = np.where(val.imag < 0.0, -val, val)
    val = np.where((val.imag == 0.0) & (val.real < 0.0), -val, val)
    return np.where(np.abs(val) < _ELL_FLOOR, _ELL_FLOOR * 1j, val)


def _fresnel(n1, n2, l1, l2, pol):
    """r, t for a wave travelling 1 -> 2 (Born & Wolf convention)."""
    if pol == "s":
        den = l1 + l2
        return (l1 - l2) / den, 2.0 * l1 / den
    den = n2 * n2 * l1 + n1 * n1 * l2
    return (n2 * n2 * l1 - n1 * n1 * l2) / den, 2.0 * n1 * n2 * l1 / den


def _flux(pol, n, l, ap, am):
    """Net upward Poynting flux (shared constant 1/(2 eta0 k0) dropped)."""
    g = np.conj(l) if pol == "s" else np.conj(l) * (n / np.conj(n))
    return 0.5 * np.real(g * (ap + am) * np.conj(ap - am))


def _solve_pol(st, u, pol, s_up, s_dn):
    """Full field solution for one polarization.

    Returns a dict with the flux-based channel spectra (WITHOUT the factor u and
    WITHOUT the azimuth weight) plus the closed-form CPS total.
    """
    k0 = st.k0
    N = st.N
    e = st.e
    nn = st.n

    l = [_ell(nn[j], u) for j in range(N)]
    phi = [None] * N          # k0 l d  (phase across a finite layer)
    P = [None] * N            # exp(2 i phi)
    for j in range(1, N - 1):
        phi[j] = k0 * l[j] * st.d[j]
        P[j] = np.exp(2j * phi[j])

    rup = [None] * (N - 1)
    tup = [None] * (N - 1)
    tdn = [None] * (N - 1)
    for i in range(N - 1):
        rup[i], tup[i] = _fresnel(nn[i], nn[i + 1], l[i], l[i + 1], pol)
        _, tdn[i] = _fresnel(nn[i + 1], nn[i], l[i + 1], l[i], pol)

    # reflection looking up from interface i, and looking down onto interface i
    Ru = [None] * (N - 1)
    Ru[N - 2] = rup[N - 2]
    for i in range(N - 3, -1, -1):
        x = Ru[i + 1] * P[i + 1]
        Ru[i] = (rup[i] + x) / (1.0 + rup[i] * x)
    Rd = [None] * (N - 1)
    Rd[0] = -rup[0]
    for i in range(1, N - 1):
        x = Rd[i - 1] * P[i]
        Rd[i] = (-rup[i] + x) / (1.0 - rup[i] * x)

    le = l[e]
    h_up = st.d[e] - st.z_off
    ph_up = k0 * le * h_up
    ph_dn = k0 * le * st.z_off
    R_u = Ru[e] * np.exp(2j * ph_up)
    R_d = Rd[e - 1] * np.exp(2j * ph_dn)
    D = 1.0 - R_u * R_d
    U = (s_up + R_d * s_dn) / D
    W = (s_dn + R_u * s_up) / D

    zero = np.zeros_like(U)

    # --- amplitudes everywhere -------------------------------------------
    # For each layer store Pb (up-going, referenced at the layer's BOTTOM) and
    # Mt (down-going, referenced at the layer's TOP).  Both stay bounded.
    Pb = [None] * N
    Mt = [None] * N

    # downward chain, starting from the emitter layer's bottom interface
    m_bot_above = W * np.exp(1j * ph_dn)           # M at the bottom of layer e
    for j in range(e - 1, -1, -1):
        if j == 0:
            Mt[0] = tdn[0] * m_bot_above
            Pb[0] = zero
        else:
            denom = 1.0 - rup[j] * Rd[j - 1] * P[j]
            Mt[j] = tdn[j] * m_bot_above / denom
            Pb[j] = Rd[j - 1] * np.exp(1j * phi[j]) * Mt[j]
            m_bot_above = Mt[j] * np.exp(1j * phi[j])

    # upward chain, starting from the emitter layer's top interface
    p_top_below = U * np.exp(1j * ph_up)           # P at the top of layer e
    for i in range(e, N - 1):
        if i + 1 == N - 1:
            Pb[N - 1] = tup[i] * p_top_below
            Mt[N - 1] = zero
        else:
            denom = 1.0 + rup[i] * Ru[i + 1] * P[i + 1]
            Pb[i + 1] = tup[i] * p_top_below / denom
            Mt[i + 1] = Ru[i + 1] * np.exp(1j * phi[i + 1]) * Pb[i + 1]
            p_top_below = Pb[i + 1] * np.exp(1j * phi[i + 1])

    # --- fluxes -----------------------------------------------------------
    def slab(j, pb, mt, thickness):
        ex = np.exp(1j * k0 * l[j] * thickness)
        s_bot = _flux(pol, nn[j], l[j], pb, mt * ex)
        s_top = _flux(pol, nn[j], l[j], pb * ex, mt)
        return s_bot, s_top

    absorption = np.zeros((N,) + U.shape)
    interface_flux = np.zeros((N - 1,) + U.shape)      # from below each interface
    interface_flux_above = np.zeros((N - 1,) + U.shape)

    for j in range(1, N - 1):
        if j == e:
            continue
        s_bot, s_top = slab(j, Pb[j], Mt[j], st.d[j])
        absorption[j] = s_bot - s_top
        interface_flux[j - 1] = s_bot          # flux at interface j-1 seen in layer j
        interface_flux_above[j] = s_top        # flux at interface j seen in layer j

    # emitter layer: two sub-slabs split at the dipole plane
    lower_pb = Rd[e - 1] * np.exp(1j * ph_dn) * W
    lower_mt = W
    lo_bot, lo_top = slab(e, lower_pb, lower_mt, st.z_off)
    upper_pb = U
    upper_mt = Ru[e] * np.exp(1j * ph_up) * U
    up_bot, up_top = slab(e, upper_pb, upper_mt, h_up)
    absorption[e] = (lo_bot - lo_top) + (up_bot - up_top)
    interface_flux[e - 1] = lo_bot
    interface_flux_above[e] = up_top

    s_top_medium = _flux(pol, nn[N - 1], l[N - 1], Pb[N - 1], zero)
    s_bot_medium = _flux(pol, nn[0], l[0], zero, Mt[0])
    interface_flux_above[0] = s_bot_medium
    interface_flux[N - 2] = s_top_medium

    p_out = s_top_medium
    p_bottom = -s_bot_medium
    p_total_flux = up_bot - lo_top             # S(z_d+) - S(z_d-)

    # closed-form CPS cross-check
    g = np.conj(le) if pol == "s" else np.conj(le) * (nn[e] / np.conj(nn[e]))
    symmetric = np.allclose(s_up, s_dn)
    if symmetric:
        p_total_cps = np.abs(s_up) ** 2 * np.real(g * (1.0 + R_u) * (1.0 + R_d) / D)
    else:
        p_total_cps = np.abs(s_up) ** 2 * np.real(np.conj(g) * (1.0 - R_u) * (1.0 - R_d) / D)

    return {
        "total": p_total_flux,
        "total_cps": p_total_cps,
        "out": p_out,
        "bottom": p_bottom,
        "absorption": absorption,
        "iface_below": interface_flux,
        "iface_above": interface_flux_above,
    }


def _source_amplitudes(st, u, orientation, pol):
    ne = st.n[st.e]
    le = _ell(ne, u)
    if orientation == "v":
        s = u / (ne * le)
        return s, s, 1.0
    if pol == "s":
        s = 1.0 / le
        return s, s, 0.5
    s = np.full_like(le, 1.0 / ne)
    return s, -s, 0.5


def components(st, u, orientation):
    """Channel spectra dP/du at the given u values, as a (M, len(u)) array.

    Row layout:  0 total, 1 outcoupled, 2 bottom, 3..3+N-1 per-layer absorption
    (rows for the semi-infinite media are identically zero), then the CPS
    cross-check total, then the largest interface flux mismatch.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    N = st.N
    pols = ["p"] if orientation == "v" else ["s", "p"]
    tot = np.zeros_like(u)
    tot_cps = np.zeros_like(u)
    out = np.zeros_like(u)
    bot = np.zeros_like(u)
    absn = np.zeros((N, u.size))
    mismatch = np.zeros_like(u)
    for pol in pols:
        s_up, s_dn, w = _source_amplitudes(st, u, orientation, pol)
        r = _solve_pol(st, u, pol, s_up, s_dn)
        tot += w * r["total"]
        tot_cps += w * r["total_cps"]
        out += w * r["out"]
        bot += w * r["bottom"]
        absn += w * r["absorption"]
        mismatch = np.maximum(mismatch,
                              np.max(np.abs(r["iface_below"] - r["iface_above"]), axis=0))
    rows = [u * tot, u * out, u * bot]
    rows += [u * absn[j] for j in range(N)]
    rows += [u * tot_cps, u * mismatch]
    return np.asarray(rows)


def n_components(st):
    return 3 + st.N + 2


IDX_TOTAL, IDX_OUT, IDX_BOTTOM = 0, 1, 2


def idx_abs(st, j):
    return 3 + j


def idx_cps(st):
    return 3 + st.N


def idx_mismatch(st):
    return 3 + st.N + 1


# ---------------------------------------------------------------------------
# Adaptive quadrature over u
# ---------------------------------------------------------------------------

_GL_N = 12
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_GL_N)


def _map_u(t, a, b):
    """Smoothstep map [0,1] -> [a,b]; du/dt vanishes at both ends, which
    regularizes any endpoint inverse-square-root singularity."""
    u = a + (b - a) * (3.0 * t ** 2 - 2.0 * t ** 3)
    du = (b - a) * 6.0 * t * (1.0 - t)
    return u, du


def _gl_batch(f, lo, hi, a, b):
    """Gauss-Legendre on many t-intervals at once -> (M, n_intervals)."""
    half = 0.5 * (hi - lo)
    t = lo[None, :] + half[None, :] * (_GL_X[:, None] + 1.0)
    u, du = _map_u(t, a, b)
    vals = f(u.ravel())
    vals = vals.reshape(vals.shape[0], t.shape[0], t.shape[1]) * du[None, :, :]
    return np.einsum("i,mij->mj", _GL_W, vals) * half[None, :], u.size


def integrate_segment(f, a, b, atol, max_intervals=2 ** 17):
    """Adaptive Gauss-Legendre on one base segment.  Returns (integral, n_nodes)."""
    lo = np.array([0.0])
    hi = np.array([1.0])
    total = None
    nodes = 0
    for _ in range(60):
        if lo.size == 0:
            break
        i1, k = _gl_batch(f, lo, hi, a, b)
        nodes += k
        mid = 0.5 * (lo + hi)
        ia, k = _gl_batch(f, lo, mid, a, b)
        nodes += k
        ib, k = _gl_batch(f, mid, hi, a, b)
        nodes += k
        i2 = ia + ib
        err = np.max(np.abs(i1 - i2), axis=0)
        ok = err <= atol * (hi - lo)
        if total is None:
            total = np.zeros(i2.shape[0])
        if np.any(ok):
            total += np.sum(i2[:, ok], axis=1)
        bad = ~ok
        if not np.any(bad):
            break
        if 2 * int(np.count_nonzero(bad)) > max_intervals:
            total += np.sum(i2[:, bad], axis=1)
            break
        lo = np.concatenate([lo[bad], mid[bad]])
        hi = np.concatenate([mid[bad], hi[bad]])
    else:
        if total is None:
            total = np.zeros(1)
    return total, nodes


def build_segments(st, u_max, extra_breaks=()):
    """Base segments: breakpoints at 0, every real layer index, the channel
    boundaries, and a geometric ladder through the evanescent tail."""
    brk = {0.0, float(u_max)}
    for j in range(st.N):
        val = float(st.n[j].real)
        if 0.0 < val < u_max:
            brk.add(val)
    for val in extra_breaks:
        if 0.0 < float(val) < u_max:
            brk.add(float(val))
    pts = sorted(brk)
    # geometric ladder above the highest real index
    top = pts[-2] if len(pts) > 1 else 0.0
    top = max(top, 1e-3)
    if u_max > 1.05 * top:
        k = 12
        for i in range(1, k):
            brk.add(top * (u_max / top) ** (i / k))
    pts = sorted(brk)
    # never let a base segment be wider than 0.4 in u (helps the adaptive start)
    refined = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        m = max(1, int(np.ceil((b - a) / 0.4)))
        for i in range(1, m + 1):
            refined.append(a + (b - a) * i / m)
    return [(refined[i], refined[i + 1]) for i in range(len(refined) - 1)]


def integrate_all(st, orientation, u_max, bands, rtol=1e-7, scale=None):
    """Integrate every channel over every band.  Returns (per_band, total, nodes)."""
    f = lambda uu: components(st, uu, orientation)
    edges = sorted({0.0, u_max} | {b[1] for b in bands} | {b[2] for b in bands})
    segs = build_segments(st, u_max, extra_breaks=edges)
    if scale is None:
        # cheap first pass to set the absolute tolerance
        rough = np.zeros(n_components(st))
        for a, b in segs:
            i, _ = _gl_batch(f, np.array([0.0]), np.array([1.0]), a, b)
            rough += i[:, 0]
        scale = max(abs(rough[IDX_TOTAL]), 1e-30)
    atol = rtol * scale / max(len(segs), 1)

    per_band = {b[0]: np.zeros(n_components(st)) for b in bands}
    nodes = 0
    for a, b in segs:
        val, k = integrate_segment(f, a, b, atol)
        nodes += k
        mid = 0.5 * (a + b)
        for name, lo, hi in bands:
            if lo - 1e-12 <= mid <= hi + 1e-12:
                per_band[name] += val
                break
    total = np.sum(list(per_band.values()), axis=0)
    return per_band, total, nodes, scale


def choose_u_max(st, orientation, bands_fn, start=None, tol=1e-9, cap=4096.0):
    """Grow u_max until the [u_max, 2 u_max] tail is negligible."""
    n_max = float(np.max(np.abs(st.n.real)))
    u_max = start or max(4.0, 2.5 * n_max)
    f = lambda uu: components(st, uu, orientation)
    ref = None
    while u_max < cap:
        bands = bands_fn(u_max)
        _, tot, _, _ = integrate_all(st, orientation, u_max, bands, rtol=1e-5)
        ref = tot[IDX_TOTAL]
        # tail contribution
        tail = 0.0
        seg_lo, seg_hi = u_max, 2.0 * u_max
        k = 8
        for i in range(k):
            a = seg_lo * (seg_hi / seg_lo) ** (i / k)
            b = seg_lo * (seg_hi / seg_lo) ** ((i + 1) / k)
            v, _ = integrate_segment(f, a, b, 1e-5 * abs(ref) + 1e-30)
            tail += v[IDX_TOTAL]
        if abs(tail) <= tol * abs(ref):
            return u_max
        u_max *= 2.0
    return u_max


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------


def make_bands(st, u_max):
    n_top = st.n_top
    n_org = st.organic_index_max()
    bands = [("air", 0.0, n_top)]
    lo = n_top
    if st.n_sub is not None and st.n_sub > lo + 1e-9:
        bands.append(("substrate", lo, st.n_sub))
        lo = st.n_sub
    if n_org > lo + 1e-9:
        bands.append(("waveguide", lo, n_org))
        lo = n_org
    bands.append(("evanescent/SPP", lo, u_max))
    return bands


def analyse(st, rtol_seq=(1e-6, 1e-8, 1e-10)):
    """Full channel decomposition for h, v and the isotropic 2:1 average."""
    free = st.free_space_stack()
    result = {"stack": st, "orientations": {}}

    # one common u_max so the h/v/iso band edges coincide exactly
    bands_fn = lambda um: make_bands(st, um)
    u_max = max(choose_u_max(st, "h", bands_fn), choose_u_max(st, "v", bands_fn))

    for orientation in ("h", "v"):
        prev = None
        used_rtol = None
        nodes = 0
        per_band = total = None
        for rtol in rtol_seq:
            bands = make_bands(st, u_max)
            per_band, total, nodes, _ = integrate_all(st, orientation, u_max, bands, rtol=rtol)
            frac = _fractions(st, per_band, total)
            if prev is not None:
                delta = max(abs(frac[k] - prev[k]) for k in frac)
                used_rtol = (rtol, delta, nodes)
                if delta < 1e-3:
                    break
            prev = frac
        if used_rtol is None:
            used_rtol = (rtol_seq[-1], float("nan"), nodes)

        # free-space reference through the identical code path
        fu_max = float(free.n[1].real) * 1.0000001
        fbands = [("air", 0.0, fu_max)]
        _, ftot, _, _ = integrate_all(free, orientation, fu_max, fbands, rtol=1e-10)

        result["orientations"][orientation] = {
            "u_max": u_max,
            "bands": make_bands(st, u_max),
            "per_band": per_band,
            "total": total,
            "free": ftot,
            "conv": used_rtol,
        }
    _combine(result)
    for orientation in ("h", "v"):
        result["orientations"][orientation]["capture"] = angular_capture(st, orientation)
    o = result["orientations"]
    o["iso"]["capture"] = 2.0 * o["h"]["capture"] + o["v"]["capture"]
    return result


CAPTURE_ANGLES_DEG = (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)


def angular_capture(st, orientation, angles_deg=CAPTURE_ANGLES_DEG, rtol=1e-8):
    """Outcoupled power emitted into air within a polar half-angle theta, i.e.
    u < n_top sin(theta).  This is what a finite-domain FDTD top monitor can
    actually see: light leaving the dipole at theta lands a distance h tan(theta)
    away, so anything past the domain half-width is eaten by the lateral PML."""
    f = lambda uu: components(st, uu, orientation)
    n_top = st.n_top
    edges = [0.0] + [n_top * np.sin(np.deg2rad(a)) for a in angles_deg]
    edges[-1] = n_top                                  # theta = 90 deg exactly
    out = []
    running = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            out.append(running)
            continue
        segs = build_segments(st, b, extra_breaks=[a])
        segs = [(x, y) for x, y in segs if x >= a - 1e-12]
        acc = 0.0
        for x, y in segs:
            val, _ = integrate_segment(f, x, y, rtol)
            acc += val[IDX_OUT]
        running += acc
        out.append(running)
    return np.asarray(out)


def _fractions(st, per_band, total):
    tot = total[IDX_TOTAL]
    frac = {"out": total[IDX_OUT] / tot, "bottom": total[IDX_BOTTOM] / tot}
    for j in range(1, st.N - 1):
        frac[f"abs{j}"] = total[idx_abs(st, j)] / tot
    for name, vec in per_band.items():
        frac[f"band_{name}"] = vec[IDX_TOTAL] / tot
    return frac


def _combine(result):
    st = result["stack"]
    o = result["orientations"]
    # isotropic emitter: 2 horizontal : 1 vertical
    tot = 2.0 * o["h"]["total"] + o["v"]["total"]
    free = 2.0 * o["h"]["free"] + o["v"]["free"]
    bands = o["h"]["bands"]
    per_band = {}
    for name, _, _ in bands:
        per_band[name] = 2.0 * o["h"]["per_band"][name] + o["v"]["per_band"][name]
    o["iso"] = {"u_max": o["h"]["u_max"], "bands": bands, "per_band": per_band,
                "total": tot, "free": free, "conv": o["h"]["conv"]}
    for key, rec in o.items():
        rec["purcell"] = rec["total"][IDX_TOTAL] / rec["free"][IDX_TOTAL]
        rec["frac"] = _fractions(st, rec["per_band"], rec["total"])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(result, fdtd_lee=None):
    st = result["stack"]
    lines = []
    w = 88
    lines.append("=" * w)
    lines.append(f"ANALYTIC DIPOLE-IN-MULTILAYER  --  {st.label}")
    lines.append("=" * w)
    lines.append(st.describe())
    lines.append("")

    o = result["orientations"]
    lines.append(f"  {'orientation':<14s}{'Purcell F':>12s}{'u_max':>10s}"
                 f"{'nodes':>10s}{'d(frac)':>12s}")
    for key, name in (("h", "horizontal"), ("v", "vertical"), ("iso", "isotropic 2:1")):
        rec = o[key]
        rtol, delta, nodes = rec["conv"]
        lines.append(f"  {name:<14s}{rec['purcell']:>12.4f}{rec['u_max']:>10.1f}"
                     f"{nodes:>10d}{delta:>12.2e}")
    lines.append("")

    # ---- sink decomposition -------------------------------------------
    lines.append("  SINK DECOMPOSITION (% of total dissipated power; sums to 100%)")
    hdr = f"  {'channel':<24s}" + "".join(f"{n:>14s}" for n in ("horizontal", "vertical", "isotropic"))
    lines.append(hdr)
    lines.append("  " + "-" * (24 + 42))

    def row(label, getter):
        vals = [getter(o[k]) for k in ("h", "v", "iso")]
        return f"  {label:<24s}" + "".join(f"{100.0 * v:>14.4f}" for v in vals)

    lines.append(row(f"outcoupled -> {st.names[-1]}",
                     lambda r: r["total"][IDX_OUT] / r["total"][IDX_TOTAL]))
    lines.append(row(f"into {st.names[0]} (bottom)",
                     lambda r: r["total"][IDX_BOTTOM] / r["total"][IDX_TOTAL]))
    if abs(st.n[0].imag) > 1e-9 or abs(st.n[-1].imag) > 1e-9:
        lines.append("    (note: a semi-infinite medium with Im(n) > 0 never returns the power "
                     "it receives,")
        lines.append("     so its 'escape' channel above is physically absorption in that medium)")
    abs_total_idx = []
    for j in range(1, st.N - 1):
        share = max(o["iso"]["total"][idx_abs(st, j)] / o["iso"]["total"][IDX_TOTAL], 0.0)
        if share > 1e-6 or abs(st.n[j].imag) > 1e-9:
            abs_total_idx.append(j)
            lines.append(row(f"absorbed: {st.names[j]}",
                             (lambda jj: lambda r: r["total"][idx_abs(st, jj)] / r["total"][IDX_TOTAL])(j)))
    lines.append(row("absorbed: TOTAL",
                     lambda r: sum(r["total"][idx_abs(st, j)] for j in range(1, st.N - 1))
                     / r["total"][IDX_TOTAL]))
    lines.append("  " + "-" * (24 + 42))
    lines.append(row("SUM (must be 100)",
                     lambda r: (r["total"][IDX_OUT] + r["total"][IDX_BOTTOM]
                                + sum(r["total"][idx_abs(st, j)] for j in range(1, st.N - 1)))
                     / r["total"][IDX_TOTAL]))
    lines.append("")

    # ---- band decomposition -------------------------------------------
    lines.append("  u-BAND DECOMPOSITION of the total dissipated power (%, sums to 100%)")
    lines.append(f"  {'band':<24s}{'u range':>16s}" +
                 "".join(f"{n:>14s}" for n in ("horizontal", "vertical", "isotropic")))
    lines.append("  " + "-" * (24 + 16 + 42))
    for name, lo, hi in o["h"]["bands"]:
        rng = f"{lo:.3f}-{hi:.1f}"
        vals = [o[k]["per_band"][name][IDX_TOTAL] / o[k]["total"][IDX_TOTAL]
                for k in ("h", "v", "iso")]
        lines.append(f"  {name:<24s}{rng:>16s}" + "".join(f"{100.0 * v:>14.4f}" for v in vals))
    lines.append("")

    # ---- band x sink cross table (isotropic + horizontal) --------------
    for key, name in (("h", "horizontal"), ("iso", "isotropic")):
        rec = o[key]
        lines.append(f"  BAND x SINK cross table, {name} dipole (% of that orientation's total)")
        cols = ["out", "bottom"] + [st.names[j] for j in abs_total_idx] + ["band total"]
        lines.append(f"  {'band':<20s}" + "".join(f"{c[:12]:>13s}" for c in cols))
        for bname, lo, hi in rec["bands"]:
            v = rec["per_band"][bname]
            tot = rec["total"][IDX_TOTAL]
            row_vals = [v[IDX_OUT], v[IDX_BOTTOM]] + [v[idx_abs(st, j)] for j in abs_total_idx]
            row_vals.append(v[IDX_TOTAL])
            lines.append(f"  {bname:<20s}" + "".join(
                f"{100.0 * (x if abs(x) > 1e-14 * tot else 0.0) / tot:>13.4f}"
                for x in row_vals))
        lines.append("")

    # ---- angular capture ----------------------------------------------
    lines.append("  OUTCOUPLED POWER WITHIN A POLAR HALF-ANGLE (% of total dissipated)")
    lines.append("  -- a finite-domain FDTD top monitor only sees the small-angle part --")
    lines.append(f"  {'theta <=':<12s}" +
                 "".join(f"{a:>10.0f}" for a in CAPTURE_ANGLES_DEG) + "  deg")
    for key, name in (("h", "horizontal"), ("v", "vertical"), ("iso", "isotropic")):
        rec = o[key]
        vals = rec["capture"] / rec["total"][IDX_TOTAL]
        lines.append(f"  {name:<12s}" + "".join(f"{100.0 * v:>10.4f}" for v in vals))
    lines.append("")

    if fdtd_lee is not None:
        h_lee = o["h"]["total"][IDX_OUT] / o["h"]["total"][IDX_TOTAL]
        iso_lee = o["iso"]["total"][IDX_OUT] / o["iso"]["total"][IDX_TOTAL]
        lines.append("  FDTD COMPARISON")
        lines.append(f"    FDTD planar LEE (horizontal x/y dipoles)      = {100.0 * fdtd_lee:.3f} %")
        lines.append(f"    analytic LEE, horizontal dipole              = {100.0 * h_lee:.3f} %")
        lines.append(f"    analytic LEE, isotropic (2h:1v)              = {100.0 * iso_lee:.3f} %")
        lines.append(f"    ratio FDTD / analytic(horizontal)            = {fdtd_lee / h_lee:.3f}")
        cap = o["h"]["capture"] / o["h"]["total"][IDX_TOTAL]
        if cap[0] < fdtd_lee < cap[-1]:
            th = float(np.interp(fdtd_lee, cap, np.asarray(CAPTURE_ANGLES_DEG)))
            lines.append(f"    the FDTD value equals the analytic cumulative emission")
            lines.append(f"    up to a polar half-angle of                  ~ {th:.1f} deg,")
            lines.append(f"    i.e. it is consistent with the top monitor capturing only")
            lines.append(f"    the small-angle cone and losing everything beyond it.")
        lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


# Categorical hues: slots 1-3 of the reference palette, assigned in fixed order
# and never cycled.  Those three slots are documented as all-pairs validated in
# light mode (CVD dE 9.2, normal-vision dE 24.0), which is the case that matters
# here because the curves overlap freely rather than sitting in a fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8880"
BAND_FILL = {"air": "#dfeafa", "substrate": "#e2f3ec",
             "waveguide": "#fbe7dc", "evanescent/SPP": "#eeece6"}


def _draw_bands(ax, bands, u_lo, u_hi):
    for name, lo, hi in bands:
        a, b = max(lo, u_lo), min(hi, u_hi)
        if b > a:
            ax.axvspan(a, b, color=BAND_FILL.get(name, "#eeece6"), lw=0, zorder=0)


def _band_ruler(ax, bands, shares, log=False):
    """Thin strip above the plots naming each u band and its share of the total,
    so the labels never sit on top of the data."""
    lo_x, hi_x = ax.get_xlim()
    _draw_bands(ax, bands, lo_x, hi_x)
    ax.set_yticks([])
    ax.xaxis.set_visible(False)
    for s in ax.spines.values():
        s.set_color(INK3)
    for name, lo, hi in bands:
        a, b = max(lo, lo_x), min(hi, hi_x)
        if b <= a:
            continue
        # width as a fraction of the axis, in the axis' own scale
        if log:
            frac = (np.log(b) - np.log(a)) / (np.log(hi_x) - np.log(lo_x))
            xm = np.sqrt(a * b)
        else:
            frac = (b - a) / (hi_x - lo_x)
            xm = 0.5 * (a + b)
        if frac < 0.10:                # too narrow to hold the label without clipping
            continue
        ax.text(xm, 0.5, f"{name}\n{100 * shares[name]:.2f}%",
                transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=7.5, linespacing=1.15, color=INK2)


def _clip(v, floor):
    """Hide sub-floor values instead of flattening them onto the axis bottom."""
    return np.where(v > floor, v, np.nan)


def plot_spectrum(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    st = result["stack"]
    o = result["orientations"]
    u_max = o["h"]["u_max"]
    bands = o["h"]["bands"]
    shares = {n: o["iso"]["per_band"][n][IDX_TOTAL] / o["iso"]["total"][IDX_TOTAL]
              for n, _, _ in bands}

    # The interesting structure (escape cone, substrate and waveguide modes) all
    # lives below the highest layer index; the SPP/evanescent tail runs decades
    # further out.  A single linear axis wastes 90 % of its width on the tail and
    # a single log axis crushes the resonances, so the axis is broken: linear on
    # the left, logarithmic for the tail on the right.
    u_split = bands[-1][1] * 1.15
    ulin = np.linspace(1e-6, u_split, 7000)
    ulog = np.geomspace(u_split, u_max, 1400)

    free = {k: o[k]["free"][IDX_TOTAL] for k in ("h", "v", "iso")}
    spec = {}
    for tag, uu in (("lin", ulin), ("log", ulog)):
        ch, cv = components(st, uu, "h"), components(st, uu, "v")
        spec[tag] = {"u": uu, "h": ch, "v": cv, "iso": 2.0 * ch + cv}

    fig = plt.figure(figsize=(11.0, 8.2))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[3.0, 1.35],
                  height_ratios=[0.16, 1.45, 1.0], hspace=0.09, wspace=0.035)
    rulers = [fig.add_subplot(gs[0, c]) for c in (0, 1)]
    axes = [[fig.add_subplot(gs[r, c]) for c in (0, 1)] for r in (1, 2)]

    def styling(ax, col, row, logy=True):
        if logy:
            ax.set_yscale("log")
        ax.grid(True, which="major", color=INK3, alpha=0.28, lw=0.5)
        ax.tick_params(colors=INK2, labelsize=8.5)
        for s in ax.spines.values():
            s.set_color(INK3)
        if col == 0:
            ax.set_xlim(0.0, u_split)
            ax.spines["right"].set_visible(False)
        else:
            from matplotlib.ticker import LogLocator, ScalarFormatter
            ax.set_xscale("log")
            ax.set_xlim(u_split, u_max)
            ax.spines["left"].set_visible(False)
            ax.tick_params(labelleft=False)
            # the tail often spans less than a decade, so label the 2/3/5 minors
            for setter in (ax.xaxis.set_major_formatter, ax.xaxis.set_minor_formatter):
                fmt = ScalarFormatter()
                fmt.set_scientific(False)
                setter(fmt)
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0),
                                                  numticks=20))
        if row == 0:
            ax.tick_params(which="both", labelbottom=False)

    # ---- row 0: total dissipation, three orientations -------------------
    curves0 = [("isotropic (2h:1v)", "iso", 1.9), ("horizontal", "h", 1.2),
               ("vertical", "v", 1.2)]
    ymin0 = 1e-6
    for col in (0, 1):
        ax = axes[0][col]
        styling(ax, col, 0)
        _draw_bands(ax, bands, *ax.get_xlim())
        for i, (lab, key, lw) in enumerate(curves0):
            d = spec["lin" if col == 0 else "log"]
            ax.plot(d["u"], _clip(d[key][IDX_TOTAL] / free[key], ymin0),
                    color=SERIES[i], lw=lw, label=lab if col == 0 else None,
                    solid_joinstyle="round")
        ax.set_ylim(ymin0, None)
        rulers[col].set_xscale(ax.get_xscale())
        rulers[col].set_xlim(ax.get_xlim())
        _band_ruler(rulers[col], bands, shares, log=(col == 1))
    axes[0][0].set_ylabel(r"$dP/du$  (normalized so $\int dP/du\,\mathrm{d}u = F$)",
                          color=INK2, fontsize=9.5)
    leg = axes[0][0].legend(loc="upper left", fontsize=8.5, framealpha=0.92,
                            edgecolor=INK3, title="total dissipated power")
    leg.get_title().set_fontsize(8.5)
    leg.get_title().set_color(INK2)
    for t in leg.get_texts():
        t.set_color(INK)
    axes[0][1].set_ylim(axes[0][0].get_ylim())

    # ---- row 1: where the isotropic power ends up -----------------------
    sinks = [("outcoupled -> " + st.names[-1], IDX_OUT)]
    order = sorted(range(1, st.N - 1),
                   key=lambda j: -o["iso"]["total"][idx_abs(st, j)])
    for j in order:
        share = o["iso"]["total"][idx_abs(st, j)] / o["iso"]["total"][IDX_TOTAL]
        if share > 1e-4:
            sinks.append((f"absorbed in {st.names[j]}", idx_abs(st, j)))
    bot = o["iso"]["total"][IDX_BOTTOM] / o["iso"]["total"][IDX_TOTAL]
    if bot > 1e-4:
        sinks.append((f"into {st.names[0]}", IDX_BOTTOM))
    sinks = sinks[:3]                     # never cycle hues; 3 is the all-pairs cap
    ymin1 = 1e-8
    for col in (0, 1):
        ax = axes[1][col]
        styling(ax, col, 1)
        _draw_bands(ax, bands, *ax.get_xlim())
        for i, (lab, row_idx) in enumerate(sinks):
            d = spec["lin" if col == 0 else "log"]
            share = o["iso"]["total"][row_idx] / o["iso"]["total"][IDX_TOTAL]
            ax.plot(d["u"], _clip(d["iso"][row_idx] / free["iso"], ymin1),
                    color=SERIES[i], lw=1.4,
                    label=f"{lab}  ({100 * share:.2f}%)" if col == 0 else None)
        ax.set_ylim(ymin1, None)
    axes[1][0].set_ylabel("power reaching each sink", color=INK2, fontsize=9.5)
    axes[1][0].set_xlabel(r"in-plane wavevector  $u = k_\parallel/k_0$   (linear)",
                          color=INK2, fontsize=9.5)
    axes[1][1].set_xlabel("(log)", color=INK2, fontsize=9.5)
    leg = axes[1][0].legend(loc="upper left", fontsize=8.5, framealpha=0.92,
                            edgecolor=INK3, title="isotropic emitter, by sink")
    leg.get_title().set_fontsize(8.5)
    leg.get_title().set_color(INK2)
    for t in leg.get_texts():
        t.set_color(INK)
    axes[1][1].set_ylim(axes[1][0].get_ylim())

    lee_h = o["h"]["total"][IDX_OUT] / o["h"]["total"][IDX_TOTAL]
    lee_i = o["iso"]["total"][IDX_OUT] / o["iso"]["total"][IDX_TOTAL]
    fig.suptitle(f"{st.label}", color=INK, fontsize=12, y=0.982)
    fig.text(0.5, 0.949,
             f"Purcell  $F_h$={o['h']['purcell']:.3f}   $F_v$={o['v']['purcell']:.3f}"
             f"   $F_{{iso}}$={o['iso']['purcell']:.3f}        "
             f"outcoupled  {100 * lee_h:.2f} % (horizontal)   "
             f"{100 * lee_i:.2f} % (isotropic)",
             ha="center", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.941))
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------


def build_microcavity_stack(color, kind=None):
    """Microcavity stack, read from oled_common.microcavity_layers.

    Single source of truth: the FDTD scripts build the very same layer list, so
    the analytic result and the simulated geometry cannot drift apart.
    kind defaults to MSOPT_MC_STACK_KIND ("optimized"; "table" is the originally
    reconstructed layer table, which sits at cavity anti-resonance).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    import oled_common as oc

    kind = kind or os.environ.get("MSOPT_MC_STACK_KIND", "optimized")
    layer_specs, spec = oc.microcavity_layers(color, kind)
    lam = spec["wavelength_um"]

    names = ["air_below"] + [nm for nm, _h, _i in layer_specs] + ["air_above"]
    n = [1.0] + [complex(float(idx["n"][0]), float(idx["k"][0]))
                 for _nm, _h, idx in layer_specs] + [1.0]
    d = [None] + [float(h) for _nm, h, _i in layer_specs] + [None]
    e = names.index(oc.MICROCAVITY_EML_LAYER)
    return Stack(names, n, d, lam, e, 0.5 * d[e], n_sub=None,
                 label=f"microcavity top-emission OLED [{kind}], {color.upper()} "
                       f"(lambda = {lam:.3f} um)")


def build_legacy_stack():
    """Legacy stack of oled_common.build_config.  Imported when possible so the
    numbers cannot drift; falls back to the mirrored table otherwise."""
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    source = "mirrored constants"
    layers = None
    try:
        import oled_common as oc
        old = os.environ.get("EIDL_RUN_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["EIDL_RUN_DIR"] = tmp
            try:
                G = oc.build_config()
            finally:
                if old is None:
                    os.environ.pop("EIDL_RUN_DIR", None)
                else:
                    os.environ["EIDL_RUN_DIR"] = old
        layers = [(nm, float(h), complex(float(ix["n"][0]), float(ix["k"][0])))
                  for nm, h, ix in G.layer_specs]
        lam = float(np.mean(G.visible_wavelengths))
        source = "imported from oled_common.build_config"
    except Exception as exc:                                # pragma: no cover
        print(f"[legacy] could not import oled_common ({exc}); using mirrored table")
        layers = [("Ag_reflector", 0.2, complex(0.76, 5.9)),
                  ("TPBi", 0.2, complex(1.75, 0.0)),
                  ("CBP_Irppy_EML", 0.2, complex(1.77, 0.0)),
                  ("TCTA", 0.2, complex(1.82, 0.0)),
                  ("ITO", 0.2, complex(1.7, 0.0)),
                  ("SiO2", 0.3, complex(1.45, 0.0))]
        lam = 0.55

    names = ["air_below"] + [nm for nm, _, _ in layers] + ["air_above"]
    n = [1.0] + [ix for _, _, ix in layers] + [1.0]
    d = [None] + [h for _, h, _ in layers] + [None]
    e = names.index("CBP_Irppy_EML")
    return Stack(names, n, d, lam, e, 0.5 * d[e], n_sub=1.45,
                 label=f"legacy planar OLED ({source}), lambda = {lam:.3f} um")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def independent_escape_cone(n1, n2=1.0, orientation="h"):
    """INDEPENDENT escape-cone integral -- deliberately shares no code with the
    transfer-matrix path above.  Ray optics for a dipole deep inside a lossless
    half-space n1 radiating through a planar interface into n2 < n1:

        f = int_0^{theta_c} A_pol(theta) T_pol(theta) sin(theta) dtheta
            / int_0^{pi} A(theta) sin(theta) dtheta

    with A_s = 1/2, A_p = cos^2(theta)/2 for a horizontal dipole (azimuth
    averaged) and A_p = sin^2(theta) for a vertical one, and T = 1 - |r|^2 built
    straight from the angle-form Fresnel coefficients.
    """
    from scipy.integrate import quad

    thc = np.arcsin(min(n2 / n1, 1.0))

    def tcoef(th):
        ci = np.cos(th)
        st_ = n1 * np.sin(th) / n2
        ct = np.sqrt(max(1.0 - st_ ** 2, 0.0))
        rs = (n1 * ci - n2 * ct) / (n1 * ci + n2 * ct)
        rp = (n2 * ci - n1 * ct) / (n2 * ci + n1 * ct)
        return 1.0 - rs ** 2, 1.0 - rp ** 2

    if orientation == "h":
        def num(th):
            ts, tp = tcoef(th)
            return (0.5 * ts + 0.5 * np.cos(th) ** 2 * tp) * np.sin(th)
        den = 4.0 / 3.0
    else:
        def num(th):
            _, tp = tcoef(th)
            return np.sin(th) ** 2 * tp * np.sin(th)
        den = 4.0 / 3.0
    val, _ = quad(num, 0.0, thc, limit=400)
    return val / den


def _matrix_M(nl, dl, u, k0, pol):
    """INDEPENDENT plain 2x2 transfer matrix for a sub-stack (bottom -> top).

    Written as an explicit product of interface and propagation matrices, i.e.
    with no reflection recursion anywhere, so it cross-checks the Moebius
    recursions used by the solver.  nl / dl list the sub-stack's indices and
    thicknesses; the first and last entries are semi-infinite (dl ignored).

    Returns M with  [a_bot+, a_bot-] = M [a_top+, a_top-],  amplitudes referenced
    at the sub-stack's outer interfaces.
    """
    m = len(nl)
    ll = [_ell(nl[j], u) for j in range(m)]
    one = np.ones_like(ll[0])
    zero = np.zeros_like(ll[0])
    M = (one, zero, zero, one)                   # (M00, M01, M10, M11)

    def mul(A, B):
        return (A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3],
                A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3])

    for j in range(m - 1):
        r, t = _fresnel(nl[j], nl[j + 1], ll[j], ll[j + 1], pol)
        M = mul(M, (1.0 / t, r / t, r / t, 1.0 / t))
        if j + 1 <= m - 2:                       # propagate across layer j+1
            ph = k0 * ll[j + 1] * dl[j + 1]
            M = mul(M, (np.exp(-1j * ph), zero, zero, np.exp(1j * ph)))
    return M


def matrix_cross_check(st, u, pol):
    """R_u, R_d and the upper-substack transmission from explicit matrix products."""
    k0 = st.k0
    e = st.e
    h_up = st.d[e] - st.z_off
    # upper sub-stack seen from the dipole plane: a fictitious semi-infinite
    # emitter medium (where the outgoing wave starts), the remaining h_up of the
    # emitter layer, then every layer above it.
    nl_up = [st.n[e], st.n[e]] + [st.n[j] for j in range(e + 1, st.N)]
    dl_up = [0.0, h_up] + [st.d[j] for j in range(e + 1, st.N)]
    M00, M01, M10, M11 = _matrix_M(nl_up, dl_up, u, k0, pol)
    t_up = 1.0 / M00
    R_u = M10 / M00
    # lower sub-stack: everything below, the z_off of the emitter layer, then the
    # fictitious semi-infinite emitter medium at the dipole plane.
    nl_dn = [st.n[j] for j in range(0, e)] + [st.n[e], st.n[e]]
    dl_dn = [st.d[j] for j in range(0, e)] + [st.z_off, 0.0]
    M00, M01, M10, M11 = _matrix_M(nl_dn, dl_dn, u, k0, pol)
    R_d = -M01 / M00
    return R_u, R_d, t_up


def _recursion_reference(st, u, pol):
    """R_u, R_d referenced at the dipole plane, straight from the solver path."""
    k0 = st.k0
    N, e = st.N, st.e
    nn = st.n
    l = [_ell(nn[j], u) for j in range(N)]
    phi = [None] * N
    P = [None] * N
    for j in range(1, N - 1):
        phi[j] = k0 * l[j] * st.d[j]
        P[j] = np.exp(2j * phi[j])
    rup = [_fresnel(nn[i], nn[i + 1], l[i], l[i + 1], pol)[0] for i in range(N - 1)]
    Ru = [None] * (N - 1)
    Ru[N - 2] = rup[N - 2]
    for i in range(N - 3, -1, -1):
        x = Ru[i + 1] * P[i + 1]
        Ru[i] = (rup[i] + x) / (1.0 + rup[i] * x)
    Rd = [None] * (N - 1)
    Rd[0] = -rup[0]
    for i in range(1, N - 1):
        x = Rd[i - 1] * P[i]
        Rd[i] = (-rup[i] + x) / (1.0 - rup[i] * x)
    h_up = st.d[e] - st.z_off
    return (Ru[e] * np.exp(2j * k0 * l[e] * h_up),
            Rd[e - 1] * np.exp(2j * k0 * l[e] * st.z_off))


def outcoupled_spectrum_matrix(st, u, orientation):
    """dP_out/du built the way the brief describes it: the upward amplitude at the
    dipole plane (1 +/- R_d)/D times the transmission amplitude of the UPPER
    sub-stack obtained from an explicit transfer matrix, times the top-medium flux
    factor.  Shares only the source-amplitude formula with the main solver."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    pols = ["p"] if orientation == "v" else ["s", "p"]
    l_top = _ell(st.n[-1], u)
    out = np.zeros_like(u)
    for pol in pols:
        s_up, s_dn, w = _source_amplitudes(st, u, orientation, pol)
        R_u, R_d, t_up = matrix_cross_check(st, u, pol)
        U = (s_up + R_d * s_dn) / (1.0 - R_u * R_d)
        amp = U * t_up
        g = np.conj(l_top) if pol == "s" else np.conj(l_top) * (st.n[-1] / np.conj(st.n[-1]))
        out += w * 0.5 * np.real(g * amp * np.conj(amp))
    return u * out


def _pass(ok, name, detail):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    return bool(ok)


def run_validation():
    print("=" * 88)
    print("VALIDATION")
    print("=" * 88)
    ok_all = True

    # ---- 1. homogeneous multilayer -> F = 1 ----------------------------
    ne = 1.8
    st = Stack(["bot", "L1", "L2", "L3", "top"],
               [ne, ne, ne, ne, ne],
               [None, 0.3, 0.25, 0.4, None], 0.55, 2, 0.125,
               label="homogeneous (all layers = n_e)")
    res = analyse(st)
    fh = res["orientations"]["h"]["purcell"]
    fv = res["orientations"]["v"]["purcell"]
    ok_all &= _pass(abs(fh - 1.0) < 1e-3, "1a homogeneous Purcell, horizontal",
                    f"F = {fh:.9f} (|F-1| = {abs(fh - 1):.2e}, tol 1e-3)")
    ok_all &= _pass(abs(fv - 1.0) < 1e-3, "1b homogeneous Purcell, vertical",
                    f"F = {fv:.9f} (|F-1| = {abs(fv - 1):.2e}, tol 1e-3)")
    # the free-space integral must equal its analytic value 2 n_e / 3
    for key in ("h", "v"):
        got = res["orientations"][key]["free"][IDX_TOTAL]
        want = 2.0 * ne / 3.0
        ok_all &= _pass(abs(got / want - 1.0) < 1e-6,
                        f"1c free-space integral vs analytic 2n/3 ({key})",
                        f"{got:.10f} vs {want:.10f}")
    # CPS closed form vs flux-based total
    for key in ("h", "v"):
        rec = res["orientations"][key]
        rel = abs(rec["total"][idx_cps(st)] - rec["total"][IDX_TOTAL]) / abs(rec["total"][IDX_TOTAL])
        ok_all &= _pass(rel < 1e-9, f"1d CPS closed form vs flux total ({key})",
                        f"rel diff = {rel:.2e} (tol 1e-9)")

    # ---- 2. deep emitter in a half space, escape cone ------------------
    n1, lam = 1.8, 0.55
    depth = 20.0 * lam / n1          # 20 wavelengths inside the medium
    st2 = Stack(["n1_semiinf", "n1_slab", "air"], [n1, n1, 1.0],
                [None, depth, None], lam, 1, 0.0,
                label=f"lossless half space n={n1}, emitter {depth * 1e3:.0f} nm deep")
    res2 = analyse(st2)
    for key, name in (("h", "horizontal"), ("v", "vertical")):
        rec = res2["orientations"][key]
        got = rec["total"][IDX_OUT] / rec["total"][IDX_TOTAL]
        want = independent_escape_cone(n1, 1.0, key)
        rel = abs(got / want - 1.0)
        ok_all &= _pass(rel < 0.01, f"2 escape cone, {name} dipole",
                        f"TMM {100 * got:.4f} % vs independent Fresnel "
                        f"{100 * want:.4f} % (rel {100 * rel:.3f} %, tol 1 %)")
        # absorption must be exactly zero here as well
    tot_abs = sum(res2["orientations"]["h"]["total"][idx_abs(st2, j)]
                  for j in range(1, st2.N - 1))
    ok_all &= _pass(abs(tot_abs) < 1e-12 * abs(res2["orientations"]["h"]["total"][IDX_TOTAL]),
                    "2b half-space absorption is identically zero",
                    f"{tot_abs:.3e}")

    # ---- 3. energy conservation on a lossy stack ----------------------
    st3 = build_legacy_stack()
    res3 = analyse(st3)
    for key, name in (("h", "horizontal"), ("v", "vertical")):
        rec = res3["orientations"][key]
        tot = rec["total"][IDX_TOTAL]
        recon = (rec["total"][IDX_OUT] + rec["total"][IDX_BOTTOM]
                 + sum(rec["total"][idx_abs(st3, j)] for j in range(1, st3.N - 1)))
        err = abs(1.0 - recon / tot)
        ok_all &= _pass(err < 1e-2, f"3 energy conservation, lossy stack ({name})",
                        f"|1 - (out+bottom+abs)/total| = {err:.3e} (tol 1e-2); "
                        f"out={100 * rec['total'][IDX_OUT] / tot:.4f}% "
                        f"bottom={100 * rec['total'][IDX_BOTTOM] / tot:.3e}% "
                        f"abs={100 * (recon - rec['total'][IDX_OUT] - rec['total'][IDX_BOTTOM]) / tot:.4f}%")
        mism = rec["total"][idx_mismatch(st3)] / tot
        ok_all &= _pass(mism < 1e-6, f"3b interface flux continuity ({name})",
                        f"max |S_below - S_above| / total = {mism:.2e} (tol 1e-6)")

    # ---- 4. lossless stack -> no absorption ---------------------------
    st4 = Stack(["glass_semiinf", "org1", "org2", "org3", "air"],
                [1.90, 1.80, 1.75, 1.70, 1.0],
                [None, 0.2, 0.15, 0.12, None], 0.55, 2, 0.075,
                label="lossless multilayer (bottom medium is the highest index)")
    res4 = analyse(st4)
    worst = 0.0
    for key in ("h", "v"):
        rec = res4["orientations"][key]
        tot = rec["total"][IDX_TOTAL]
        a = sum(abs(rec["total"][idx_abs(st4, j)]) for j in range(1, st4.N - 1))
        worst = max(worst, a / tot)
    ok_all &= _pass(worst < 1e-6, "4 lossless stack absorption",
                    f"max total absorption / total = {worst:.3e} (tol 1e-6)")
    # and energy still balances
    for key in ("h", "v"):
        rec = res4["orientations"][key]
        tot = rec["total"][IDX_TOTAL]
        recon = rec["total"][IDX_OUT] + rec["total"][IDX_BOTTOM]
        ok_all &= _pass(abs(1 - recon / tot) < 1e-4,
                        f"4b lossless energy balance ({key})",
                        f"(out+bottom)/total = {recon / tot:.8f}")

    # ---- 5. convergence of the u integration --------------------------
    st5 = build_microcavity_stack("green")
    print("  [ .. ] 5 convergence sweep on the microcavity/green stack")
    prev = None
    conv_ok = False
    for rtol in (1e-4, 1e-6, 1e-8, 1e-10):
        u_max = choose_u_max(st5, "h", lambda um: make_bands(st5, um))
        bands = make_bands(st5, u_max)
        per_band, total, nodes, _ = integrate_all(st5, "h", u_max, bands, rtol=rtol)
        frac = _fractions(st5, per_band, total)
        if prev is not None:
            delta = max(abs(frac[k] - prev[k]) for k in frac)
            print(f"         rtol={rtol:.0e}  nodes={nodes:7d}  u_max={u_max:.1f}  "
                  f"max |d fraction| = {delta:.3e}")
            if delta < 1e-3:
                conv_ok = True
                break
        else:
            print(f"         rtol={rtol:.0e}  nodes={nodes:7d}  u_max={u_max:.1f}  "
                  f"(reference pass)")
        prev = frac
    ok_all &= _pass(conv_ok, "5 u-integration convergence",
                    "fractions stable to < 1e-3 under tolerance refinement "
                    "(adaptive Gauss-Legendre, smoothstep-mapped segments)")

    # ---- 6. recursions vs an independent explicit 2x2 transfer matrix --
    for st_x, nm in ((st3, "legacy"), (st5, "microcavity/green")):
        # the plain matrix product is exponentially unstable for evanescent
        # waves, so restrict the comparison to where it still has any precision
        thick = float(np.sum(st_x.d[1:-1]))
        u_lim = max(3.0, 250.0 / (st_x.k0 * thick))
        uu = np.concatenate([np.linspace(1e-4, 0.999, 400),
                             np.linspace(1.001, 3.0, 400),
                             np.geomspace(3.0, u_lim, 200)])
        worst_r = 0.0
        for pol in ("s", "p"):
            Rum, Rdm, _ = matrix_cross_check(st_x, uu, pol)
            Rur, Rdr = _recursion_reference(st_x, uu, pol)
            worst_r = max(worst_r, float(np.max(np.abs(Rum - Rur))),
                          float(np.max(np.abs(Rdm - Rdr))))
        ok_all &= _pass(worst_r < 1e-9, f"6a R_u/R_d recursion vs matrix TMM ({nm})",
                        f"max |dR| = {worst_r:.2e} over u <= {u_lim:.1f} (tol 1e-9)")
        worst_o = 0.0
        for orientation in ("h", "v"):
            uo = np.linspace(1e-4, 0.9999, 900)
            a = components(st_x, uo, orientation)[IDX_OUT]
            b = outcoupled_spectrum_matrix(st_x, uo, orientation)
            scl = float(np.max(np.abs(a)))
            worst_o = max(worst_o, float(np.max(np.abs(a - b))) / max(scl, 1e-30))
        ok_all &= _pass(worst_o < 1e-9,
                        f"6b outcoupled spectrum: field chain vs matrix route ({nm})",
                        f"max rel dev = {worst_o:.2e} (tol 1e-9)")

    print("=" * 88)
    print("VALIDATION RESULT: " + ("ALL PASS" if ok_all else "FAILURE"))
    print("=" * 88)
    return ok_all


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def out_dir():
    run = os.environ.get("EIDL_RUN_DIR")
    d = os.path.join(os.path.abspath(run), "A") if run else os.getcwd()
    os.makedirs(d, exist_ok=True)
    return d


FDTD_PLANAR_LEE_LEGACY = 0.0949      # measured LEE, planar legacy stack, horizontal dipole


def run_case(st, tag, fdtd_lee=None, outdir=None, plots=True):
    outdir = outdir or out_dir()
    res = analyse(st)
    report = format_report(res, fdtd_lee=fdtd_lee)
    print(report)
    with open(os.path.join(outdir, f"OLED_layered_dipole_{tag}.txt"), "w") as fp:
        fp.write(report + "\n")
    if plots:
        png = os.path.join(outdir, f"OLED_layered_dipole_{tag}.png")
        plot_spectrum(res, png)
        print(f"  [plot] {png}")
    o = res["orientations"]
    summary = {
        "tag": tag, "label": st.label, "lambda_um": st.lam,
        "purcell": {k: float(o[k]["purcell"]) for k in ("h", "v", "iso")},
        "outcoupled": {k: float(o[k]["total"][IDX_OUT] / o[k]["total"][IDX_TOTAL])
                       for k in ("h", "v", "iso")},
        "bottom": {k: float(o[k]["total"][IDX_BOTTOM] / o[k]["total"][IDX_TOTAL])
                   for k in ("h", "v", "iso")},
        "absorption_per_layer": {
            st.names[j]: {k: float(o[k]["total"][idx_abs(st, j)] / o[k]["total"][IDX_TOTAL])
                          for k in ("h", "v", "iso")}
            for j in range(1, st.N - 1)},
        "bands": {name: {k: float(o[k]["per_band"][name][IDX_TOTAL] / o[k]["total"][IDX_TOTAL])
                         for k in ("h", "v", "iso")}
                  for name, _, _ in o["h"]["bands"]},
        "u_max": float(o["h"]["u_max"]),
        "capture_angles_deg": list(CAPTURE_ANGLES_DEG),
        "outcoupled_within_angle": {
            k: [float(x / o[k]["total"][IDX_TOTAL]) for x in o[k]["capture"]]
            for k in ("h", "v", "iso")},
    }
    if fdtd_lee is not None:
        summary["fdtd_planar_lee_horizontal"] = float(fdtd_lee)
    return res, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true",
                    help="run the mandatory validation suite and exit non-zero on failure")
    ap.add_argument("--stack", default="both",
                    choices=["microcavity", "legacy", "both", "none"])
    ap.add_argument("--color", default="red,green,blue",
                    help="comma list of microcavity colors")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    outdir = args.outdir or out_dir()
    os.makedirs(outdir, exist_ok=True)
    print(f"[output] {outdir}")

    if args.validate:
        ok = run_validation()
        if not ok:
            return 1

    summaries = []
    if args.stack in ("microcavity", "both"):
        for color in [c.strip() for c in args.color.split(",") if c.strip()]:
            st = build_microcavity_stack(color)
            _, s = run_case(st, f"microcavity_{color}", outdir=outdir,
                            plots=not args.no_plots)
            summaries.append(s)
    if args.stack in ("legacy", "both"):
        st = build_legacy_stack()
        _, s = run_case(st, "legacy", fdtd_lee=FDTD_PLANAR_LEE_LEGACY,
                        outdir=outdir, plots=not args.no_plots)
        summaries.append(s)

    if summaries:
        path = os.path.join(outdir, "OLED_layered_dipole_summary.json")
        with open(path, "w") as fp:
            json.dump(summaries, fp, indent=2)
        print(f"[summary] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
