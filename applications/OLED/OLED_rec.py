"""OLED_rec.py -- k-mapped reciprocal OLED out-coupling optimizer.

The idea, in one line: probe the stack ONCE with a normal-incidence plane wave,
look at the EML plane, and reward the design for putting field into exactly the
in-plane momenta that a periodic out-coupler can turn into the wanted emission
angles.

Why this formulation
--------------------
A planar stack conserves the in-plane wavevector u = k_par/k0, and a periodic
out-coupler of pitch P shifts it in steps of lambda/P:

    sin(theta_air) = u_EML - m * lambda / P                        (m = order)

so every target emission angle corresponds to ONE momentum inside the organic,
u_m = sin(theta) + m*lambda/P -- this is the mapping k_mapping.py tabulates and
this script imports rather than re-deriving.

The design region is periodic with that pitch, so a normal-incidence plane wave
entering from air is diffracted into exactly the discrete set {m*lambda/P}. The
field on the EML mid-plane therefore already carries the momenta of interest,
and its Fourier component at u_m measures how strongly the structure connects
the m-th order to the outside. By reciprocity that same coefficient governs an
EML dipole radiating outward through that order, which is what the incoherent
postprocess ends up measuring.

FoM
    J = sum_i  w_i * |<E_EML , exp(-i k_i . r)>|^2 / |E_EML|^2

    one term per target (angle, order) pair, so the score is a normalized modal
    purity: what fraction of the EML field sits in the momenta we asked for.

Gradient
    Requested shape:  g = sum_i ( fwd x adj_i )  -- run one adjoint PER MODE,
    multiply each against the forward field, and only then sum.

    PER_MODE_ADJOINT = True does exactly that: ONE forward run, then N adjoint
    runs, each driven by a single mode's source, each producing its own gradient,
    summed at the end. per_mode_gradient() drives msopt's INIT -> FWD -> Adj
    state machine directly and restores the cached forward fields between modes,
    so the forward is paid for once.

    Note for the record: because the adjoint source is LINEAR in the FoM's field
    derivative, sum_i (fwd x adj_i) == fwd x (sum_i adj_i) exactly -- the two
    routes give the same numbers, and CHECK_GRADIENT_EQUIVALENCE verifies it on
    the first iteration. The reason to keep the per-mode route is not the value
    but the breakdown: it exposes which target angle is actually driving the
    design, and it is the only form that survives if the modes are ever combined
    non-linearly (minimax, ratio windows) instead of summed.

    PER_MODE_ADJOINT = False collapses to the single summed adjoint (1/N cost).

    Beware msopt's own multi-objective path: passing several objective_functions
    to one LumericalOptimizationProblem sums the adjoint SOURCES into a single
    run and then combines per-frequency gradients with Minimax, which is a
    different objective from the weighted sum used here. This script therefore
    drives the per-mode loop itself instead of handing msopt a list.

Relation to the other optimizers
    OLED_opt probes at each target angle separately (N forward runs) and scores
    the EML |E|^2. This script probes once at normal incidence and scores the
    MOMENTUM CONTENT instead, which is what makes a single forward run cover all
    target angles.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import autograd.numpy as npa
    from autograd import jacobian as ag_jacobian
    from autograd.tracer import getval        # detach a boxed value for the
                                               # reporting cache; see modal_purity_terms
except Exception:                                   # pragma: no cover
    npa, ag_jacobian = np, None
    def getval(x):
        return x

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import msopt as ms
import oled_common as oc
import k_mapping as km


# =============================================================================
# Run switches
# =============================================================================
#   RUN_OPTIMIZATION  False -> skip the optimizer and only run the postprocess
#   TARGET_ANGLES     emission angles in air the design must serve (deg)
#   ORDERS            diffraction orders allowed to carry them
#   PERIOD_UM         out-coupler pitch; None -> lambda/(n_org-1), the pitch
#                     whose first order just spans organic -> normal
#   SOURCE_POL        "x" or "y": the normal-incidence probe polarization
#   PER_MODE_ADJOINT  True -> one forward + one adjoint PER MODE, gradients
#                     multiplied per mode and then summed: g = sum_i(fwd x adj_i)
#   CHECK_GRADIENT_EQUIVALENCE  first iteration only, assert that the per-mode
#                     sum equals the single summed-adjoint gradient
RUN_OPTIMIZATION = True
PLANAR_PP = False
PP_MODE = "supercell"
PP_DIPOLE_GRID = 6
PP_CAPTURE_DEG = 60
PP_KEEP_FSP = False
RESOLUTION = 50
DESIGN_H_UM = 0.30
DESIGN_X_UM = None
DESIGN_Y_UM = None
DESIGN_N = 2.2
DESIGN_LOW_N = 1.0
PROBE_GAP_UM = 0.7
# The forward probe is a plane-wave source, and the first run of this script
# diverged (auto-shutoff climbing 97 -> 69577 before Lumerical killed it) because
# the default margin left it 2 cells clear of the top PML. Give it real room.
TOP_MARGIN_UM = 0.1
# Same story underneath: at the default 0.10 um the 0.10 um Ag anode mirror STARTS
# inside the bottom PML and ends outside it. Metal terminating inside the absorber
# is a standard divergence source; 0.35 um puts the whole mirror clear of it. The
# mirror is opaque at 100 nm, so nothing of interest reaches the bottom anyway.
AIR_BOT_UM = 0.1
os.environ.setdefault("MSOPT_OLED_AIR_BOT_UM", f"{AIR_BOT_UM:.6f}")
STACK = "microcavity"
MC_COLOR = "green"
MC_STACK_KIND = "optimized"

# SYMMETRY decides what the FoM plane is and which orders are scored:
# SYMMETRY picks the ORDER SET and the design parameterisation. The FoM monitor
# is the full 2-D plane either way (see the note above eml_s).
#   "radial"    rotationally symmetric design (RADIAL_DESIGN below should be True):
#               only (m, 0) orders are requested, because the ring-summing mapping
#               makes every azimuth equivalent. One adjoint per order.
#   "fourfold"  4-fold symmetric design: representative azimuths phi = 0 / 45 / 90
#               from the (m,0), (m,m) and (0,n) orders. The adjoint count grows
#               with the number of azimuth samples -- one per (order, azimuth).
SYMMETRY = "fourfold"

# TARGET MODE
#   "ladder"  (default) give only the OUTERMOST angle. The pitch is set so the
#             top order M lands there, and orders m = 1..M-1 fill the ramp in
#             between automatically, evenly spaced in sin(theta). Every rung is a
#             real Bloch order, so nothing has to be approximated -- this is what
#             makes an arbitrary target angle usable at all. Larger N_ORDERS =
#             finer ramp, longer pitch, bigger design grid.
#   "explicit" list the angles yourself; k_mapping then has to find one pitch
#             that puts them ALL on lattice points, which most hand-picked sets
#             (0/30/45 among them) simply do not admit.
TARGET_MODE = "ladder"

TARGET_MAX_ANGLE = 45.0        # ladder: the outermost emission angle, in air
N_ORDERS = 3                   # ladder: how many rungs (top order M). P = M*lam/
                               # sin(TARGET_MAX_ANGLE) -- rolled back from the
                               # 6/doubled-period experiment (cancelled) to match
                               # the first run's period, P~2.24um, so this run
                               # isolates the MFS/MGS filter fix instead
AZIMUTHS_DEG = [0.0, 90.0]
RAMP = "linear"                # "linear" interpolates W_AT_0 -> W_AT_MAX, "flat" equal
W_AT_0 = 1.00                  # requested share at normal incidence
W_AT_MAX = 0.85                # requested share at TARGET_MAX_ANGLE; the rungs in
                               # between are interpolated, so 1.00 -> 0.85 is a
                               # profile that peaks on axis and falls off gently.
RAMP_AXIS = "theta"            # interpolate linearly in "theta" or in "u"=sin(theta)

# How the per-mode purities are turned into one score.
#
# Two constraints have to hold at once. The requested RATIO must be the optimum --
# a weighted sum is a linear program under the Parseval budget sum(p) <= 1, so it
# lands on a vertex and 1.00/0.93/0.85 collapses to 1/0/0. And the score must stay
# SEPARABLE, J = sum_i J_i(p_i), because the per-mode adjoint route builds the
# gradient as sum_i(fwd x adj_i); a product form like prod p_i^w_i does not
# decompose that way -- SEPARATE from that concern, DIST_MATCH below (the modal
# block's actual scoring) uses exactly that product form anyway, deliberately
# NOT decomposed into per-mode adjoints; see its own comment for why a sum, even
# this concave one, still cannot punish "everything in one order" hard enough.
# COMBINE / _sep_score below now only cover the transmission term (score_level)
# and the suppress term (grouped_suppress_J calls _sep_score directly). Both
# hold for the family
#
#       J_i = w_i * h(p_i / w_i),   h concave
#
# whose stationary condition h'(T) = lambda is the same for every i, hence p ∝ w.
#
#   "sqrt" J_i = sqrt(w_i * p_i). Cauchy-Schwarz gives sum_i sqrt(w_i p_i) <=
#          sqrt(sum w * sum p), so the modal block is bounded by sqrt(W_MODAL) and
#          NON-NEGATIVE by construction -- which is what msopt's optimizer assumes
#          (see _sep_score). Default for that reason.
#   "log"  J_i = w_i * ln(p_i / eps + 1). Same ratio optimum, but the natural form
#          w_i*ln(p_i) is negative for p < 1 and has to be shifted by a constant
#          that depends on LOG_EPS. Kept for comparison.
#   "sum"  J_i = w_i * p_i. Do NOT use to request a profile; see above.
COMBINE = "sqrt"
LOG_EPS = 1e-3                 # floor inside both scores: guards log at p=0 and
                               # keeps sqrt's derivative finite there (both blow up
                               # as p -> 0, which is the self-balancing that pulls a
                               # starved mode back, but it has to stay finite)

# ---- FoM split: how much arrived, versus how it is shared out -----------------
# Term 1      how much of the probe reaches the EML at all (a level).
# Terms 2..N  how that is divided among the target orders (a shape).
# The two are independent -- p_i is a SHARE of what arrived, so p_i * T is the
# absolute power in mode i -- which is why they need separate weights rather than
# one score. W_TRANS + W_MODAL is normalized internally, so 0.1/0.9 == 1/9.
# What the per-mode terms measure.
#   "purity"   p_i = overlap / TOTAL EML intensity: a share in [0, 1]. SHAPE only.
#              The FoM is then sqrt(WT*T) + sum sqrt(w_i p_i) -- T appears in its
#              own small term and is ADDED, so it cannot buy score by scaling the
#              angular distribution.
#   "absolute" a_i = overlap / (N * planar reference) = p_i * T. Level and shape in
#              one number, which is tidier but puts T where it does not belong:
#              sum sqrt(w_i a_i) = sqrt(T) * sum sqrt(w_i p_i), so T MULTIPLIES the
#              whole shape score. The shape factor only spans a factor ~2 between
#              "all in one order" and the requested ratio, while sqrt(T) has no
#              ceiling -- raising the total is always cheaper than fixing the
#              ratio. Seen in the v8 run: T 3.4 -> 5.1 (sqrt 1.23x) carried more of
#              the FoM rise than the shape did, and the first-order ratio went
#              BACKWARDS, 0.369 -> 0.178. Getting the angles right is the point, so
#              "purity" is the default.
MODAL_METRIC = "purity"
_MLAB = "a" if MODAL_METRIC == "absolute" else "p"

W_TRANS = 0.05                 # share of the FoM the level term should carry
W_MODAL = 0.95                 # share the modal block should carry, split by RAMP
# W_TRANS / W_MODAL are the CONTRIBUTION split. FoM = _ST*level_score + _SM*match
# (see transmission_J / match_J) is a literal convex combination of two [0,1]
# -bounded terms, so _ST = W_TRANS/(W_TRANS+W_MODAL) and _SM = 1-_ST ARE the
# actual blend weights directly -- no squaring/compensation needed the way the
# old sqrt(w*x)-summed design required (a concave score's contribution at its
# own reference point is sqrt(w), not w, which is what WT/WM below still
# exists to correct for -- that machinery now only feeds W_MODAL's absolute
# scale into build_target_modes' per-mode ramp weights, used by
# grouped_suppress_J's still-sqrt-scored suppress term; match_J's own w_hat
# renormalizes it away, so WM's exact value no longer matters there).
TRANS_REF = None               # |E|^2 on the EML plane that counts as T = 1.
                               # None -> measure the PLANAR device (full stack, no
                               # out-coupler) before the run starts, so T = 1 is the
                               # flat OLED and T > 1 means the design helped.

# N_ORDERS is free -- 45 deg does not have to be order 2. Raising it only costs a
# longer pitch and the orders that then escape ABOVE the target (~41% as many as
# there are rungs, at 45 deg, a ratio that does not shrink with M).
#
# Off by default, and the reason is worth recording: the modal terms are PURITIES,
# normalized by the total EML energy, so power that leaks above the target already
# shrinks every p_i. Solving the allocation with and without an explicit push-down
# term gives an identical optimum (leak = 0 either way) whenever the design can set
# the orders independently. The term only earns its adjoint run when the orders are
# COUPLED -- when making the top order unavoidably makes the one above it -- and
# then what it really controls is how much of the target angle to give up in
# exchange. Turn it on for a hard viewing-angle spec, not for efficiency.
SUPPRESS_ABOVE_TARGET = False
SUPPRESS_WEIGHT = 1.0          # relative to the mean target weight; 0 disables
SUPPRESS_AGGREGATE = True      # score all leaking orders in ONE objective, so
                               # suppression costs 1 adjoint run instead of one
                               # per order. False keeps them separate, which only
                               # buys a per-order breakdown in the logs.

# ---- Modal scoring: distribution MATCH, not weighted SUM ----------------------
# This replaced two earlier attempts at the same underlying problem (both kept
# only in the oled-outcoupling-optimization memory now, not in this file):
#   1. Plain per-mode score_one summed across 7 adjoints, COMBINE="sqrt". A
#      weighted sum can't tell "everything in one order" apart from "correctly
#      spread across all of them" as long as that one order also happens to
#      carry the biggest individual weight -- measured on the 0806 run
#      (20260806_092004_OLED_rec_gpu1_th8) and reproduced live on this run's
#      first iteration: 96% of the field sitting in the (0,0) order with the
#      other 6 modes at ~0 still scored 0.665 out of a ~1.0 ceiling.
#   2. RATIO_WINDOW: a per-rung grow_boost/shrink_weight multiplier bolted onto
#      score_one's weight, pushing purity-ratio-to-(0,0) into an explicit
#      [target-tol, target+tol] window (oc.make_ratio_performance_spec, the
#      same one OLED_opt.py / OLED_Min.py use). It worked -- verified with
#      autograd that the gradient sign flips correctly on both sides of the
#      window -- but it is still a SUM underneath, so an aggressively boosted
#      weight just raises the same ceiling problem one level up: the reported
#      FoM stopped being comparable iteration-to-iteration (needed a second,
#      un-boosted "ref=" number just to see whether the shape was actually
#      improving), and it needed a getval stop-gradient fix and a boost cap to
#      avoid a self-reinforcing feedback loop and 50-60x early weight spikes.
#
# DIST_MATCH fixes the root cause instead of compensating for it: score the
# modal block as a weighted GEOMETRIC MEAN of the captured purity's own shares,
# not a weighted arithmetic sum of the raw purities. See match_J for the exact
# formula; the short version is that a PRODUCT of per-mode terms crashes toward
# 0 the moment even one required order is empty, which a sum structurally
# cannot do no matter how any single term's weight is scaled. This is a
# genuinely joint function of every target purity at once (through the shared
# normalizer each mode's share is measured against), so it is NOT decomposed
# into per-mode adjoints the way score_one's terms are -- one joint objective,
# traced by autograd as a whole, replaces what used to be 7 separate ones (and
# is cheaper for it: 1 adjoint instead of 7).

# explicit mode only: (theta_air_deg, phi_deg)
TARGET_ANGLES = [(0.0, 0.0), (45.0, 0.0), (45.0, 90.0)]
PERIOD_UM = None               # None -> pitch derived from the targets
SOURCE_POL = "x"
MODE_WEIGHTS = None            # None -> equal weight per (angle, order)
_WSUM = float(W_TRANS) + float(W_MODAL)
# Squared, because the score is sqrt(w*x): a term at x = 1 contributes sqrt(w), so
# squaring the requested shares makes the CONTRIBUTIONS come out in the requested
# ratio. With COMBINE="sum" the score is linear and the shares are already the
# coefficients, so no squaring there.
_ST, _SM = float(W_TRANS) / _WSUM, float(W_MODAL) / _WSUM
if COMBINE == "sqrt":
    _q = _ST ** 2 + _SM ** 2
    WT, WM = _ST ** 2 / _q, _SM ** 2 / _q
else:
    WT, WM = _ST, _SM
PER_MODE_ADJOINT = True      # -> msopt Incoherent=True
BROADBAND_ADJOINT = False    # True: wavelengths nested under each mode
PLOT_EVERY_ITER = True       # structure + per-order purity, one figure per iteration
# VERTICAL_GRATING is decided below, with USE_MFS_FILTER and GRATING_PROFILE,
# right next to the Mapping() call that's the only thing that reads them --
# see the "Design mapping" section.

# Start from a ring grating carrying the requested orders instead of uniform grey.
# See seeded_x0() for why: a weakly modulated start does not survive the first
# beta step under a tight parameterisation.
SEED_HARMONICS = False
# Symmetry-breaking noise for a uniform start. A perfectly uniform slab is a SADDLE
# POINT for every target order above zero: c_m = 0 there, so d|c_m|^2/drho =
# 2*Re(c_m* dc_m/drho) = 0 and no gradient exists to grow them. sqrt(w*p) would
# regularize that (dJ/drho ~ sqrt(w)*d|c|/drho stays finite at c = 0) except the
# LOG_EPS floor caps dJ/dp and cancels it. Measured on the 4-fold uniform run: the
# target orders sat at 0.0000/0.0015 and SHRANK over three iterations while only T
# and the zeroth order moved. This much noise breaks the slab without biasing the
# design toward any particular order -- unlike SEED_HARMONICS, which picks them.
SEED_NOISE = 0.05
SEED_NOISE_RNG = 0             # fixed so a rerun reproduces the same start
SEED_AMPLITUDE = 0.50        # swing about grating_initial_density after std-normalization

oc.export_run_knobs(
    run_optimization=RUN_OPTIMIZATION, planar_pp=PLANAR_PP, pp_mode=PP_MODE,
    pp_dipole_grid=PP_DIPOLE_GRID, pp_capture_deg=PP_CAPTURE_DEG,
    pp_keep_fsp=PP_KEEP_FSP, resolution=RESOLUTION, design_h_um=DESIGN_H_UM,
    design_x_um=DESIGN_X_UM, design_y_um=DESIGN_Y_UM,
    design_n=DESIGN_N, design_low_n=DESIGN_LOW_N,
    probe_gap_um=PROBE_GAP_UM, top_margin_um=TOP_MARGIN_UM,
    stack=STACK, mc_color=MC_COLOR, mc_stack_kind=MC_STACK_KIND,
)

# The cell period is an OUTPUT of the target angles, so it has to be resolved
# before select_stack builds the geometry; the design region then defaults to it.
_st_probe = (km.ld.build_microcavity_stack(MC_COLOR, MC_STACK_KIND)
             if STACK == "microcavity" else km.ld.build_legacy_stack())
if PERIOD_UM:
    CELL_PERIOD = float(PERIOD_UM)
elif TARGET_MODE == "ladder":
    # snap_per_um here too, not just in build_target_modes: this is the value the
    # CELL is built from, and if the two calls disagree the projector momenta
    # kx = 2*pi*m/PERIOD stop being Bloch orders of the cell that was actually
    # simulated -- a silent wrong answer rather than a crash.
    _L = km.angle_ladder(TARGET_MAX_ANGLE, _st_probe.lam, N_ORDERS,
                         snap_per_um=RESOLUTION)
    CELL_PERIOD = float(_L["period_um"])
else:
    _p, _o, _r = km.minimum_period_2d(
        [(float(t), float(p)) for t, p in TARGET_ANGLES], _st_probe.lam, symmetry=SYMMETRY)
    if _p is None:
        raise ValueError("\n".join(_r))
    CELL_PERIOD = float(_p)
# The design footprint defaults to the full unit cell in oled_common, which is
# exactly what a periodic out-coupler wants -- so it is left alone. Writing the
# rounded period into MSOPT_OLED_DESIGN_X_UM would make it a few 1e-6 um LARGER
# than the cell and trip the overlap guard.

G, _mc_spec = oc.select_stack(STACK, MC_COLOR, MC_STACK_KIND, period_mc=CELL_PERIOD)
LAM = float(np.mean(G.visible_wavelengths))
K0 = 2.0 * np.pi / LAM


# =============================================================================
# Target momenta, straight from k_mapping
# =============================================================================
def build_target_modes():
    """Targets -> pitch -> momenta, in that order.

    The pitch is NOT a free knob: with a normal-incidence probe a cell of pitch P
    only carries u = (m, n) lambda/P, so the requested angles fix the pitch. Ask
    k_mapping for the smallest pitch that puts every target on a lattice point,
    and let the cell (and hence the design region) follow from it.
    """
    st = (km.ld.build_microcavity_stack(MC_COLOR, MC_STACK_KIND)
          if STACK == "microcavity" else km.ld.build_legacy_stack())
    lam = st.lam
    # n_org is the HIGHEST organic index in the stack (the CPL here, n=2.2) and is
    # the right bound for "can this momentum exist in the organic at all". It is NOT
    # the EML index -- the FoM plane sits in the EML (n=1.79), so the angle a target
    # momentum actually makes at the monitor has to be taken against that. The two
    # differ a lot: u=0.7071 is 18.75 deg in the CPL but 23.27 deg in the EML.
    n_org = st.organic_index_max()
    try:
        n_eml = float(np.real(st.n[st.names.index("EML")]))
    except (ValueError, AttributeError, IndexError):
        n_eml = float(n_org)
    if TARGET_MODE == "ladder":
        L = km.angle_ladder(TARGET_MAX_ANGLE, lam, N_ORDERS, n_org=n_org,
                            snap_per_um=RESOLUTION)
        period = float(PERIOD_UM) if PERIOD_UM else float(L["period_um"])
        for line in L["report"]:
            print(f"[k-map] {line}")
        base_w = km.ramp_weights(L["rungs"], RAMP, W_AT_0, W_AT_MAX, RAMP_AXIS)
        targets, orders, weights, suppress = [], [], [], []
        phis = [0.0] if SYMMETRY == "radial" else [float(a) for a in AZIMUTHS_DEG]

        def _emit(rung, w, is_suppress):
            if rung["m"] == 0:                    # k-space origin: no azimuth
                targets.append((0.0, 0.0)); orders.append((0, 0))
                weights.append(w); suppress.append(is_suppress)
                return
            for ph in phis:
                c, sn = np.cos(np.deg2rad(ph)), np.sin(np.deg2rad(ph))
                mm, nn = int(round(rung["m"] * c)), int(round(rung["m"] * sn))
                if (mm, nn) == (0, 0):
                    continue
                targets.append((rung["theta_air_deg"], ph))
                orders.append((mm, nn))
                weights.append(w / max(len(phis), 1))
                suppress.append(is_suppress)

        for r, w in zip(L["rungs"], base_w):
            _emit(r, w, False)
        # Orders past the target that still escape. Nothing forbids raising M -- the
        # leak count is a fixed ~41% of the rung count at 45 deg and does not shrink
        # with M -- so instead of capping M, name those orders and push them DOWN.
        # The log FoM already penalizes them through the purity denominator (leaked
        # energy shrinks every p_i), but that is the same generic pressure applied to
        # waveguide and SPP loss; an explicit term is direct control over the one
        # thing being asked for, at one extra adjoint run per suppressed mode.
        if SUPPRESS_ABOVE_TARGET and L["leaking_orders"]:
            v = SUPPRESS_WEIGHT * float(np.mean(base_w))
            for r in L["leaking_orders"]:
                _emit(r, v, True)
            print(f"[k-map] suppressing {len(L['leaking_orders'])} order(s) above "
                  f"{TARGET_MAX_ANGLE:g} deg at weight {v:.4f} each")
        rep = []
        MODE_W = weights
        MODE_SUP = suppress
    elif PERIOD_UM:
        period = float(PERIOD_UM)
        MODE_W = MODE_SUP = None
        step = lam / period
        orders = [(int(round(np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(p)) / step)),
                   int(round(np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(p)) / step)))
                  for t, p in targets]
        rep = [f"pitch fixed by hand: P = {period:g} um"]
        targets = [(float(t), float(p)) for t, p in TARGET_ANGLES]
    else:
        MODE_W = MODE_SUP = None
        targets = [(float(t), float(p)) for t, p in TARGET_ANGLES]
        period, orders, rep = km.minimum_period_2d(targets, lam, symmetry=SYMMETRY)
        if period is None:
            near = km.nearest_achievable([t for t, _p in targets], lam)
            msg = ["\n".join(rep), "", "closest sets that DO share one pitch:"]
            for r in near[:4]:
                msg.append(f"  P={r['period_um']:.4f} um  orders={r['orders']}  "
                           f"angles={[round(v, 2) for v in r['angles_deg']]}  "
                           f"max err {r['max_abs_error_deg']:.2f} deg")
            raise ValueError("\n".join(msg))
    for line in rep:
        print(f"[k-map] {line}")

    k0 = 2.0 * np.pi / lam
    # Weight and suppress flag travel WITH their target through the n_organic drop
    # filter, so a dropped mode in the middle cannot shift the rest by one.
    src_w = MODE_WEIGHTS if MODE_WEIGHTS is not None else MODE_W
    if src_w is None:
        src_w = [1.0] * len(targets)
    if len(src_w) != len(targets):
        raise ValueError(f"MODE_WEIGHTS has {len(src_w)} entries for {len(targets)} targets")
    if MODE_SUP is None:
        MODE_SUP = [False] * len(targets)
    modes, dropped = [], []
    for (t, ph), (m, n), wi, si in zip(targets, orders, src_w, MODE_SUP):
        ux, uy = m * lam / period, n * lam / period
        u = float(np.hypot(ux, uy))
        if u > n_org + 1e-9:
            dropped.append((t, ph, m, n, u))
            continue
        modes.append({
            "name": f"th{t:g}_phi{ph:g}_m{m}n{n}" + ("_SUP" if si else ""),
            "theta_air_deg": t, "phi_deg": ph, "m": int(m), "n": int(n),
            "ux": float(ux), "uy": float(uy), "u": u,
            "kx": float(k0 * ux), "ky": float(k0 * uy),
            "theta_org_deg": float(np.rad2deg(np.arcsin(min(u / n_org, 1.0)))),
            "theta_eml_deg": float(np.rad2deg(np.arcsin(min(u / n_eml, 1.0)))),
            "suppress": bool(si), "weight": float(wi),
        })
    if not modes:
        raise ValueError(f"every target exceeds n_organic={n_org:.3f}")
    if not any(not mo["suppress"] for mo in modes):
        raise ValueError("every surviving mode is a suppression term")
    w = np.asarray([mo["weight"] for mo in modes], dtype=float)
    # The modal weights share out W_MODAL; the transmission term holds W_TRANS.
    # Normalizing the pair means 0.1/0.9 and 1/9 are the same request.
    w = WM * w / max(float(np.sum(w)), 1e-30)
    for mo, wi in zip(modes, w):
        mo["weight"] = float(wi)
    print(f"[k-map] symmetry={SYMMETRY}, pitch={period:.5f} um, lambda={lam:g} um, "
          f"n_organic_max={n_org:.4f}, n_EML={n_eml:.4f}")
    for mo in modes:
        print(f"[k-map]   {mo['name']:<24s} theta={mo['theta_air_deg']:6.2f} "
              f"phi={mo['phi_deg']:6.1f}  (m,n)=({mo['m']},{mo['n']})  u={mo['u']:.4f}  "
              f"th_EML={mo['theta_eml_deg']:5.2f}  w={mo['weight']:.3f}  "
              f"{'SUPPRESS' if mo['suppress'] else 'target'}")
    for t, ph, m, n, u in dropped:
        print(f"[k-map]   DROPPED theta={t:g} phi={ph:g} (m,n)=({m},{n}): u={u:.4f} "
              f"> n_organic={n_org:.4f}")
    return modes, period, n_org, n_eml


target_modes, PERIOD, N_ORG, N_EML = build_target_modes()
# The projectors are exp(-i 2*pi*m*x/PERIOD); they are only Bloch orders of the
# simulated cell if PERIOD is that cell. Nothing downstream would complain if it
# drifted -- the FoM would just quietly score the wrong momenta -- so check it.
if abs(PERIOD - G.Sx) > 1e-9 or abs(PERIOD - G.Sy) > 1e-9:
    raise RuntimeError(
        f"target pitch {PERIOD:.9f} um != FDTD cell {G.Sx:.9f} x {G.Sy:.9f} um. "
        f"The cell is built from CELL_PERIOD before the geometry exists, so both "
        f"angle_ladder calls must use the same snap_per_um.")
# And the cell has to hold a whole number of mesh cells, or Lumerical retunes dx
# under Bloch and the design grid no longer lines up with the design monitor.
_cells = PERIOD * G.resolution
if abs(_cells - round(_cells)) > 1e-6:
    raise RuntimeError(
        f"pitch {PERIOD:.9f} um is {_cells:.4f} mesh cells at resolution "
        f"{G.resolution}/um. Bloch needs an integer; pass snap_per_um.")
N_fom = 1                       # ONE forward probe covers every target momentum


# =============================================================================
# Geometry: normal-incidence probe above the stack, FoM monitor at the EML plane
# =============================================================================
src_c = [0.0, 0.0, G.probe_plane_z]
src_s = [G.Sx, G.Sy, 0.0]
bandwidth = 0.0

# The EML mid-plane. Everything the FoM sees is this one 2-D cut, so it has to be
# the plane the emitters actually live on.
# The FoM surface is the FULL 2-D EML mid-plane in BOTH symmetry modes.
#
# A radial cut is tempting when the design is rotationally symmetric, but it is
# wrong here for two reasons:
#   * a linearly polarised normal-incidence probe is NOT azimuthally symmetric
#     even on a rotationally symmetric structure -- the response carries cos^2 phi
#     / sin^2 phi terms, and the residual symmetry only holds if the polarisation
#     is rotated along with the coordinates. One radial line does not see that.
#   * the adjoint source IS dJ/dE on the monitor surface. A line monitor gives a
#     LINE source, whose adjoint field differs from the plane source's, so the
#     result would be the exact gradient of a DIFFERENT objective.
#
# Symmetry belongs in the design PARAMETERISATION instead: with RADIAL_DESIGN the
# mapping sums the 3-D gradient over each ring, and that is where the azimuthal
# coverage properly comes from.
eml_monitor_name = "eml_plane"
eml_c = [0.0, 0.0, float(G.eml_c[2])]
eml_s = [G.design_s[0], G.design_s[1], 0.0]


def eml_grid(full=False):
    """Sample coordinates of the FoM plane, in um.

    The monitor spans the closed cell, so it reports BOTH edges: x = -P/2 and
    x = +P/2 are the same point under periodicity and the duplicate breaks the
    orthogonality of the Bloch basis (a pure single mode scored 1.000078 instead
    of 1). `full=True` gives the monitor's own grid; the default drops the
    duplicated last sample, which is the grid the projectors are built on.
    """
    nx = max(int(round(G.design_s[0] * G.resolution)) + 1, 2)
    ny = max(int(round(G.design_s[1] * G.resolution)) + 1, 2)
    x = np.linspace(-0.5 * G.design_s[0], 0.5 * G.design_s[0], nx)
    y = np.linspace(-0.5 * G.design_s[1], 0.5 * G.design_s[1], ny)
    return (x, y) if full else (x[:-1], y[:-1])


MONITOR_SHAPE = (len(eml_grid(full=True)[0]), len(eml_grid(full=True)[1]))


def mode_projectors():
    """exp(-i k . r) on the FoM surface, one per target mode.

    Both signs of k are scored: the cell is symmetric, so a lobe at +k always has
    its mirror at -k and asking for only one would demand a design that cannot be
    built. Normalization is 1/sqrt(N) so a pure single mode scores exactly 1
    (Parseval), which makes the FoM a share in [0, 1] instead of an amplitude.

    EXCEPT at the origin. (m, n) = (0, 0) is its own mirror -- exp(-i0) and
    exp(+i0) are the same projector -- so scoring both counts normal emission
    TWICE and lets it reach 2. Left in, the optimizer finds that immediately: the
    live run reached purity 1.2164 at 0 deg while orders 2 and 3 collapsed to
    0.0053 and 0.0008, i.e. it bought the doubled term by abandoning the ramp.
    The origin therefore gets a single projector.
    """
    x, y = eml_grid()
    X, Y = np.meshgrid(x, y, indexing="ij")
    N = float(X.size)
    out = []
    for mo in target_modes:
        phase = mo["kx"] * X + mo["ky"] * Y      # always the full 2-D phase
        plus = np.exp(-1j * phase) / np.sqrt(N)
        if mo["m"] == 0 and mo["n"] == 0:
            out.append((plus,))
        else:
            out.append((plus, np.exp(+1j * phase) / np.sqrt(N)))
    return out


PROJECTORS = mode_projectors()


# =============================================================================
# FoM
# =============================================================================
def _on_projector_grid(E):
    """Monitor field -> the grid the projectors live on (duplicate edge dropped)."""
    return npa.reshape(E, MONITOR_SHAPE)[:-1, :-1]


_LAST_PURITIES = [None]        # see modal_purity_terms's cache note below


def modal_purity_terms(Ex, Ey):
    """Per-mode share of the EML field that sits at the requested momentum.

    Normalizing by the total field energy on the plane makes the score a PURITY
    in [0, 1] rather than an amplitude: without it the optimizer can win by
    simply pushing more light into the EML plane while the angular content stays
    wrong, which is the failure the OLED_opt header documents for the older
    unbounded scores.
    """
    E = _on_projector_grid(Ex if SOURCE_POL == "x" else Ey)
    if MODAL_METRIC == "absolute":
        # ABSOLUTE modal coupling, referenced to the planar device: the mean
        # intensity sitting in mode i, in units of the planar EML intensity. Equal
        # to purity * T, so it carries level and shape in one number and the FoM
        # needs no separate level term to stop the optimizer from perfecting the
        # shape of nothing. sqrt(w*a) is concave, so dJ/da = 0.5*sqrt(w/a) is
        # largest for the starved order -- raising the total uniformly is always
        # worth less than feeding whichever order is behind, which is the job the
        # purity denominator used to do.
        denom = float(E.shape[0] * E.shape[1]) * _trans_ref(None)
    else:
        denom = npa.sum(npa.abs(E) ** 2) + 1e-30
    terms = []
    for projs in PROJECTORS:
        terms.append(sum(npa.abs(npa.sum(E * p)) ** 2 for p in projs) / denom)
    # Detached snapshot for reporting. match_J scores every target mode jointly
    # in ONE objective (see its own docstring), so there is no per-mode f0 for
    # _split_per_J to recover an individual purity from -- this cache is the
    # only place a per-mode number still exists. Every objective that reaches
    # here this iteration shares the same forward-field evaluation (see
    # per_mode_gradient), so caching once and reading it back is both correct
    # and free.
    _LAST_PURITIES[0] = [float(getval(t)) for t in terms]
    return terms


_TRANS_REF_CAL = [None]        # measured once, before the optimization starts


def measure_planar_reference():
    """|E|^2 at the FoM plane for the PLANAR device -- full stack, no out-coupler.

    T then reads directly as "how much better than the flat OLED", which is the
    comparison the device is actually judged on. Two references were tried and
    rejected: the starting DESIGN (T = 1 would mean "whatever the initial guess
    gave", so it moves with the initial density and two runs are incomparable), and
    empty AIR (well defined, but T = 3.4 just restates the microcavity's standing-
    wave enhancement, which no design choice controls and which therefore only
    inflates the level term).

    Same domain, same source, the stack built exactly as the optimization builds
    it -- only add_design_grid is left out.
    """
    if _TRANS_REF_CAL[0] is not None:
        return _TRANS_REF_CAL[0]
    sim = oc.make_sim(G, [G.Sx, G.Sy, G.Sz])
    sim.add_source(mode="plane", name="source", center=src_c, size=src_s,
                   direction="backward", src_wl=G.visible_wavelengths,
                   bandwidth=bandwidth,
                   pol=(0.0 if SOURCE_POL == "x" else 90.0),
                   theta=0.0, phi=0.0, broadband=False)
    oc.add_stack(G, sim)          # full stack; the design region is what is omitted
    sim.add_monitor(name=eml_monitor_name, center=eml_c, size=eml_s, N_f=1)
    sim.run(name=os.path.join(G.design_dir, "Planar_reference"), save=True)
    E = np.asarray(sim.fdtd.getresult(eml_monitor_name, "E")["E"],
                   dtype=np.complex128)
    E = np.squeeze(E)
    if E.ndim == 3 and E.shape[-1] == 3:
        inten = np.abs(E[..., 0]) ** 2 + np.abs(E[..., 1]) ** 2
    else:
        inten = np.sum(np.abs(E) ** 2, axis=-1)
    inten = np.asarray(inten, float)
    if inten.shape[:2] == MONITOR_SHAPE:
        inten = inten[:-1, :-1]                  # same grid the projectors use
    ref = float(np.mean(inten))
    _TRANS_REF_CAL[0] = max(ref, 1e-12)
    print(f"[fom] planar reference measured: mean |E|^2 at the EML plane with the "
          f"full stack and NO out-coupler = {ref:.6f}  ->  T = 1 means 'as much as "
          f"the flat OLED', T > 1 means the design helped")
    try:
        sim.fdtd.close()
    except Exception:
        pass
    return _TRANS_REF_CAL[0]


def _trans_ref(observed=None):
    """Reference intensity that counts as T = 1 (a CONSTANT -- a reference that
    tracked the current field would cancel out of the gradient)."""
    if TRANS_REF is not None:
        return float(TRANS_REF)
    if _TRANS_REF_CAL[0] is None:
        raise RuntimeError("planar reference not measured; call "
                           "measure_planar_reference() before the first FoM eval")
    return _TRANS_REF_CAL[0]


def transmission_term(Ex, Ey):
    """Captured level: how much MORE target-momentum power this design delivers
    to the EML plane than the flat OLED, i.e. raw level T times the CAPTURED
    share S = sum of the (non-suppressed) target purities.

    Used to be raw T alone (mean |E|^2 on the EML plane, relative to the flat
    planar reference -- not a transmittance and not bounded by 1: it is ~3.4
    raw because |E|^2 inside a resonant microcavity -- Ag mirror below,
    semi-transparent Ag cathode above -- exceeds the incident |E|^2. Stored
    field is not power flux, so nothing is violated, and it is the RIGHT
    quantity rather than a Poynting flux: by reciprocity it is what a dipole
    at that plane couples out with).

    Raw T turned out to fight the steering it was meant to only accompany:
    it is dominated by whatever configuration keeps the WHOLE EML-plane field
    highest, and for this microcavity that is the flat, ON-RESONANCE,
    UNDIFFRACTED state -- diffracting power out to the target angles disturbs
    that resonance and lowers T, so a raw-T level term directly penalizes the
    one thing the modal term is trying to do. T*S instead only credits
    arrived power that ALSO lands in a target momentum, so growing it moves
    the same direction as steering (steering is literally how S grows)
    instead of against it -- "how much more target-angle power this design
    delivers than the flat OLED," which is the LEE-floor guarantee this term
    was for in the first place.

    DIST_MATCH's shape score is deliberately SCALE-INVARIANT (p_hat = p / S
    divides S back out, see match_J), so it alone gives no pull to grow S --
    a vanishingly small but perfectly-shaped allocation scores the same as a
    large one. T*S is what supplies that pull, and because match_J never sees
    T, there is no risk of the older "buy shape score by raising T" failure
    the MODAL_METRIC="absolute" comment above warns about -- shape and level
    stay cleanly separated by construction.

    Both E components are summed regardless of SOURCE_POL: the probe is
    polarized but a structured design converts between components, and power
    that ends up in the orthogonal one has still arrived.
    """
    ex = _on_projector_grid(Ex)
    ey = _on_projector_grid(Ey)
    inten = npa.sum(npa.abs(ex) ** 2 + npa.abs(ey) ** 2) / float(ex.size)
    T = inten / _trans_ref(inten)
    all_p = modal_purity_terms(Ex, Ey)
    S = sum(all_p[i] for i, mo in enumerate(target_modes) if not mo["suppress"])
    return T * S


def _sep_score(x, weight):
    """One separable term J_i, on whichever scale COMBINE selects.

    Both concave options put the requested ratio at the optimum; they differ in
    range, and range matters because msopt's optimizer assumes a POSITIVE FoM:

      Opt_MS2.Updater   the global best only ever rises, `if Best[0] > Best[6]`,
                        and Best[6] starts at 0 -- a negative FoM never beats it.
      Warm_restarter    `if FoM < Best[0]*1.01` is the no-improvement branch. With
                        Best[0] = 0 a negative FoM always takes it, so
                        Updater(is_Best=True) is never reached: the best geometry
                        is NEVER SAVED, the warm-restart counter climbs every
                        iteration, and the learning rate is multiplied by 5 every
                        third one. Observed live on the plain-log run: "Best FoM: 0"
                        throughout and Max dv 165 on a density that lives in [0, 1].

    sqrt is non-negative by construction, so it needs no offset to satisfy that.
    """
    x = npa.maximum(x, 0.0) + LOG_EPS
    if COMBINE == "sqrt":
        return npa.sqrt(x * weight)
    if COMBINE == "log":
        return npa.log(x / LOG_EPS) * weight
    if COMBINE == "sum":
        return x * weight
    raise ValueError(f"COMBINE must be 'sqrt', 'log' or 'sum', got {COMBINE!r}")


LEVEL_SCORE_REF = oc.env_float("MSOPT_OLED_LEVEL_REF", 1.0)
                               # T*S value that counts as "half credit" --
                               # see level_score


def level_score(TS):
    """Bounded-to-[0,1) transform of the captured level T*S (see
    transmission_term), so the combined FoM can be a proper convex
    combination with match (also bounded [0,1]) instead of two differently
    -scaled quantities just added together with no shared ceiling.

    TS / (TS + ref): 0 at TS=0, 0.5 at TS=ref, -> 1 as TS -> infinity. Same
    saturating-ratio shape oc.oled_performance_metrics already uses for its
    own throughput_score, so this stays consistent with the rest of the
    OLED_* scripts rather than inventing a third way to bound a level term.
    ref=1 (default) means "captured target-order power alone equal to the
    ENTIRE planar reference's total EML intensity" is the 50%-credit point --
    a deliberately ambitious anchor, not a claim that T*S=1 is merely OK.
    """
    return TS / (TS + max(LEVEL_SCORE_REF, 1e-9))




# =============================================================================
# Design mapping (same radial/freeform switch as OLED_opt)
# =============================================================================
# Three independent choices, in the order Opt_MS2.Mapping actually decides them:
#   1. symmetry            radial (Is_radial_3d, rho(r,z)) vs fourfold (Sym_geo_C8)
#   2. filter vs freeform   real MFS/MGS vs raw per-pixel topology optimization
#   3. grating profile      vertical (one layer extruded through z) vs slanted
#                           (sidewall angle varies with z) -- only meaningful
#                           together with the filter; see below.

# ---- 1. symmetry ------------------------------------------------------------
# Symmetry is ALWAYS imposed -- there is no free-form-SYMMETRY option (not to
# be confused with choice 2's "freeform", which is about the FILTER, not the
# symmetry). The two symmetry choices differ in how it is imposed:
#   radial    Is_radial_3d: the design IS rho(r, z), so rotational symmetry is
#             exact by construction and the parameter count collapses.
#   fourfold  Sym_geo_C8 (msopt's Pseudo_Cyl): fliplr + flipud + transpose, which
#             generates D4 / C4v -- invariant under both mirrors, the diagonal
#             mirror and 90/180 deg rotation. Verified on a random array.
# Because the structure is symmetric either way, +k and -k always carry equal
# power, so merging them into one FoM term (and one adjoint) loses nothing.
if SYMMETRY not in km.SYMMETRY_MODES:
    raise ValueError(f"SYMMETRY must be one of {km.SYMMETRY_MODES}")
RADIAL_DESIGN = (SYMMETRY == "radial")
FOURFOLD_DESIGN = (SYMMETRY == "fourfold")

# ---- 2. filter vs freeform ---------------------------------------------------
# Opt_MS2.Mapping.__call__ branches on Is_freeform[0] (non-radial path) /
# Is_radial_3d["apply_filter"] (radial path) alone: True routes to a plain
# tanh projection with NO conic dilation/erosion anywhere in that branch,
# False routes to the actual MFS/MGS filter (Sub_Mapping.get_reference_layer /
# radial_cross_section_to_3d). This confirms a 65-day-old framework note
# ("Is_freeform=True in Mapping SKIPS the conic filter (only tanh)") that this
# script had NOT been following: every OLED_rec.py run through 20260807
# (including the 0806 run this script's ramp/weights were tuned against) used
# Is_freeform=[True, ...] and therefore never actually applied MFS/MGS -- they
# were live-but-inert config the whole time.
USE_MFS_FILTER = True

# ---- 3. grating profile ------------------------------------------------------
# Only meaningful within the FILTER path (Is_slanted_grating is read at
# Opt_MS2.py's `elif self.N_height > 1: if self.slanted: Slant_sidewall ...
# else: Vertical_sidewall`, which only that path reaches -- see choice 2).
# Freeform's own single-vs-full-3D choice is a different axis (below) and has
# no slanted option at all.
GRATING_PROFILE = "vertical"   # "vertical" or "slanted"
if GRATING_PROFILE not in ("vertical", "slanted"):
    raise ValueError(f"GRATING_PROFILE must be 'vertical' or 'slanted', got {GRATING_PROFILE!r}")
VERTICAL_GRATING = (GRATING_PROFILE == "vertical")   # also freeform's single-
                                                      # layer-vs-full-3D flag

DR_info = [G.design_s[0], G.design_s[1], G.design_s[2], 0, 1, 2]
DR_N_info = [G.Nx, G.Ny, G.Nz, G.resolution]
radial_radius = oc.env_float("MSOPT_OLED_RADIAL_RADIUS", 0.5 * min(G.design_s[0], G.design_s[1]))
radial_grids = oc.env_int("MSOPT_OLED_RADIAL_GRIDS", int(round(radial_radius * G.resolution)) + 1)
mapping = ms.Opt_MS2.Mapping(
    Symmetry_sim=False, Sym_geo_width=False, Sym_geo_C8=FOURFOLD_DESIGN,
    Sym_geo_length=False, Sym_geo_C2=False,
    DR_info=DR_info, DR_N_info=DR_N_info, Mask_pixels=0, MFS=0.1, MGS=0.05,
    Is_radial_3d={"enabled": bool(RADIAL_DESIGN), "N_radius": radial_grids,
                  "radius": radial_radius, "outside_value": 0.0,
                  "apply_filter": USE_MFS_FILTER, "vertical_grating": VERTICAL_GRATING},
    # Is_freeform = [freeform?, gray-scale?, SINGLE LAYER?]. Is_freeform[2] is
    # only read INSIDE the freeform branch, so it is irrelevant whenever
    # USE_MFS_FILTER is True -- there is no "filtered + freeform-single-layer-
    # flag" interaction to worry about.
    **({} if RADIAL_DESIGN else
       {"Is_freeform": [not USE_MFS_FILTER, False, VERTICAL_GRATING]}),
    Is_slanted_grating=(USE_MFS_FILTER and not VERTICAL_GRATING),
)
# msopt only sets parameter_count on the radial path. Both the filter path
# (always a single 2-D reference layer, vertical or slanted -- Slant_sidewall
# takes the same 2-D Reference_layer Vertical_sidewall does) and freeform's own
# single-layer mode take ONE layer (N_width*N_length) and extrude/slant it, so
# falling back to design_cells for either would hand it the full 3-D vector --
# "cannot reshape array of size 204304 into shape (113,113)" on the first
# iteration. Only genuine per-z-layer freeform (USE_MFS_FILTER=False and
# VERTICAL_GRATING=False) actually wants the full 3-D parameter count.
design_parameters = getattr(mapping, "parameter_count", None)
if design_parameters is None:
    _single_layer_params = USE_MFS_FILTER or VERTICAL_GRATING
    design_parameters = (G.Nx * G.Ny) if _single_layer_params else G.design_cells
def seeded_x0():
    """Start from a ring grating that already carries the requested orders.

    A uniform grey start is what killed the fabrication-constrained run. With the
    vertical-grating constraint the design has only 57 parameters, so msopt
    reached its beta=1 convergence in ~10 iterations while the profile was still
    0th-order dominant and barely modulated (p = 0.64/0.027/0.010/0.040). The
    first beta step then projected that weak modulation flat: binarization 100%,
    a uniform layer, every order but m=0 gone, and no gradient left to escape --
    the FoM sat at 0.5989 for ten identical evaluations.

    Order m is carried by a radial period P/m, so seeding cos(2*pi*m*r/P) for
    every requested m puts the rings there before the projection sharpens. It
    costs nothing and helps the unconstrained parameterisation too.
    """
    base = float(G.grating_initial_density)
    n = int(design_parameters)
    if not SEED_HARMONICS:
        x = base * np.ones(n)
        if SEED_NOISE > 0:
            rng = np.random.default_rng(int(SEED_NOISE_RNG))
            # BAND-LIMITED, not white. Per-pixel noise does not survive the MFS
            # filter: at MFS = 0.1 um and dx = 0.02 um the conic kernel averages
            # ~78 cells, so sigma 0.02 comes out at 0.002 and the slab is still a
            # slab -- measured, p_0 stayed 0.9917 against 0.9918 for no noise at
            # all. Keeping only |k| <= 2*pi*(M+1)/P puts the randomness on the
            # scales the out-coupler actually uses, which the filter passes
            # (period P/3 = 0.75 um against a 0.1 um feature size), while leaving
            # the phases random so no particular order is favoured.
            kmax = (N_ORDERS + 1) / PERIOD                 # cycles per um
            if n == G.Nx * G.Ny:                           # single layer, 2-D
                f = rng.standard_normal((G.Nx, G.Ny))
                kx = np.fft.fftfreq(G.Nx, d=1.0 / G.resolution)
                ky = np.fft.fftfreq(G.Ny, d=1.0 / G.resolution)
                mask = (np.hypot(*np.meshgrid(kx, ky, indexing="ij")) <= kmax)
                f = np.real(np.fft.ifft2(np.fft.fft2(f) * mask)).ravel()
            else:
                f = rng.standard_normal(n)
                kf = np.fft.fftfreq(n, d=1.0 / G.resolution)
                f = np.real(np.fft.ifft(np.fft.fft(f) * (np.abs(kf) <= kmax)))
            f = f / max(float(f.std()), 1e-12)
            x = np.clip(x + SEED_NOISE * f, 0.02, 0.98)
            print(f"[seed] uniform {base:g} + band-limited noise sigma={SEED_NOISE:g}, "
                  f"|k| <= {kmax:.4f}/um (rng {SEED_NOISE_RNG}): "
                  f"rho in [{x.min():.3f}, {x.max():.3f}]")
        return x
    orders_2d = sorted({max(mo["m"], mo["n"]) for mo in target_modes
                        if not mo["suppress"]} - {0})
    if not RADIAL_DESIGN:
        # Single-layer 2-D: cos(2*pi*m*x/P) + cos(2*pi*m*y/P) puts content on the
        # (m,0) and (0,m) orders, which are exactly the fourfold targets. Same
        # std-normalize-then-clip as the radial branch.
        if n != G.Nx * G.Ny or not orders_2d:
            print("[seed] harmonic seed needs the single-layer 2-D parameterisation; "
                  "starting uniform")
            return base * np.ones(n)
        x_um = np.linspace(-0.5 * G.design_s[0], 0.5 * G.design_s[0], G.Nx)
        y_um = np.linspace(-0.5 * G.design_s[1], 0.5 * G.design_s[1], G.Ny)
        X, Y = np.meshgrid(x_um, y_um, indexing="ij")
        prof = np.zeros_like(X)
        for m in orders_2d:
            prof += np.cos(2.0 * np.pi * m * X / PERIOD) + np.cos(2.0 * np.pi * m * Y / PERIOD)
        prof = prof / max(float(prof.std()), 1e-12)
        out = np.clip(base + SEED_AMPLITUDE * prof, 0.02, 0.98).ravel()
        print(f"[seed] 2-D harmonic seed on orders {orders_2d}, amplitude "
              f"{SEED_AMPLITUDE}: rho in [{out.min():.3f}, {out.max():.3f}], "
              f"{n:,} parameters")
        return out
    nr = int(radial_grids)
    if n % nr:
        print(f"[seed] unexpected parameter count {n} for {nr} radial samples; "
              f"starting uniform")
        return base * np.ones(n)
    orders = sorted({max(mo["m"], mo["n"]) for mo in target_modes
                     if not mo["suppress"]} - {0})
    if not orders:
        return base * np.ones(n)
    r = np.linspace(0.0, float(radial_radius), nr)
    prof = np.zeros(nr)
    for m in orders:
        prof += np.cos(2.0 * np.pi * m * r / PERIOD)
    # Normalize by the STD, not by the sum length or the peak. The harmonics are
    # all in phase at r = 0, so the raw sum is a spike: dividing it by len(orders)
    # or by its own maximum leaves every harmonic at 1/M of the available swing,
    # and at M = 6 that seeded orders 3-6 at purity ~0.004. Scaling by the std and
    # letting the profile clip gives a quasi-binary ring pattern -- measured
    # harmonic content 0.133-0.191 per order against 0.075 before, and still even
    # across orders (staggering the phases instead spreads them 0.05-0.32).
    prof = prof / max(float(prof.std()), 1e-12)
    prof = np.clip(base + SEED_AMPLITUDE * prof, 0.02, 0.98)
    # parameter_shape is (N_radius,) with vertical_grating, else (N_radius,
    # N_height) in r-major order -- repeat the same profile down z.
    x = prof if n == nr else np.repeat(prof[:, None], n // nr, axis=1).ravel()
    print(f"[seed] harmonic seed on orders {orders}, amplitude {SEED_AMPLITUDE}: "
          f"rho in [{x.min():.3f}, {x.max():.3f}], {n:,} parameters")
    return x


x0 = seeded_x0()
dJ_0 = np.zeros(G.design_cells)
print(f"[design] symmetry={SYMMETRY} "
      f"({'Is_radial_3d rho(r,z)' if RADIAL_DESIGN else 'Sym_geo_C8 = D4/C4v'}) "
      f"-> {design_parameters:,} parameters")


# =============================================================================
# Modal scoring: ONE joint distribution-match objective, plus per-group
# suppression (still separable, still per-mode adjoint where it applies)
# =============================================================================
def match_J(mode_indices):
    """Distribution-match FoM: a weighted GEOMETRIC MEAN of how the purity
    CAPTURED across the target modes (mode_indices) is actually shared out,
    against the requested ramp share, NORMALIZED so 1.0 means a perfect match
    and not just "as good as this target profile's own shape allows" -- see
    the DIST_MATCH config comment for why a weighted SUM was replaced at all.

        p_hat_i = p_i / sum_j(p_j)              j ranges over mode_indices
        w_hat_i = w_i / sum_j(w_j)               target_modes' own ramp weight
        raw     = prod_i (p_hat_i) ** w_hat_i  =  exp(sum_i w_hat_i * ln(p_hat_i))
        ceiling = prod_i (w_hat_i) ** w_hat_i    (raw's own value AT p_hat = w_hat)
        match   = raw / ceiling

    A PRODUCT, not a sum: any single p_hat_i collapsing toward 0 collapses the
    whole match toward 0, no matter how big the other terms are -- measured on
    this run's first evaluation, p = [0.9642, 0.0003, 0.0031, 0.0012, 0.0135,
    0.0001, 0.0006] (96% in the (0,0) order, everything else ~empty) scores
    match = 0.086, against 0.665 under the old weighted-sum score for the
    identical field.

    The /ceiling division is NOT optional. raw alone is the weighted geometric
    mean of shares that individually sum to 1 -- even at the perfect point
    p_hat = w_hat it is bounded by w_hat's own entropy, not by 1 (measured on
    this run's 7-mode ramp: ceiling = 0.152, so raw tops out at 0.15 even
    for an exact match, which is not the 0-to-1 "match rate" asked for).
    ln(raw) - ln(ceiling) = sum_i w_hat_i * ln(p_hat_i / w_hat_i), which is
    exactly the negative KL divergence from w_hat to p_hat -- a standard,
    principled distribution-similarity measure, non-positive by Gibbs'
    inequality, so match = exp(that) lands in (0, 1] with equality only at a
    perfect match. ceiling is a CONSTANT (w_hat is fixed at setup), so
    subtracting its log changes nothing about the gradient, only the scale.

    This is a genuinely JOINT function of every p_i at once, through the
    shared normalizer sum_j(p_j) -- unlike every other term in this file it is
    therefore NOT split across one adjoint per mode. It is traced by autograd
    as a single objective instead, which is also simply cheaper: one adjoint
    covers what used to take 7.

    Always in (0, ~1]: exp() of a real number is never negative, so this needs
    none of the non-negativity care _sep_score's docstring warns is required
    elsewhere.
    """
    idx = list(mode_indices)
    w = np.asarray([target_modes[i]["weight"] for i in idx], dtype=float)
    w_hat = w / max(float(np.sum(w)), 1e-30)
    ceiling_ln = float(np.sum(w_hat * np.log(w_hat + LOG_EPS)))

    def J(Ex, Ey):
        all_p = modal_purity_terms(Ex, Ey)
        p_list = [all_p[i] for i in idx]
        S = sum(p_list) + LOG_EPS
        ln_raw = sum(wh * npa.log(npa.maximum(pi / S, 0.0) + LOG_EPS)
                     for wh, pi in zip(w_hat, p_list))
        # Weighted by _SM so FoM = _ST*level_score + _SM*match is a proper
        # convex combination (_ST + _SM = 1) of two [0,1]-bounded terms,
        # keeping the WHOLE FoM in [0,1] instead of just this term -- see
        # transmission_J / level_score for the matching half of this.
        return _SM * npa.exp(ln_raw - ceiling_ln)
    return J


def grouped_suppress_J(mode_indices, weight):
    """ALL above-target orders in ONE objective, hence one adjoint run.

    The targets have to stay split because the per-mode log is what pins their
    RATIO -- collapsing them would only fix their sum. The suppressed orders carry
    no ratio to hold: the request is just "no power above the target", so their
    purities can be added first and scored once. dJ/dE of that sum is a single
    field on the FoM plane, so msopt builds a single adjoint source from it and the
    suppression cost stops scaling with the leak count -- it is 1 adjoint whether
    one order escapes or five.
    """
    def J(Ex, Ey):
        terms = modal_purity_terms(Ex, Ey)
        tot = sum(terms[i] for i in mode_indices)
        return _sep_score(1.0 - tot, weight)
    return J


def transmission_J(Ex, Ey):
    """Objective 1: the bounded level score, weighted by _ST so FoM =
    _ST*level_score + _SM*match is a proper convex combination -- see
    level_score and match_J's matching _SM weighting."""
    return _ST * level_score(transmission_term(Ex, Ey))


# Objective 1 is transmission (now T*S, see transmission_term); objective 2 is
# the ONE joint distribution-match term over every non-suppressed target mode;
# suppressed modes (if any) share one more.
OBJ_SPECS = [{"kind": "trans", "idx": []}]
_target_idx = [i for i, mo in enumerate(target_modes) if not mo["suppress"]]
if _target_idx:
    OBJ_SPECS.append({"kind": "match", "idx": _target_idx})
_sup_idx = [i for i, mo in enumerate(target_modes) if mo["suppress"]]
if _sup_idx:
    if SUPPRESS_AGGREGATE:
        OBJ_SPECS.append({"kind": "suppress", "idx": _sup_idx})
    else:
        OBJ_SPECS += [{"kind": "suppress", "idx": [i]} for i in _sup_idx]


def objective_for(spec):
    if spec["kind"] == "trans":
        return transmission_J
    if spec["kind"] == "match":
        return match_J(spec["idx"])
    w = float(np.sum([target_modes[i]["weight"] for i in spec["idx"]]))
    return grouped_suppress_J(spec["idx"], w)


def spec_label(s):
    if s["kind"] == "trans":
        return "eml_intensity"
    if s["kind"] == "match":
        return "distribution_match"
    if len(s["idx"]) == 1:
        return target_modes[s["idx"][0]]["name"]
    return f"SUPPRESS_x{len(s['idx'])}"


OBJECTIVES = [objective_for(s) for s in OBJ_SPECS]
OBJ_LABELS = [spec_label(s) for s in OBJ_SPECS]
print(f"[fom] {len(OBJECTIVES)} objective(s) = 1 captured level (w={_ST:.3f})"
      f" + 1 distribution-match (w={_SM:.3f}, over {len(_target_idx)} target order(s))"
      + (f" + {sum(1 for s in OBJ_SPECS if s['kind']=='suppress')} suppress"
         f" (covering {len(_sup_idx)} order(s))" if _sup_idx else "")
      + f"  -> {len(OBJECTIVES)} adjoint run(s) per iteration")


# =============================================================================
def build_problem():
    sim = oc.make_sim(G, [G.Sx, G.Sy, G.Sz])
    # Normal incidence, from air down into the stack: theta = phi = 0.
    sim.add_source(mode="plane", name="source", center=src_c, size=src_s,
                   direction="backward", src_wl=G.visible_wavelengths,
                   bandwidth=bandwidth,
                   pol=(0.0 if SOURCE_POL == "x" else 90.0),
                   theta=0.0, phi=0.0, broadband=False)
    oc.add_stack(G, sim)
    sim.add_design_grid("design", G.design_c, G.design_s, G.design_high_index,
                        G.design_low_index, G.design_grids,
                        G.grating_initial_density * np.ones(G.design_grids), LAM)
    sim.add_design_monitor()

    fom_history = []
    # objective_arguments [0, 1] = Ex, Ey on the FoM plane, and that plane IS the
    # EML mid-plane -- the whole formulation is about the momentum content there.
    # ONE objective PER TARGET MODE: with Incoherent=True msopt runs the forward
    # once and then one adjoint per objective, summing fwd x adj_j, which is the
    # requested g = sum_i(fwd x adj_i). Wavelength stays a sub-level, so turning
    # BROADBAND_ADJOINT on gives J1_lam1..J1_lamN, J2_lam1, ... unchanged.
    opt = ms.Lumerical_utill.LumericalOptimizationProblem(
        sim,
        objective_functions=OBJECTIVES,
        objective_arguments=[0, 1],
        FoM_size=eml_s,
        FoM_center=eml_c,
        adj_fwd=False,
        opt_idx=0,
        broadband_adjoint=BROADBAND_ADJOINT,
        Incoherent=PER_MODE_ADJOINT,
    )
    return sim, opt, fom_history


# One entry per msopt ITERATION, not per FDTD evaluation. msopt calls the loop
# several times per iteration -- the gradient point plus every line-search /
# backtracking probe -- and recording all of them made the history a mix of
# accepted designs and rejected trials, which reads as noise. The optimizer's own
# cur_iter is the only counter that means "a step was taken".
_FOM_TRACE = []          # (iteration, fom, level, match, {mode: purity})
_OPT_REF = [None]        # the OPT_Ms instance, set once main() builds it


def _msopt_iter():
    o = _OPT_REF[0]
    try:
        return int(o.cur_iter[0])
    except Exception:
        return len(_FOM_TRACE)


def _split_per_J(per_J):
    """msopt's per-objective f0 -> the purities, the bounded level score, and
    the distribution-match rate they were scored from.

    Purities are read straight from _LAST_PURITIES, not inverted out of
    per_J: there is no longer one objective per mode to invert (match_J
    scores all of them jointly), so per-mode purities only ever exist as the
    cache modal_purity_terms already left behind. level and match are each
    read back UNWEIGHTED (divide out _ST / _SM) so what gets reported and
    plotted is the same [0,1] diagnostic quantity regardless of the FoM split
    -- transmission_J / match_J each hand msopt the WEIGHTED value, since
    that is what has to sum to a bounded total FoM.
    """
    level, match = float("nan"), float("nan")
    for i, s in enumerate(OBJ_SPECS):
        if i >= len(per_J):
            break
        if s["kind"] == "trans":
            level = float(per_J[i]) / _ST if _ST > 0 else float("nan")
        elif s["kind"] == "match":
            match = float(per_J[i]) / _SM if _SM > 0 else float("nan")
    cached = _LAST_PURITIES[0]
    purities = ({i: cached[i] for i, mo in enumerate(target_modes) if not mo["suppress"]}
                if cached is not None else {})
    return purities, level, match


def _plot_state(X, purities, level, match, val, opt, hist, trial):
    """Refresh the rolling figure, ONE point per msopt iteration.

    Every call still has a design worth looking at, so the picture is redrawn each
    time; what is guarded is the HISTORY. Only the best score seen within an
    iteration is kept, so the trace holds accepted steps rather than a mixture of
    steps and the trials msopt threw away.
    """
    # The very first FoM-only call lands before msopt has populated f0_per_J, so
    # there is nothing per-mode to draw yet ("iteration plot skipped: 0" came from
    # taking max() of the resulting empty array).
    if not purities:
        return
    it = _msopt_iter()
    if _FOM_TRACE and _FOM_TRACE[-1][0] == it:
        if val > _FOM_TRACE[-1][1]:               # same iteration: keep the best
            _FOM_TRACE[-1] = (it, float(val), float(level), float(match), dict(purities))
    else:
        _FOM_TRACE.append((it, float(val), float(level), float(match), dict(purities)))
    if not PLOT_EVERY_ITER or isinstance(X, str):
        return
    try:
        plot_iteration_state(npa.clip(X, 0.0, 1.0), purities, level, match, it,
                             beta=getattr(opt, "beta", None), fom=val, trial=trial)
    except Exception as exc:                      # never kill a run over a plot
        print(f"[plot] iteration plot skipped: {exc}")


def make_rec_loop(opt, hist, mode_records):
    """Adjoint loop for OPT_Ms. The per-mode machinery lives in msopt now
    (Incoherent=True); this only records the breakdown msopt exposes."""
    def loop(X, N_cases, Case=True):
        if Case == 3:
            return npa.where(npa.isfinite(npa.array(X[0][0])), npa.array(X[0][0]), 0.0)
        rho = None if isinstance(X, str) else [npa.clip(X, 0.0, 1.0)]
        if not Case:
            # FoM-only evaluation: msopt's line search / backtracking calls this,
            # and it is a real design being scored, so it gets a plot too. Skipping
            # it was why the figure only refreshed on some iterations.
            f0, _ = opt(rho_vector=rho, need_gradient=False)
            per_J = list(getattr(opt, "f0_per_J", []) or [])
            val = float(np.sum(per_J)) if per_J else float(np.real(np.sum(f0)))
            purities, level, match = _split_per_J(per_J)
            print(f"[rec] FoM={val:.6e}  level={level:.4f}  match={match:.4f}  "
                  f"{_MLAB}={[f'{purities[k]:.4f}' for k in sorted(purities)]}"
                  f"  (eval only)")
            _plot_state(X, purities, level, match, val, opt, hist, trial=True)
            return val, [val]
        f0, g = opt(rho_vector=rho, need_gradient=True)
        per_J = list(getattr(opt, "f0_per_J", []) or [float(np.real(np.sum(f0)))])
        val = float(np.sum(per_J))
        hist.append([float(v) for v in per_J])
        gn = [float(np.linalg.norm(gj)) for gj in getattr(opt, "gradient_per_J", [])]
        # One record per OBJECTIVE, not per mode: with SUPPRESS_AGGREGATE the last
        # objective stands for several orders at once, so pairing per_J against
        # target_modes would mis-align every entry after the first suppressed one.
        rec = []
        for i, s in enumerate(OBJ_SPECS):
            lead = target_modes[s["idx"][0]] if s["idx"] else None
            rec.append({"mode": OBJ_LABELS[i],
                        "theta_air_deg": lead["theta_air_deg"] if lead else -1.0,
                        "theta_org_deg": lead["theta_org_deg"] if lead else -1.0,
                        "m": lead["m"] if lead else -1,
                        "n": lead["n"] if lead else -1,
                        "u": lead["u"] if lead else -1.0,
                        "n_orders": len(s["idx"]),
                        "kind": s["kind"],
                        "suppress": int(s["kind"] == "suppress"),
                        "f0": per_J[i] if i < len(per_J) else float("nan"),
                        "grad_norm": gn[i] if i < len(gn) else float("nan")})
        mode_records.append(rec)
        purities, level, match = _split_per_J(per_J)
        print(f"[rec] FoM={val:.6e}  level={level:.4f}  match={match:.4f}  "
              f"{_MLAB}={[f'{purities[k]:.4f}' for k in sorted(purities)]}")
        _plot_state(X, purities, level, match, val, opt, hist, trial=False)
        return (g if isinstance(X, str) else (val, [val], [g]))
    return loop


def _draw_stack(ax, x0, x1, show_names=True):
    """The layer stack in cross-section, drawn from the geometry actually built."""
    for L in G.stack_layers:
        zc, dz = L["center"][2], L["size"][2]
        nr = float(L["index"]["n"][0])
        kr = float(L["index"].get("k", [0.0])[0])
        metal = kr > 0.5
        ax.add_patch(plt.Rectangle((x0, zc - 0.5 * dz), x1 - x0, dz,
                                   facecolor="#8a8a8a" if metal else
                                   plt.cm.Blues(0.15 + 0.55 * (nr - 1.7) / 0.6),
                                   edgecolor="none", zorder=1))
        if show_names and dz > 0.02 and L["name"] != "EML":
            ax.text(x1 - 0.02 * (x1 - x0), zc, f"{L['name']}  n={nr:.2f}",
                    fontsize=6.0, ha="right", va="center",
                    color="w" if metal else "#123", zorder=6)
    dz = G.design_s[2]
    ax.add_patch(plt.Rectangle((x0, G.design_c[2] - 0.5 * dz), x1 - x0, dz,
                               facecolor="#ffd9a0", edgecolor="#e08b00",
                               hatch="///", lw=1.2, zorder=2))
    ax.text(x0 + 0.02 * (x1 - x0), G.design_c[2], "design (out-coupler)",
            fontsize=7, va="center", color="#7a4a00", zorder=6)
    ax.axhline(eml_c[2], color="#b00020", lw=1.4, ls="--", zorder=4)
    ax.text(x0 + 0.02 * (x1 - x0), eml_c[2] - 0.035,
            f"EML plane  n={N_EML:.2f}", fontsize=7, color="#b00020", zorder=6)


def plot_kmap_geometry(path=None):
    """What the pitch k_mapping chose actually does, in real space.

    Left  the forward run this script optimizes: one normal-incidence plane wave
          from air, diffracted by the design into the discrete internal angles.
    Right the emission it stands for. By reciprocity the same coefficient governs
          a ray leaving the EML at that internal angle and exiting into air at the
          target angle -- which is the pairing the FoM is really requesting.
    """
    path = path or os.path.join(G.design_dir, "OLED_rec_kmap_geometry.png")
    P = float(PERIOD)
    x0, x1 = -0.62 * P, 0.62 * P
    z_des_top = G.design_c[2] + 0.5 * G.design_s[2]
    z_des_bot = G.design_c[2] - 0.5 * G.design_s[2]
    # Air is trimmed to what the outgoing rays need. The panels are set to EQUAL
    # aspect further down so the drawn angles ARE the angles -- with the default
    # auto aspect a 23 deg ray in a 0.4 um stack under a 2.2 um period renders
    # almost vertical, which is exactly the thing this figure exists to show.
    z_top = z_des_top + 0.62
    z_bot = G.Z_min
    z_anode = max(L["center"][2] + 0.5 * L["size"][2] for L in G.stack_layers
                  if "anode" in L["name"].lower()) if any(
        "anode" in L["name"].lower() for L in G.stack_layers) else z_bot

    tgt, seen = [], set()
    for mo in target_modes:
        if mo["suppress"] or round(mo["u"], 9) in seen:
            continue
        seen.add(round(mo["u"], 9))
        tgt.append(mo)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(tgt), 1)))

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)
    for ax in (aL, aR):
        _draw_stack(ax, x0, x1, show_names=(ax is aR))
        ax.set_xlim(x0, x1)
        ax.set_ylim(z_bot, z_top)
        ax.set_aspect("equal")                    # so the drawn angles are the angles
        ax.set_xlabel(r"x ($\mu$m)")
        for xb in (-0.5 * P, 0.5 * P):
            ax.axvline(xb, color="#e08b00", lw=0.9, ls=":", zorder=3)
        zp = z_des_top + 0.07
        ax.annotate("", xy=(0.5 * P, zp), xytext=(-0.5 * P, zp),
                    arrowprops=dict(arrowstyle="<->", color="#e08b00", lw=1.4), zorder=6)
        ax.text(0.0, zp + 0.02, f"P = {P:.4f} $\\mu$m", ha="center", fontsize=8.5,
                color="#7a4a00", zorder=6)
    aL.set_ylabel(r"z ($\mu$m)")

    # ---- left: the forward probe ---------------------------------------------
    for xs in np.linspace(-0.5 * P, 0.5 * P, 7):
        aL.annotate("", xy=(xs, z_des_top), xytext=(xs, z_top - 0.06),
                    arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=1.5), zorder=5)
    aL.text(0.0, z_top - 0.04, r"normal-incidence probe, $\theta_{air}=0^\circ$",
            ha="center", va="top", fontsize=9, color="#1f4e79", zorder=6)
    # Carry the diffracted rays all the way to the anode mirror, not just to the
    # EML: over the 0.14 um from the design to the EML even a 23 deg ray shifts by
    # 0.06 um and reads as vertical.
    run = z_des_bot - z_anode
    for c, mo in zip(cols, tgt):
        t = np.deg2rad(mo["theta_eml_deg"])
        dx = run * np.tan(t)
        for sgn in ((1, -1) if dx > 1e-9 else (1,)):
            aL.annotate("", xy=(sgn * dx, z_anode), xytext=(0.0, z_des_bot),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8), zorder=5)
            xe = (z_des_bot - eml_c[2]) * np.tan(t) * sgn
            aL.plot([xe], [eml_c[2]], "o", ms=4.5, color=c, mec="w", mew=0.6, zorder=7)
        aL.plot([], [], "-", color=c, lw=1.8,
                label=f"m={max(mo['m'], mo['n'])}: $u$={mo['u']:.3f}, "
                      f"$\\theta_{{EML}}$={mo['theta_eml_deg']:.1f}$^\\circ$")
    aL.set_title("forward: 0$^\\circ$ plane wave $\\rightarrow$ target diffraction "
                 "angles in the EML", fontsize=10)
    aL.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    # ---- right: the emission it is equivalent to ------------------------------
    rise = z_des_bot - eml_c[2]
    out = z_top - z_des_top - 0.10
    for c, mo in zip(cols, tgt):
        ti, ta = np.deg2rad(mo["theta_eml_deg"]), np.deg2rad(mo["theta_air_deg"])
        xs = -0.30 * P                            # common launch point on the EML
        xm = xs + rise * np.tan(ti)
        aR.annotate("", xy=(xm, z_des_bot), xytext=(xs, eml_c[2]),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8), zorder=5)
        aR.plot([xm, xm], [z_des_bot, z_des_top], "-", color=c, lw=1.0,
                alpha=0.5, zorder=5)
        xe = xm + out * np.tan(ta)
        aR.annotate("", xy=(xe, z_des_top + out), xytext=(xm, z_des_top),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8), zorder=5)
        aR.text(xe, z_des_top + out + 0.03, f"{mo['theta_air_deg']:.0f}$^\\circ$",
                color=c, fontsize=9, ha="center", va="bottom", zorder=6,
                fontweight="bold")
        aR.plot([], [], "-", color=c, lw=1.8,
                label=f"{mo['theta_eml_deg']:.1f}$^\\circ$ in EML "
                      f"$\\rightarrow$ {mo['theta_air_deg']:.1f}$^\\circ$ in air")
    aR.plot([-0.30 * P], [eml_c[2]], "*", ms=13, color="#b00020", zorder=7)
    aR.set_title("emission (reciprocal): each target order $\\rightarrow$ its air angle",
                 fontsize=10)
    aR.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    fig.suptitle(f"k-mapping result: pitch {P:.4f} $\\mu$m "
                 f"($\\lambda$={LAM:g} $\\mu$m, $\\lambda/P$={LAM / P:.4f}), "
                 f"{len(tgt)} target order(s), {SYMMETRY} symmetry", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] k-map geometry -> {path}")


def plot_angle_mapping(path=None):
    """Free-space emission angle vs the EML-internal angle it is fed by.

    The whole formulation lives in u = k_par/k0 because that is what a planar stack
    conserves; this is the picture of what that means in angles. u = sin(theta_air)
    = n_org*sin(theta_EML), so the internal angles are compressed by refraction --
    the entire escape cone is theta_EML < asin(1/n_org), and everything past it is
    trapped no matter what the out-coupler does.
    """
    path = path or os.path.join(G.design_dir, "OLED_rec_angle_mapping.png")
    n = float(N_ORG)
    th_air = np.linspace(0.0, 90.0, 721)
    u = np.sin(np.deg2rad(th_air))
    th_org = np.rad2deg(np.arcsin(np.clip(u / n, 0.0, 1.0)))
    crit = float(np.rad2deg(np.arcsin(min(1.0 / n, 1.0))))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))
    ax.plot(th_air, th_org, "-", color="#1f4e79", lw=2.0,
            label=f"$u=\\sin\\theta_{{air}}=n\\,\\sin\\theta_{{EML}}$, n={n:.3f}")
    ax.axhline(crit, color="#b00020", ls="--", lw=1.4,
               label=f"escape cone limit  {crit:.2f}$^\\circ$")
    ax.fill_between([0, 90], crit, 90, color="#b00020", alpha=0.07)
    tgt = [m for m in target_modes if not m["suppress"]]
    sup = [m for m in target_modes if m["suppress"]]
    seen = set()
    for mo in tgt:
        key = round(mo["u"], 9)
        if key in seen:
            continue
        seen.add(key)
        ax.plot([mo["theta_air_deg"]], [mo["theta_org_deg"]], "o", ms=8,
                color="#e08b00", zorder=5)
        ax.annotate(f"m={max(mo['m'], mo['n'])}\n{mo['theta_air_deg']:.1f}$^\\circ$"
                    f"$\\to${mo['theta_org_deg']:.1f}$^\\circ$",
                    (mo["theta_air_deg"], mo["theta_org_deg"]),
                    textcoords="offset points", xytext=(6, -22), fontsize=8)
    for mo in sup:
        ax.plot([mo["theta_air_deg"]], [mo["theta_org_deg"]], "x", ms=9,
                color="#b00020", mew=2, zorder=5)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, max(90.0 / 3.0, crit * 1.35))
    ax.set_xlabel(r"free-space emission angle  $\theta_{air}$  (deg)")
    ax.set_ylabel(r"matching angle inside the EML  $\theta_{EML}$  (deg)")
    ax.set_title("angle mapping: what the design actually has to steer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Same information as momentum, which is where the orders are evenly spaced.
    ax2.axhline(1.0, color="#b00020", ls="--", lw=1.4, label="air light line  u=1")
    ax2.axhline(n, color="#444", ls=":", lw=1.4, label=f"$n_{{org}}$={n:.3f}")
    ax2.fill_between([-0.5, len(tgt) - 0.5], 1.0, n, color="#b00020", alpha=0.07)
    ax2.text(len(tgt) / 2.0 - 0.5, (1.0 + n) / 2.0, "trapped\n(waveguide / SPP)",
             ha="center", va="center", fontsize=9, color="#b00020")
    for k, mo in enumerate(tgt):
        ax2.plot([k], [mo["u"]], "o", ms=8, color="#e08b00", zorder=5)
        ax2.annotate(f"({mo['m']},{mo['n']})\nu={mo['u']:.3f}", (k, mo["u"]),
                     textcoords="offset points", xytext=(0, 9), fontsize=7.5,
                     ha="center", linespacing=1.15)
    ax2.set_xticks(range(len(tgt)))
    ax2.set_xticklabels([f"{m['theta_air_deg']:.0f}$^\\circ$\n$\\phi$={m['phi_deg']:.0f}"
                         for m in tgt], fontsize=8)
    ax2.set_ylim(0, n * 1.08)
    ax2.set_xlim(-0.5, len(tgt) - 0.5)
    ax2.set_ylabel(r"in-plane momentum  $u=k_\parallel/k_0$")
    ax2.set_title(f"targets in momentum, pitch {PERIOD:.4f} $\\mu$m "
                  f"($\\lambda/P$={LAM / PERIOD:.4f})")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] angle mapping -> {path}")


def plot_iteration_state(rho, purities, level, match, it, path=None, beta=None,
                         fom=float("nan"), trial=False):
    """Structure and per-order purity side by side, once per iteration.

    Plotted together on purpose: the question being asked of every iteration is
    which feature of the structure is feeding which order, and that is unreadable
    from either panel alone.
    """
    # One rolling file, overwritten each iteration -- the iteration number lives in
    # the title, not in a pile of filenames.
    path = path or os.path.join(G.design_dir, "OLED_rec_current.png")
    try:
        # oled_common.design_to_grid already accepts all three input sizes OPT_Ms
        # can hand over -- the full voxel vector, the mapped-parameter vector, or an
        # Nx*Ny sheet -- so use it instead of calling mapping() again. Doing that by
        # hand is what fed 204304 values to a radial map wanting its 912 parameters
        # ("cannot reshape array of size 204304 into shape (57,16)").
        vol = oc.design_to_grid(G, np.asarray(rho, float).ravel(), mapping)
    except Exception as exc:                          # never kill a run over a plot
        print(f"[plot] structure panel skipped: {exc}")
        vol = None

    tgt = [(k, mo) for k, mo in enumerate(target_modes) if not mo["suppress"]]
    # Column 2 is three stacked x-z cross-sections instead of one panel. The
    # earlier x-z-at-y=0 / y-z-at-x=0 pairing was redundant -- Sym_geo_C8
    # imposes D4/C4v (fliplr + flipud + transpose), so those two cuts are
    # identical by construction -- so this instead shows the SAME kind of cut
    # (x-z, i.e. constant y) at three different y, which is actually new
    # information: how the profile changes moving across the cell.
    #
    # The design region is wide and thin (period ~um across, a fraction of
    # that tall), so an aspect="equal" cut wants to render far wider than one
    # unit-width column can hold -- aspect="auto" (the original choice) fills
    # the column but draws the cross-section at the WRONG aspect ratio
    # instead, which is the actual bug being fixed here. Sizing the column
    # from the real x:z ratio (rather than guessing a fixed width_ratio) is
    # what stops the aspect-correct image (and its colorbar) from spilling
    # into column 3 -- the failure mode the first attempt at this had, at a
    # column width picked before checking whether it was wide enough.
    aspect_xz = float(G.design_s[0]) / max(float(G.design_s[2]), 1e-9)
    BASE_COL_W = 4.6            # approx. inches one width_ratio=1.0 unit renders at
    row_h_est = 3.3 / 3.0       # rough usable plot height per row, inches
    col2_ratio = max(row_h_est * aspect_xz / BASE_COL_W, 1.0)
    width_ratios = [1.0, col2_ratio, 1.05, 1.2]
    fig = plt.figure(figsize=(BASE_COL_W * sum(width_ratios), 4.6))
    outer_gs = fig.add_gridspec(1, 4, width_ratios=width_ratios)
    a1 = fig.add_subplot(outer_gs[0, 0])
    # A narrow SECOND inner column holds one colorbar axis spanning all three
    # rows -- carved from the same subgridspec as the image axes, so it is
    # guaranteed to stay inside column 2's own bounds instead of relying on
    # fig.colorbar's automatic ax-shrinking (matplotlib's make_axes), which is
    # what let the previous version's colorbar drift past the column boundary.
    inner_gs = outer_gs[0, 1].subgridspec(3, 2, width_ratios=[1.0, 0.05],
                                          hspace=0.75, wspace=0.25)
    a1b = fig.add_subplot(inner_gs[0, 0])
    a1c = fig.add_subplot(inner_gs[1, 0])
    a1d = fig.add_subplot(inner_gs[2, 0])
    cax = fig.add_subplot(inner_gs[:, 1])
    a2 = fig.add_subplot(outer_gs[0, 2])
    a3 = fig.add_subplot(outer_gs[0, 3])
    # Same sections oled_common.save_current_design_sections uses, and the same
    # binary colormap: a z-averaged single panel hid whether the design had actually
    # binarized or was still grey, and hid any z structure entirely.
    if vol is not None:
        Nx, Ny, Nz = G.design_grids
        xa = np.linspace(-0.5 * G.design_s[0], 0.5 * G.design_s[0], Nx)
        ya = np.linspace(-0.5 * G.design_s[1], 0.5 * G.design_s[1], Ny)
        za = np.linspace(G.design_c[2] - 0.5 * G.design_s[2],
                         G.design_c[2] + 0.5 * G.design_s[2], Nz)
        gray = float(np.mean((vol > 1e-3) & (vol < 1 - 1e-3)))
        a1.imshow(vol[:, :, Nz // 2].T, origin="lower", cmap="binary",
                  extent=(xa[0], xa[-1], ya[0], ya[-1]), vmin=0.0, vmax=1.0,
                  aspect="equal", interpolation="nearest")
        a1.set_ylabel(r"y ($\mu$m)")
        a1.set_title(f"x-y at z=center   grey {gray*100:.1f}%")

        y_max = float(ya[-1])
        rows = [(a1b, 0.8 * y_max, r"0.8\,y_{max}"),
                (a1c, 0.5 * y_max, r"0.5\,y_{max}"),
                (a1d, 0.0, r"0")]
        im = None
        for ax, y_target, y_lbl in rows:
            iy = int(np.argmin(np.abs(ya - y_target)))
            im = ax.imshow(vol[:, iy, :].T, origin="lower", cmap="binary",
                           extent=(xa[0], xa[-1], za[0], za[-1]), vmin=0.0, vmax=1.0,
                           aspect="equal", interpolation="nearest")
            ax.set_ylabel(r"z ($\mu$m)")
            ax.set_title(f"x-z at $y={y_lbl}={ya[iy]:.3f}\\,\\mu$m", fontsize=9)
        fig.colorbar(im, cax=cax, label=r"$\rho$")
    a1.set_xlabel(r"x ($\mu$m)")
    a1d.set_xlabel(r"x ($\mu$m)")

    got = np.array([purities[k] for k, _mo in tgt], dtype=float)
    want = np.array([target_modes[k]["weight"] for k, _mo in tgt], dtype=float)
    want = want / max(want.sum(), 1e-30) * max(got.sum(), 1e-30)   # same total
    x = np.arange(len(tgt))
    a2.bar(x - 0.2, got, 0.4, color="#1f4e79", label="achieved")
    a2.bar(x + 0.2, want, 0.4, color="#e08b00", alpha=0.85,
           label="requested ratio (rescaled to the same total)")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{mo['theta_air_deg']:.0f}$^\\circ$\n({mo['m']},{mo['n']})"
                        for _k, mo in tgt], fontsize=8)
    a2.set_ylabel("modal coupling  $a_i$" if MODAL_METRIC == "absolute" else "modal purity  $p_i$")
    a2.set_title(f"order {'coupling' if MODAL_METRIC == 'absolute' else 'purity'}   "
                 f"$\\Sigma$={got.sum():.4f}   level={level:.3f}   match={match:.3f}")
    a2.grid(True, alpha=0.3, axis="y")
    a2.legend(fontsize=8)

    # Third panel: the trend. FoM, level and match are ALL bounded to [0,1] now
    # (see level_score / match_J), so they share one axis instead of the old
    # FoM-vs-twin-axis-T*S split, which needed two scales because raw T*S was
    # not bounded. Trial points (msopt's line search) are marked apart from
    # accepted ones, because a run that looks like it is going backwards is
    # usually just showing the rejected probes alongside the accepted steps.
    if _FOM_TRACE:
        idx = np.array([r[0] for r in _FOM_TRACE], float)
        f = np.array([r[1] for r in _FOM_TRACE], float)
        lv = np.array([r[2] for r in _FOM_TRACE], float)
        mt = np.array([r[3] for r in _FOM_TRACE], float)
        a3.plot(idx, f, "-o", ms=4.5, color="#1f4e79", lw=1.6, label="FoM")
        a3.plot(idx, mt, "-", color="#7a4a00", lw=1.2, alpha=0.85, label="match")
        a3.plot(idx, lv, "-", color="#e08b00", lw=1.2, alpha=0.85, label="level")
        a3.axhline(f.max(), color="#2e7d32", ls="--", lw=1.0,
                   label=f"best FoM {f.max():.4f}")
        a3.set_xlabel("msopt iteration")
        a3.set_ylabel("score (all bounded 0-1)")
        a3.set_ylim(-0.02, 1.05)
        a3.set_title("FoM / match / level history (accepted steps)")
        a3.grid(True, alpha=0.3)
        a3.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(f"OLED_rec  --  iteration {it}"
                 f"{'  (line-search trial)' if trial else ''}   "
                 f"FoM={fom:.4f}   level={level:.3f}   match={match:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def iter_plot(state):
    """Per-iteration record of which mode is carrying the score."""
    hist = state.get("history") or []
    if not hist:
        return
    arr = np.asarray(hist, dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for k, s in enumerate(OBJ_SPECS):
        if k >= arr.shape[1]:
            break
        if s["kind"] == "trans":
            ax.plot(arr[:, k], lw=2.0, color="k",
                    label=f"captured level, weighted (_ST={_ST:.2f})")
            continue
        if s["kind"] == "match":
            ax.plot(arr[:, k], lw=2.0, color="#7a4a00",
                    label=f"distribution match, weighted (_SM={_SM:.2f}, "
                          f"{len(s['idx'])} order(s))")
            continue
        lead = target_modes[s["idx"][0]]
        lbl = (f"SUPPRESS >{TARGET_MAX_ANGLE:g} deg ({len(s['idx'])} order(s))"
               if s["kind"] == "suppress" else
               f"{lead['theta_air_deg']:g} deg air / {lead['theta_org_deg']:.1f} deg EML "
               f"(m,n)=({lead['m']},{lead['n']})")
        ax.plot(arr[:, k], lw=1.4, ls="--" if s["kind"] == "suppress" else "-",
                label=lbl)
    ax.plot(arr.sum(axis=1), "k--", lw=1.8, label="total FoM")
    ax.set_xlabel("FoM evaluation")
    ax.set_ylabel("per-objective score (weighted contribution)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"OLED_rec: EML momentum content, pitch {PERIOD:g} um, pol {SOURCE_POL}")
    fig.tight_layout()
    fig.savefig(os.path.join(G.design_dir, "OLED_rec_mode_history.png"), dpi=150)
    plt.close(fig)


def main():
    if oc.env_flag("MSOPT_OLED_SESSION_TEST", "0"):
        oc.session_test_banner(G, len(OBJECTIVES))
        plot_kmap_geometry(); plot_angle_mapping()
        return
    start = time.time()
    post_only = oc.env_flag("MSOPT_OLED_POSTPROCESS_ONLY", "0")

    if not post_only:
        if ag_jacobian is None:
            raise RuntimeError("autograd is required; install it or set "
                               "MSOPT_OLED_POSTPROCESS_ONLY=1")
        plot_kmap_geometry()
        plot_angle_mapping()
        # Empty-domain reference BEFORE build_problem: the transmission term is
        # measured against it so it has to exist before any FoM is evaluated, and
        # running it first means its Lumerical session is opened and closed while
        # the optimization session does not yet exist -- two live sessions would
        # ask the licence server for 18 tasks at once.
        if TRANS_REF is None:
            measure_planar_reference()
        sim, opt, hist = build_problem()
        print(f"[setup] period={G.Sx:g}x{G.Sy:g} um, design={G.design_grids}, "
              f"modes={len(target_modes)}, probe=normal-incidence {SOURCE_POL}-pol, "
              f"FoM plane z={eml_c[2]:+.4f} um (EML)")
        optimizer = ms.Opt_MS2.OPT_Ms(
            x0, dJ_0, design_dir=G.design_dir, local_best_dir=G.local_dir,
            Born_k=90, Initial_LR=oc.env_float("MSOPT_OLED_INITIAL_LR", 0.2),
            # Raw=True sets Armijo_cond = -1, so `f0 < Armijo_cond` can never fire
            # on a positive FoM and the backtracking line search is dead: a step
            # that makes the FoM worse is simply kept. Observed here as every
            # gradient step losing ground (0.7446 -> 0.6869 -> 0.7431 -> 0.6866)
            # with the learning rate pinned at 0.2 and warm restarts MULTIPLYING it
            # by 5. Raw=False uses Armijo_cond ~ f_old, which shrinks alpha by 10x
            # and retries whenever a step does not improve. OLED_lens -- the run
            # that actually converged -- is the one that had Raw=False.
            Raw=False)
        optimizer.flag = True
        _OPT_REF[0] = optimizer          # so the plots can read msopt's cur_iter
        mode_records = []
        optimizer(mapping, N_fom, make_rec_loop(opt, hist, mode_records))
        iter_plot({"history": hist})
        if mode_records:
            np.savetxt(os.path.join(G.design_dir, "OLED_rec_mode_gradients.txt"),
                       np.array([[r["theta_air_deg"], r["theta_org_deg"],
                                  r["m"], r["n"], r["u"],
                                  r["suppress"], r["n_orders"],
                                  r["f0"], r["grad_norm"]]
                                 for it in mode_records for r in it]),
                       header="theta_air_deg theta_org_deg m n u suppress "
                              "n_orders f0 grad_norm")
        oc.save_result_plots(optimizer, G.design_dir)

    if oc.env_flag("MSOPT_OLED_POSTPROCESS", "1"):
        design_path = os.environ.get("MSOPT_OLED_POSTPROCESS_DESIGN", "").strip()
        if not (design_path and os.path.exists(design_path)):
            design_path = os.path.join(G.design_dir, "lastdesign.txt")
        if oc.planar_requested() and not os.path.exists(design_path):
            print("[postprocess] PLANAR stack characterization (no design required)")
            oc.run_postprocess(G, np.zeros(int(np.prod(G.design_grids)), float), mapping=None)
        elif os.path.exists(design_path):
            print(f"[postprocess] using design: {design_path}")
            oc.run_postprocess(G, np.loadtxt(design_path), mapping=mapping)
        else:
            print(f"[postprocess] skipped: {design_path} not found")
    print(f"Runtime setup time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
