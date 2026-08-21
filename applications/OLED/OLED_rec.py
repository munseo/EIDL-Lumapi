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
    J = level_score(T*S) * match(p)                  <- MERGE_LEVEL_INTO_MATCH
    J = _ST * level_score(T*S) + _SM * match(p)      <- the older split form

    p_i = |<E_EML , exp(-i k_i . r)>|^2 / |E_EML|^2, one per target (angle,
    order) pair -- the normalized modal purity, what fraction of the EML
    field sits in the momentum we asked for. level_score(T*S) is a bounded,
    saturating reward for how much power actually lands in the target momenta
    (T = EML-plane level relative to the flat OLED, S = sum_i p_i); match(p)
    is a KL-divergence-based weighted geometric mean scoring how that
    CAPTURED purity is SHARED across the target set against the requested
    ramp, normalized to 1.0 at a perfect match. See level_score / match_J for
    the exact formulas, and match_J's own docstring for why a weighted SUM of
    purities (this script's original design, and OLED_opt's/OLED_new's)
    cannot punish "everything in one order" the way a product-based score can.

    The two are combined by a PRODUCT, not a sum, and they are not
    interchangeable halves: match is deliberately SCALE-INVARIANT (it divides
    by S), so nothing in it pulls S up, and as a SEPARATE objective the level
    term was skipped by minimax on essentially every iteration for sitting
    above the mean -- leaving S free to collapse while the shape score stayed
    happy. Multiplying GATES the shape score by the level, so a
    perfectly-shaped but vanishing allocation scores ~0. See
    MERGE_LEVEL_INTO_MATCH and merged_J for the measurement that forced this.

Gradient
    ONE top-level objective (two with suppression on), not one per target
    mode: merged_J covers the level and every target (angle, order) pair at
    once. msopt's Incoherent=True (PER_MODE_ADJOINT) still runs one forward +
    one adjoint PER OBJECTIVE FUNCTION, restoring the cached forward field
    between them -- "per objective" just no longer means "per mode": match_J's
    joint KL-divergence form is NOT separable across modes the way a weighted
    sum is (see its own docstring), so it is traced by autograd as a single
    objective instead of being split into one adjoint per mode. That single
    joint adjoint gives the mathematically EXACT gradient of the whole
    product, not an approximation -- no per-mode decomposition trick is needed
    the way RATIO_WINDOW (an earlier, now-removed attempt at the same
    underlying problem; see match_J and the oled-outcoupling-optimization
    memory) required one.

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
#   TARGET_ANGLES     emission angles in air the design must serve (deg);
#                     explicit mode only -- see TARGET_MODE / N_ORDERS below
#                     for the default ladder mode
#   PERIOD_UM         out-coupler pitch; None -> derived from the targets
#   SOURCE_POL        "x" or "y": the normal-incidence probe polarization
#   PER_MODE_ADJOINT  True -> msopt Incoherent=True: one forward + one adjoint
#                     PER OBJECTIVE FUNCTION (now trans + match, not one per
#                     mode -- see the header docstring's Gradient section)
RUN_OPTIMIZATION = True
PLANAR_PP = False
PP_MODE = "supercell"
PP_DIPOLE_GRID = None          # None -> let oled_common.resolve_dipole_grid decide
                               # (12x12 folded for a mirror-symmetric design, 6x6
                               # otherwise -- 36 FDTD runs either way).
                               # MUST stay None rather than a number: this value is
                               # pushed into MSOPT_OLED_PP_DIPOLE_GRID by
                               # export_run_knobs below, and env_int cannot tell an
                               # env var the SCRIPT exported from one the USER set,
                               # so any number here silently overrides the default
                               # for every run. Set the env var to override.
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
AIR_BOT_UM = 0.35
os.environ.setdefault("MSOPT_OLED_AIR_BOT_UM", f"{AIR_BOT_UM:.6f}")
STACK = "microcavity"
MC_COLOR = "green"
MC_STACK_KIND = "onaxis"   # +93% on-axis luminance; costs 24% LEE and doubles the
                           # guided-mode share, so the postprocess needs more tiles
                           # than "optimized" before its LEE means anything.

# SYMMETRY decides what the FoM plane is and which orders are scored:
# SYMMETRY picks the ORDER SET and the design parameterisation. The FoM monitor
# is the full 2-D plane either way (see the note above eml_s).
#   "radial"    rotationally symmetric design (RADIAL_DESIGN below should be True):
#               only (m, 0) orders are requested, because the ring-summing mapping
#               makes every azimuth equivalent. One adjoint per order.
#   "fourfold"  4-fold symmetric design: representative azimuths phi = 0 / 45 / 90
#               from the (m,0), (m,m) and (0,n) orders. The adjoint count grows
#               with the number of azimuth samples -- one per (order, azimuth).
SYMMETRY = oc.env_str("MSOPT_OLED_SYMMETRY", "fourfold")  # radial|fourfold|none
# Resolved here rather than next to Mapping() because build_target_modes()
# reads KM_SYMMETRY, and that runs at import time long before the design
# section: defining it there raised NameError on every mode.
if SYMMETRY not in km.SYMMETRY_MODES + ("none",):
    raise ValueError(f"SYMMETRY must be one of {km.SYMMETRY_MODES + ('none',)}")
RADIAL_DESIGN = (SYMMETRY == "radial")
FOURFOLD_DESIGN = (SYMMETRY == "fourfold")
NOSYM_DESIGN = (SYMMETRY == "none")
# SYMMETRY names the DESIGN parameterisation; the ORDER SET only cares whether
# the lattice is 1-D (radial) or 2-D. "none" is a square lattice like fourfold,
# so it shares fourfold's momentum bookkeeping and differs only in Mapping().
KM_SYMMETRY = "fourfold" if NOSYM_DESIGN else SYMMETRY

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
N_ORDERS = 3                   # ladder: rungs / top order M. P = M*lam/sin(TARGET_MAX_ANGLE)
AZIMUTHS_DEG = [0.0, 90.0]
RAMP = "linear"                # "linear" interpolates W_AT_0 -> W_AT_MAX, "flat" equal
W_AT_0 = oc.env_float("MSOPT_OLED_W_AT_0", 1.00)     # requested share at normal incidence
W_AT_MAX = oc.env_float("MSOPT_OLED_W_AT_MAX", 0.85)  # requested share at TARGET_MAX_ANGLE;
                               # rungs between are interpolated, giving a profile
                               # that peaks on axis and falls off gently.
                               # env-overridable so a ramp sweep (e.g. 0.9 vs 0.8 at
                               # the top angle) launches without editing the file.
RAMP_AXIS = "theta"            # interpolate linearly in "theta" or in "u"=sin(theta)

# How a per-mode purity becomes one score. Only the SUPPRESS term still uses this
# (grouped_suppress_J calls _sep_score); the level term uses level_score and the
# modal block uses match_J, each with its own bounded form.
#
# J_i = sqrt(w_i * p_i): concave, so the requested RATIO is the optimum rather than
# a vertex, and non-negative by construction -- which msopt's optimizer requires.
# See _sep_score for what a negative FoM does to Opt_MS2.Updater.
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
#   "purity"   p_i = overlap / TOTAL EML intensity, a share in [0, 1]: SHAPE only.
#              match_J divides by sum_i(p_i) anyway, so any common scale cancels.
#
# An "absolute" variant, a_i = overlap / (N * planar reference) = p_i * T, used to
# be selectable and is REMOVED. It could not pay for itself: match_J divides by
# sum(p) anyway, so the SHAPE score was identical either way, while
# transmission_term already multiplies by T -- absolute fed T into the level term
# a second time (T ~ 3.4 in this microcavity, not a rounding error). It also
# rescaled the raw suppressed sum that LEAK_TOL thresholds, so the same tolerance
# bit ~T times harder and no run could separate "metric" from "suppression
# strength". Nothing measurable was left once both defects were accounted for.
_MLAB = "p"

MULTIOBJ = oc.env_str("MSOPT_OLED_MULTIOBJ", "minimax")   # "minimax" | "weighted"
# HOW THE OBJECTIVES COMBINE.
#   "weighted"  FoM = W_TRANS*level + W_MODAL*match (+ suppress), gradients
#               summed. Needs the ratio below to be right, and a weighted sum
#               of competing objectives is a linear program in the trade-off:
#               its optimum sits at a VERTEX, so it will starve one objective
#               to feed another rather than balance them. Picking W_TRANS was
#               always a guess at where that vertex should be.
#   "minimax"   every objective is its own raw [0,1] score and msopt steers
#               with the gradients of the BELOW-AVERAGE ones only
#               (Opt_MS2.Minimax, via Multiobj="minimax"). The worst
#               objective is what improves, so no objective can be traded
#               away and NO RATIO IS REQUIRED -- W_TRANS/W_MODAL are unused.
#               This is the answer to "what T:P ratio is right": none, if the
#               level term is written as a CONSTRAINT that saturates (see
#               level_score/LEVEL_KNEE) rather than a quantity to maximize.

# ---- Level and match: ONE objective (a product), not two ----------------------
MERGE_LEVEL_INTO_MATCH = oc.env_flag("MSOPT_OLED_MERGE_LEVEL", "1")
# ON  J_merged = level_score(T*S) * match(p)          -- one objective, one adjoint
# OFF J1 = _ST*level_score(T*S), J2 = _SM*match(p)    -- the older split form
#
# WHY THE SPLIT FORM FAILS. match_J divides by sum(p), so it is SCALE-INVARIANT:
# it scores the RATIO among the captured orders and is indifferent to how much was
# captured at all. The only term that sees the captured share S is the level term,
# and under minimax that term is skipped on every iteration it sits above the mean
# of the three -- which, with the knee at the planar level (LEVEL_KNEE_SCORE=0.9),
# is essentially always. So nothing pushes S up. Measured live on run 1_diag at
# eval 107: sum(p) = 0.168 over 7 cones against 0.562 over 4 cones for 2_nodiag,
# with level = 0.870 -> T*S = 0.743, i.e. T ~ 4.4. Field piles up in the cavity
# instead of in the target momenta -- the old "buy score by raising T" failure
# reappearing in a new place.
#
# WHY A PRODUCT, and specifically NOT the removed "absolute" metric: feeding p_i*T
# into the modal terms gets divided straight back out by match_J's own normalizer,
# which is why absolute measured nothing (see _MLAB above). Multiplying the
# FINISHED match by the FINISHED level score cannot be normalized away, keeps both
# factors in [0,1] so the result is too, and leaves transmission_term as the single
# place T enters -- no double count. A zero in either factor zeroes the objective,
# which is exactly the gating the split form lacked.
#
# Objective count drops 3 -> 2 (or 2 -> 1 with suppression off), so minimax's mean
# threshold changes with it: with two objectives one is always at or below the
# mean, and the merged term drives whenever it is the laggard.
#
# NOT IMPLEMENTED, the alternative considered -- cheaper but less fundamental:
# keep the objectives separate and change the SELECTION rule from "f_j <= mean" to
# "f_j < its own target" (0.9 for level, 1.0 for the others), so an objective short
# of its own knee is never skipped just because the others happen to be worse.
# Worth testing as its own factor; it is not obvious which is better.
W_TRANS = 0.05                 # ("weighted" only, and only with the merge OFF)
                               # share the level term carries
W_MODAL = 0.95                 # share the modal block should carry, split by RAMP
# W_TRANS / W_MODAL are the CONTRIBUTION split. FoM = _ST*level_score + _SM*match
# (see transmission_J / match_J) is a literal convex combination of two [0,1]
# -bounded terms, so _ST = W_TRANS/(W_TRANS+W_MODAL) and _SM = 1-_ST ARE the
# actual blend weights directly -- no squaring/compensation needed the way the
# old sqrt(w*x)-summed design required (a concave score's contribution at its
# own reference point is sqrt(w), not w, which is what WM below still
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
SUPPRESS_ABOVE_TARGET = oc.env_flag("MSOPT_OLED_SUPPRESS_ABOVE", "1")  # env override
SUPPRESS_WEIGHT = 1.0          # relative to the mean target weight; 0 disables
LEAK_TOL = oc.env_float("MSOPT_OLED_LEAK_TOL", 0.05)
                               # minimax only: the share of EML power allowed to
                               # escape ABOVE the target before the suppression
                               # objective is judged half-spent. See
                               # grouped_suppress_J.
SUPPRESS_AGGREGATE = True      # score all leaking orders in ONE objective, so
                               # suppression costs 1 adjoint run instead of one
                               # per order. False keeps them separate, which only
                               # buys a per-order breakdown in the logs.

# ---- Modal scoring: distribution MATCH, not weighted SUM ----------------------
# score_one-summed-across-adjoints and RATIO_WINDOW both failed the same way and
# are gone (details in the oled-outcoupling-optimization memory): a SUM cannot
# tell "everything in one order" from "correctly spread" whenever that order also
# carries the largest weight -- 96% of the field in (0,0) with the other 6 modes
# at ~0 still scored 0.665 of a ~1.0 ceiling.
#
# match_J scores a weighted GEOMETRIC MEAN of the captured shares instead. A
# product crashes toward 0 the moment one required order is empty, which no sum
# can do at any weighting. It is joint in every purity at once (shared
# normalizer), so it is ONE objective traced by autograd, not one adjoint per
# mode -- also 1 adjoint instead of 7.

# explicit mode only: (theta_air_deg, phi_deg)
TARGET_ALL_ORDERS = oc.env_flag("MSOPT_OLED_ALL_ORDERS", "1")   # env override.
                               # score EVERY escaping (m,n), grouped into cones by |u|
TARGET_ANGLES = [(0.0, 0.0), (45.0, 0.0), (45.0, 90.0)]
PERIOD_UM = None               # None -> pitch derived from the targets
SOURCE_POL = oc.env_str("MSOPT_OLED_SOURCE_POL", "xy")   # "x", "y" or "xy"
                               # ("xy" = one 45-deg linear probe, both components
                               #  scored -- see _pol_components)
if SOURCE_POL not in ("x", "y", "xy"):
    raise ValueError(f"SOURCE_POL must be x, y or xy, got {SOURCE_POL!r}")
_SOURCE_POL_DEG = {"x": 0.0, "y": 90.0, "xy": 45.0}[SOURCE_POL]
# TWO FORWARDS. Under radial/fourfold the geometry forces the x and y responses
# to be identical, so a single probe measures both and the second run would be a
# duplicate. Drop the symmetry and that stops being true: an x-polarized probe
# alone leaves the design free to do anything it likes to y, and the FoM would
# never know. One problem per probe polarization, each scored on BOTH field
# components (see _pol_components), gradients averaged -- averaged, not
# minimaxed, because OLED emission is unpolarized, so what the device delivers
# IS the incoherent sum over the two polarizations.
FORWARD_POLS = [0.0, 90.0] if NOSYM_DESIGN else [_SOURCE_POL_DEG]
MODE_WEIGHTS = None            # None -> equal weight per (angle, order)
# Both are dead weights under MERGE_LEVEL_INTO_MATCH -- level and match are no
# longer two contributions to blend, and merged_J takes no prefactor -- but they
# are still computed unconditionally because WM below is derived from them and
# feeds the per-mode ramp weights the SUPPRESS term still scores with.
_WSUM = float(W_TRANS) + float(W_MODAL)
_ST, _SM = float(W_TRANS) / _WSUM, float(W_MODAL) / _WSUM
if MULTIOBJ == "minimax":
    # Commensurability, not weighting, is what minimax needs: each objective
    # must be on its OWN [0,1] scale (1 = satisfied) for the below-average
    # test to compare like with like. Any prefactor < 1 would make that
    # objective look permanently behind and capture the whole gradient.
    _ST = _SM = 1.0
# Squared because the score is sqrt(w*x): a term at x = 1 contributes sqrt(w),
# so squaring the requested shares makes the CONTRIBUTIONS come out in the
# requested ratio.
_q = _ST ** 2 + _SM ** 2
WM = _SM ** 2 / _q          # feeds build_target_modes' ramp weights
PER_MODE_ADJOINT = True      # -> msopt Incoherent=True
BROADBAND_ADJOINT = False    # True: wavelengths nested under each mode
PLOT_EVERY_ITER = True       # structure + per-order purity, one figure per iteration
# VERTICAL_GRATING is decided below, with USE_MFS_FILTER and GRATING_PROFILE,
# right next to the Mapping() call that's the only thing that reads them --
# see the "Design mapping" section.

# Start from a ring grating carrying the requested orders instead of uniform grey.
# See seeded_x0() for why: a weakly modulated start does not survive the first
# beta step under a tight parameterisation.
# Symmetry-breaking noise for a uniform start. A perfectly uniform slab is a SADDLE
# POINT for every target order above zero: c_m = 0 there, so d|c_m|^2/drho =
# 2*Re(c_m* dc_m/drho) = 0 and no gradient exists to grow them. sqrt(w*p) would
# regularize that (dJ/drho ~ sqrt(w)*d|c|/drho stays finite at c = 0) except the
# LOG_EPS floor caps dJ/dp and cancels it. Measured on the 4-fold uniform run: the
# target orders sat at 0.0000/0.0015 and SHRANK over three iterations while only T
# and the zeroth order moved. This much noise breaks the slab without biasing the
# design toward any particular order.
SEED_NOISE = 0.05
SEED_NOISE_RNG = 0             # fixed so a rerun reproduces the same start

# Postprocess polarizations. The dipole sweep runs once per entry, so "x" alone
# halves the postprocess -- 36 FDTD cases instead of 72, ~1.5 h saved.
#
# Justified by measurement, not assumption: on a D4 design the y-dipole ensemble
# is the x one rotated 90 deg, so the two ensembles are related by a symmetry the
# structure has. Measured across finished runs --
#     20260807_073422   x=0.5138084  y=0.5138084   0.0000%
#     20260807_090536   x=0.5043221  y=0.5037662   0.1102%
# -- against the ~1.9% LEE differences these runs are meant to resolve. Set it
# back to "x,y" for a design that is NOT symmetric under a 90 deg rotation.
PP_POLARIZATIONS = "x"
os.environ.setdefault("MSOPT_OLED_POSTPROCESS_POLARIZATIONS", PP_POLARIZATIONS)

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
        [(float(t), float(p)) for t, p in TARGET_ANGLES], _st_probe.lam, symmetry=KM_SYMMETRY)
    if _p is None:
        raise ValueError("\n".join(_r))
    CELL_PERIOD = float(_p)
# The design footprint defaults to the full unit cell in oled_common, which is
# exactly what a periodic out-coupler wants -- so it is left alone. Writing the
# rounded period into MSOPT_OLED_DESIGN_X_UM would make it a few 1e-6 um LARGER
# than the cell and trip the overlap guard.

G, _mc_spec = oc.select_stack(STACK, MC_COLOR, MC_STACK_KIND, period_mc=CELL_PERIOD)
LAM = float(np.mean(G.visible_wavelengths))


# =============================================================================
# Target momenta, straight from k_mapping
# =============================================================================
def _all_escaping_orders(period, lam, L):
    """EVERY (m, n) that escapes into air, grouped into emission cones by |u|.

    The axial-only target set is what broke the correspondence between the FoM
    and the actual emission. A square cell diffracts into the whole 2-D lattice,
    but AZIMUTHS_DEG = [0, 90] only ever named (m, 0) and (0, m). Measured on
    20260807_090536: the postprocess found eleven populated orders and the seven
    scored ones carried 0.330 of the emission -- 67% went into diagonals the FoM
    could not see, with the LARGEST single share (0.177) at 31.9 deg = (1, 2),
    an angle that was never a target. Rewarding only the axes leaves the
    diagonals free to grow, so the better the FoM scored the more it leaked.

    Under a radial design this could not happen: rho(r) has no azimuthal
    structure, so only (m, 0)-type rings exist and the scored set WAS the
    reachable set. That is why the two used to agree.

    Lattice points that share |u| leave at the same polar angle and land in the
    same measured cone, so they are grouped and scored together -- the same
    quantity the postprocess integrates over its Voronoi region per order.
    """
    step = lam / period
    mmax = int(np.floor(1.0 / step)) + 1
    groups = {}
    for m in range(-mmax, mmax + 1):
        for n in range(-mmax, mmax + 1):
            u = float(np.hypot(m, n) * step)
            if u >= 1.0:                       # outside the escape cone
                continue
            groups.setdefault(round(u, 9), []).append((m, n))

    # Boundary is the ACHIEVED top rung, not sin(TARGET_MAX_ANGLE). The pitch snap
    # moves the top order slightly off the requested angle -- here 45.22 deg, u =
    # 0.7098 against sin(45) = 0.7071 -- so testing against the requested value
    # classifies the run's own top target as a leak and drops it.
    s_max = float(L["rungs"][-1]["u"]) if L.get("rungs") else float(
        np.sin(np.deg2rad(TARGET_MAX_ANGLE)))
    theta_max_real = float(np.rad2deg(np.arcsin(min(s_max, 1.0))))
    targets, orders, weights, suppress, lattice = [], [], [], [], []
    above = []
    for u in sorted(groups):
        pts = groups[u]
        theta = float(np.rad2deg(np.arcsin(min(u, 1.0))))
        # Requested share straight off the ramp, and zero above the target angle:
        # those orders are leaks, so they get a suppression term rather than a
        # share to hit.
        if u <= s_max + 1e-9:
            frac = (theta / theta_max_real) if theta_max_real > 0 else 0.0
            if RAMP == "flat":
                w = 1.0
            else:
                w = float(W_AT_0 + (W_AT_MAX - W_AT_0) * frac)
            is_sup = False
        else:
            w = SUPPRESS_WEIGHT * float(0.5 * (W_AT_0 + W_AT_MAX))
            is_sup = True
            if not SUPPRESS_ABOVE_TARGET:
                above.append((theta, len(pts)))
                continue
        rep = min(pts, key=lambda t: (-t[0], -t[1]))     # a representative label
        targets.append((theta, float(np.rad2deg(np.arctan2(rep[1], rep[0])))))
        orders.append(rep)
        weights.append(w)
        suppress.append(is_sup)
        lattice.append(pts)
    n_t = sum(1 for s in suppress if not s)
    print(f"[k-map] full lattice: {len(groups)} escaping order group(s) over "
          f"{sum(len(v) for v in groups.values())} lattice points -- "
          f"{n_t} target up to {theta_max_real:.2f} deg, "
          f"{len(suppress) - n_t} suppressed")
    if above:
        print("[k-map] NOT scored (above the target, and SUPPRESS_ABOVE_TARGET is "
              "off): " + ", ".join(f"{t:.1f} deg" for t, _ in above)
              + " -- they still cost the level term, which only counts captured "
                "target-order power")
    return targets, orders, weights, suppress, lattice


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
        targets, orders, weights, suppress, lattice = [], [], [], [], []
        phis = [0.0] if KM_SYMMETRY == "radial" else [float(a) for a in AZIMUTHS_DEG]

        def _emit(rung, w, is_suppress):
            """ONE cone per rung, carrying the rung's COMPLETE symmetry orbit.

            This used to emit one mode per entry in AZIMUTHS_DEG, which for
            [0, 90] scored (m,0) and (0,m) but never (-m,0) or (0,-m). A 4-fold
            design puts identical power in all four, so half of every off-axis
            rung sat in the purity DENOMINATOR without ever appearing in a target
            term -- a blind spot that made the axial configuration look worse for
            a reason that has nothing to do with which angles are targeted.

            Grouping the orbit is also what makes the diagonal contrast mean
            something: with the old form, "diagonals off" changed the target set
            AND left that blind spot, so the two could not be separated. Now the
            only difference is whether the diagonal cones are added.
            """
            if rung["m"] == 0:                    # k-space origin: no azimuth
                targets.append((0.0, 0.0)); orders.append((0, 0))
                weights.append(w); suppress.append(is_suppress)
                lattice.append([(0, 0)])
                return
            m = int(rung["m"])
            if KM_SYMMETRY == "radial":
                # rho(r,z) has no square-lattice orbit to complete; keep the
                # historical single representative per azimuth.
                for ph in phis:
                    c, sn = np.cos(np.deg2rad(ph)), np.sin(np.deg2rad(ph))
                    mm, nn = int(round(m * c)), int(round(m * sn))
                    if (mm, nn) == (0, 0):
                        continue
                    targets.append((rung["theta_air_deg"], ph))
                    orders.append((mm, nn))
                    weights.append(w)
                    suppress.append(is_suppress)
                    lattice.append([(mm, nn)])
                return
            # FULL rung weight on the cone, and the cone holds every lattice point
            # at this |u|. build_target_modes multiplies by len(points) so each
            # DIRECTION in it asks for ramp(theta) -- a radiance profile, which is
            # what the postprocess reports (ratio_to_0 in channel_ratios.txt).
            pts = [(m, 0), (-m, 0), (0, m), (0, -m)]
            targets.append((rung["theta_air_deg"], 0.0))
            orders.append((m, 0))
            weights.append(w)
            suppress.append(is_suppress)
            lattice.append(pts)

        if TARGET_ALL_ORDERS and KM_SYMMETRY != "radial":
            targets, orders, weights, suppress, lattice = _all_escaping_orders(
                period, lam, L)
        else:
            # lattice stays a list: _emit now fills it with each rung's complete
            # symmetry orbit (it used to be None because this path scored single
            # lattice points).
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
        MODE_LATTICE = lattice
    elif PERIOD_UM:
        period = float(PERIOD_UM)
        MODE_W = MODE_SUP = MODE_LATTICE = None
        step = lam / period
        orders = [(int(round(np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(p)) / step)),
                   int(round(np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(p)) / step)))
                  for t, p in targets]
        rep = [f"pitch fixed by hand: P = {period:g} um"]
        targets = [(float(t), float(p)) for t, p in TARGET_ANGLES]
    else:
        MODE_W = MODE_SUP = MODE_LATTICE = None
        targets = [(float(t), float(p)) for t, p in TARGET_ANGLES]
        period, orders, rep = km.minimum_period_2d(targets, lam, symmetry=KM_SYMMETRY)
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
    lat = MODE_LATTICE if MODE_LATTICE is not None else [None] * len(targets)
    modes, dropped = [], []
    for (t, ph), (m, n), wi, si, pts in zip(targets, orders, src_w, MODE_SUP, lat):
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
            # every lattice point that leaves at this polar angle; the projector
            # scores their SUM, which is the cone the postprocess measures
            "points": [(int(a), int(b)) for a, b in (pts if pts else [(m, n)])],
        })
    if not modes:
        raise ValueError(f"every target exceeds n_organic={n_org:.3f}")
    if not any(not mo["suppress"] for mo in modes):
        raise ValueError("every surviving mode is a suppression term")
    w = np.asarray([mo["weight"] for mo in modes], dtype=float)
    # PER-DIRECTION, not per cone. RAMP asks for a RADIANCE profile -- how bright
    # the panel looks from each angle -- and that is what the postprocess reports
    # (radiance_per_emitted_power / ratio_to_0 in channel_ratios.txt). But a cone
    # is several lattice points and modal_purity_terms scores their SUM, so a raw
    # ramp weight on the cone asks for the RING TOTAL instead. The two differ by
    # the point count, which is not a constant: 1 at normal incidence, 4 for
    # (m,0)/(0,m) and (m,m), and 8 for (2,1)-type cones. Measured on the 45 deg /
    # 3-order ladder, a requested 1.00 -> 0.90 came out as 1.00 -> 0.23 per
    # direction, and 31.94 deg (8 points) landed at 1/8.6 of normal -- which is
    # why in-opt match 0.847 sat against a postprocess angular_shape_score of
    # 0.394 on run 20260811_055547, and why its 45/0 radiance ratio missed the
    # window at 0.712. Multiplying by the multiplicity makes the cone total
    # ramp(theta) * n_points, so each DIRECTION in it carries ramp(theta).
    npts = np.asarray([len(mo["points"]) for mo in modes], dtype=float)
    w = w * npts
    # SOLID ANGLE, the second half of the same correction. The multiplicity above
    # fixes "a cone holds several directions"; this fixes "a direction is not a
    # solid angle". Every diffraction order owns the same area in DIRECTION-COSINE
    # space -- one lattice cell, (lambda/P)^2, independent of the order -- but
    # dOmega = du_x du_y / cos(theta), so that fixed cell subtends MORE solid angle
    # the further off axis it sits. Radiance is power per solid angle, hence
    #     radiance(theta) = power(theta) * cos(theta)
    # which is exactly what the postprocess computes (see radiance_from_spectrum:
    # "multiply by cos(theta) before averaging to recover a per-solid-angle
    # quantity"). Asking for power = ramp(theta) therefore delivers radiance =
    # ramp(theta)*cos(theta), so the request has to be divided by cos(theta).
    #
    # Measured, not argued: run s4 (20260812_144932) drove match to 0.9779 -- it
    # satisfied the old target almost exactly -- and its postprocess reported a
    # 45/0 radiance ratio of 0.631, against 0.90*cos(45.22) = 0.6339 predicted by
    # this very omission. 0.5% apart. Without this line the requested window
    # [0.80, 0.90] is unreachable no matter how good the design is: 0.634 is the
    # ceiling.
    w = w / np.cos(np.deg2rad(np.asarray([mo["theta_air_deg"] for mo in modes],
                                         dtype=float)))
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
N_fom = len(FORWARD_POLS)       # one forward probe per polarization; a symmetric
                                # design needs exactly one (see FORWARD_POLS)


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


def _check_fom_plane_in_eml(z):
    """The FoM plane must sit inside the EML, and nothing else checks it.

    Every claim this script makes rests on that plane being where the emitters
    are: reciprocity relates the field THERE to what a dipole THERE radiates. A
    plane one layer off still produces a perfectly well-behaved FoM that rises
    for 5 hours and means something else entirely, with no error anywhere.

    That is not hypothetical. Run 20260807_073422 optimized at z = -0.4990 um,
    which is in NPB_HTL -- the EML spans [-0.4865, -0.4615] -- and it ran to
    completion, converged, and returned the best LEE of the three runs that day
    (0.5138). Nothing in the logs flagged it. Fail loudly instead.
    """
    layers = [(L["name"],
               L["center"][2] - 0.5 * L["size"][2],
               L["center"][2] + 0.5 * L["size"][2]) for L in G.stack_layers]
    eml = [(n, lo, hi) for n, lo, hi in layers if n == "EML"]
    if not eml:
        print("[fom] WARNING: no layer named EML in the stack; FoM plane unchecked")
        return
    _, lo, hi = eml[0]
    if lo <= z <= hi:
        frac = (z - lo) / max(hi - lo, 1e-12)
        print(f"[fom] FoM plane z={z:+.4f} um is inside the EML [{lo:+.4f}, {hi:+.4f}] "
              f"({100 * frac:.0f}% up from its bottom face)")
        return
    where = next((n for n, a, b in layers if a <= z <= b), "outside every layer")
    raise RuntimeError(
        f"FoM plane z={z:+.4f} um is NOT in the EML [{lo:+.4f}, {hi:+.4f}] -- it is "
        f"in {where!r}. The whole reciprocity argument is about the field at the "
        f"EMITTERS, so optimizing any other plane silently answers a different "
        f"question (see run 20260807_073422).")


_check_fom_plane_in_eml(eml_c[2])


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

    With the full lattice, one mode is an emission CONE, not a single lattice
    point: every (m, n) sharing |u| leaves at the same polar angle. Each of those
    points gets its own projector and modal_purity_terms sums their |c|^2, which
    is the same quantity the postprocess integrates over that order's Voronoi
    region. Both signs are already in the point list, so no separate +/- pairing
    is needed there and the (0, 0) double-count cannot recur.
    """
    x, y = eml_grid()
    X, Y = np.meshgrid(x, y, indexing="ij")
    N = float(X.size)
    k1 = 2.0 * np.pi / PERIOD
    out = []
    for mo in target_modes:
        pts = mo.get("points") or [(mo["m"], mo["n"])]
        if len(pts) > 1 or (mo["m"], mo["n"]) != (0, 0):
            projs = [np.exp(-1j * (k1 * a * X + k1 * b * Y)) / np.sqrt(N)
                     for a, b in pts]
            if len(pts) == 1:                    # single point: keep its mirror
                projs.append(np.conj(projs[0]))
            out.append(tuple(projs))
        else:
            out.append((np.ones_like(X, dtype=complex) / np.sqrt(N),))
    return out


PROJECTORS = mode_projectors()


# =============================================================================
# FoM
# =============================================================================
def _on_projector_grid(E):
    """Monitor field -> the grid the projectors live on (duplicate edge dropped)."""
    return npa.reshape(E, MONITOR_SHAPE)[:-1, :-1]


def _pol_components(Ex, Ey):
    """Field components the modal score reads, on the projector grid.

    SOURCE_POL "x"/"y" score the ONE component the probe launched. That quietly
    builds an axis bias into the FoM the moment the design is a 2-D lattice:
    an x-polarized probe drives the (m,0) and (0,m) order families through
    different field components, so scoring Ex alone weighs one family far more
    than the other even though a 4-fold structure emits into both equally
    (measured 2.8x between the two families on run 20260811_002009).

    "xy" is the 45-degree linear probe -- one source, one forward run, still --
    that excites Ex and Ey together and is scored on BOTH. Under a C4v design
    that restores the x/y symmetry of the FoM itself instead of relying on the
    geometry constraint to paper over an asymmetric score.
    """
    if SOURCE_POL == "xy" or NOSYM_DESIGN:
        # NOSYM: each probe is singly polarized, but the design is free to rotate
        # power into the orthogonal component, and power that arrived is power
        # that arrived. Scoring one component would leave that conversion
        # invisible to the FoM -- and unconstrained.
        return [_on_projector_grid(Ex), _on_projector_grid(Ey)]
    return [_on_projector_grid(Ex if SOURCE_POL == "x" else Ey)]


_LAST_PURITIES = [None]        # see modal_purity_terms's cache note below


def modal_purity_terms(Ex, Ey):
    """Per-mode share of the EML field that sits at the requested momentum.

    Normalizing by the total field energy on the plane makes the score a PURITY
    in [0, 1] rather than an amplitude: without it the optimizer can win by
    simply pushing more light into the EML plane while the angular content stays
    wrong, which is the failure the OLED_opt header documents for the older
    unbounded scores.
    """
    Es = _pol_components(Ex, Ey)
    denom = sum(npa.sum(npa.abs(E) ** 2) for E in Es) + 1e-30
    terms = []
    for projs in PROJECTORS:
        # Every (lattice point, field component) pair in this cone. With a single
        # -component probe that is the historical sum; with SOURCE_POL="xy" it
        # adds the Ey half, so a cone's score is its TOTAL power, not one
        # component's -- which is the whole point of the diagonal probe.
        terms.append(sum(npa.abs(npa.sum(E * p)) ** 2
                         for p in projs for E in Es) / denom)
    # Detached snapshot for reporting. match_J scores every target mode jointly
    # in ONE objective (see its own docstring), so there is no per-mode f0 for
    # _split_per_J to recover an individual purity from -- this cache is the
    # only place a per-mode number still exists. Every objective this
    # iteration (transmission_J, match_J, ...) is evaluated at the SAME
    # forward field before its own adjoint runs, so caching once here and
    # reading it back on every call is both correct and free.
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
                   pol=_SOURCE_POL_DEG,
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
    T, there is no risk of the older "buy shape score by raising T" failure --
    shape and level stay cleanly separated by construction. That separation is
    also why the removed "absolute" metric had nothing to offer: it folded T back
    into the modal terms this function already carries it for.

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
    """One separable term J_i = sqrt(w_i * p_i).

    Concave, so the requested ratio is the optimum, and NON-NEGATIVE, which is
    not a stylistic choice -- msopt's optimizer assumes a positive FoM, and the
    log form that used to be selectable here is not:

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
    return npa.sqrt((npa.maximum(x, 0.0) + LOG_EPS) * weight)


def _scalar_fom(per_J):
    """The single number msopt line-searches on. Under minimax the quantity
    actually being maximized is min_j J_j, so reporting the SUM would let
    Armijo accept a step that raised the total while pushing the worst
    objective down -- exactly the trade minimax exists to forbid."""
    if not per_J:
        return None
    return float(np.min(per_J) if MULTIOBJ == "minimax" else np.sum(per_J))


LEVEL_SCORE_REF = oc.env_float("MSOPT_OLED_LEVEL_REF", 1.0)
LEVEL_KNEE = oc.env_float("MSOPT_OLED_LEVEL_KNEE", 1.0)          # minimax only:
LEVEL_KNEE_SCORE = oc.env_float("MSOPT_OLED_LEVEL_KNEE_SCORE", 0.9)
                               # T*S value that must not be undercut, and the
                               # score it earns. See level_score.
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
    if MULTIOBJ == "minimax":
        # Constraint form. The spec is "LEE must not fall BELOW the bare
        # stack", not "maximize LEE", so the knee is placed at the planar
        # level (T*S = 1 by construction: the planar device puts essentially
        # all of its EML power at u = 0, which is a scored order, so its own
        # T*S is ~1) and scored LEVEL_KNEE_SCORE there. Reaching the planar
        # level therefore already scores 0.9 -- above almost any achievable
        # match, so minimax stops spending gradient on it -- while falling
        # below it drops fast and makes the level term the worst objective,
        # which is precisely when it should take the wheel.
        ref = LEVEL_KNEE * (1.0 / min(max(LEVEL_KNEE_SCORE, 1e-3), 0.999) - 1.0)
        return TS / (TS + max(ref, 1e-9))
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
#   none      no symmetry at all: every design voxel is independent. The design
#             can then respond differently to x and to y, so ONE probe no longer
#             characterizes it -- see FORWARD_POLS below, which is why this mode
#             costs two forwards per iteration instead of one.

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
USE_MFS_FILTER = oc.env_flag("MSOPT_OLED_USE_MFS_FILTER", "0")   # env override
                               # so a filter-vs-freeform A/B pair can launch
                               # from the same file without editing between runs

# ---- 3. grating profile ------------------------------------------------------
# Only meaningful within the FILTER path (Is_slanted_grating is read at
# Opt_MS2.py's `elif self.N_height > 1: if self.slanted: Slant_sidewall ...
# else: Vertical_sidewall`, which only that path reaches -- see choice 2).
# Freeform's own single-vs-full-3D choice is a different axis (below) and has
# no slanted option at all.
# "vertical"  one 2-D layer extruded straight down  (single etch depth)
# "slanted"   the same layer with a slanted sidewall
# "freeform"  FULL 3-D: every z layer is its own design. With SYMMETRY="fourfold"
#             msopt folds EACH LAYER independently (fliplr+flipud+transpose, the
#             per-z loop in Opt_MS2), so the stack is 4-fold symmetric in plane
#             while still being multilayer in z -- verified below.
GRATING_PROFILE = oc.env_str("MSOPT_OLED_GRATING_PROFILE", "freeform")
                               # env-overridable so the grating-vs-freeform pair
                               # launches without editing the file. Note the
                               # interaction with USE_MFS_FILTER: "vertical" with
                               # the filter OFF is freeform restricted to a single
                               # extruded layer (Is_freeform[2]), and with it ON is
                               # the filtered vertical-sidewall grating.
if GRATING_PROFILE not in ("vertical", "slanted", "freeform"):
    raise ValueError(f"GRATING_PROFILE must be 'vertical' or 'slanted' or 'freeform', got {GRATING_PROFILE!r}")
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
    """Uniform density plus BAND-LIMITED noise.

    Per-pixel white noise does not survive the MFS filter -- at MFS = 0.1 um and
    dx = 0.02 um the conic kernel averages ~78 cells, so sigma 0.02 arrives as
    0.002 and the slab is still a slab (measured: p_0 = 0.9917 with noise against
    0.9918 without). Keeping only |k| <= (M+1)/P puts the randomness on the scales
    the out-coupler actually uses, with random phases so no order is favoured.

    A harmonic seed (cos(2*pi*m*r/P) per requested order) used to live here behind
    SEED_HARMONICS. It was hardwired off and is gone; it PICKED the orders, which
    is the optimizer's job, and the backup is legacy/OLED_rec_preclean_20260811.py.
    """
    base = float(G.grating_initial_density)
    n = int(design_parameters)
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

x0 = seeded_x0()
dJ_0 = np.zeros(G.design_cells)
print(f"[design] symmetry={SYMMETRY} "
      f"({'Is_radial_3d rho(r,z)' if RADIAL_DESIGN else 'Sym_geo_C8 = D4/C4v'}) "
      f"-> {design_parameters:,} parameters")


# =============================================================================
# Modal scoring: ONE joint distribution-match objective, plus one grouped
# suppression objective (all leaking orders together, when enabled) -- see
# grouped_suppress_J. Neither is per-mode anymore.
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

    The math itself lives in _match_rate, which returns the bare rate; this
    function only applies the _SM weight the SPLIT form needs. merged_J calls
    _match_rate directly, because a product with the level score is bounded by
    construction and wants no prefactor at all.
    """
    core = _match_rate(mode_indices)

    def J(Ex, Ey):
        # Weighted by _SM so FoM = _ST*level_score + _SM*match is a proper
        # convex combination (_ST + _SM = 1) of two [0,1]-bounded terms,
        # keeping the WHOLE FoM in [0,1] instead of just this term -- see
        # transmission_J / level_score for the matching half of this.
        return _SM * core(Ex, Ey)
    return J


def _match_rate(mode_indices):
    """The distribution-match rate itself, unweighted, in (0, ~1].

    Everything about WHY it is shaped this way is in match_J's docstring; this
    is only the closure both match_J (times _SM) and merged_J (times the level
    score) evaluate.
    """
    idx = list(mode_indices)
    w = np.asarray([target_modes[i]["weight"] for i in idx], dtype=float)
    w_hat = w / max(float(np.sum(w)), 1e-30)
    ceiling_ln = float(np.sum(w_hat * np.log(w_hat + LOG_EPS)))

    def rate(Ex, Ey):
        all_p = modal_purity_terms(Ex, Ey)
        p_list = [all_p[i] for i in idx]
        S = sum(p_list) + LOG_EPS
        ln_raw = sum(wh * npa.log(npa.maximum(pi / S, 0.0) + LOG_EPS)
                     for wh, pi in zip(w_hat, p_list))
        return npa.exp(ln_raw - ceiling_ln)
    return rate


def grouped_suppress_J(mode_indices, weight):
    """ALL above-target orders in ONE objective, hence one adjoint run.

    The suppressed orders carry no ratio to hold -- unlike the targets (scored
    jointly by match_J to pin their RAMP, not individually), the request here
    is just "no power above the target", so their purities can be added first
    and scored once. dJ/dE of that sum is a single field on the FoM plane, so
    msopt builds a single adjoint source from it and the suppression cost
    stops scaling with the leak count -- it is 1 adjoint whether one order
    escapes or five.
    """
    def J(Ex, Ey):
        terms = modal_purity_terms(Ex, Ey)
        tot = sum(terms[i] for i in mode_indices)
        if MULTIOBJ == "minimax":
            # LEAK BUDGET, not "1 - leak". Measured on run A (minimax, diagonals
            # + suppression ON): minimax reported driving J[2] on EVERY
            # single iteration -- the suppression objective never steered once --
            # and the postprocess then showed the entire LEE gain landing ABOVE
            # the target (40-80 deg went from 26.0% of extracted light on the bare
            # stack to 34.2%). The cause is scale, not the mechanism: sqrt(1-leak)
            # reads 0.81 even at 34% leak, so against a match of 0.55 it sits
            # comfortably above the mean and minimax correctly -- but uselessly --
            # judges it satisfied. minimax's below-average rule only works when the
            # objectives are comparably HARD, and "1 - leak" is far too easy.
            # tol/(tol+leak) fixes exactly that: 1 at no leak, 0.5 when the whole
            # budget is spent, 0.13 at the 34% leak actually observed -- well under
            # match, so it takes the wheel. Same saturating-ratio form as
            # level_score, and positive everywhere, so the gradient never dies at
            # large violation the way a clipped hinge would.
            return LEAK_TOL / (LEAK_TOL + npa.maximum(tot, 0.0))
        return _sep_score(1.0 - tot, weight)
    return J


def transmission_J(Ex, Ey):
    """SPLIT form only (MERGE_LEVEL_INTO_MATCH off): the bounded level score,
    weighted by _ST so FoM = _ST*level_score + _SM*match is a proper convex
    combination -- see level_score and match_J's matching _SM weighting."""
    return _ST * level_score(transmission_term(Ex, Ey))


_LAST_LEVEL_MATCH = [None]     # see merged_J's reporting-cache note


def merged_J(mode_indices):
    """The whole steering FoM as ONE objective: level_score(T*S) * match(p).

    The two factors answer different questions and neither can substitute for
    the other -- how MUCH of the probe reached the target momenta at all
    (level, the only term that sees the captured share S) and how that capture
    is SHARED OUT against the requested ramp (match, deliberately blind to S).
    Scored separately they do not cooperate; multiplied they do. The full
    argument, and the measurement that forced the change, is at
    MERGE_LEVEL_INTO_MATCH.

    No prefactor. Both factors are already [0, 1], so the product is too --
    _ST/_SM exist only to make the SPLIT form's sum bounded and have nothing
    to weight here. Under minimax they are 1.0 anyway, so this is literally
    the J1*J2 the split form was already computing, just multiplied instead of
    added -- and therefore directly comparable against those runs' numbers.

    Non-negative by construction (a ratio of non-negatives times an exp), so
    it satisfies the positive-FoM requirement _sep_score's docstring spells
    out. It is also strictly gated: either factor going to 0 takes the
    objective with it, which is the property the sum could not express.

    One objective means ONE adjoint. autograd traces the product exactly --
    d(L*M) = M*dL + L*dM comes out of the same single backward pass, so this
    is not just cheaper than two adjoints, it is the exact gradient of the
    quantity actually being maximized.

    S IS NOT DOUBLE-COUNTED. transmission_term multiplies T by the same sum of
    non-suppressed purities that _match_rate normalizes by, but match divides
    it back out completely (p_hat = p/S), so S enters the product exactly
    once, through the level factor. This is the distinction that made the
    removed "absolute" metric pointless and makes this merge work.

    The level/match split is still what gets REPORTED, so both are cached
    detached here on the way through. Same scheme, and same caveat, as
    modal_purity_terms' _LAST_PURITIES: with two forwards the cache holds the
    last one, while per_J is averaged over both, so the printed level/match
    are that forward's rather than the mean. The product in per_J is exact
    either way; only the breakdown is approximate.
    """
    core = _match_rate(mode_indices)

    def J(Ex, Ey):
        lvl = level_score(transmission_term(Ex, Ey))
        mat = core(Ex, Ey)
        _LAST_LEVEL_MATCH[0] = (float(getval(lvl)), float(getval(mat)))
        return lvl * mat
    return J


# Objective 1 is the steering term over every non-suppressed target mode --
# either the merged level*match product (default) or, with the merge off, the
# level (now T*S, see transmission_term) and the ONE joint distribution-match
# term as two separate objectives. Suppressed modes (if any) share one more.
# See MERGE_LEVEL_INTO_MATCH for why the merged form is the default.
_target_idx = [i for i, mo in enumerate(target_modes) if not mo["suppress"]]
_sup_idx = [i for i, mo in enumerate(target_modes) if mo["suppress"]]
# Merging needs something to merge WITH: with no unsuppressed target there is no
# match term at all, so the level term stands alone exactly as in the split form.
_MERGED = bool(MERGE_LEVEL_INTO_MATCH and _target_idx)
if _MERGED:
    OBJ_SPECS = [{"kind": "merged", "idx": _target_idx}]
else:
    OBJ_SPECS = [{"kind": "trans", "idx": []}]
    if _target_idx:
        OBJ_SPECS.append({"kind": "match", "idx": _target_idx})
if _sup_idx:
    if SUPPRESS_AGGREGATE:
        OBJ_SPECS.append({"kind": "suppress", "idx": _sup_idx})
    else:
        OBJ_SPECS += [{"kind": "suppress", "idx": [i]} for i in _sup_idx]


def objective_for(spec):
    if spec["kind"] == "merged":
        return merged_J(spec["idx"])
    if spec["kind"] == "trans":
        return transmission_J
    if spec["kind"] == "match":
        return match_J(spec["idx"])
    w = float(np.sum([target_modes[i]["weight"] for i in spec["idx"]]))
    if MULTIOBJ == "minimax":
        w = 1.0                # sqrt(1 - leaked) in [0,1]; see _ST/_SM above
    return grouped_suppress_J(spec["idx"], w)


def spec_label(s):
    if s["kind"] == "merged":
        return "level_x_match"
    if s["kind"] == "trans":
        return "eml_intensity"
    if s["kind"] == "match":
        return "distribution_match"
    if len(s["idx"]) == 1:
        return target_modes[s["idx"][0]]["name"]
    return f"SUPPRESS_x{len(s['idx'])}"


OBJECTIVES = [objective_for(s) for s in OBJ_SPECS]
OBJ_LABELS = [spec_label(s) for s in OBJ_SPECS]
print(f"[fom] {len(OBJECTIVES)} objective(s) = "
      + (f"1 merged level*match (unweighted, over {len(_target_idx)} target order(s))"
         if _MERGED else
         f"1 captured level (w={_ST:.3f})"
         f" + 1 distribution-match (w={_SM:.3f}, over {len(_target_idx)} target order(s))")
      + (f" + {sum(1 for s in OBJ_SPECS if s['kind']=='suppress')} suppress"
         f" (covering {len(_sup_idx)} order(s))" if _sup_idx else "")
      + f"  -> {len(OBJECTIVES)} adjoint run(s) per iteration")


# =============================================================================
def build_problem():
    """One (sim, problem) pair per entry in FORWARD_POLS.

    Everything except the source polarization is identical between them: same
    stack, same design grid, same FoM plane, same objectives. They are separate
    SIMULATIONS because a Lumerical source has one polarization, and separate
    PROBLEMS because msopt drives one forward per problem -- which is exactly
    the two-forward scheme the no-symmetry design needs.
    """
    sims, opts = [], []
    for k, pol_deg in enumerate(FORWARD_POLS):
        sims.append(_build_one(k, pol_deg))
        opts.append(_wrap_one(sims[-1], k))
    fom_history = []
    if len(opts) > 1:
        print(f"[fom] {len(opts)} forward run(s) per iteration, "
              f"pol = {FORWARD_POLS} deg (SYMMETRY={SYMMETRY})")
    return sims, opts, fom_history


def _build_one(idx, pol_deg):
    sim = oc.make_sim(G, [G.Sx, G.Sy, G.Sz])
    # Normal incidence, from air down into the stack: theta = phi = 0.
    sim.add_source(mode="plane", name="source", center=src_c, size=src_s,
                   direction="backward", src_wl=G.visible_wavelengths,
                   bandwidth=bandwidth,
                   pol=pol_deg,
                   theta=0.0, phi=0.0, broadband=False)
    oc.add_stack(G, sim)
    sim.add_design_grid("design", G.design_c, G.design_s, G.design_high_index,
                        G.design_low_index, G.design_grids,
                        G.grating_initial_density * np.ones(G.design_grids), LAM)
    sim.add_design_monitor()
    return sim


def _wrap_one(sim, idx):
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
        # The adjoint source sits ON the FoM plane (the EML mid-plane) and has to
        # reach the DESIGN, which is 0.30 um ABOVE it -- design z = -0.1785 against
        # FoM z = -0.4740. Lumerical injects "forward" as +axis, so +z is what
        # points at the design. False sent it downward, away from the design; the
        # Ag mirror underneath bounced enough of it back up that a gradient still
        # existed and the FoM still climbed, which is exactly why this went
        # unnoticed -- the adjoint field in the design region was a mirror-delayed
        # copy with the wrong phase and amplitude, so every step was a poor
        # approximation of the true gradient. That is the likeliest reason a
        # 204k-parameter freeform run crawled and stalled where it should have had
        # the most room to move.
        #
        # ROOT CAUSE, so the same class of defect is easier to find elsewhere:
        # this script was ported from the dipole-EMISSION formulation, where the
        # source sits AT the EML and the FoM plane is above it -- there the
        # adjoint really does travel downward. The reciprocal form swaps those
        # two: the probe moved to the top and the FoM plane became the EML. Every
        # direction-bearing parameter had to flip with them, and this one did not.
        # Anything else inherited from that ancestor which encodes an orientation
        # (monitor normal, injection axis, source direction, which side of the
        # design the FoM sits on) is worth re-deriving rather than trusted. Not reached when broadband_adjoint is on, which
        # takes a different source-creation path (Lumerical_utill ~L1874 vs L1932).
        adj_fwd=True,
        opt_idx=idx,
        broadband_adjoint=BROADBAND_ADJOINT,
        Incoherent=PER_MODE_ADJOINT,
        Multiobj=MULTIOBJ,
    )
    return opt


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
    cache modal_purity_terms already left behind. Under the merged FoM the
    same is true of level and match themselves -- one objective carries their
    PRODUCT, which cannot be factored back apart, so merged_J caches the pair
    on its way through and this reads it back.

    In the split form level and match are each read back UNWEIGHTED (divide
    out _ST / _SM) so what gets reported and plotted is the same [0,1]
    diagnostic quantity regardless of the FoM split -- transmission_J /
    match_J each hand msopt the WEIGHTED value, since that is what has to sum
    to a bounded total FoM. The merged term is unweighted to begin with.
    """
    level, match = float("nan"), float("nan")
    for i, s in enumerate(OBJ_SPECS):
        if i >= len(per_J):
            break
        if s["kind"] == "merged":
            lm = _LAST_LEVEL_MATCH[0]
            if lm is not None:
                level, match = lm
        elif s["kind"] == "trans":
            level = float(per_J[i]) / _ST if _ST > 0 else float("nan")
        elif s["kind"] == "match":
            match = float(per_J[i]) / _SM if _SM > 0 else float("nan")
    cached = _LAST_PURITIES[0]
    purities = ({i: cached[i] for i, mo in enumerate(target_modes) if not mo["suppress"]}
                if cached is not None else {})
    return purities, level, match


def save_angular_target(path):
    """Dump the angular target this run optimized against, for the postprocess.

    oled_common.load_optimization_angular_target looks for this file next to
    lastdesign.txt, and when it finds one the postprocess re-evaluates the FoM's
    OWN match on the measured far field -- the only number that puts the
    optimizer's score and the real emission in the same units. Nothing was
    writing it, despite the loader's docstring claiming "every optimization run
    writes" it, so that comparison had never run once: every manifest so far
    carried an empty optimization_target_match.

    The convention is fixed by the consumer, optimization_target_match:
      angle_thetas    EVERY ring, suppressed ones included -- it assigns each
                      propagating direction to its nearest ring, so a missing
                      suppressed ring would fold leaked power into a target's
      target_profile  normalized over the IN-RANGE rings alone, because match
                      compares it against q = profile / throughput
      in_range        the rings the FoM actually scores

    Failure here must not cost a finished optimization its postprocess, so the
    write is reported and swallowed rather than raised.
    """
    try:
        merged = {}
        for mo in target_modes:
            t = round(float(mo["theta_air_deg"]), 4)
            scored = not bool(mo["suppress"])
            w_prev, keep_prev = merged.get(t, (0.0, False))
            # A repeated theta (the k-map can list one twice) merges instead of
            # letting argmin silently pick whichever came first.
            merged[t] = (w_prev + (float(mo["weight"]) if scored else 0.0),
                         keep_prev or scored)
        thetas = np.array(sorted(merged), dtype=float)
        weights = np.array([merged[t][0] for t in thetas], dtype=float)
        in_range = np.array([merged[t][1] for t in thetas], dtype=bool)
        total = float(weights[in_range].sum())
        if total <= 0.0:
            raise ValueError("no positive in-range weight")
        profile = np.zeros_like(weights)
        profile[in_range] = weights[in_range] / total
        np.savez(path, angle_thetas=thetas, target_profile=profile, in_range=in_range)
        print("[target] wrote " + path + ": " + ", ".join(
            f"{t:.2f}deg->{p:.4f}" + ("" if r else " (SUP)")
            for t, p, r in zip(thetas, profile, in_range)))
    except Exception as exc:
        print(f"[target] could not write the angular target ({exc}); the "
              f"postprocess will skip the FoM-vs-measured comparison")


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


def make_rec_loop(opts, hist, mode_records):
    """Adjoint loop for OPT_Ms, over one problem per probe polarization.

    Per-OBJECTIVE adjoint machinery lives in msopt (Incoherent=True); this
    layer only walks the FORWARDS and records the breakdown msopt exposes. With
    a symmetric design there is exactly one forward and the loops below collapse
    to what they always did.
    """
    opts = list(opts) if isinstance(opts, (list, tuple)) else [opts]
    opt = opts[0]                 # the one whose beta/state the plots report

    def _mean_per_J(rows):
        """Average the objective breakdown across forwards. Every problem scores
        the SAME objectives on its own forward, so entry j means the same thing
        in each row and the mean is the unpolarized device's value for it."""
        rows = [r for r in rows if r]
        if not rows:
            return []
        n = min(len(r) for r in rows)
        return [float(np.mean([r[j] for r in rows])) for j in range(n)]

    def loop(X, N_cases, Case=True):
        if Case == 3:
            # X[0] is one gradient per forward. Unpolarized emission is the
            # incoherent sum over probe polarizations, so the design gradient is
            # their mean -- with a single forward this is the identity.
            gs = [npa.where(npa.isfinite(npa.array(g)), npa.array(g), 0.0)
                  for g in X[0]]
            return gs[0] if len(gs) == 1 else sum(gs) / float(len(gs))
        rho = None if isinstance(X, str) else [npa.clip(X, 0.0, 1.0)]
        if not Case:
            # FoM-only evaluation: msopt's line search / backtracking calls this,
            # and it is a real design being scored, so it gets a plot too. Skipping
            # it was why the figure only refreshed on some iterations.
            f0s, rows = [], []
            # Re-evaluate the objectives on THIS forward's fields. msopt only
            # writes f0_per_J inside adjoint_dipole_run_incoherent -- the single
            # assignment in the whole file -- so a need_gradient=False call leaves
            # it holding the previous ADJOINT iteration's numbers. Reading it here
            # returned a stale FoM for every line-search probe: three consecutive
            # eval-only lines reported FoM=8.587858e-01 for three different designs
            # while the purities (cached fresh by modal_purity_terms) plainly
            # differed. That value is what this loop hands back to msopt, so the
            # Armijo test compared a trial point against itself, backtracking could
            # never reject a bad step, and the run cycled 0.891 -> 0.789 -> 0.674 ->
            # collapse three times over. The forward fields are already in memory,
            # so this costs arithmetic, not another FDTD run.
            for o in opts:
                f0k, _ = o(rho_vector=rho, need_gradient=False)
                f0s.append(f0k)
                try:
                    args = [o.FoM_fields[k] for k in o.objective_arguments]
                    rows.append([float(np.real(np.sum(J(*args))))
                                 for J in o.objective_functions])
                except Exception as exc:
                    print(f"[rec] per-objective re-evaluation unavailable ({exc}); "
                          f"falling back to msopt's f0")
                    rows.append(list(getattr(o, "f0_per_J", []) or []))
            per_J = _mean_per_J(rows)
            val = _scalar_fom(per_J)
            if val is None:
                val = float(np.mean([np.real(np.sum(f)) for f in f0s]))
            purities, level, match = _split_per_J(per_J)
            print(f"[rec] FoM={val:.6e}  level={level:.4f}  match={match:.4f}  "
                  f"{_MLAB}={[f'{purities[k]:.4f}' for k in sorted(purities)]}"
                  f"  (eval only)")
            _plot_state(X, purities, level, match, val, opt, hist, trial=True)
            return val, [val]
        f0s, gs, rows, gnorms = [], [], [], []
        for o in opts:
            f0k, gk = o(rho_vector=rho, need_gradient=True)
            f0s.append(f0k)
            gs.append(gk)
            rows.append(list(getattr(o, "f0_per_J", []) or
                             [float(np.real(np.sum(f0k)))]))
            # None entries are objectives minimax pre-selected away (above the
            # mean, so their gradient would have been zero-weighted anyway) --
            # msopt never runs their adjoint now, so report nan rather than
            # feeding None to np.linalg.norm.
            gnorms.append([float("nan") if gj is None else float(np.linalg.norm(gj))
                           for gj in getattr(o, "gradient_per_J", [])])
        f0, g = f0s[0], gs[0]
        per_J = _mean_per_J(rows)
        val = _scalar_fom(per_J)
        hist.append([float(v) for v in per_J])
        gn = _mean_per_J(gnorms)
        if len(opts) > 1:
            print("[rec] per-forward FoM " + ", ".join(
                f"pol{int(p)}={_scalar_fom(r):.6e}"
                for p, r in zip(FORWARD_POLS, rows)))
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
        # Element 1 is the per-CASE FoM list and element 2 the per-case
        # gradient list -- one entry per forward, which is what Case == 3 above
        # then averages. With one forward both are singletons, exactly as before.
        per_case = [(_scalar_fom(r) if r else val) for r in rows]
        return (g if isinstance(X, str) else (val, per_case, gs))
    return loop


# ---- figures -----------------------------------------------------------------
# Moved to oled_rec_plots.py; bound here so the call sites below stay unchanged.
# bind() must run AFTER every configuration global above exists.
import oled_rec_plots as _plots
_plots.bind(globals())
_draw_stack = _plots._draw_stack
plot_kmap_geometry = _plots.plot_kmap_geometry
plot_angle_mapping = _plots.plot_angle_mapping
plot_iteration_state = _plots.plot_iteration_state
iter_plot = _plots.iter_plot


def pp_grid_for_this_run():
    """The dipole grid this run's POSTPROCESS will end up using.

    resolve_dipole_grid decides from the DESIGN's mirror symmetry, which does not
    exist yet before the optimization runs. The symmetry is a CONSTRAINT of this
    run though, so it is already known: fourfold/radial designs come out
    mirror-symmetric and fold to 12x12, a SYMMETRY="none" design does not and
    stays 6x6. Probing with a matching dummy density asks resolve_dipole_grid the
    question in its own terms rather than duplicating its rule here.
    """
    if NOSYM_DESIGN:
        probe = np.random.default_rng(0).random(G.design_grids)   # not mirror-symmetric
    else:
        probe = np.zeros(G.design_grids, dtype=float)             # trivially symmetric
    return oc.resolve_dipole_grid(probe)[0]


def ensure_planar_reference():
    """Measure the PLANAR reference first if this configuration has none.

    Every design postprocess overlays that curve as the black dashed line, and it
    is keyed to stack, wavelength, pitch, monitor, resolution, dipole grid AND
    source layout (planar_reference_identity). Discovering it is missing only at
    the END -- after a multi-hour optimization -- costs the whole overlay for that
    run, and the layout/grid defaults have already changed twice. Measuring it
    HERE costs ~30 min once; every later run in the same configuration reuses it.

    Placed before build_problem for the same reason measure_planar_reference is:
    its Lumerical session opens and closes while the optimization session does not
    yet exist, so the licence server is never asked for two solves at once.

    Note the planar postprocess writes its own artifacts into this run's design
    dir; the run's real postprocess overwrites them at the end. The durable output
    is Planer_data.txt, which lives next to the scripts.
    """
    if not oc.env_flag("MSOPT_OLED_REQUIRE_PLANAR_REF", "1"):
        return
    if oc.planar_requested():
        return                      # this run IS the planar measurement
    grid_n = pp_grid_for_this_run()
    ident = oc.planar_reference_identity(G, grid_n)
    if oc.load_planar_reference_curve(identity=ident)[0] is not None:
        print(f"[planar] reference matches this configuration ({ident})")
        return
    print(f"[planar] no reference for {ident}")
    print("[planar] measuring the bare stack FIRST so the postprocess overlay exists")
    # PIN THE GRID to the one this run's postprocess will use. The planar density
    # is uniform, hence trivially mirror-symmetric, so resolve_dipole_grid would
    # always fold it to 12x12 -- even for a SYMMETRY="none" run whose own
    # postprocess stays at 6x6. The reference would then be measured on different
    # dipole positions than the design it is meant to be overlaid against, and the
    # identity check would reject it right after paying for it.
    os.environ["MSOPT_OLED_PP_PLANAR"] = "low"
    os.environ["MSOPT_OLED_PP_DIPOLE_GRID"] = str(grid_n)
    try:
        oc.run_postprocess(G, np.zeros(int(np.prod(G.design_grids)), float), mapping=None)
    finally:
        os.environ.pop("MSOPT_OLED_PP_PLANAR", None)
        os.environ.pop("MSOPT_OLED_PP_DIPOLE_GRID", None)
    if oc.load_planar_reference_curve(identity=ident)[0] is None:
        raise RuntimeError(
            "the planar measurement did not produce a matching reference "
            f"({ident}). Set MSOPT_OLED_REQUIRE_PLANAR_REF=0 to optimize without "
            "the overlay, or investigate before spending GPU hours."
        )
    print("[planar] reference written; continuing to the optimization")


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
        # The postprocess overlay's reference, before anything expensive starts.
        ensure_planar_reference()
        # Empty-domain reference BEFORE build_problem: the transmission term is
        # measured against it so it has to exist before any FoM is evaluated, and
        # running it first means its Lumerical session is opened and closed while
        # the optimization session does not yet exist -- two live sessions would
        # ask the licence server for 18 tasks at once.
        if TRANS_REF is None:
            measure_planar_reference()
        sims, opts, hist = build_problem()
        print(f"[setup] period={G.Sx:g}x{G.Sy:g} um, design={G.design_grids}, "
              f"modes={len(target_modes)}, probe=normal-incidence "
              f"{'+'.join(f'{p:g}deg' for p in FORWARD_POLS)}, "
              f"FoM plane z={eml_c[2]:+.4f} um (EML)")
        optimizer = ms.Opt_MS2.OPT_Ms(
            x0, dJ_0, design_dir=G.design_dir, local_best_dir=G.local_dir,
            Born_k=90, Initial_LR=oc.env_float("MSOPT_OLED_INITIAL_LR", 0.2),
            # Filterless designs have no projection-driven snap; see Opt_MS2.
            # FREEFORM ONLY. 1.5 when there is no filter, 1.0 (unchanged) with one:
            # the filtered path already gets its sharpening from the
            # filter+projection schedule -- it jumped match 0.2 -> 0.5
            # in 3 evaluations while freeform crawled the same span over ~90 --
            # so raising it there would change a path that already works.
            # so it starts projection sharper. Deliberately under 2.0, which is
            # where Opt_MS2 switches mapping branches: this buys the sharpening
            # without also changing the code path being measured.
            Initial_beta=oc.env_float("MSOPT_OLED_INITIAL_BETA",
                                      1.0 if USE_MFS_FILTER else 1.5),
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
        optimizer(mapping, N_fom, make_rec_loop(opts, hist, mode_records))
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
        save_angular_target(os.path.join(
            os.path.dirname(os.path.abspath(design_path)), "OLED_angular_target.npz"))
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
