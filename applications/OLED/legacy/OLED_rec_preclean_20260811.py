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
    J = _ST * level_score(T*S)  +  _SM * match(p)

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

Gradient
    Two top-level objectives, not one per target mode: transmission_J (the
    level term) and match_J (ONE joint distribution-match term covering every
    target (angle, order) pair at once). msopt's Incoherent=True
    (PER_MODE_ADJOINT) still runs one forward + one adjoint PER OBJECTIVE
    FUNCTION, restoring the cached forward field between them -- "per
    objective" just no longer means "per mode": match_J's joint KL-divergence
    form is NOT separable across modes the way a weighted sum is (see its own
    docstring), so it is traced by autograd as a single objective instead of
    being split into one adjoint per mode. That single joint adjoint gives the
    mathematically EXACT gradient of match(p), not an approximation -- no
    per-mode decomposition trick is needed the way RATIO_WINDOW (an earlier,
    now-removed attempt at the same underlying problem; see match_J and the
    oled-outcoupling-optimization memory) required one.

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
W_AT_0 = 1.00                  # requested share at normal incidence
W_AT_MAX = 0.9                 # requested share at TARGET_MAX_ANGLE; rungs between
                               # are interpolated, giving a profile that peaks on
                               # axis and falls off gently.
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
#   "absolute" a_i = overlap / (N * planar reference) = p_i * T: level and shape in
#              one number. NOTE transmission_term already multiplies by T, so with
#              "absolute" that factor enters the level score twice.
MODAL_METRIC = oc.env_str("MSOPT_OLED_MODAL_METRIC", "purity")   # env override:
                               # "purity" (share of the plane) or "absolute"
                               # (purity x level). Swept by the factor study.
_MLAB = "a" if MODAL_METRIC == "absolute" else "p"

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
W_TRANS = 0.05                 # ("weighted" only) share the level term carries
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
SUPPRESS_ABOVE_TARGET = oc.env_flag("MSOPT_OLED_SUPPRESS_ABOVE", "0")  # env override
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
TARGET_ALL_ORDERS = oc.env_flag("MSOPT_OLED_ALL_ORDERS", "0")   # env override.
                               # score EVERY escaping (m,n), grouped into cones by |u|
TARGET_ANGLES = [(0.0, 0.0), (45.0, 0.0), (45.0, 90.0)]
PERIOD_UM = None               # None -> pitch derived from the targets
SOURCE_POL = oc.env_str("MSOPT_OLED_SOURCE_POL", "x")   # "x", "y" or "xy"
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
        targets, orders, weights, suppress = [], [], [], []
        phis = [0.0] if KM_SYMMETRY == "radial" else [float(a) for a in AZIMUTHS_DEG]

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

        if TARGET_ALL_ORDERS and KM_SYMMETRY != "radial":
            targets, orders, weights, suppress, lattice = _all_escaping_orders(
                period, lam, L)
        else:
            lattice = None
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
    if MODAL_METRIC == "absolute":
        # ABSOLUTE modal coupling, referenced to the planar device: the mean
        # intensity sitting in mode i, in units of the planar EML intensity. Equal
        # to purity * T, so it carries level and shape in one number and the FoM
        # needs no separate level term to stop the optimizer from perfecting the
        # shape of nothing. sqrt(w*a) is concave, so dJ/da = 0.5*sqrt(w/a) is
        # largest for the starved order -- raising the total uniformly is always
        # worth less than feeding whichever order is behind, which is the job the
        # purity denominator used to do.
        denom = float(Es[0].shape[0] * Es[0].shape[1]) * _trans_ref(None)
    else:
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
GRATING_PROFILE = "freeform"
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
            # LEAK BUDGET, not "1 - leak". Measured on run A (minimax, absolute,
            # diagonals + suppression ON): minimax reported driving J[2] on EVERY
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
    if MULTIOBJ == "minimax":
        w = 1.0                # sqrt(1 - leaked) in [0,1]; see _ST/_SM above
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
        adj_fwd=False,
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
            gnorms.append([float(np.linalg.norm(gj))
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
        sims, opts, hist = build_problem()
        print(f"[setup] period={G.Sx:g}x{G.Sy:g} um, design={G.design_grids}, "
              f"modes={len(target_modes)}, probe=normal-incidence "
              f"{'+'.join(f'{p:g}deg' for p in FORWARD_POLS)}, "
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
