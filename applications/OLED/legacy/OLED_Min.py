import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from autograd import jacobian as ag_jacobian
from autograd import numpy as npa

import msopt as ms
import oled_common as oc


# =============================================================================
# PML/Bloch-selectable OLED reciprocity optimization (compact, on oled_common)
# Coordinate: 3D Cartesian; propagation axis: z.
# Boundary: Bloch in x/y and PML in z by default (MSOPT_OLED_BOUNDARY_MODE).
#
# Design idea:
# - Treat the OLED/pixel as a finite supercell/window.
# - Launch symmetric (reciprocal) plane-wave probes from +z toward the stack.
# - Maximize EML-plane coupling with an active IPR uniformity factor: this
#   script's assigned role is EML-layer UNIFORMITY maximization (see
#   MSOPT_OLED_UNIFORMITY_POWER below).
#
# Shared geometry/config, sim helpers, result readers, spectrum/analysis,
# plotting, and the incoherent-dipole postprocess all come from oled_common;
# this file keeps only Min's unique reciprocal-channel machinery.
# =============================================================================

# =============================================================================
# Run switches -- edit these, no environment variables needed
# =============================================================================
# Values here are exported as environment DEFAULTS, so an explicitly set
# MSOPT_OLED_* still wins (sweeps keep working without touching this file).
#
#   RUN_OPTIMIZATION  False -> skip the optimizer and only run the postprocess
#   PLANAR_PP         True  -> discard the design and characterize the BARE stack
#                              ("low" = design region filled with air, i.e. nothing
#                               on top; "high" = a flat slab of design material,
#                               which isolates PATTERNING from the layer's presence)
#   PP_MODE           "supercell" (NxN tiles, the validated protocol) or "single"
#                              (1 cell + pad; cheap and correct for a planar stack)
#   PP_DIPOLE_GRID    NxN incoherent dipole grid; 1 is exact for a planar stack
#   PP_CAPTURE_DEG    the domain is widened until this polar angle still reaches
#                              the monitor -- light past it is eaten by the lateral
#                              PML and is MISSING from the near2far input
#   PP_KEEP_FSP       keep every run's .fsp (hundreds of MB each)
#   RESOLUTION        cells/um for the simulation AND the design grid (locked
#                              together by msopt). Layers thinner than 1/RESOLUTION
#                              still get resolved: add_stack adds a z-only mesh
#                              override derived from the stack's thinnest layer.
#   DESIGN_H_UM       design-region thickness
#   DESIGN_X/Y_UM     design footprint; None = the full period
#   DESIGN_N          design-region index at rho=1 (rho=0 is DESIGN_LOW_N).
#                              1.45 suits the legacy stack; on the microcavity the
#                              design sits on an n=2.2 CPL, so a higher DESIGN_N
#                              gives the pattern real contrast
#   DESIGN_LOW_N      index at rho=0 (1.0 = air)
#   PROBE_GAP_UM      source/monitor plane height above the design top
#   TOP_MARGIN_UM     air above that plane (auto-raised to clear the PML)
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
TOP_MARGIN_UM = 0.1
STACK = "microcavity"        # "microcavity" (optimized stack) or "legacy"
MC_COLOR = "green"           # red / green / blue -- also sets the wavelength
MC_STACK_KIND = "optimized"  # "optimized" (literature-derived) or "table"

oc.export_run_knobs(
    run_optimization=RUN_OPTIMIZATION, planar_pp=PLANAR_PP, pp_mode=PP_MODE,
    pp_dipole_grid=PP_DIPOLE_GRID, pp_capture_deg=PP_CAPTURE_DEG,
    pp_keep_fsp=PP_KEEP_FSP, resolution=RESOLUTION, design_h_um=DESIGN_H_UM,
    design_x_um=DESIGN_X_UM, design_y_um=DESIGN_Y_UM,
    design_n=DESIGN_N, design_low_n=DESIGN_LOW_N,
    probe_gap_um=PROBE_GAP_UM, top_margin_um=TOP_MARGIN_UM,
    stack=STACK, mc_color=MC_COLOR, mc_stack_kind=MC_STACK_KIND,
)


G, _mc_spec = oc.select_stack(STACK, MC_COLOR, MC_STACK_KIND, period_mc=2.0)

# Aliases into the shared config so the validated function bodies below stay
# verbatim with the legacy script.
visible_wavelengths = G.visible_wavelengths
design_grids = G.design_grids
design_cells = G.design_cells
Nx, Ny, Nz = G.design_grids
design_s, design_c = G.design_s, G.design_c
eml_c, eml_s = G.eml_c, G.eml_s
boundary_label = G.bc_xy
bandwidth = 0.0

src_s = [G.Sx, G.Sy, 0]
# Reciprocal probe: the symmetric plane wave is launched on the PROBE PLANE,
# which build_config puts PROBE_GAP (0.7 um) above the design top -- exactly
# where the dipole-radiation cases put their far-field monitor. Same plane for
# both formulations, so reciprocal and dipole results refer to the same surface.
src_c = [0.0, 0.0, G.probe_plane_z]

source_norm_monitor_name = "source_norm_monitor"
source_norm_s = [G.Sx, G.Sy, 0]
source_norm_c = [0, 0, src_c[2] - 0.05]

# Floor for the RECIPROCAL FoM channel scores (EML-plane |E|^2 sums, O(1e-3..1)),
# where 1e-12 is a harmless divide-by-zero guard.
#
# It is deliberately NOT copied into G: G.channel_power_floor guards ABSOLUTE
# watts in oled_common.run_postprocess, and this device emits ~1e-15 W per
# dipole.  Writing 1e-12 there made max(emitted_power, floor) return the FLOOR
# for every dipole, so every postprocess LEE came out ~500x too small.  The two
# floors measure different quantities and must stay separate.
channel_power_floor = oc.env_float("MSOPT_OLED_FOM_POWER_FLOOR", 1e-12)
unstable_candidate_fom = G.unstable_candidate_fom
combined_fom_history = []


# -----------------------------------------------------------------------------
# Reciprocal radiation channels
#
# Validated 2026-07-30 against the OLED_opt Done run (20260728_122008): unit-amplitude
# plane-wave probes with EML-plane mean |Ex|^2 reproduce the incoherent 9x9/PML dipole
# verification ring shares (reciprocal 0.020/0.364/0.616 vs verified 0.023/0.362/0.615
# at theta=0/30/45), while the coherent dipole-grid FoM of OLED_new/OLED_opt claimed
# 0.606/0.328/0.066 for the same design -- the exact-normal (0,0) order only exists for
# an in-phase source array, never for the real incoherent ensemble. Keep the FoM on
# reciprocal EML coupling; do not switch back to coherent dipole grids.
# -----------------------------------------------------------------------------
theta_channel_centers_deg = np.array([0.0, 30.0, 45.0, 60.0])

# Per-direction efficiency ramp, same spec as the OLED_new/OLED_opt order-share runs:
# 1.0 at 0 deg falling linearly to 0.85 at 45 deg, suppressed (0) above 45 deg. The
# min/max windows are the ramp +- MSOPT_OLED_RATIO_TOL; suppression stays a hard zero.
target_angle_efficiency_ratio = np.array([1.0, 0.90, 0.85, 0.0], dtype=float)
if target_angle_efficiency_ratio.size != theta_channel_centers_deg.size:
    raise ValueError("target_angle_efficiency_ratio length must match theta_channel_centers_deg.")

ratio_window_tol = oc.env_float("MSOPT_OLED_RATIO_TOL", 0.05)
_suppressed = target_angle_efficiency_ratio <= 0.0
target_angle_efficiency_ratio_min = np.clip(target_angle_efficiency_ratio - ratio_window_tol, 0.0, 1.0)
target_angle_efficiency_ratio_max = np.clip(target_angle_efficiency_ratio + ratio_window_tol, 0.0, 1.0)
target_angle_efficiency_ratio_min[0] = target_angle_efficiency_ratio_max[0] = 1.0
target_angle_efficiency_ratio_min[_suppressed] = 0.0
target_angle_efficiency_ratio_max[_suppressed] = 0.0

channel_polarizations = ("x",)
polarization_angles = {"x": 0.0, "y": 90.0}
# Horizontal-ensemble coupling |Ex|^2 + |Ey|^2 (incoherent x+y dipoles). Under the
# C4v cell symmetry this equals the ring-level x-dipole verification target, and it
# removes the dipole-orientation geometry factor that a single component picks up
# once the probe azimuth is jittered away from phi=0.
eml_components_by_polarization = {"x": ("Ex", "Ey"), "y": ("Ex", "Ey")}
ez_component_weight = oc.env_float("MSOPT_OLED_EZ_WEIGHT", 0.0)

target_distribution_weight = oc.env_float("MSOPT_OLED_DISTRIBUTION_WEIGHT", 10.0)
relaxed_distribution_weight = oc.env_float("MSOPT_OLED_RELAXED_DISTRIBUTION_WEIGHT", 3.0)
penalty_ramp_start = oc.env_float("MSOPT_OLED_PENALTY_RAMP_START", 0.20)
penalty_ramp_end = oc.env_float("MSOPT_OLED_PENALTY_RAMP_END", 0.90)
current_distribution_weight = target_distribution_weight
current_binarization_fraction = 1.0


target_channels = []
for angle_idx, (center_deg, min_ratio, max_ratio) in enumerate(
    zip(theta_channel_centers_deg, target_angle_efficiency_ratio_min, target_angle_efficiency_ratio_max)
):
    for pol in channel_polarizations:
        theta_rad = np.deg2rad(center_deg)
        target_channels.append(
            {
                "name": f"theta_{center_deg:.1f}deg_{pol}",
                "angle_idx": angle_idx,
                "theta_deg": float(center_deg),
                "phi_deg": 0.0,
                "polarization": pol,
                "polarization_angle": polarization_angles[pol],
                "eml_components": eml_components_by_polarization[pol],
                "target_ratio_to_zero_min": float(min_ratio),
                "target_ratio_to_zero_max": float(max_ratio),
                "source_power_norm": max(float(np.cos(theta_rad)), 1e-6),
                "wavelengths": np.asarray(visible_wavelengths, dtype=float),
            }
        )
N_fom = len(target_channels)

angle_channel_indices = [
    [idx for idx, channel in enumerate(target_channels) if channel["angle_idx"] == angle_idx]
    for angle_idx in range(len(theta_channel_centers_deg))
]

# PP angular-target overlay = the reciprocity channel ramp itself (angle, ratio),
# not the tolerance-window edges. Written back into G so the shared
# run_postprocess / build_target_orders use exactly this ramp (instead of the
# env-driven MSOPT_OLED_TARGET_EFFICIENCY_CURVE default).
G.target_efficiency_curve = [
    (float(d), float(r)) for d, r in zip(theta_channel_centers_deg, target_angle_efficiency_ratio)
]
G.target_angle_pairs = list(G.target_efficiency_curve)


def _interp_ramp(theta_deg, curve=tuple(G.target_efficiency_curve)):
    theta, value = np.asarray(curve, dtype=float).T
    return float(np.interp(float(theta_deg), theta, value, left=value[0], right=value[-1]))


G.interp_curve = _interp_ramp

# -----------------------------------------------------------------------------
# Stochastic probe-angle sampling (keeps N_fom / sim count fixed)
#
# Fixed probes only pin the ramp AT the probe angles; the continuum in between is
# unconstrained, so sharp guided-mode features can hide between probes. One Bloch
# sim can only probe one incidence angle (two plane waves in one cell leave |E|^2
# cross terms that cannot be separated), so instead of adding channels each
# non-anchor channel redraws its probe angle inside a band once per gradient
# evaluation, and the ratio windows are rebuilt from the ramp at the drawn angles.
# In expectation the whole 0-45 deg ramp and the >45 deg suppression band are
# enforced at the same per-iteration cost. Line-search (Case=False) evaluations
# reuse the last draw so candidate comparisons stay consistent within a step.
# -----------------------------------------------------------------------------
angle_jitter_enabled = oc.env_flag("MSOPT_OLED_ANGLE_JITTER", "1")
angle_jitter_rng = np.random.default_rng(oc.env_int("MSOPT_OLED_ANGLE_JITTER_SEED", 12345))

# Azimuth/polarization sampling rides on the same redraw. The square lattice breaks
# the radial design's rotational symmetry down to C4v, so phi=0-only probes miss
# diagonal (Gamma-M) band features; phi is therefore drawn per channel inside the
# irreducible wedge [0, MSOPT_OLED_PHI_WEDGE_DEG]. One source polarization (p or s)
# is drawn per evaluation and shared by all channels so the ratio windows compare
# like with like. Enforcing the ramp per polarization is slightly stronger than the
# s+p-summed target (mainly near 45 deg); the ratio tolerance absorbs that.
phi_jitter_enabled = oc.env_flag("MSOPT_OLED_PHI_JITTER", "1")
phi_wedge_deg = oc.env_float("MSOPT_OLED_PHI_WEDGE_DEG", 45.0)
pol_jitter_enabled = oc.env_flag("MSOPT_OLED_POL_JITTER", "1")


def _parse_jitter_bands(raw):
    bands = []
    for part in raw.split(";"):
        lo, hi = (float(v) for v in part.split(",")[:2])
        bands.append((min(lo, hi), max(lo, hi)))
    return bands


# One "lo,hi" band per channel angle; a zero-width band anchors that probe.
angle_jitter_bands = _parse_jitter_bands(
    os.environ.get("MSOPT_OLED_ANGLE_JITTER_BANDS", "0,0;5,37.5;37.5,52.5;52.5,70")
)
if len(angle_jitter_bands) != theta_channel_centers_deg.size:
    raise ValueError("MSOPT_OLED_ANGLE_JITTER_BANDS must provide one lo,hi band per channel angle.")


def ramp_efficiency_ratio(theta_deg):
    # Continuous per-direction target: 1.0 at 0 deg -> 0.85 at 45 deg, 0 above.
    th = float(theta_deg)
    return 0.0 if th > 45.0 else 1.0 - 0.15 * th / 45.0


def redraw_channel_angles(sims):
    """Redraw probe theta per band (plus phi in the C4v wedge and a shared source
    polarization), rebuild ratio windows from the ramp at the drawn thetas, and
    point each channel's plane-wave source at its new direction (Bloch k follows
    the source angles). Build-time measured source norms stay valid across
    redraws: unit-amplitude Bloch plane-wave injection is angle-independent in
    |E| (reciprocal validation, 2026-07-30)."""
    global theta_channel_centers_deg
    global target_angle_efficiency_ratio_min, target_angle_efficiency_ratio_max
    drawn = [
        lo if hi <= lo else float(angle_jitter_rng.uniform(lo, hi))
        for lo, hi in angle_jitter_bands
    ]
    theta_channel_centers_deg = np.asarray(drawn, dtype=float)
    ratios = np.asarray([ramp_efficiency_ratio(t) for t in drawn], dtype=float)
    suppressed = ratios <= 0.0
    target_angle_efficiency_ratio_min = np.clip(ratios - ratio_window_tol, 0.0, 1.0)
    target_angle_efficiency_ratio_max = np.clip(ratios + ratio_window_tol, 0.0, 1.0)
    target_angle_efficiency_ratio_min[0] = target_angle_efficiency_ratio_max[0] = 1.0
    target_angle_efficiency_ratio_min[suppressed] = 0.0
    target_angle_efficiency_ratio_max[suppressed] = 0.0
    shared_pol_angle = float(angle_jitter_rng.choice([0.0, 90.0])) if pol_jitter_enabled else None
    for idx, channel in enumerate(target_channels):
        theta = drawn[channel["angle_idx"]]
        channel["theta_deg"] = float(theta)
        channel["source_power_norm"] = max(float(np.cos(np.deg2rad(theta))), 1e-6)
        if phi_jitter_enabled:
            channel["phi_deg"] = float(angle_jitter_rng.uniform(0.0, phi_wedge_deg))
        if shared_pol_angle is not None:
            channel["polarization_angle"] = shared_pol_angle
        if sims:
            fdtd = sims[idx].fdtd
            try:
                fdtd.switchtolayout()
            except Exception:
                pass
            fdtd.setnamed("source", "angle theta", float(channel["theta_deg"]))
            fdtd.setnamed("source", "angle phi", float(channel["phi_deg"]))
            fdtd.setnamed("source", "polarization angle", float(channel["polarization_angle"]))
    print(
        f"[angle jitter] probes -> theta={np.round(theta_channel_centers_deg, 2).tolist()} "
        f"phi={[round(float(c['phi_deg']), 1) for c in target_channels]} "
        f"pol_angle={float(target_channels[0]['polarization_angle']):.0f}"
    )


# FoM: each reciprocal channel evaluates its matching transverse EML component.
# Ez can be added as a weighted intensity term, but it is not balanced against Ex/Ey.
# The uniformity (IPR) factor is ON by default here: this script's assigned role is
# EML-layer UNIFORMITY maximization, so the position-averaged coupling is weighted
# by the IPR uniformity factor. The pure position-average FoM (power 0, ~ what the
# incoherent dipole verification measures) stays available via
# MSOPT_OLED_UNIFORMITY_POWER=0 and is the default in OLED_opt.py.
uniformity_power = oc.env_float("MSOPT_OLED_UNIFORMITY_POWER", 0.2)


""" FoM subfunctions for OLED optimization """

def real_scalar_or_none(value):
    try:
        return float(np.real(value))
    except (TypeError, ValueError):
        return None


def flatten_channel_values(channel_values):
    return npa.array(
        [v[0] if isinstance(v, (list, tuple, np.ndarray)) else v for v in channel_values]
    )


def angle_powers_from_channel_values(vals):
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), channel_power_floor)
    return npa.array([npa.sum(vals[indices]) for indices in angle_channel_indices])


def binarization_fraction_from_design(X):
    rho = np.asarray(npa.clip(X, 0.0, 1.0), dtype=float).ravel()
    if rho.size == 0:
        return 1.0
    return float(np.mean((rho <= 1e-3) | (rho >= 1.0 - 1e-3)))


def penalty_ramp_fraction(binarization_fraction):
    if penalty_ramp_end <= penalty_ramp_start:
        return 1.0
    return float(np.clip(
        (binarization_fraction - penalty_ramp_start) / (penalty_ramp_end - penalty_ramp_start),
        0.0,
        1.0,
    ))


def update_oled_penalty_weights(X):
    global current_distribution_weight
    global current_binarization_fraction

    if isinstance(X, str):
        current_distribution_weight = target_distribution_weight
        current_binarization_fraction = 1.0
        return current_distribution_weight, current_binarization_fraction

    current_binarization_fraction = binarization_fraction_from_design(X)
    ramp = penalty_ramp_fraction(current_binarization_fraction)
    current_distribution_weight = (
        relaxed_distribution_weight
        + ramp * (target_distribution_weight - relaxed_distribution_weight)
    )
    return current_distribution_weight, current_binarization_fraction


def combine_oled_scalar_from_values(vals):
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), channel_power_floor)
    angle_powers = angle_powers_from_channel_values(vals)
    zero_power = npa.maximum(angle_powers[0], channel_power_floor)
    ratios_to_zero = angle_powers / zero_power
    low_violation = npa.maximum(target_angle_efficiency_ratio_min - ratios_to_zero, 0.0)
    high_violation = npa.maximum(ratios_to_zero - target_angle_efficiency_ratio_max, 0.0)
    # ratios_to_zero are already normalized to the 0-deg channel (O(1)), so penalize the
    # violations on that scale. Dividing by the target itself blew up for suppression
    # channels (target=0): high_violation/(0+1e-30) -> ~1e30, squared -> ~1e60, which
    # collapsed the whole FoM to ~1e-63. Floor the reference at 1.0 so target=0 is safe.
    lo_ref = npa.maximum(target_angle_efficiency_ratio_min, 1.0)
    hi_ref = npa.maximum(target_angle_efficiency_ratio_max, 1.0)
    distribution_penalty = npa.sum((low_violation / lo_ref) ** 2 + (high_violation / hi_ref) ** 2)
    penalty = current_distribution_weight * distribution_penalty
    penalty_score = 1.0 / (1.0 + penalty)
    return zero_power * penalty_score


def combined_oled_summary_from_values(vals):
    vals = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, channel_power_floor)
    angle_powers = np.asarray(angle_powers_from_channel_values(vals), dtype=float)
    fractions = angle_powers / max(float(np.sum(angle_powers)), 1e-30)
    ratios_to_zero = angle_powers / max(float(angle_powers[0]), 1e-30)
    low_violation = np.maximum(target_angle_efficiency_ratio_min - ratios_to_zero, 0.0)
    high_violation = np.maximum(ratios_to_zero - target_angle_efficiency_ratio_max, 0.0)
    lo_ref = np.maximum(target_angle_efficiency_ratio_min, 1.0)
    hi_ref = np.maximum(target_angle_efficiency_ratio_max, 1.0)
    distribution_penalty = np.sum((low_violation / lo_ref) ** 2 + (high_violation / hi_ref) ** 2)
    return {
        "angle_powers": angle_powers,
        "fractions": fractions,
        "ratios_to_zero": ratios_to_zero,
        "distribution_penalty": float(distribution_penalty),
    }



""" Initialization and optimization loop setup for OLED design problem """

def build_optimization_problem():
    use_source_normalization = oc.env_flag("MSOPT_OLED_SOURCE_NORMALIZATION", "1")
    source_norms = np.asarray(
        [channel["source_power_norm"] for channel in target_channels],
        dtype=float,
    )
    fom_history = [[] for _ in range(N_fom)]
    sim = [None] * N_fom
    opt = [None] * N_fom

    for idx, channel in enumerate(target_channels):
        sim[idx] = oc.make_sim(G, [G.Sx, G.Sy, G.Sz])

        sim[idx].add_source(
            mode="plane",
            name="source",
            center=src_c,
            size=src_s,
            direction="backward",
            src_wl=visible_wavelengths,
            bandwidth=bandwidth,
            pol=channel["polarization_angle"],
            theta=channel["theta_deg"],
            phi=channel["phi_deg"],
            broadband=True,
        )

        if use_source_normalization:
            sim[idx].add_monitor(
                name=source_norm_monitor_name,
                center=source_norm_c,
                size=source_norm_s,
            )
            sim[idx].run(name=f"source_norm_{idx}", save=True)
            Eres = sim[idx].fdtd.getresult(source_norm_monitor_name, "E")
            Eall = np.asarray(Eres["E"], dtype=np.complex128)
            score = 0.0
            if Eall.ndim >= 5:
                n_freq = min(len(visible_wavelengths), Eall.shape[-2])
                for fidx in range(n_freq):
                    intensity = (
                        np.abs(Eall[..., fidx, 0]) ** 2
                        + np.abs(Eall[..., fidx, 1]) ** 2
                        + np.abs(Eall[..., fidx, 2]) ** 2
                    )
                    score += float(np.nanmean(intensity))
            else:
                intensity = (
                    np.abs(Eall[..., 0]) ** 2
                    + np.abs(Eall[..., 1]) ** 2
                    + np.abs(Eall[..., 2]) ** 2
                )
                score = float(np.nanmean(intensity))
            source_norms[idx] = max(
                float(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)),
                channel_power_floor,
            )
            sim[idx].fdtd.switchtolayout()
            oc.delete_object(sim[idx].fdtd, source_norm_monitor_name)
            print(
                f"[source normalization] channel {idx} {channel['name']}: "
                f"measured_mean_absE2={source_norms[idx]:.16e}, "
                f"analytic_cos={channel['source_power_norm']:.16e}"
            )

        oc.add_stack(G, sim[idx])
        sim[idx].add_design_grid(
            name="design",
            center=design_c,
            size=design_s,
            index1=G.design_high_index,
            index2=G.design_low_index,
            design_grids=design_grids,
            density=G.grating_initial_density * np.ones(design_grids),
            wavelength=float(np.mean(visible_wavelengths)),
        )
        sim[idx].add_design_monitor()
        sim[idx].add_monitor(name="eml_preview_monitor", center=eml_c, size=eml_s)

        source_norm = max(float(source_norms[idx]), 1e-30)

        def J_oled(E_x, E_y, E_z, channel_idx=idx, channel=channel, source_norm=source_norm):
            score_terms = []
            component_intensities = []
            component_uniformities = []
            field_components = {"Ex": E_x, "Ey": E_y, "Ez": E_z}
            weighted_components = [(component, 1.0) for component in channel["eml_components"]]
            if ez_component_weight > 0.0 and "Ez" not in channel["eml_components"]:
                weighted_components.append(("Ez", ez_component_weight))
            for component, component_weight in weighted_components:
                E_component = field_components[component]
                raw_intensity = 0.0
                uniformity = 0.0
                n_freq = len(visible_wavelengths) if E_component.ndim == 4 else 1
                for fidx in range(n_freq):
                    Ei = E_component[:, :, :, fidx] if E_component.ndim == 4 else E_component
                    Ei = npa.where(npa.isfinite(Ei), Ei, 0.0)
                    intensity = npa.where(npa.isfinite(npa.abs(Ei) ** 2), npa.abs(Ei) ** 2, 0.0)
                    mean_intensity = npa.mean(intensity)
                    mean_intensity_sq = npa.mean(intensity ** 2)
                    raw_intensity += mean_intensity
                    uniformity += mean_intensity ** 2 / (mean_intensity_sq + 1e-30)
                component_score = raw_intensity * (uniformity + 1e-30) ** uniformity_power / source_norm
                score_terms.append(component_weight * component_score)
                component_intensities.append((component, component_weight, raw_intensity))
                component_uniformities.append((component, uniformity))

            score_terms = npa.array(score_terms)
            fom = npa.maximum(npa.sum(score_terms), channel_power_floor)
            fom_value = real_scalar_or_none(fom)
            if fom_value is not None:
                fom_history[channel_idx].append(fom_value)
                score_values = [float(np.real(v)) for v in score_terms]
                print(
                    f"[{boundary_label} channel {channel_idx}] {channel['name']} "
                    f"reciprocal EML proxy: {fom} "
                    f"(source_pol={channel['polarization']}, "
                    f"components={weighted_components}, "
                    f"weighted_scores={score_values}, "
                    f"mean_absE2={component_intensities}, "
                    f"uniformity={component_uniformities}, uniformity_power={uniformity_power})"
                )
            return fom

        opt[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim[idx],
            objective_functions=[J_oled],
            objective_arguments=[0, 1, 2],
            FoM_size=eml_s,
            FoM_center=eml_c,
            adj_fwd=True,
            opt_idx=idx,
            broadband_adjoint=True,
        )
    return sim, opt, fom_history


DR_info = [design_s[0], design_s[1], design_s[2], 0, 1, 2]
DR_N_info = [Nx, Ny, Nz, G.resolution]
radial_design_radius = oc.env_float("MSOPT_OLED_RADIAL_RADIUS", 0.5 * min(design_s[0], design_s[1]))
radial_design_grids = oc.env_int("MSOPT_OLED_RADIAL_GRIDS", int(round(radial_design_radius * G.resolution)) + 1)
mapping = ms.Opt_MS2.Mapping(
    Symmetry_sim=False,
    Sym_geo_width=False,
    Sym_geo_C8=False,
    Sym_geo_length=False,
    Sym_geo_C2=False,
    DR_info=DR_info,
    DR_N_info=DR_N_info,
    Mask_pixels=0,
    MFS=0.1,
    MGS=0.1,
    # Is_radial_3d={
    #     "enabled": True,
    #     "N_radius": radial_design_grids,
    #     "radius": radial_design_radius,
    #     "outside_value": 0.0,
    #     "apply_filter": True,
    #     "vertical_grating": True,
    # },
    Is_slanted_grating=False,
    Is_freeform=[True, True, False],
)
design_parameters = Nx * Ny * Nz
x0 = G.grating_initial_density * np.ones(design_parameters)
dJ_0 = np.zeros(design_cells)


def make_adjoint_loop(opt, sims=None):
    def Adjoint_loop(X, N_cases, Case=True):
        if Case == 3:
            dJ_dus = X[0]
            channel_values = N_cases
            vals = flatten_channel_values(channel_values)
            vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), channel_power_floor)
            grads = [npa.where(npa.isfinite(npa.array(grad)), npa.array(grad), 0.0) for grad in dJ_dus]
            coeffs = ag_jacobian(combine_oled_scalar_from_values)(vals)
            coeffs = npa.where(npa.isfinite(coeffs), coeffs, 0.0)
            grad = 0.0
            for coeff, channel_grad in zip(coeffs, grads):
                grad += coeff * channel_grad
            grad = npa.where(npa.isfinite(grad), grad, 0.0)
            print(f"combined grad mean: {np.mean(np.abs(grad))}")
            print(f"combined grad max: {np.max(np.abs(grad))}")
            return grad

        if angle_jitter_enabled and Case and not isinstance(X, str):
            redraw_channel_angles(sims)
        update_oled_penalty_weights(X)
        f0s = [0] * N_fom
        dJ_dus = [0] * N_fom
        for idx in range(N_fom):
            if isinstance(X, str):
                f0s[idx], dJ_dus[idx] = opt[idx](need_gradient=Case)
            else:
                rho = npa.clip(X, 0.0, 1.0)
                f0s[idx], dJ_dus[idx] = opt[idx](rho_vector=[rho], need_gradient=Case)

        if not isinstance(X, str):
            try:
                rho_temp = np.asarray(npa.clip(X, 0.0, 1.0), dtype=float)
                if rho_temp.size == design_cells:
                    rho_temp = rho_temp.reshape(design_grids)
                elif rho_temp.size == Nx * Ny:
                    rho_temp = np.repeat(rho_temp.reshape(Nx, Ny)[:, :, None], Nz, axis=2)
                else:
                    raise ValueError(f"unexpected design size {rho_temp.size}")

                x_axis = np.linspace(-0.5 * design_s[0], 0.5 * design_s[0], Nx)
                y_axis = np.linspace(-0.5 * design_s[1], 0.5 * design_s[1], Ny)
                z_axis = np.linspace(
                    design_c[2] - 0.5 * design_s[2],
                    design_c[2] + 0.5 * design_s[2],
                    Nz,
                )
                xy_aspect = design_s[1] / max(design_s[0], 1e-30)
                xz_aspect = design_s[2] / max(design_s[0], 1e-30)
                panel_width = 5.0
                fig_height = max(2.2, panel_width * max(xy_aspect, xz_aspect))
                fig, axes = plt.subplots(1, 2, figsize=(2 * panel_width, fig_height))
                axes[0].imshow(
                    rho_temp[:, :, Nz // 2].T,
                    origin="lower",
                    extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
                    cmap="binary",
                    vmin=0.0,
                    vmax=1.0,
                    aspect="equal",
                    interpolation="nearest",
                )
                axes[0].set_xlabel("x (um)")
                axes[0].set_ylabel("y (um)")
                axes[0].set_title("x-y section")
                axes[1].imshow(
                    rho_temp[:, Ny // 2, :].T,
                    origin="lower",
                    extent=(x_axis[0], x_axis[-1], z_axis[0], z_axis[-1]),
                    cmap="binary",
                    vmin=0.0,
                    vmax=1.0,
                    aspect="equal",
                    interpolation="nearest",
                )
                axes[1].set_xlabel("x (um)")
                axes[1].set_ylabel("z (um)")
                axes[1].set_title("x-z section at y=0")
                fig.suptitle("Current design sections")
                fig.tight_layout()
                path = os.path.join(G.design_dir, "design_iter_temp.png")
                fig.savefig(path, dpi=200)
                plt.close(fig)
                print(f"[optimized] saved temporary design section: {path}")
            except Exception as exc:
                print(f"[optimized] skipped temporary design section: {exc}")

        unstable_candidate = any(getattr(problem, "last_forward_had_nonfinite", False) for problem in opt)
        if unstable_candidate:
            print(
                f"[{boundary_label}] unstable candidate detected: non-finite Lumerical field/FoM. "
                "Rejecting this geometry through backtracking."
            )
            zero_grads = [
                np.zeros_like(grad, dtype=float)
                if not isinstance(grad, (int, float))
                else np.zeros(design_cells, dtype=float)
                for grad in dJ_dus
            ]
            f0s = [channel_power_floor for _ in range(N_fom)]
            if Case:
                if isinstance(X, str):
                    return zero_grads
                return unstable_candidate_fom, f0s, zero_grads
            return unstable_candidate_fom, f0s

        f0 = combine_oled_scalar_from_values(flatten_channel_values(f0s))
        f0_value = real_scalar_or_none(f0)
        if f0_value is not None:
            combined_fom_history.append(f0_value)

        vals = [
            max(float(np.nan_to_num(np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v))), channel_power_floor)
            for v in f0s
        ]
        summary = combined_oled_summary_from_values(vals)
        angle_powers = summary["angle_powers"]
        fractions = summary["fractions"]
        ratios_to_zero = summary["ratios_to_zero"]
        for idx, channel in enumerate(target_channels):
            print(
                f"[{boundary_label}] channel={channel['name']} reciprocal EML proxy={vals[idx]} "
                f"(theta={channel['theta_deg']:.1f}, pol={channel['polarization']}, "
                f"components={channel['eml_components']}, "
                f"source_power_norm={channel['source_power_norm']:.6g})"
            )
        for angle_idx, theta_deg in enumerate(theta_channel_centers_deg):
            print(
                f"[{boundary_label}] theta={theta_deg:.1f} deg angle_power={angle_powers[angle_idx]} "
                f"fraction={fractions[angle_idx] * 100:.3f}% "
                f"ratio_to_0={ratios_to_zero[angle_idx]:.4f} "
                f"(target_range={target_angle_efficiency_ratio_min[angle_idx]:.4f}-"
                f"{target_angle_efficiency_ratio_max[angle_idx]:.4f})"
            )
        print(
            f"combined {boundary_label} OLED FoM: {f0} "
            f"(distribution_weight={current_distribution_weight:.4g}/{target_distribution_weight:.4g}, "
            f"Ez_weight={ez_component_weight:.4g}, binarization={current_binarization_fraction:.3f})"
        )

        if Case:
            if isinstance(X, str):
                return dJ_dus
            return f0, f0s, dJ_dus
        return f0, f0s

    return Adjoint_loop


if __name__ == "__main__":
    if oc.env_flag("MSOPT_OLED_SESSION_TEST", "0"):
        channel_idx = oc.env_int("MSOPT_OLED_SESSION_TEST_CHANNEL", 0)
        channel = target_channels[channel_idx]
        print(f"[OLED session test] channel {channel_idx}: {channel['name']}")
        print(
            f"[OLED session test] theta={channel['theta_deg']}, pol={channel['polarization']}, "
            f"source_power_norm={channel['source_power_norm']}"
        )
        oc.session_test_banner(G, N_fom, extra=f"uniformity_power={uniformity_power}")
        raise SystemExit(0)

    start = time.time()

    sim, opt, fom_history = build_optimization_problem()
    print(f"{boundary_label} OLED reciprocity scaffold built.")
    print(
        "OLED periodic 3D freeform setup: "
        f"period={G.window_x}x{G.window_y} um, active area={G.active_x}x{G.active_y} um, "
        f"air={G.air_top_h} um, design={G.grating_design_h} um, "
        f"SiO2={G.sio2_h} um, ITO={G.ito_h} um, TCTA={G.tcta_h} um, "
        f"EML={G.eml_h} um, TPBi={G.tpbi_h} um, Ag={G.ag_h} um, "
        f"bottom_air_pad={G.air_bot_h} um, background_index={G.background_index}"
    )
    print(
        "Target theta channels: "
        + ", ".join(
            f"{ch['name']} target_ratio_to_zero="
            f"{ch['target_ratio_to_zero_min']:.4f}-{ch['target_ratio_to_zero_max']:.4f} "
            f"source_pol={ch['polarization']} components={ch['eml_components']} "
            f"source_power_norm={ch['source_power_norm']:.4f}"
            for ch in target_channels
        )
    )
    print(
        f"N_fom={N_fom}, design_grids={design_grids}, design_cells={design_cells}, "
        f"radial_grating_shape=({radial_design_grids},), design_parameters={design_parameters}, "
        f"radial_radius={radial_design_radius}"
    )
    print(f"boundary_mode={boundary_label}, bc_x={G.bc_xy}, bc_y={G.bc_xy}, bc_z=PML")
    print(
        "FoM=sum of source-power-normalized matching transverse EML coupling "
        "from one representative reciprocal polarization, weighted by target "
        "angular distribution; Ez is optional via MSOPT_OLED_EZ_WEIGHT"
    )
    print(f"visible_wavelengths={visible_wavelengths}")
    print(f"EML FoM plane center={eml_c}, size={eml_s}")
    print(
        f"FoM controls: uniformity_power={uniformity_power}, "
        f"Ez_weight={ez_component_weight}, distribution_weight={target_distribution_weight}"
    )
    print(
        "Postprocess settings: "
        f"MSOPT_OLED_POSTPROCESS={oc.env_flag('MSOPT_OLED_POSTPROCESS', '1')}, "
        f"MSOPT_OLED_POSTPROCESS_ONLY={oc.env_flag('MSOPT_OLED_POSTPROCESS_ONLY', '0')}, "
        f"MSOPT_OLED_POSTPROCESS_ANGLE_RES={os.environ.get('MSOPT_OLED_POSTPROCESS_ANGLE_RES', '181')}"
    )

    postprocess_only = oc.env_flag("MSOPT_OLED_POSTPROCESS_ONLY", "0")
    if postprocess_only:
        print("[optimized] skipped optimizer: MSOPT_OLED_POSTPROCESS_ONLY is enabled")
    else:
        optimizer = ms.Opt_MS2.OPT_Ms(
            x0,
            dJ_0,
            design_dir=G.design_dir,
            local_best_dir=G.local_dir,
            Born_k=50,
            Initial_LR=0.2,
            Raw=True,
        )
        optimizer.flag = True
        optimizer(mapping, N_fom, make_adjoint_loop(opt, sims=sim))
        np.savetxt(os.path.join(G.design_dir, "FoM_history.txt"), np.array(fom_history, dtype=object), fmt="%s")

        if combined_fom_history:
            values = np.asarray(combined_fom_history, dtype=float)
            np.savetxt(os.path.join(G.design_dir, "OLED_optimized_combined_fom_history.txt"), values)
            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, values.size + 1), values, linewidth=1.5)
            plt.xlabel("combined FoM evaluation")
            plt.ylabel("combined FoM")
            plt.title("OLED optimized FoM curve")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            path = os.path.join(G.design_dir, "OLED_optimized_fom_curve.png")
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"[optimized] saved FoM curve: {path}")
        else:
            print("[optimized] skipped FoM curve: no combined FoM history")

    if oc.env_flag("MSOPT_OLED_POSTPROCESS", "1"):
        design_path = os.environ.get("MSOPT_OLED_POSTPROCESS_DESIGN", "").strip()
        if not (design_path and os.path.exists(design_path)):
            cand = os.path.join(G.design_dir, "lastdesign.txt")
            design_path = cand if os.path.exists(cand) else ""
        if not design_path:
            refs = sorted(glob.glob(os.path.join(G.local_dir, "ref_layer_*.txt")), key=os.path.getmtime)
            design_path = refs[-1] if refs else ""
        # A PLANAR run characterizes the bare stack, so it needs no design at
        # all: the design region is filled with the low index either way. Do not
        # let the "no design file" check abort it.
        if oc.planar_requested() and not (design_path and os.path.exists(design_path)):
            print("[postprocess] PLANAR stack characterization (no design required)")
            design_path = ""
            final_design_planar = np.zeros(int(np.prod(G.design_grids)), dtype=float)
        if design_path and os.path.exists(design_path):
            print(f"[postprocess] using design: {design_path}")
            oc.run_postprocess(G, np.loadtxt(design_path), mapping=mapping)
        elif oc.planar_requested():
            oc.run_postprocess(G, final_design_planar, mapping=None)
        else:
            print("[postprocess] skipped: no design file (set MSOPT_OLED_POSTPROCESS_DESIGN or produce lastdesign.txt)")
    else:
        print("[postprocess] skipped all postprocess: MSOPT_OLED_POSTPROCESS is disabled")

    print(f"Runtime setup time: {time.time() - start:.2f} seconds")
