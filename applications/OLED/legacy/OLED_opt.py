"""OLED_opt.py -- the lab's best-method OLED outcoupling optimizer.

Best method: the RECIPROCAL ramp-target engine (from OLED_Min) run at the
1.1 um period (from the old OLED_opt), on the shared oled_common module.

Why this combination:
  * Engine (validated 2026-07-30 against the Done run 20260728_122008):
    unit-amplitude backward plane-wave probes scoring the EML-plane mean
    |Ex|^2+|Ey|^2 reproduced the incoherent-dipole verification ring shares
    (reciprocal 0.020/0.364/0.616 vs verified 0.023/0.362/0.615 at
    theta=0/30/45 deg), while the coherent 25-dipole-array FoM was ~7x off
    at 0 deg for the same design. By reciprocity, maximizing EML |E|^2 under
    a normal-incidence probe is equivalent to maximizing the incoherent
    ensemble's normal-direction radiance -- the FoM finally matches what the
    incoherent-dipole postprocess measures.
  * Period 1.1 um: at lambda=0.55 um, lambda/P=0.5, so the only propagating
    diffraction orders are 0 deg, 30 deg (1,0) and 45 deg (1,1) (plus
    grazing); everything >=2nd order is evanescent, so high-angle leakage is
    physically impossible. Period is THE dominant angular lever
    (P = lambda/sin(theta) for the (1,0) order).

Channels: theta = [0, 30, 45, 60] deg with ramp targets [1.0, 0.90, 0.85, 0].
Both reciprocal source polarizations are summed to match the postprocess x+y
incoherent horizontal-dipole ensemble. Deterministic probes are the default;
theta/phi jitter remains an explicit experimental opt-in.

The previous coherent 25-dipole-array version of this file is preserved at
legacy_20260731/OLED_opt.py.
"""

import glob
import hashlib
import json
import os
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from autograd import jacobian as ag_jacobian
    from autograd import numpy as npa
except ModuleNotFoundError:
    ag_jacobian = None
    npa = np

import msopt as ms

import oled_common as oc


# =============================================================================
# Shared configuration (oled_common). period default 1.1 um -- see header.
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
#   STACK             "microcavity" (the optimized top-emission stack, analytic
#                              outcoupling R 61.8 / G 57.3 / B 50.3 %) or "legacy"
#   MC_COLOR          red | green | blue
#   MC_STACK_KIND     "optimized" or "table" (the original layer table, which sits
#                              at cavity ANTI-resonance)
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
STACK = "microcavity"
MC_COLOR = "green"
MC_STACK_KIND = "optimized"

# Cylindrical (radial) design symmetry. True collapses the design to rho(r, z),
# 51 x 16 = 816 parameters instead of the full 101 x 101 x 16 = 163,216 -- and it
# is what makes a SINGLE reciprocal pole sufficient, because a y dipole is then
# the x one rotated by 90 deg. Turn it OFF to resume a full-freeform run.
RADIAL_DESIGN = True

# Warm start. RESUME_FROM points at a previous run's Local_bests/, RESUME_ITER at
# the saved iteration; param_<it>.txt (beta, LR) and ref_layer_<it>.txt (the design
# vector) are copied in and handed to OPT_Ms(Load=True). The design vector length
# must match this run's parameter count, so the geometry AND RADIAL_DESIGN have to
# be the same as the run being resumed -- the code checks and refuses otherwise.
# Fresh start. The 237-iteration checkpoint below belongs to a FULL-FREEFORM run
# (163,216 parameters); it cannot seed a radial run (816), so resuming is only
# possible with RADIAL_DESIGN = False.
# RESUME_FROM = "/home/eidl/Lumerical_data/Failed/20260804_035941_OLED_opt_gpu5_th30/Local_bests"
RESUME_FROM = ""
RESUME_ITER = 0

# Pick up automatically where the last compatible run left off.
#
# The FDTD engine asks the licence server for 9 tasks NO MATTER what -th is set to
# (verified: th3, th4 and th8 all request 9), so when the pool is saturated a run
# dies wherever it happens to be and no launcher tuning prevents it. What CAN be
# fixed is losing the work: on startup, scan previous runs of this script for the
# newest Local_bests checkpoint whose design vector matches this configuration's
# parameter count, and resume from it. Explicit RESUME_FROM always wins.
AUTO_RESUME = True
AUTO_RESUME_ROOT = "/home/eidl/Lumerical_data"

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

bandwidth = 0.0
boundary_label = G.bc_xy

# Reciprocal probe geometry: backward plane wave launched in the top air,
# normalization monitor just below the source (both OLED_Min values).
src_s = [G.Sx, G.Sy, 0.0]
# Reciprocal probe: the symmetric plane wave is launched on the PROBE PLANE,
# which build_config puts PROBE_GAP (0.7 um) above the design top -- exactly
# where the dipole-radiation cases put their far-field monitor. Same plane for
# both formulations, so reciprocal and dipole results refer to the same surface.
src_c = [0.0, 0.0, G.probe_plane_z]
source_norm_monitor_name = "source_norm_monitor"
source_norm_s = [G.Sx, G.Sy, 0.0]
source_norm_c = [0.0, 0.0, src_c[2] - 0.05]


# =============================================================================
# Reciprocal radiation channels (OLED_Min engine, 1.1 um channel set)
#
# At period 1.1 um the propagating orders sit exactly at the first three
# channel centers; the 60 deg channel (jitter band 55-70) only exists to
# hard-suppress the grazing band.
# =============================================================================
canonical_theta_channel_centers_deg = np.array([0.0, 30.0, 45.0, 60.0])
theta_channel_centers_deg = canonical_theta_channel_centers_deg.copy()

# Per-direction efficiency ramp: 1.0 at 0 deg falling linearly to 0.85 at
# 45 deg, suppressed (0) above 45 deg. The min/max windows are the ramp
# +- MSOPT_OLED_RATIO_TOL; suppression stays a hard zero.
target_angle_efficiency_ratio = np.array([1.0, 0.90, 0.85, 0.0], dtype=float)
if target_angle_efficiency_ratio.size != theta_channel_centers_deg.size:
    raise ValueError("target_angle_efficiency_ratio length must match theta_channel_centers_deg.")

ratio_window_tol = oc.env_float("MSOPT_OLED_RATIO_TOL", 0.05)
canonical_performance_spec = oc.make_ratio_performance_spec(
    canonical_theta_channel_centers_deg,
    target_angle_efficiency_ratio,
    ratio_window_tol,
)
target_angle_efficiency_ratio_min = canonical_performance_spec["ratio_min"].copy()
target_angle_efficiency_ratio_max = canonical_performance_spec["ratio_max"].copy()
# Keep the diffraction-order target used by the supercell verifier identical to
# the reciprocal optimization channels. Suppression channels are represented by
# the performance spec rather than a positive target order.
G.target_angle_pairs = [
    (float(a), float(r))
    for a, r in zip(canonical_theta_channel_centers_deg, target_angle_efficiency_ratio)
    if r > 0.0
]

# SINGLE POLE. The design is cylindrically symmetric (Is_radial_3d below), so an
# x- and a y-oriented horizontal dipole are the same problem rotated by 90 deg and
# the second one carries no new information -- one polarization halves the cost of
# every FoM evaluation and every adjoint.
#
# NOTE these must be TUPLES. ("x") and ("Ex") are plain strings, so iterating
# channel["eml_components"] yielded 'E' then 'x' and field_components['E'] raised
# KeyError before the first FoM ever came back.
channel_polarizations = ("x",)
polarization_angles = {"x": 0.0, "y": 90.0}
# |Ex|^2 alone is only defensible when the design is radially symmetric, where
# |Ey|^2 is its 90-deg image. Without RADIAL_DESIGN the pattern has no symmetry to
# lean on and both in-plane components have to be measured.
eml_components_by_polarization = (
    {"x": ("Ex",), "y": ("Ex", "Ey")} if RADIAL_DESIGN
    else {"x": ("Ex", "Ey"), "y": ("Ex", "Ey")}
)
ez_component_weight = oc.env_float("MSOPT_OLED_EZ_WEIGHT", 0.0)

current_binarization_fraction = 1.0
ratio_violation_scale = oc.env_float("MSOPT_OLED_RATIO_VIOLATION_SCALE", max(ratio_window_tol, 0.05))
throughput_reference = oc.env_float("MSOPT_OLED_OPT_NORMAL_REFERENCE", 1.0)
throughput_weight = oc.env_float("MSOPT_OLED_THROUGHPUT_WEIGHT", 0.10)

# Floor for the RECIPROCAL FoM channel scores (EML-plane |E|^2 sums), a pure
# divide-by-zero guard.  It must NOT share an env name with
# MSOPT_OLED_CHANNEL_POWER_FLOOR, which guards ABSOLUTE watts in the
# postprocess: this device emits ~1e-15 W per dipole, so a 1e-12 value there
# replaces every real power and destroys the LEE.
channel_power_floor = oc.env_float("MSOPT_OLED_FOM_POWER_FLOOR", 1e-12)
unstable_candidate_fom = G.unstable_candidate_fom
combined_fom_history = []

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
                "wavelengths": np.asarray(G.visible_wavelengths, dtype=float),
            }
        )
N_fom = len(target_channels)

angle_channel_indices = [
    [idx for idx, channel in enumerate(target_channels) if channel["angle_idx"] == angle_idx]
    for angle_idx in range(len(theta_channel_centers_deg))
]

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
angle_jitter_enabled = oc.env_flag("MSOPT_OLED_ANGLE_JITTER", "0")
angle_jitter_rng = np.random.default_rng(oc.env_int("MSOPT_OLED_ANGLE_JITTER_SEED", 12345))

# Azimuth/polarization sampling rides on the same redraw. The square lattice breaks
# the radial design's rotational symmetry down to C4v, so phi=0-only probes miss
# diagonal (Gamma-M) band features; phi is therefore drawn per channel inside the
# irreducible wedge [0, MSOPT_OLED_PHI_WEDGE_DEG]. One source polarization (p or s)
# is drawn per evaluation and shared by all channels so the ratio windows compare
# like with like. Enforcing the ramp per polarization is slightly stronger than the
# s+p-summed target (mainly near 45 deg); the ratio tolerance absorbs that.
phi_jitter_enabled = oc.env_flag("MSOPT_OLED_PHI_JITTER", "0")
phi_wedge_deg = oc.env_float("MSOPT_OLED_PHI_WEDGE_DEG", 45.0)
pol_jitter_enabled = oc.env_flag("MSOPT_OLED_POL_JITTER", "0")


def _parse_jitter_bands(raw):
    bands = []
    for part in raw.split(";"):
        lo, hi = (float(v) for v in part.split(",")[:2])
        bands.append((min(lo, hi), max(lo, hi)))
    return bands


# One "lo,hi" band per channel angle; a zero-width band anchors that probe.
# Bands hug the 1.1 um-period orders (0 / 30 / 45 deg); 55-70 sweeps the
# hard-suppression guard band.
angle_jitter_bands = _parse_jitter_bands(
    os.environ.get("MSOPT_OLED_ANGLE_JITTER_BANDS", "0,0;25,35;40,50;55,70")
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


# FoM: each reciprocal channel evaluates its matching transverse EML components.
# Ez can be added as a weighted intensity term, but it is not balanced against Ex/Ey.
# uniformity_power stays 0.0 here: the reciprocity-validated quantity is the plain
# position-averaged mean |E|^2 (exactly what the incoherent-dipole verification
# measures); uniformity emphasis is OLED_Min.py's role.
uniformity_power = oc.env_float("MSOPT_OLED_UNIFORMITY_POWER", 0.0)


""" FoM subfunctions (OLED_Min, verbatim) """


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


def update_oled_binarization(X):
    global current_binarization_fraction

    if isinstance(X, str):
        return current_binarization_fraction

    current_binarization_fraction = binarization_fraction_from_design(X)
    return current_binarization_fraction


def combine_oled_scalar_from_values(vals):
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), channel_power_floor)
    angle_powers = angle_powers_from_channel_values(vals)
    return oc.oled_constrained_score(
        angle_powers,
        target_angle_efficiency_ratio_min,
        target_angle_efficiency_ratio_max,
        power_floor=channel_power_floor,
        violation_scale=ratio_violation_scale,
        throughput_reference=throughput_reference,
        throughput_weight=throughput_weight,
    )


def combined_oled_summary_from_values(vals):
    vals = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, channel_power_floor)
    angle_powers = np.asarray(angle_powers_from_channel_values(vals), dtype=float)
    fractions = angle_powers / max(float(np.sum(angle_powers)), 1e-30)
    spec = {
        "ratio_min": target_angle_efficiency_ratio_min,
        "ratio_max": target_angle_efficiency_ratio_max,
        "tolerance": ratio_violation_scale,
    }
    metrics = oc.oled_performance_metrics(
        angle_powers,
        spec,
        power_floor=channel_power_floor,
        violation_scale=ratio_violation_scale,
        throughput_reference=throughput_reference,
        throughput_weight=throughput_weight,
    )
    metrics["fractions"] = fractions
    return metrics


# =============================================================================
# Optimization problem: one backward plane-wave sim per reciprocal channel
# (OLED_Min build, on oled_common helpers)
# =============================================================================

def run_source_normalization_with_retry(simulator, channel_idx):
    """Run a source-normalization solve and require its monitor result.

    Lumerical may return from a session job without raising when an external
    engine could not acquire an HPC license.  Treat the missing result as a
    retryable solver failure instead of aborting optimizer construction.
    """
    retries = max(
        1,
        int(os.environ.get("LUMERICAL_SOURCE_NORM_RESULT_RETRIES", "6")),
    )
    retry_delay = float(
        os.environ.get("LUMERICAL_SOURCE_NORM_RESULT_RETRY_DELAY", "10")
    )
    run_name = f"source_norm_{channel_idx}"
    last_error = None
    for attempt in range(retries):
        try:
            simulator.run(name=run_name, save=True)
            return simulator.fdtd.getresult(source_norm_monitor_name, "E")
        except Exception as exc:
            last_error = exc
            log_tail = simulator._run_log_tail(run_name)
            if attempt >= retries - 1:
                raise RuntimeError(
                    f"{run_name} finished without a readable "
                    f"{source_norm_monitor_name} result after {retries} attempt(s). "
                    f"Last Python error: {exc}\n"
                    f"{run_name}_p0.log tail:\n{log_tail}"
                ) from exc
            print(
                f"[source normalization] {run_name} returned without readable "
                f"monitor results; retrying in {retry_delay:g}s "
                f"({attempt + 1}/{retries - 1}). Last error: {exc}"
            )
            if log_tail:
                print(f"{run_name}_p0.log tail:\n{log_tail}")
            time.sleep(retry_delay)
            simulator.fdtd.switchtolayout()
            simulator.fdtd.load(
                getattr(
                    simulator,
                    "_last_run_fsp_path",
                    os.path.abspath(f"{run_name}.fsp"),
                )
            )
    raise last_error


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
            src_wl=G.visible_wavelengths,
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
            Eres = run_source_normalization_with_retry(sim[idx], idx)
            Eall = np.asarray(Eres["E"], dtype=np.complex128)
            score = 0.0
            if Eall.ndim >= 5:
                n_freq = min(len(G.visible_wavelengths), Eall.shape[-2])
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
            center=G.design_c,
            size=G.design_s,
            index1=G.design_high_index,
            index2=G.design_low_index,
            design_grids=G.design_grids,
            density=G.grating_initial_density * np.ones(G.design_grids),
            wavelength=float(np.mean(G.visible_wavelengths)),
        )
        sim[idx].add_design_monitor()
        sim[idx].add_monitor(name="eml_preview_monitor", center=G.eml_c, size=G.eml_s)

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
                n_freq = len(G.visible_wavelengths) if E_component.ndim == 4 else 1
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
            FoM_size=G.eml_s,
            FoM_center=G.eml_c,
            adj_fwd=True,
            opt_idx=idx,
            broadband_adjoint=True,
        )
    return sim, opt, fom_history


# =============================================================================
# Design mapping: binary full-freeform projection.  Grayscale mode is disabled
# so convergence cannot terminate on a non-manufacturable continuous density.
# =============================================================================
DR_info = [G.design_s[0], G.design_s[1], G.design_s[2], 0, 1, 2]
DR_N_info = [G.Nx, G.Ny, G.Nz, G.resolution]
radial_design_radius = oc.env_float("MSOPT_OLED_RADIAL_RADIUS", 0.5 * min(G.design_s[0], G.design_s[1]))
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
    Is_radial_3d={
        "enabled": bool(RADIAL_DESIGN),
        "N_radius": radial_design_grids,
        "radius": radial_design_radius,
        "outside_value": 0.0,
        "apply_filter": False,
        "vertical_grating": False,
    },
    # Full 3-D freeform: one variable per design cell (101 x 101 x 16 = 163,216).
    # WITHOUT it the design collapses to a single 101 x 101 reference layer that is
    # extruded over z (10,201 variables) -- a different parameterisation, and the
    # reason a resume of the binary_full_freeform run failed with
    # "cannot reshape array of size 163216 into shape (101,101)".
    # Mutually exclusive with RADIAL_DESIGN, which is the far smaller rho(r,z).
    **({} if RADIAL_DESIGN else {"Is_freeform": [True, False, False]}),
    Is_slanted_grating=False,
)
# Mapping only exposes parameter_count for the radial parameterisation; the
# full-freeform design is one variable per design-grid cell.
design_parameters = getattr(mapping, "parameter_count", G.design_cells)
print(f"[design] RADIAL_DESIGN={RADIAL_DESIGN} -> {design_parameters:,} parameters "
      f"(full grid {G.Nx} x {G.Ny} x {G.Nz} = {G.Nx * G.Ny * G.Nz:,})")
x0 = G.grating_initial_density * np.ones(design_parameters)
dJ_0 = np.zeros(G.design_cells)


def find_latest_checkpoint(n_params):
    """Newest Local_bests checkpoint from a previous run of this script whose design
    vector length matches n_params.

    Length is the compatibility test that matters: a radial run stores 816 values and
    a full-freeform one 163,216, and feeding the wrong one to OPT_Ms fails deep inside
    the mapping. Runs are searched newest-first and the first size-compatible one wins;
    incompatible ones are reported so a silent mismatch cannot look like "no checkpoint".
    """
    pats = [os.path.join(AUTO_RESUME_ROOT, d, "*OLED_opt*", "Local_bests")
            for d in ("Ongoing", "Done", "Failed")]
    cands = []
    for pat in pats:
        cands.extend(glob.glob(pat))
    # exclude this run's own (empty) directory
    cands = [c for c in cands if os.path.abspath(c) != os.path.abspath(G.local_dir)]
    cands.sort(key=lambda c: os.path.getmtime(c), reverse=True)
    skipped = []
    for c in cands:
        refs = glob.glob(os.path.join(c, "ref_layer_*.txt"))
        if not refs:
            continue
        best_it, best_f = -1, None
        for f in refs:
            try:
                it = int(os.path.basename(f).split("_")[-1][:-4])
            except ValueError:
                continue
            if it > best_it and os.path.isfile(os.path.join(c, f"param_{it}.txt")):
                best_it, best_f = it, f
        if best_f is None:
            continue
        try:
            n = np.loadtxt(best_f).size
        except Exception:
            continue
        if n == n_params:
            print(f"[auto-resume] {c} iteration {best_it} ({n:,} design values)")
            return c, best_it
        skipped.append((os.path.basename(os.path.dirname(c)), best_it, n))
    for name, it, n in skipped[:3]:
        print(f"[auto-resume] skipped {name} iter {it}: {n:,} values != {n_params:,}")
    print(f"[auto-resume] no compatible checkpoint for {n_params:,} parameters; starting fresh")
    return "", 0


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
        update_oled_binarization(X)
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
                path = oc.save_current_design_sections(G, X, mapping=mapping)
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
                else np.zeros(G.design_cells, dtype=float)
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
        if not isinstance(X, str):
            try:
                rho_report = oc.design_to_grid(G, X, mapping=mapping)
                report = {
                    "schema_version": 1,
                    "design_sha256": hashlib.sha256(
                        np.ascontiguousarray(rho_report, dtype=np.float64).tobytes()
                    ).hexdigest(),
                    "angles_deg": [float(v) for v in theta_channel_centers_deg],
                    "channel_powers": [float(v) for v in angle_powers],
                    "ratios_to_zero": [float(v) for v in ratios_to_zero],
                    "ratio_min": [float(v) for v in target_angle_efficiency_ratio_min],
                    "ratio_max": [float(v) for v in target_angle_efficiency_ratio_max],
                    "shape_score": float(summary["shape_score"]),
                    "throughput_score": float(summary["throughput_score"]),
                    "optimization_score": float(summary["score"]),
                    "all_ratio_windows_met": bool(summary["all_ratio_windows_met"]),
                }
                with open(
                    os.path.join(G.design_dir, "OLED_optimization_latest_metrics.json"),
                    "w",
                    encoding="utf-8",
                ) as fp:
                    json.dump(report, fp, indent=2, sort_keys=True)
                    fp.write("\n")
            except Exception as exc:
                print(f"[optimized] warning: performance report failed: {exc}")
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
            f"(shape_score={summary['shape_score']:.6f}, "
            f"throughput_score={summary['throughput_score']:.6f}, "
            f"ratio_windows_met={summary['all_ratio_windows_met']}, "
            f"Ez_weight={ez_component_weight:.4g}, binarization={current_binarization_fraction:.3f})"
        )

        if Case:
            if isinstance(X, str):
                return dJ_dus
            return f0, f0s, dJ_dus
        return f0, f0s

    return Adjoint_loop


# =============================================================================
# Main: session test, optimizer, history outputs, incoherent-dipole postprocess
# =============================================================================


if __name__ == "__main__":
    if oc.env_flag("MSOPT_OLED_SESSION_TEST", "0"):
        channel_idx = int(os.environ.get("MSOPT_OLED_SESSION_TEST_CHANNEL", "0"))
        channel = target_channels[channel_idx]
        oc.session_test_banner(
            G, N_fom,
            extra=f"engine=reciprocal ramp-target, period={G.window_x:g}x{G.window_y:g}um, "
                  f"design_parameters={design_parameters}",
        )
        print(f"[OLED session test] channel {channel_idx}: {channel['name']}")
        print(
            f"[OLED session test] theta={channel['theta_deg']}, pol={channel['polarization']}, "
            f"source_power_norm={channel['source_power_norm']}"
        )
        raise SystemExit(0)

    start = time.time()

    postprocess_only = oc.env_flag("MSOPT_OLED_POSTPROCESS_ONLY", "0")
    final_reciprocal_evaluator = None
    if postprocess_only:
        print("[optimized] skipped optimizer: MSOPT_OLED_POSTPROCESS_ONLY is enabled")
    else:
        if ag_jacobian is None:
            raise RuntimeError("autograd is required for optimization. Install autograd or run with MSOPT_OLED_POSTPROCESS_ONLY=1.")
        sim, opt, fom_history = build_optimization_problem()
        print(f"{boundary_label} OLED reciprocity best-method optimizer built.")
        print(
            "OLED periodic 3D radial setup: "
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
            f"N_fom={N_fom}, design_grids={G.design_grids}, design_cells={G.design_cells}, "
            f"design_parameters={design_parameters}, mapping=binary_full_freeform"
        )
        print(f"boundary_mode={G.boundary_mode}, bc_x={G.bc_xy}, bc_y={G.bc_xy}, bc_z=PML")
        print(
            "FoM = bounded shared angular-compliance score with saturated normal-channel "
            "throughput (two reciprocal source polarizations, EML-plane mean "
            "|Ex|^2+|Ey|^2, source-normalized); Ez is optional via MSOPT_OLED_EZ_WEIGHT"
        )
        print(f"visible_wavelengths={G.visible_wavelengths}")
        print(f"EML FoM plane center={G.eml_c}, size={G.eml_s}")
        print(
            f"FoM controls: uniformity_power={uniformity_power}, "
            f"Ez_weight={ez_component_weight}, ratio_tol={ratio_window_tol}, "
            f"violation_scale={ratio_violation_scale}, throughput_weight={throughput_weight}, "
            f"jitter_enabled={angle_jitter_enabled}, jitter_bands={angle_jitter_bands}"
        )

        # --- warm start ------------------------------------------------------
        # OPT_Ms(Load=True, Load_iter=N) chdirs into local_best_dir and reads
        # param_N.txt / ref_layer_N.txt, so the pair has to be copied in first.
        resume_iter = 0
        resume_src = os.environ.get("MSOPT_OLED_RESUME_FROM", RESUME_FROM or "").strip()
        if not resume_src and AUTO_RESUME:
            resume_src, resume_auto_iter = find_latest_checkpoint(design_parameters)
            if resume_src:
                os.environ.setdefault("MSOPT_OLED_RESUME_ITER", str(resume_auto_iter))
        if resume_src:
            resume_iter = int(oc.env_float("MSOPT_OLED_RESUME_ITER", RESUME_ITER or 0))
            src_par = os.path.join(resume_src, f"param_{resume_iter}.txt")
            src_ref = os.path.join(resume_src, f"ref_layer_{resume_iter}.txt")
            for f in (src_par, src_ref):
                if not os.path.isfile(f):
                    raise FileNotFoundError(f"resume source missing: {f}")
            saved = np.loadtxt(src_ref).reshape(-1)
            if saved.size != design_parameters:
                raise ValueError(
                    f"cannot resume: {src_ref} holds {saved.size} design values but this "
                    f"run has {design_parameters} parameters. The geometry and "
                    f"RADIAL_DESIGN (now {RADIAL_DESIGN}) must match the run being "
                    f"resumed -- radial collapses the design to rho(r,z)."
                )
            os.makedirs(G.local_dir, exist_ok=True)
            shutil.copy2(src_par, os.path.join(G.local_dir, f"param_{resume_iter}.txt"))
            shutil.copy2(src_ref, os.path.join(G.local_dir, f"ref_layer_{resume_iter}.txt"))
            beta0, lr0 = np.loadtxt(src_par).reshape(-1)[:2]
            print(f"[resume] iteration {resume_iter} from {resume_src}: "
                  f"{saved.size} design values, beta={beta0:.4f}, LR={lr0:.4g}, "
                  f"binarized={100.0 * np.mean((saved < 0.01) | (saved > 0.99)):.2f}%")

        optimizer = ms.Opt_MS2.OPT_Ms(
            x0,
            dJ_0,
            design_dir=G.design_dir,
            local_best_dir=G.local_dir,
            Load=bool(resume_src),
            Load_iter=resume_iter,
            Born_k=90,
            Initial_LR=oc.env_float("MSOPT_OLED_INITIAL_LR", 0.2),
            Raw=True,
        )
        optimizer.flag = True
        adjoint_loop = make_adjoint_loop(opt, sims=sim)
        optimizer(mapping, N_fom, adjoint_loop)
        final_reciprocal_evaluator = adjoint_loop
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

        try:
            oc.save_result_plots(optimizer, G.design_dir)
        except Exception as exc:
            print(f"[optimized] skipped optimizer-history plots: {exc}")

    if oc.env_flag("MSOPT_OLED_POSTPROCESS", "1"):
        # MSOPT_OLED_POSTPROCESS_DESIGN lets the postprocess re-run on a design from
        # a previous run instead of this run's lastdesign.txt; the final fallback is
        # the newest local-best snapshot.
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
            final_design = np.loadtxt(design_path)
            if (
                final_reciprocal_evaluator is not None
                and oc.env_flag("MSOPT_OLED_FINAL_RECIPROCAL_CHECK", "1")
            ):
                try:
                    print("[optimized] evaluating the exact postprocess design with reciprocal channels")
                    final_reciprocal_evaluator(final_design, N_fom, Case=False)
                except Exception as exc:
                    print(f"[optimized] warning: final reciprocal consistency evaluation failed: {exc}")
            oc.run_postprocess(
                G,
                final_design,
                mapping=mapping,
                performance_spec=canonical_performance_spec,
            )
        elif oc.planar_requested():
            oc.run_postprocess(
                G,
                final_design_planar,
                mapping=None,
                performance_spec=canonical_performance_spec,
            )
        else:
            print("[postprocess] skipped: no design file (set MSOPT_OLED_POSTPROCESS_DESIGN or produce lastdesign.txt)")
    else:
        print("[postprocess] skipped all postprocess: MSOPT_OLED_POSTPROCESS is disabled")

    print(f"Runtime setup time: {time.time() - start:.2f} seconds")
