import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from autograd import jacobian as ag_jacobian
from autograd import numpy as npa

import msopt as ms




# =============================================================================
# PML/Bloch-selectable OLED reciprocity optimization scaffold
# Coordinate: 3D Cartesian
# Propagation axis: z
# Boundary: Periodic in x/y and PML in z by default. Set boundary_mode below
# to "Bloch" for oblique periodic validation or "PML" for finite-window tests.
#
# Design idea:
# - Treat the OLED/pixel as a finite supercell/window.
# - Back-propagate the desired radiation pattern to a finite source plane.
# - Launch that near-field from +z toward the OLED stack.
# - Maximize local incoherent Ex/Ey/Ez dipole coupling on a 2D EML plane.
# - Use an active-pixel mask so lateral edge/PML artifacts do not dominate FoM.
#
# Bloch/PBC validation is in OLED_bloch_validation.py.
# =============================================================================


seed = 240
np.random.seed(seed)

RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
os.makedirs(design_dir, exist_ok=True)
local_dir = os.path.join(RUN_DIR, "Local_bests") + os.sep
os.makedirs(local_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Wavelength / channel setup
# -----------------------------------------------------------------------------
visible_wavelengths = np.array([0.55])
resolution = 50
bandwidth = 0.0


# -----------------------------------------------------------------------------
# Periodic 3D setup
# -----------------------------------------------------------------------------
boundary_mode = "Bloch"  # "Periodic", "Bloch", or "PML"
boundary_mode_key = boundary_mode.strip().upper()
if boundary_mode_key not in ("PML", "BLOCH", "PERIODIC"):
    raise ValueError("boundary_mode must be 'Periodic', 'Bloch', or 'PML'.")
bc_xy = {"PML": "PML", "BLOCH": "Bloch", "PERIODIC": "Periodic"}[boundary_mode_key]
boundary_label = {"PML": "PML", "BLOCH": "Bloch", "PERIODIC": "Periodic"}[boundary_mode_key]

window_x = 2.5
window_y = 2.5
active_x = window_x
active_y = window_y

air_top_h = 0.7
sio2_h = 0.3
grating_design_h = 0.25
ito_h = 0.2
tcta_h = 0.2
eml_h = 0.2
tpbi_h = 0.2
ag_h = 0.2
air_bot_h = 0.10
Sz = air_bot_h + ag_h + tpbi_h + eml_h + tcta_h + ito_h  + grating_design_h + sio2_h + air_top_h

Sx = window_x
Sy = window_y
Z_min = -0.5 * Sz
Z_max = 0.5 * Sz

grating_initial_density = 0.5
background_index = 1.0


# -----------------------------------------------------------------------------
# Layer materials from the provided 550 nm stack. Wavelength unit: um.
# -----------------------------------------------------------------------------
air_index = [1.0]
design_high_index = {
    "name": "OLED_grating_high_sampled",
    "wavelength": [0.55],
    "n": [1.45],
    "k": [0.0],
}
design_low_index = air_index
sio2_index = {
    "name": "OLED_SiO2_sampled",
    "wavelength": [0.55],
    "n": [1.45],
    "k": [0.0],
}
ito_index = {
    "name": "OLED_ITO_sampled",
    "wavelength": [0.55],
    "n": [1.7],
    "k": [0.0],
}
tcta_index = {
    "name": "OLED_TCTA_sampled",
    "wavelength": [0.55],
    "n": [1.82],
    "k": [0.0],
}
eml_index = {
    "name": "OLED_CBP_Irppy_sampled",
    "wavelength": [0.55],
    "n": [1.77],
    "k": [0.0],
}
tpbi_index = {
    "name": "OLED_TPBi_sampled",
    "wavelength": [0.55],
    "n": [1.75],
    "k": [0.0],
}
ag_index = {
    "name": "OLED_Ag_sampled",
    "wavelength": [0.55],
    "n": [0.76],
    "k": [5.9],
}



# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
layer_specs = [
    ("Ag_reflector", ag_h, ag_index),
    ("TPBi", tpbi_h, tpbi_index),
    ("CBP_Irppy_EML", eml_h, eml_index),
    ("TCTA", tcta_h, tcta_index),
    ("ITO", ito_h, ito_index),
    ("SiO2", sio2_h, sio2_index),
]

z_cursor = Z_min + air_bot_h
stack_layers = []
for layer_name, layer_h, layer_index in layer_specs:
    center = [0, 0, z_cursor + 0.5 * layer_h]
    size = [Sx, Sy, layer_h]
    stack_layers.append(
        {
            "name": layer_name,
            "center": center,
            "size": size,
            "index": layer_index,
        }
    )
    if layer_name == "CBP_Irppy_EML":
        eml_layer_c = center
        eml_layer_s = size
        eml_c = [0, 0, center[2]]
        eml_s = [active_x, active_y, 0]
    z_cursor += layer_h

design_s = [Sx, Sy, grating_design_h]
design_c = [0, 0, z_cursor + 0.5 * grating_design_h]

src_s = [Sx, Sy, 0]
src_c = [0, 0, Z_max - 0.35]

source_norm_monitor_name = "source_norm_monitor"
source_norm_s = [Sx, Sy, 0]
source_norm_c = [0, 0, src_c[2] - 0.05]

out_s = [Sx, Sy, 0]
out_c = [0, 0, Z_max - 0.15]

Nx = int(round(design_s[0] * resolution)) + 1
Ny = int(round(design_s[1] * resolution)) + 1
Nz = int(round(design_s[2] * resolution)) + 1
design_grids = [Nx, Ny, Nz]
design_cells = Nx * Ny * Nz


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

ratio_window_tol = float(os.environ.get("MSOPT_OLED_RATIO_TOL", "0.05"))
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
ez_component_weight = float(os.environ.get("MSOPT_OLED_EZ_WEIGHT", "0.0"))

target_distribution_weight = float(os.environ.get("MSOPT_OLED_DISTRIBUTION_WEIGHT", "10.0"))
relaxed_distribution_weight = float(os.environ.get("MSOPT_OLED_RELAXED_DISTRIBUTION_WEIGHT", "3.0"))
penalty_ramp_start = float(os.environ.get("MSOPT_OLED_PENALTY_RAMP_START", "0.20"))
penalty_ramp_end = float(os.environ.get("MSOPT_OLED_PENALTY_RAMP_END", "0.90"))
current_distribution_weight = target_distribution_weight
current_binarization_fraction = 1.0

channel_power_floor = float(os.environ.get("MSOPT_OLED_CHANNEL_POWER_FLOOR", "1e-12"))
unstable_candidate_fom = float(os.environ.get("MSOPT_OLED_UNSTABLE_CANDIDATE_FOM", "-1e30"))
combined_fom_history = []


def env_flag(name, default="1"):
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


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
angle_jitter_enabled = env_flag("MSOPT_OLED_ANGLE_JITTER", "1")
angle_jitter_rng = np.random.default_rng(int(os.environ.get("MSOPT_OLED_ANGLE_JITTER_SEED", "12345")))

# Azimuth/polarization sampling rides on the same redraw. The square lattice breaks
# the radial design's rotational symmetry down to C4v, so phi=0-only probes miss
# diagonal (Gamma-M) band features; phi is therefore drawn per channel inside the
# irreducible wedge [0, MSOPT_OLED_PHI_WEDGE_DEG]. One source polarization (p or s)
# is drawn per evaluation and shared by all channels so the ratio windows compare
# like with like. Enforcing the ramp per polarization is slightly stronger than the
# s+p-summed target (mainly near 45 deg); the ratio tolerance absorbs that.
phi_jitter_enabled = env_flag("MSOPT_OLED_PHI_JITTER", "1")
phi_wedge_deg = float(os.environ.get("MSOPT_OLED_PHI_WEDGE_DEG", "45.0"))
pol_jitter_enabled = env_flag("MSOPT_OLED_POL_JITTER", "1")


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
# The uniformity (IPR) factor is opt-in: the reciprocity-validated quantity is the
# plain position-averaged mean |E|^2, and any extra factor biases the FoM away from
# what the incoherent dipole verification measures.
uniformity_power = float(os.environ.get("MSOPT_OLED_UNIFORMITY_POWER", "0.0"))


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

def delete_lumerical_object(fdtd, name):
    fdtd.eval(
        f'if (getnamednumber("{name}") > 0) {{'
        f'select("{name}");'
        f'delete;'
        f'}}'
    )


def add_oled_stack(sim, wavelength):
    for layer in stack_layers:
        sim.add_geo(
            center=layer["center"],
            size=layer["size"],
            index=layer["index"],
            name=layer["name"],
            wavelength=wavelength,
        )


def build_optimization_problem():
    use_source_normalization = env_flag("MSOPT_OLED_SOURCE_NORMALIZATION", "1")
    source_norms = np.asarray(
        [channel["source_power_norm"] for channel in target_channels],
        dtype=float,
    )
    fom_history = [[] for _ in range(N_fom)]
    sim = [None] * N_fom
    opt = [None] * N_fom

    for idx, channel in enumerate(target_channels):
        sim[idx] = ms.Lumerical_utill.LumericalFDTDSimulator(
            sim_size=[Sx, Sy, Sz],
            resolution=resolution,
            unit=1e-6,
            background_index=background_index,
            center_wl=float(np.mean(visible_wavelengths)),
            N_f=len(visible_wavelengths),
            bc_x=bc_xy,
            bc_y=bc_xy,
            bc_z="PML",
        )

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
            delete_lumerical_object(sim[idx].fdtd, source_norm_monitor_name)
            print(
                f"[source normalization] channel {idx} {channel['name']}: "
                f"measured_mean_absE2={source_norms[idx]:.16e}, "
                f"analytic_cos={channel['source_power_norm']:.16e}"
            )

        add_oled_stack(sim[idx], float(np.mean(visible_wavelengths)))
        sim[idx].add_design_grid(
            name="design",
            center=design_c,
            size=design_s,
            index1=design_high_index,
            index2=design_low_index,
            design_grids=design_grids,
            density=grating_initial_density * np.ones(design_grids),
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
DR_N_info = [Nx, Ny, Nz, resolution]
radial_design_radius = float(os.environ.get("MSOPT_OLED_RADIAL_RADIUS", str(0.5 * min(design_s[0], design_s[1]))))
radial_design_grids = int(os.environ.get("MSOPT_OLED_RADIAL_GRIDS", str(int(round(radial_design_radius * resolution)) + 1)))
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
        "enabled": True,
        "N_radius": radial_design_grids,
        "radius": radial_design_radius,
        "outside_value": 0.0,
        "apply_filter": True,
        "vertical_grating": True,
    },
    Is_slanted_grating=False,
)
design_parameters = mapping.parameter_count
x0 = grating_initial_density * np.ones(design_parameters)
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
                path = os.path.join(design_dir, "design_iter_temp.png")
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


# =============================================================================
# Postprocess: incoherent-dipole far-field emission (ported from OLED_new.py)
# Reference plane-wave optimization above; here dipoles emit and we read the
# far-field angular distribution in a tiled PML window.
# =============================================================================

def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


target_monitor_name = "FoM_monitor"
angular_order_soft_sigma_deg = env_float("MSOPT_OLED_ANGULAR_ORDER_SIGMA_DEG", 5.0)
pp_far_z_um = env_float("MSOPT_OLED_PP_FAR_Z_UM", 2.0)
pp_max_angle_deg = env_float("MSOPT_OLED_PP_MAX_ANGLE_DEG", 60.0)
pp_min_tiles = env_int("MSOPT_OLED_PP_MIN_TILES", 3)
pp_resolution = env_int("MSOPT_OLED_PP_RESOLUTION", resolution)
# PP angular-target overlay = the reciprocity channel ramp itself (angle, ratio),
# not the tolerance-window edges.
target_efficiency_curve = [(float(d), float(r)) for d, r in zip(theta_channel_centers_deg, target_angle_efficiency_ratio)]
target_angle_pairs = list(target_efficiency_curve)
delete_object = delete_lumerical_object


def design_to_grid(design, beta=1.0):
    rho = np.asarray(design, dtype=float)
    if rho.size == design_cells:
        return rho.reshape(design_grids)
    if rho.size == design_parameters:
        return np.asarray(mapping(rho, beta), dtype=float).reshape(design_grids)
    if rho.size == Nx * Ny:
        return np.repeat(rho.reshape(Nx, Ny)[:, :, None], Nz, axis=2)
    raise ValueError(f"expected {design_cells}, {design_parameters}, or {Nx * Ny} design values, got {rho.size}")


def parse_curve(text):
    pts = []
    for token in text.split(","):
        parts = token.strip().split(":")
        if len(parts) == 2:
            pts.append((float(parts[0]), float(parts[1])))
    pts = pts or [(0.0, 1.0)]
    merged = {float(a): float(v) for a, v in pts}
    for a, v in list(merged.items()):
        merged.setdefault(-a, v)
    return sorted(merged.items())


def interp_curve(theta_deg, curve=target_efficiency_curve):
    theta, value = np.asarray(curve, dtype=float).T
    return float(np.interp(float(theta_deg), theta, value, left=value[0], right=value[-1]))


def dipole_angles(pol):
    pol = str(pol).strip().lower()
    if pol == "x":
        return 90.0, 0.0
    if pol == "y":
        return 90.0, 90.0
    if pol == "z":
        return 0.0, 0.0
    raise ValueError(f"Unsupported dipole polarization {pol!r}.")


def build_target_orders(wavelength_um, period_x_um, period_y_um):
    mx = os.environ.get("MSOPT_OLED_MAX_DIFFRACTION_ORDER_X") or os.environ.get("MSOPT_OLED_MAX_DIFFRACTION_ORDER")
    my = os.environ.get("MSOPT_OLED_MAX_DIFFRACTION_ORDER_Y") or os.environ.get("MSOPT_OLED_MAX_DIFFRACTION_ORDER")
    mx = max(0, int(mx)) if mx else int(np.floor(period_x_um / wavelength_um))
    my = max(0, int(my)) if my else int(np.floor(period_y_um / wavelength_um))
    orders = []
    for m in range(-mx, mx + 1):
        ux = m * wavelength_um / period_x_um
        for n in range(-my, my + 1):
            uy = n * wavelength_um / period_y_um
            s = float(np.hypot(ux, uy))
            if s <= 1.0 + 1e-12:
                theta = float(np.rad2deg(np.arcsin(min(s, 1.0))))
                orders.append(
                    {
                        "m": m,
                        "n": n,
                        "ux": float(ux),
                        "uy": float(uy),
                        "theta_deg": theta,
                        "phi_deg": 0.0 if s <= 1e-12 else float(np.rad2deg(np.arctan2(uy, ux))),
                        "efficiency": max(interp_curve(theta), 0.0),
                    }
                )
    # Multiplicity correction: many (m,n) orders share the same polar angle theta
    # (azimuthal degeneracy grows with theta). The target is a per-DIRECTION curve,
    # so split each angle's curve value across its degenerate orders. Without this
    # the summed target mass is biased toward large theta (0 deg ends up smallest).
    mult = {}
    for o in orders:
        key = round(o["theta_deg"], 3)
        mult[key] = mult.get(key, 0) + 1
    for o in orders:
        o["efficiency"] = o["efficiency"] / float(mult[round(o["theta_deg"], 3)])
    orders.sort(key=lambda v: (v["theta_deg"], v["m"], v["n"]))
    return {"orders": orders, "wavelength_um": wavelength_um, "period_x_um": period_x_um, "period_y_um": period_y_um}


def order_mask(ukx, uky, ux, uy, sigma_deg, propagating):
    sigma = max(float(np.sin(np.deg2rad(max(float(sigma_deg), 1e-9)))), 1e-9)
    w = np.exp(-0.5 * (((ukx - ux) ** 2 + (uky - uy) ** 2) ** 0.5 / sigma) ** 2)
    w = np.where(propagating, w, 0.0)
    peak = float(np.max(w))
    return w / peak if peak > 0.0 else w


def finite_sum(values):
    vals = np.real(np.asarray(values, dtype=np.complex128)).reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    return float(np.sum(vals)) if vals.size else None


def read_source_power(fdtd, freqs_hz):
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    try:
        return finite_sum(fdtd.sourcepower(freqs_hz))
    except Exception:
        fdtd.putv("msopt_sourcepower_freqs", freqs_hz)
        fdtd.eval("msopt_sourcepower_values = sourcepower(msopt_sourcepower_freqs);")
        return finite_sum(fdtd.getv("msopt_sourcepower_values"))


def read_dipole_power(fdtd, freqs_hz):
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    try:
        return finite_sum(fdtd.dipolepower(freqs_hz))
    except Exception:
        fdtd.putv("msopt_dipolepower_freqs", freqs_hz)
        fdtd.eval("msopt_dipolepower_values = dipolepower(msopt_dipolepower_freqs);")
        return finite_sum(fdtd.getv("msopt_dipolepower_values"))


def read_transmission(fdtd, monitor_name):
    try:
        return float(np.real(np.asarray(fdtd.transmission(monitor_name)).reshape(-1)[0]))
    except Exception:
        fdtd.eval(f'msopt_T = transmission("{monitor_name}");')
        return float(np.real(np.asarray(fdtd.getv("msopt_T")).reshape(-1)[0]))


def load_run_results(sim):
    # A GPU session run may leave the live session EITHER in analysis mode (results
    # already loaded) or in layout mode. Only reload the saved project when we are
    # actually in layout mode -- reloading while results are present would load the
    # PRE-RUN layout .fsp and DISCARD the results (reverting to layout mode), which is
    # exactly the "dipolepower ... you are in layout mode" failure.
    try:
        in_layout = int(sim.fdtd.layoutmode()) == 1
    except Exception:
        in_layout = True                       # unknown -> assume layout and reload
    if not in_layout:
        return                                 # results already in session; do not revert
    fsp = getattr(sim, "_last_run_fsp_path", None)
    if not fsp:
        return
    try:
        sim.fdtd.load(fsp)
    except Exception as exc:
        print(f"[run] warning: could not reload results from {fsp}: {exc}")


def source_freqs(sim):
    wl = np.asarray(getattr(sim, "src_wl", []), dtype=float).reshape(-1)
    if wl.size == 0:
        wl = visible_wavelengths * sim.unit
    return sim.c / wl


def valid_power(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > 0.0 else None


def make_sim(size, bc_x=bc_xy, bc_y=bc_xy, res=None):
    return ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=size,
        resolution=int(res or resolution),
        unit=1e-6,
        background_index=background_index,
        center_wl=float(np.mean(visible_wavelengths)),
        N_f=len(visible_wavelengths),
        bc_x=bc_x,
        bc_y=bc_y,
        bc_z="PML",
    )


def add_stack(sim, span_x=Sx, span_y=Sy, z_offset=0.0):
    # z_offset shifts the whole stack (used by the postprocess, which grows the
    # domain upward for far field and must keep the stack at the same height
    # above the bottom boundary).
    for layer in stack_layers:
        c = layer["center"]
        center = [c[0], c[1], c[2] + z_offset]
        size = [span_x, span_y, layer["size"][2]]
        sim.add_geo(center, size, layer["index"], layer["name"], float(np.mean(visible_wavelengths)))


def add_dipole(sim, x, y, z, pol, name="source", enabled=True, group_name=None):
    theta, phi = dipole_angles(pol)
    sim.fdtd.adddipole()
    if group_name is not None and name != group_name:
        sim.fdtd.eval(f'addtogroup("{group_name}");')
    sim.fdtd.set("name", name)
    sim.fdtd.set("enabled", bool(enabled))
    sim.fdtd.set("x", x * 1e-6)
    sim.fdtd.set("y", y * 1e-6)
    sim.fdtd.set("z", z * 1e-6)
    sim.fdtd.set("theta", theta)
    sim.fdtd.set("phi", phi)
    sim.fdtd.set("wavelength start", float(np.min(visible_wavelengths)) * 1e-6)
    sim.fdtd.set("wavelength stop", float(np.max(visible_wavelengths)) * 1e-6)
    sim.src_wl = visible_wavelengths.reshape(-1) * sim.unit
    sim.src_bw = 0.0


def render_xz_field_image(res, path, title, cmap="hot"):
    # |E| on an XZ-normal monitor plane -> PNG. Shared by the optimization snapshot and
    # the per-dipole postprocess emission images.
    E = np.asarray(res["E"], dtype=np.complex128)
    if E.size == 0:
        return None
    field_mag = np.sqrt(np.sum(np.abs(E) ** 2, axis=-1)) if E.shape[-1] == 3 else np.abs(E)
    field_mag = np.squeeze(field_mag)
    if field_mag.ndim == 1:
        field_mag = field_mag.reshape(1, -1)
    x = np.asarray(res.get("x", np.arange(field_mag.shape[1])), dtype=float).reshape(-1)
    z = np.asarray(res.get("z", np.arange(field_mag.shape[0])), dtype=float).reshape(-1)
    if field_mag.shape != (z.size, x.size):
        field_mag = field_mag.T if field_mag.shape == (x.size, z.size) else np.reshape(field_mag, (z.size, x.size))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    im = ax.imshow(field_mag.T, origin="lower", extent=(x[0], x[-1], z[0], z[-1]), cmap=cmap, aspect="equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, pad=0.03, label="|E|")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def monitor_spectrum(sim, monitor_name, wavelength_um, flux_sign=1.0):
    Eres, Hres = sim.fdtd.getresult(monitor_name, "E"), sim.fdtd.getresult(monitor_name, "H")
    E, H = np.asarray(Eres["E"], dtype=np.complex128), np.asarray(Hres["H"], dtype=np.complex128)
    x, y = np.ravel(np.asarray(Eres["x"], dtype=float)), np.ravel(np.asarray(Eres["y"], dtype=float))
    sx = E.shape[:-1]
    ix = next(i for i, s in enumerate(sx) if s == x.size)
    iy = next((i for i, s in enumerate(sx) if s == y.size and i != ix), ix)
    E, H = np.moveaxis(E, [ix, iy], [0, 1]), np.moveaxis(H, [ix, iy], [0, 1])
    while E.ndim > 3:
        E = np.mean(E, axis=2)
    while H.ndim > 3:
        H = np.mean(H, axis=2)
    dx, dy = float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(x.size, d=dx))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(y.size, d=dy))
    Ex, Ey = np.fft.fftshift(np.fft.fft2(E[:, :, 0])), np.fft.fftshift(np.fft.fft2(E[:, :, 1]))
    Hx, Hy = np.fft.fftshift(np.fft.fft2(H[:, :, 0])), np.fft.fftshift(np.fft.fft2(H[:, :, 1]))
    spectrum = np.maximum(0.5 * float(flux_sign) * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx)), 0.0)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k0 = 2 * np.pi / (wavelength_um * 1e-6)
    ukx, uky = KX / k0, KY / k0
    theta = np.rad2deg(np.arcsin(np.clip(np.sqrt(ukx ** 2 + uky ** 2), 0.0, 1.0)))
    return theta, spectrum, ukx, uky


def angle_profile(theta, spectrum, angles):
    centers = np.asarray(angles, dtype=float)
    edges = np.r_[0.0, 0.5 * (centers[:-1] + centers[1:]), 90.0] if centers.size > 1 else np.asarray([0.0, 90.0])
    out = []
    for i in range(centers.size):
        hi_cmp = theta <= edges[i + 1] if i == centers.size - 1 else theta < edges[i + 1]
        out.append(float(np.sum(spectrum[(theta >= edges[i]) & hi_cmp])))
    return np.asarray(out, dtype=float)


def radiance_from_spectrum(theta_grid, spectrum, angles):
    # Per-direction radiance (solid-angle Jacobian removed): ring flux / ring cells.
    ring_flux = angle_profile(theta_grid, spectrum, angles)
    ring_count = angle_profile(theta_grid, np.ones_like(spectrum), angles)
    radiance = ring_flux / np.maximum(ring_count, 1.0)
    return radiance / max(float(np.max(radiance)), 1e-30)


def signed_angle_axis(angles):
    # [-90 .. +90] axis built from a [0 .. 90] grid: mirror without duplicating 0.
    angles = np.asarray(angles, dtype=float)
    return np.r_[-angles[1:][::-1], angles]


def directional_radiance(theta_grid, ukx, spectrum, signed_angles):
    # SIGNED polar angle (sign taken from kx): +theta on the +kx half of k-space,
    # -theta on the -kx half. This keeps the real left/right emission asymmetry that an
    # off-center dipole has -- the |k|-binned angle_profile is azimuthally integrated
    # (theta = arcsin(|k|/k0)) so it cannot distinguish +kx from -kx and is symmetric
    # by construction. Returns (signed ring flux, signed per-direction radiance).
    signed = np.sign(ukx) * theta_grid
    centers = np.asarray(signed_angles, dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    edges = np.r_[centers[0] - 90.0, mids, centers[-1] + 90.0]
    flux = np.empty(centers.size)
    cnt = np.empty(centers.size)
    for i in range(centers.size):
        m = (signed >= edges[i]) & (signed < edges[i + 1])
        flux[i] = float(np.sum(spectrum[m]))
        cnt[i] = float(np.sum(m))
    return flux, flux / np.maximum(cnt, 1.0)


def render_emission_figure(angles, radiance_signed_norm, target_norm, order_thetas, achieved_share, target_share, path, label):
    # Shared 2-panel emission figure: (left) SIGNED per-direction radiance (real +/-
    # asymmetry, not a mirror) vs the symmetric target, (right) order power share.
    fig = plt.figure(figsize=(12, 4.6))
    ax0 = fig.add_subplot(121, projection="polar")
    signed = signed_angle_axis(angles)
    ax0.plot(np.deg2rad(signed), radiance_signed_norm, label=label)   # already over signed axis
    ax0.plot(np.deg2rad(signed), np.r_[target_norm[1:][::-1], target_norm], label="target")
    ax0.set_thetamin(-90)
    ax0.set_thetamax(90)
    ax0.set_theta_zero_location("N")
    ax0.set_theta_direction(-1)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Per-direction radiance vs target")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18))

    ax1 = fig.add_subplot(122)
    xpos = np.arange(len(order_thetas))
    bw = 0.4
    ax1.bar(xpos - 0.5 * bw, target_share, width=bw, label="target share", color="tab:orange")
    ax1.bar(xpos + 0.5 * bw, achieved_share, width=bw, label="achieved share", color="tab:blue")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([f"{t:.0f}" for t in order_thetas])
    ax1.set_xlabel("diffraction order angle (deg)")
    ax1.set_ylabel("power share")
    ax1.set_title("Order power share: achieved vs target")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_per_dipole_emission_plot(angles, per_dipole, path):
    """Angular emission of every postprocess dipole in one figure, one color per
    dipole keyed to its in-cell radial position, so how the emission-angle
    characteristic changes with dipole position is directly visible."""
    if not per_dipole:
        return None
    radii = np.asarray([d["r"] for d in per_dipole], dtype=float)
    lo, hi = float(np.min(radii)), float(np.max(radii))
    norm = plt.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-9)
    cmap = plt.get_cmap("viridis")
    signed = signed_angle_axis(angles)
    mean_eff = np.mean(np.stack([d["eff"] for d in per_dipole]), axis=0)
    mean_rad_s = np.mean(np.stack([d["rad_signed"] for d in per_dipole]), axis=0)

    fig = plt.figure(figsize=(14.5, 5.2))
    fig.subplots_adjust(wspace=0.32)
    ax0 = fig.add_subplot(121, projection="polar")
    ax1 = fig.add_subplot(122)
    for d in per_dipole:
        c = cmap(norm(d["r"]))
        # SIGNED radiance (real +/- asymmetry from the dipole's x-offset), not mirrored.
        shape = d["rad_signed"] / max(float(np.max(d["rad_signed"])), 1e-30)
        ax0.plot(np.deg2rad(signed), shape, color=c, lw=1.0, alpha=0.75)
        ax1.plot(angles, d["eff"], color=c, lw=1.0, alpha=0.75)
    ax0.plot(np.deg2rad(signed), mean_rad_s / max(float(np.max(mean_rad_s)), 1e-30), "k--", lw=2.0, label="mean")
    ax1.plot(angles, mean_eff, "k--", lw=2.0, label="mean over dipoles")

    ax0.set_thetamin(-90)
    ax0.set_thetamax(90)
    ax0.set_theta_zero_location("N")
    ax0.set_theta_direction(-1)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Per-dipole SIGNED radiance (each self-normalized): real +/- shape")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), fontsize=8)

    for a, _ in target_angle_pairs:
        ax1.axvline(float(a), color="tab:red", ls=":", lw=1.0)
    ax1.set_xlabel("emission angle theta (deg)")
    ax1.set_ylabel("extraction efficiency per angle bin")
    ax1.set_title(f"Per-dipole angular efficiency ({len(per_dipole)} dipoles); sum over theta = LEE")
    ax1.set_xlim(0.0, 90.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=[ax0, ax1], fraction=0.03, pad=0.04)
    cb.set_label("dipole radial position from cell center (um)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    np.savez(
        os.path.splitext(path)[0] + ".npz",
        angles=angles,
        efficiency=np.stack([d["eff"] for d in per_dipole]),
        radiance=np.stack([d["rad"] for d in per_dipole]),
        x=np.asarray([d["x"] for d in per_dipole], dtype=float),
        y=np.asarray([d["y"] for d in per_dipole], dtype=float),
        r=radii,
        lee=np.asarray([d["lee"] for d in per_dipole], dtype=float),
    )
    return path


def central_cell_dipoles(n_samples, pol="x"):
    # Same seeded positions regardless of pol, so a polarization sweep re-samples the
    # identical dipole locations and only the orientation changes.
    rng = np.random.default_rng(seed)
    pol = str(pol).strip().lower()
    return [
        (float(rng.uniform(-0.5 * active_x, 0.5 * active_x)), float(rng.uniform(-0.5 * active_y, 0.5 * active_y)), float(eml_c[2]), pol)
        for _ in range(n_samples)
    ]


def postprocess_polarizations():
    # "x,y" (or add "z") -> route that runs the incoherent single-dipole sweep once per
    # polarization; spectra add incoherently (independent emitter orientations).
    raw = os.environ.get(
        "MSOPT_OLED_POSTPROCESS_POLARIZATIONS",
        os.environ.get("MSOPT_OLED_POSTPROCESS_POLARIZATION", "x"),
    )
    pols = [p.strip().lower() for p in raw.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    for p in pols:
        if p not in ("x", "y", "z"):
            raise ValueError(f"Unsupported postprocess polarization {p!r}.")
    return pols or ["x"]


def run_postprocess(final_design):
    rho = design_to_grid(final_design)

    # --- Far-field geometry ----------------------------------------------------
    # Grow the domain upward by pp_far_z_um so the monitor sits further from the
    # emitting plane, then widen it (and the design tiling) until emission at
    # pp_max_angle_deg from the CENTRAL cell still lands on the monitor.
    post_sz = Sz + pp_far_z_um
    z_shift = -0.5 * pp_far_z_um                    # keep the stack at the same height above the bottom
    monitor_z = 0.5 * post_sz - 0.15
    emission_z = design_c[2] + z_shift
    h = max(monitor_z - emission_z, 1e-6)
    half_needed = h * float(np.tan(np.deg2rad(pp_max_angle_deg)))
    tile_n = max(int(pp_min_tiles), int(np.ceil(2.0 * half_needed / Sx)))
    if tile_n % 2 == 0:
        tile_n += 1                                  # odd -> there is a central cell
    post_sx, post_sy = tile_n * Sx, tile_n * Sy
    post_monitor_s = [post_sx, post_sy, 0.0]
    post_monitor_c = [0.0, 0.0, monitor_z]

    cells = (post_sx * pp_resolution) * (post_sy * pp_resolution) * (post_sz * pp_resolution)
    print(
        f"[postprocess] far field: +{pp_far_z_um:g}um air -> Sz={post_sz:g}um, monitor {h:.2f}um above design; "
        f"{pp_max_angle_deg:g}deg needs half-width {half_needed:.2f}um -> {tile_n}x{tile_n} tiles "
        f"({post_sx:g}x{post_sy:g}um), res={pp_resolution}, ~{cells/1e6:.0f}M cells"
    )

    sim = make_sim([post_sx, post_sy, post_sz], bc_x="PML", bc_y="PML", res=pp_resolution)
    add_stack(sim, span_x=post_sx, span_y=post_sy, z_offset=z_shift)
    half = tile_n // 2
    for ix in range(-half, half + 1):
        for iy in range(-half, half + 1):
            sim.add_design_grid(
                f"design_{ix}_{iy}",
                [ix * Sx, iy * Sy, design_c[2] + z_shift],
                design_s,
                design_high_index,
                design_low_index,
                design_grids,
                rho,
                float(np.mean(visible_wavelengths)),
            )
    sim.add_monitor(target_monitor_name, post_monitor_c, post_monitor_s)

    # Full-domain XZ field monitor so every dipole's emission can be saved as an |E|
    # cross-section image (dipole_emission_N.png). The monitor plane is re-centered on
    # each dipole's y so the slice passes through that dipole.
    pp_field_images = env_flag("MSOPT_OLED_PP_FIELD_IMAGES", "1")
    pp_xz_monitor_name = "pp_xz_field"
    if pp_field_images:
        sim.add_monitor(pp_xz_monitor_name, [0.0, 0.0, 0.0], [post_sx, 0.0, post_sz])

    angles = np.linspace(0.0, 90.0, env_int("MSOPT_OLED_POSTPROCESS_ANGLE_RES", 181))
    signed_angles = signed_angle_axis(angles)
    n_dipoles = env_int("MSOPT_OLED_POSTPROCESS_N_DIPOLES", 20)
    pols = postprocess_polarizations()
    records, spectrum_sum, per_dipole = [], None, []
    pol_spectra = {}                                   # per-polarization incoherent sum
    ukx = uky = None
    run_idx = 0
    for pol in pols:
        for i, (x, y, z, _p) in enumerate(central_cell_dipoles(n_dipoles, pol)):
            print(f"[postprocess] {tile_n}x{tile_n}/PML pol={pol} dipole {i + 1}/{n_dipoles}: x={x:.3f}, y={y:.3f}")
            sim.fdtd.switchtolayout()
            delete_object(sim.fdtd, "postprocess_dipole")
            add_dipole(sim, x, y, z + z_shift, pol, "postprocess_dipole")   # same shift as the stack
            if pp_field_images:
                sim.fdtd.setnamed(pp_xz_monitor_name, "y", y * 1e-6)        # slice through this dipole
            sim.run(name=f"postprocess_pml_{run_idx:03d}", save=True)
            run_idx += 1
            load_run_results(sim)
            if pp_field_images:
                try:
                    render_xz_field_image(
                        sim.fdtd.getresult(pp_xz_monitor_name, "E"),
                        os.path.join(design_dir, f"dipole_emission_{run_idx - 1}.png"),
                        f"dipole {run_idx - 1}  pol={pol}  x={x:.2f} y={y:.2f} um  |E|",
                    )
                except Exception as exc:
                    print(f"[postprocess] warning: field image {run_idx - 1} failed: {exc}")
            try:
                T = read_transmission(sim.fdtd, target_monitor_name)
                theta, spectrum, ukx, uky = monitor_spectrum(sim, target_monitor_name, float(np.mean(visible_wavelengths)), 1.0 if T >= 0 else -1.0)
                # Incoherent sum over BOTH position and polarization.
                spectrum_sum = spectrum.copy() if spectrum_sum is None else spectrum_sum + spectrum
                pol_spectra[pol] = spectrum.copy() if pol not in pol_spectra else pol_spectra[pol] + spectrum
                freqs = source_freqs(sim)
                src_power = read_source_power(sim.fdtd, freqs)
                dip_power = read_dipole_power(sim.fdtd, freqs)
                # Full LEE = extracted top power / total dipole emission. transmission()
                # is normalized to source power, so absolute top power = |T|*source_power;
                # total emission = dipolepower (falls back to source power if unavailable).
                src = valid_power(src_power)
                dip = valid_power(dip_power)
                total_emitted = dip or src or channel_power_floor
                top_power_abs = abs(float(T)) * (src if src is not None else total_emitted)
                lee = top_power_abs / max(total_emitted, channel_power_floor)
                records.append((run_idx - 1, x, y, z, pol, float(T), src or np.nan, dip or np.nan, top_power_abs, lee))
                # Per-dipole angular breakdown: ring flux split into absolute extraction
                # efficiency per angle bin (sums to this dipole's LEE), plus the
                # per-direction radiance (solid-angle Jacobian removed) for the shape.
                ring_flux = angle_profile(theta, spectrum, angles)
                ring_count = angle_profile(theta, np.ones_like(spectrum), angles)
                _, rad_signed = directional_radiance(theta, ukx, spectrum, signed_angles)
                per_dipole.append({
                    "idx": run_idx - 1, "x": float(x), "y": float(y), "r": float(np.hypot(x, y)), "pol": pol, "lee": float(lee),
                    "eff": ring_flux / max(float(np.sum(ring_flux)), 1e-30) * float(lee),
                    "rad": ring_flux / np.maximum(ring_count, 1.0),
                    "rad_signed": rad_signed,     # signed (+/-) radiance, real left/right shape
                })
            except Exception as exc:
                print(f"[postprocess] warning: pol={pol} dipole {i} failed: {exc}")
    print(f"[postprocess] polarizations {pols}: {len(records)} total dipole runs (incoherent sum)")

    try:
        sim.fdtd.close()
    except Exception:
        pass
    if spectrum_sum is None:
        print("[postprocess] skipped: no valid spectra")
        return

    # Per-direction radiance (solid-angle Jacobian removed). Both the |k|-binned (theta,
    # symmetric) profile and the SIGNED profile (real +/- asymmetry) are kept; the polar
    # figure uses the signed one so an asymmetric emission is no longer hidden by mirroring.
    ring_flux = angle_profile(theta, spectrum_sum, angles)
    radiance_norm = radiance_from_spectrum(theta, spectrum_sum, angles)
    _, rad_signed_sum = directional_radiance(theta, ukx, spectrum_sum, signed_angles)
    radiance_signed_norm = rad_signed_sum / max(float(np.max(rad_signed_sum)), 1e-30)
    target = np.asarray([interp_curve(a) for a in angles], dtype=float)
    target_norm = target / max(float(np.max(target)), 1e-30)

    np.savetxt(
        os.path.join(design_dir, "OLED_postprocess_3x3_angle_profile.txt"),
        np.column_stack([angles, ring_flux, radiance_norm, target_norm]),
        header="theta_deg ring_flux radiance_norm target_norm",
    )
    lee_values = np.asarray([rec[9] for rec in records], dtype=float)
    lee_values = lee_values[np.isfinite(lee_values)]
    mean_lee = float(np.mean(lee_values)) if lee_values.size else float("nan")
    min_lee = float(np.min(lee_values)) if lee_values.size else float("nan")
    max_lee = float(np.max(lee_values)) if lee_values.size else float("nan")
    print(f"[postprocess] full LEE (extracted top / total dipole power): mean={mean_lee:.6e}, min={min_lee:.6e}, max={max_lee:.6e} over {lee_values.size} dipoles")

    # Per-polarization breakdown (only when the sweep ran more than one polarization):
    # each polarization's own incoherent angular profile and mean LEE, so x vs y (vs z)
    # can be inspected separately in addition to their combined sum.
    if len(pol_spectra) > 1:
        for pol, spec in pol_spectra.items():
            pol_ring = angle_profile(theta, spec, angles)
            pol_rad = radiance_from_spectrum(theta, spec, angles)
            np.savetxt(
                os.path.join(design_dir, f"OLED_postprocess_pol_{pol}_angle_profile.txt"),
                np.column_stack([angles, pol_ring, pol_rad]),
                header="theta_deg ring_flux radiance_norm",
            )
            pol_lee = np.asarray([rec[9] for rec in records if rec[4] == pol], dtype=float)
            pol_lee = pol_lee[np.isfinite(pol_lee)]
            print(f"[postprocess] pol={pol}: mean LEE={float(np.mean(pol_lee)) if pol_lee.size else float('nan'):.6e} over {pol_lee.size} dipoles")

    with open(os.path.join(design_dir, "OLED_postprocess_3x3_records.txt"), "w", encoding="utf-8") as fp:
        fp.write("method final_3x3_array_pml_central_cell_dipoles\n")
        fp.write(f"mean_LEE {mean_lee:.8e}\n")
        fp.write(f"min_LEE {min_lee:.8e}\n")
        fp.write(f"max_LEE {max_lee:.8e}\n")
        fp.write("dipole_idx x_um y_um z_um pol top_monitor_transmission source_power dipole_power extracted_top_power LEE\n")
        for rec in records:
            fp.write("%d %.8e %.8e %.8e %s %.8e %.8e %.8e %.8e %.8e\n" % rec)

    # B: decompose the incoherent emission into diffraction orders and compare
    # order power shares against the optimization's target order shares. This is
    # the apples-to-apples comparison (the optimization matches order shares, not
    # a continuous angular curve). Orders are grouped by their polar angle.
    order_info = build_target_orders(float(np.mean(visible_wavelengths)), window_x, window_y)
    propagating = np.sqrt(ukx ** 2 + uky ** 2) <= 1.0 + 1e-12
    order_rows = {}
    for o in order_info["orders"]:
        mask = order_mask(ukx, uky, o["ux"], o["uy"], angular_order_soft_sigma_deg, propagating)
        p = float(np.sum(spectrum_sum * mask))
        key = round(float(o["theta_deg"]), 1)
        row = order_rows.setdefault(key, {"achieved": 0.0, "target": 0.0})
        row["achieved"] += p
        row["target"] += max(float(o["efficiency"]), 0.0)
    order_thetas = sorted(order_rows)
    achieved = np.asarray([order_rows[t]["achieved"] for t in order_thetas], dtype=float)
    target_eff = np.asarray([order_rows[t]["target"] for t in order_thetas], dtype=float)
    achieved_share = achieved / max(float(np.sum(achieved)), 1e-30)
    target_share = target_eff / max(float(np.sum(target_eff)), 1e-30)

    with open(os.path.join(design_dir, "OLED_postprocess_order_shares.txt"), "w", encoding="utf-8") as fp:
        fp.write("theta_deg achieved_power achieved_share target_efficiency target_share\n")
        for t, a, ash, te, tsh in zip(order_thetas, achieved, achieved_share, target_eff, target_share):
            fp.write(f"{t:.3f} {a:.6e} {ash:.6e} {te:.6e} {tsh:.6e}\n")

    path = os.path.join(design_dir, "OLED_postprocess_3x3_emission.png")
    render_emission_figure(angles, radiance_signed_norm, target_norm, order_thetas, achieved_share, target_share, path, "signed radiance")
    print(f"[postprocess] saved emission (radiance + order-share) plot: {path}")

    per_path = save_per_dipole_emission_plot(angles, per_dipole, os.path.join(design_dir, "OLED_postprocess_per_dipole_emission.png"))
    if per_path:
        print(f"[postprocess] saved per-dipole angular emission plot ({len(per_dipole)} dipoles): {per_path}")


if __name__ == "__main__":
    if os.environ.get("MSOPT_OLED_SESSION_TEST", "").lower() in ("1", "true", "yes"):
        channel_idx = int(os.environ.get("MSOPT_OLED_SESSION_TEST_CHANNEL", "0"))
        channel = target_channels[channel_idx]
        print(f"[OLED session test] channel {channel_idx}: {channel['name']}")
        print(
            f"[OLED session test] theta={channel['theta_deg']}, pol={channel['polarization']}, "
            f"source_power_norm={channel['source_power_norm']}"
        )
        raise SystemExit(0)

    start = time.time()

    sim, opt, fom_history = build_optimization_problem()
    print(f"{boundary_label} OLED reciprocity scaffold built.")
    print(
        "OLED periodic 3D freeform setup: "
        f"period={window_x}x{window_y} um, active area={active_x}x{active_y} um, "
        f"air={air_top_h} um, design={grating_design_h} um, "
        f"SiO2={sio2_h} um, ITO={ito_h} um, TCTA={tcta_h} um, "
        f"EML={eml_h} um, TPBi={tpbi_h} um, Ag={ag_h} um, "
        f"bottom_air_pad={air_bot_h} um, background_index={background_index}"
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
    print(f"boundary_mode={boundary_mode}, bc_x={bc_xy}, bc_y={bc_xy}, bc_z=PML")
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
        f"MSOPT_OLED_POSTPROCESS={env_flag('MSOPT_OLED_POSTPROCESS', '1')}, "
        f"MSOPT_OLED_POSTPROCESS_ONLY={env_flag('MSOPT_OLED_POSTPROCESS_ONLY', '0')}, "
        f"MSOPT_OLED_DIPOLE_POSTPROCESS={env_flag('MSOPT_OLED_DIPOLE_POSTPROCESS', '1')}, "
        f"MSOPT_OLED_DIPOLE_SAMPLES_PER_POL={os.environ.get('MSOPT_OLED_DIPOLE_SAMPLES_PER_POL', '20')}, "
        f"MSOPT_OLED_POSTPROCESS_ANGLE_RES={os.environ.get('MSOPT_OLED_POSTPROCESS_ANGLE_RES', '181')}"
    )

    postprocess_only = env_flag("MSOPT_OLED_POSTPROCESS_ONLY", "0")
    if postprocess_only:
        print("[optimized] skipped optimizer: MSOPT_OLED_POSTPROCESS_ONLY is enabled")
    else:
        optimizer = ms.Opt_MS2.OPT_Ms(
            x0,
            dJ_0,
            Born_k=50,
            Initial_LR=0.2,
            Raw=True,
        )
        optimizer.flag = True
        optimizer(mapping, N_fom, make_adjoint_loop(opt, sims=sim))
        np.savetxt(os.path.join(design_dir, "FoM_history.txt"), np.array(fom_history, dtype=object), fmt="%s")

        if combined_fom_history:
            values = np.asarray(combined_fom_history, dtype=float)
            np.savetxt(os.path.join(design_dir, "OLED_optimized_combined_fom_history.txt"), values)
            plt.figure(figsize=(6, 4))
            plt.plot(np.arange(1, values.size + 1), values, linewidth=1.5)
            plt.xlabel("combined FoM evaluation")
            plt.ylabel("combined FoM")
            plt.title("OLED optimized FoM curve")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            path = os.path.join(design_dir, "OLED_optimized_fom_curve.png")
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"[optimized] saved FoM curve: {path}")
        else:
            print("[optimized] skipped FoM curve: no combined FoM history")





    if env_flag("MSOPT_OLED_POSTPROCESS", "1"):
        design_path = os.environ.get("MSOPT_OLED_POSTPROCESS_DESIGN", "").strip()
        if not (design_path and os.path.exists(design_path)):
            cand = os.path.join(design_dir, "lastdesign.txt")
            design_path = cand if os.path.exists(cand) else ""
        if not design_path:
            import glob
            refs = sorted(glob.glob(os.path.join(local_dir, "ref_layer_*.txt")), key=os.path.getmtime)
            design_path = refs[-1] if refs else ""
        if design_path and os.path.exists(design_path):
            print(f"[postprocess] using design: {design_path}")
            run_postprocess(np.loadtxt(design_path))
        else:
            print("[postprocess] skipped: no design file (set MSOPT_OLED_POSTPROCESS_DESIGN or produce lastdesign.txt)")
    else:
        print("[postprocess] skipped all postprocess: MSOPT_OLED_POSTPROCESS is disabled")

    print(f"Runtime setup time: {time.time() - start:.2f} seconds")
