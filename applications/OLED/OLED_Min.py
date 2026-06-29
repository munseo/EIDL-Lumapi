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
# -----------------------------------------------------------------------------
theta_channel_centers_deg = np.array([0.0, 45.0])

target_angle_efficiency_ratio_min = np.array([1.0, 0.85], dtype=float)
if target_angle_efficiency_ratio_min.size != theta_channel_centers_deg.size:
    raise ValueError("target_angle_efficiency_ratio_min length must match theta_channel_centers_deg.")


target_angle_efficiency_ratio_max = np.array([1.0, 1.0], dtype=float)
if target_angle_efficiency_ratio_max.size != theta_channel_centers_deg.size:
    raise ValueError("target_angle_efficiency_ratio_max length must match theta_channel_centers_deg.")

channel_polarizations = ("x",)
polarization_angles = {"x": 0.0, "y": 90.0}
eml_components_by_polarization = {"x": ("Ex",), "y": ("Ey",)}
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

# FoM: each reciprocal channel evaluates its matching transverse EML component.
# Ez can be added as a weighted intensity term, but it is not balanced against Ex/Ey.
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
    distribution_penalty = npa.sum(
        (low_violation / (target_angle_efficiency_ratio_min + 1e-30)) ** 2
        + (high_violation / (target_angle_efficiency_ratio_max + 1e-30)) ** 2
    )
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
    distribution_penalty = np.sum(
        (low_violation / (target_angle_efficiency_ratio_min + 1e-30)) ** 2
        + (high_violation / (target_angle_efficiency_ratio_max + 1e-30)) ** 2
    )
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


def make_adjoint_loop(opt):
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
        optimizer(mapping, N_fom, make_adjoint_loop(opt))
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
        if True: # post process with incoherent dipole sampling
            def postprocess_design_array(design, beta=1.0):
                rho = np.asarray(design, dtype=float)
                if rho.size == design_cells:
                    return rho.reshape(design_grids)
                if rho.size == design_parameters:
                    return np.asarray(mapping(rho, beta), dtype=float).reshape(design_grids)
                if rho.size == Nx * Ny:
                    return np.repeat(rho.reshape(Nx, Ny)[:, :, None], Nz, axis=2)
                raise ValueError(
                    f"expected {design_cells}, {design_parameters}, or {Nx * Ny} design values, got {rho.size}"
                )

            def design_beta_for_path(path):
                basename = os.path.basename(path)
                if not (basename.startswith("ref_layer_") and basename.endswith(".txt")):
                    return 1.0
                idx = basename[len("ref_layer_"):-len(".txt")]
                param_path = os.path.join(os.path.dirname(path), f"param_{idx}.txt")
                if not os.path.exists(param_path):
                    return 1.0
                try:
                    return float(np.asarray(np.loadtxt(param_path), dtype=float).reshape(-1)[0])
                except Exception as exc:
                    print(f"[postprocess] warning: failed to read beta from {param_path}: {exc}")
                    return 1.0

            def latest_local_best_design_path():
                override = os.environ.get("MSOPT_OLED_POSTPROCESS_DESIGN", "").strip()
                if override:
                    return os.path.abspath(override)
                candidates = []
                default_path = os.path.join(design_dir, "lastdesign.txt")
                if os.path.exists(default_path):
                    candidates.append(default_path)
                if os.path.isdir(local_dir):
                    for name in os.listdir(local_dir):
                        if name.startswith("ref_layer_") and name.endswith(".txt"):
                            candidates.append(os.path.join(local_dir, name))
                if not candidates:
                    return default_path
                return max(candidates, key=lambda path: os.path.getmtime(path))

            def save_final_design_images(final_design, suffix="final", beta=1.0):
                try:
                    rho = postprocess_design_array(final_design, beta=beta)
                except ValueError as exc:
                    print(
                        f"[postprocess] skipped design image: {exc}"
                    )
                    return

                x = np.linspace(-0.5 * design_s[0], 0.5 * design_s[0], Nx)
                y = np.linspace(-0.5 * design_s[1], 0.5 * design_s[1], Ny)
                z = np.linspace(
                    design_c[2] - 0.5 * design_s[2],
                    design_c[2] + 0.5 * design_s[2],
                    Nz,
                )
                ix = Nx // 2
                iy = Ny // 2
                iz = Nz // 2

                plots = [
                    (
                        rho[:, iy, :].T,
                        (x[0], x[-1], z[0], z[-1]),
                        "x (um)",
                        "z (um)",
                        "center_y_xz",
                        "Final design x-z section at y=0",
                    ),
                    (
                        rho[ix, :, :].T,
                        (y[0], y[-1], z[0], z[-1]),
                        "y (um)",
                        "z (um)",
                        "center_x_yz",
                        "Final design y-z section at x=0",
                    ),
                    (
                        rho[:, :, iz].T,
                        (x[0], x[-1], y[0], y[-1]),
                        "x (um)",
                        "y (um)",
                        "center_z_xy",
                        "Final design x-y section at center z",
                    ),
                    (
                        np.mean(rho, axis=2).T,
                        (x[0], x[-1], y[0], y[-1]),
                        "x (um)",
                        "y (um)",
                        "z_average_xy",
                        "Final design x-y z-average",
                    ),
                ]

                for data, extent, xlabel, ylabel, name, title in plots:
                    path = os.path.join(design_dir, f"OLED_design_{suffix}_{name}.png")
                    plt.figure(figsize=(6, 5))
                    image = plt.imshow(
                        data,
                        origin="lower",
                        extent=extent,
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0,
                        aspect="auto",
                        interpolation="nearest",
                    )
                    plt.colorbar(image, label="density")
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.title(title)
                    plt.tight_layout()
                    plt.savefig(path, dpi=200)
                    plt.close()
                    print(f"[postprocess] saved final design image: {path}")

            def run_dipole_postprocess(final_design, n_samples=20, sim=None):
                try:
                    rho = postprocess_design_array(final_design)
                except ValueError as exc:
                    print(
                        f"[postprocess] skipped dipole validation: {exc}"
                    )
                    return None

                monitor_name = "postprocess_source_plane"
                pattern_resolution = int(os.environ.get("MSOPT_OLED_POSTPROCESS_ANGLE_RES", "181"))
                radial_sample_spec = os.environ.get("MSOPT_OLED_DIPOLE_RADIAL_SAMPLES", "0,0.5,0.85").strip()
                positions = []
                if radial_sample_spec:
                    active_radius = 0.5 * min(active_x, active_y)
                    for token in radial_sample_spec.split(","):
                        frac = float(token.strip())
                        radius = np.clip(frac, 0.0, 0.98) * active_radius
                        positions.append((float(radius), 0.0, float(eml_c[2])))
                else:
                    nx = int(np.ceil(np.sqrt(n_samples)))
                    ny = int(np.ceil(n_samples / nx))
                    for y in np.linspace(-0.4 * active_y, 0.4 * active_y, ny):
                        for x in np.linspace(-0.4 * active_x, 0.4 * active_x, nx):
                            positions.append((float(x), float(y), float(eml_c[2])))
                            if len(positions) == n_samples:
                                break
                        if len(positions) == n_samples:
                            break
                records = []
                angular_patterns = []

                external_sim = sim is not None
                if sim is None:
                    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
                        sim_size=[Sx, Sy, Sz],
                        resolution=resolution,
                        unit=1e-6,
                        background_index=background_index,
                        center_wl=float(np.mean(visible_wavelengths)),
                        N_f=len(visible_wavelengths),
                        bc_x="PML",
                        bc_y="PML",
                        bc_z="PML",
                    )
                    sim.src_wl = np.asarray(visible_wavelengths, dtype=float) * sim.unit
                    add_oled_stack(sim, float(np.mean(visible_wavelengths)))
                    sim.add_design_grid(
                        name="design",
                        center=design_c,
                        size=design_s,
                        index1=design_high_index,
                        index2=design_low_index,
                        design_grids=design_grids,
                        density=rho,
                        wavelength=float(np.mean(visible_wavelengths)),
                    )
                else:
                    sim.fdtd.switchtolayout()
                    for name in (
                        "source",
                        "source_0",
                        "adjoint_source",
                        "adjoint_source_0",
                        "postprocess_dipole",
                        monitor_name,
                    ):
                        delete_lumerical_object(sim.fdtd, name)
                sim.add_monitor(name=monitor_name, center=src_c, size=src_s)

                def source_plane_angular_spectrum():
                    result = sim.fdtd.getresult(monitor_name, "E")
                    E = np.asarray(result["E"])
                    x_axis_data = np.ravel(np.asarray(result["x"], dtype=float))
                    y_axis_data = np.ravel(np.asarray(result["y"], dtype=float))
                    if E.ndim < 3 or E.shape[-1] != 3:
                        raise ValueError(f"unexpected monitor E shape {E.shape}")

                    spatial_shape = E.shape[:-1]
                    x_candidates = [idx for idx, size in enumerate(spatial_shape) if size == x_axis_data.size]
                    y_candidates = [idx for idx, size in enumerate(spatial_shape) if size == y_axis_data.size]
                    if not x_candidates or not y_candidates:
                        raise ValueError(
                            f"could not map monitor axes: E shape={E.shape}, "
                            f"len(x)={x_axis_data.size}, len(y)={y_axis_data.size}"
                        )
                    x_axis_idx = x_candidates[0]
                    y_axis_idx = next((idx for idx in y_candidates if idx != x_axis_idx), y_candidates[0])
                    E = np.moveaxis(E, [x_axis_idx, y_axis_idx], [0, 1])

                    # Collapse singleton z and frequency axes, preserving vector components.
                    while E.ndim > 3:
                        E = np.mean(E, axis=2)
                    if E.shape[0] != x_axis_data.size or E.shape[1] != y_axis_data.size:
                        raise ValueError(
                            f"unexpected moved monitor E shape {E.shape}; "
                            f"expected first axes {x_axis_data.size}, {y_axis_data.size}"
                        )

                    dx = float(np.mean(np.diff(x_axis_data)))
                    dy = float(np.mean(np.diff(y_axis_data)))
                    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(x_axis_data.size, d=dx))
                    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(y_axis_data.size, d=dy))
                    spectrum = np.zeros((x_axis_data.size, y_axis_data.size), dtype=float)
                    for component_idx in range(3):
                        fft_field = np.fft.fftshift(np.fft.fft2(E[:, :, component_idx]))
                        spectrum += np.abs(fft_field) ** 2

                    KX, KY = np.meshgrid(kx, ky, indexing="ij")
                    wavelength_m = float(np.mean(visible_wavelengths)) * 1e-6
                    k0 = 2.0 * np.pi / wavelength_m
                    normalized_kr = np.sqrt(KX ** 2 + KY ** 2) / k0
                    propagating = normalized_kr <= 1.0
                    theta_abs = np.rad2deg(np.arcsin(np.clip(normalized_kr, 0.0, 1.0)))
                    phi_deg = np.rad2deg(np.arctan2(KY, KX))
                    signed_theta_x = np.rad2deg(np.arcsin(np.clip(KX / k0, -1.0, 1.0)))
                    spectrum = np.where(propagating, spectrum, 0.0)
                    return spectrum, theta_abs, phi_deg, signed_theta_x, KY / k0

                def sample_source_plane_angles():
                    spectrum, theta_abs, phi_deg, signed_theta_x, normalized_ky = source_plane_angular_spectrum()
                    angle_samples = {}
                    for angle in theta_channel_centers_deg:
                        if abs(float(angle)) < 1e-12:
                            metric = theta_abs + 1000.0 * np.abs(normalized_ky)
                        else:
                            metric = np.sqrt((theta_abs - float(angle)) ** 2 + phi_deg ** 2)
                        sample_index = np.unravel_index(np.nanargmin(metric), metric.shape)
                        angle_samples[float(angle)] = float(np.real(spectrum[sample_index]))

                    pattern_angles = np.linspace(-90.0, 90.0, pattern_resolution)
                    pattern = []
                    for angle in pattern_angles:
                        metric = np.abs(signed_theta_x - angle) + 1000.0 * np.abs(normalized_ky)
                        sample_index = np.unravel_index(np.nanargmin(metric), metric.shape)
                        pattern.append(float(np.real(spectrum[sample_index])))
                    return angle_samples, pattern_angles, np.asarray(pattern, dtype=float), float(np.sum(spectrum))

                try:
                    for polarization in channel_polarizations:
                        for sample_idx, position in enumerate(positions):
                            print(
                                f"[postprocess] dipole {polarization} sample "
                                f"{sample_idx + 1}/{len(positions)} at {position}"
                            )
                            sim.fdtd.switchtolayout()
                            x, y, z = position
                            theta, phi = {"x": (90.0, 0.0), "y": (90.0, 90.0)}[polarization]
                            delete_lumerical_object(sim.fdtd, "postprocess_dipole")
                            sim.fdtd.adddipole()
                            sim.fdtd.set("name", "postprocess_dipole")
                            sim.fdtd.set("x", x * 1e-6)
                            sim.fdtd.set("y", y * 1e-6)
                            sim.fdtd.set("z", z * 1e-6)
                            sim.fdtd.set("theta", theta)
                            sim.fdtd.set("phi", phi)
                            sim.fdtd.set("wavelength start", float(np.min(visible_wavelengths)) * 1e-6)
                            sim.fdtd.set("wavelength stop", float(np.max(visible_wavelengths)) * 1e-6)
                            sim.run(name=f"postprocess_{polarization}_{sample_idx:02d}", save=True)

                            try:
                                angle_samples, pattern_angles, pattern, transmission = sample_source_plane_angles()
                                spectrum_error = ""
                                pattern_error = ""
                            except Exception as exc:
                                angle_samples = {float(angle): np.nan for angle in theta_channel_centers_deg}
                                pattern_angles = np.linspace(-90.0, 90.0, pattern_resolution)
                                pattern = np.full(pattern_angles.shape, np.nan, dtype=float)
                                transmission = np.nan
                                spectrum_error = str(exc)
                                pattern_error = str(exc)

                            if not pattern_error:
                                angular_patterns.append((pattern_angles, pattern))
                            elif not spectrum_error:
                                spectrum_error = pattern_error
                            records.append(
                                {
                                    "polarization": polarization,
                                    "sample_idx": sample_idx,
                                    "x": position[0],
                                    "y": position[1],
                                    "z": position[2],
                                    "top_efficiency": abs(float(np.real(transmission))),
                                    "angle_samples": angle_samples,
                                    "angle_efficiencies": {
                                        angle: (
                                            angle_samples[angle] / transmission
                                            if np.isfinite(transmission) and abs(transmission) > 1e-30
                                            else np.nan
                                        )
                                        for angle in angle_samples
                                    },
                                    "spectrum_error": spectrum_error,
                                }
                            )
                finally:
                    if not external_sim:
                        try:
                            sim.fdtd.close()
                        except Exception:
                            pass

                if not records:
                    print("[postprocess] dipole validation produced no records")
                    return None

                csv_path = os.path.join(design_dir, "OLED_postprocess_dipole_samples.csv")
                angle_cols = [float(angle) for angle in theta_channel_centers_deg]
                with open(csv_path, "w", encoding="utf-8") as fp:
                    fp.write(
                        "polarization,sample_idx,x_um,y_um,z_um,source_plane_power,"
                        + ",".join(f"source_plane_{angle:g}deg" for angle in angle_cols)
                        + ","
                        + ",".join(f"source_plane_{angle:g}deg_efficiency" for angle in angle_cols)
                        + ",angular_spectrum_error\n"
                    )
                    for rec in records:
                        fp.write(
                            f"{rec['polarization']},{rec['sample_idx']},{rec['x']:.16e},"
                            f"{rec['y']:.16e},{rec['z']:.16e},{rec['top_efficiency']:.16e},"
                            + ",".join(f"{rec['angle_samples'][angle]:.16e}" for angle in angle_cols)
                            + ","
                            + ",".join(f"{rec['angle_efficiencies'][angle]:.16e}" for angle in angle_cols)
                            + f",{rec['spectrum_error']!r}\n"
                        )

                efficiencies = np.asarray([rec["top_efficiency"] for rec in records], dtype=float)
                by_pol = {
                    pol: np.asarray([rec["top_efficiency"] for rec in records if rec["polarization"] == pol], dtype=float)
                    for pol in channel_polarizations
                }
                angle_means = {}
                angle_efficiency_means = {}
                for angle in angle_cols:
                    values = np.asarray([rec["angle_samples"][angle] for rec in records], dtype=float)
                    angle_means[angle] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
                    eff_values = np.asarray([rec["angle_efficiencies"][angle] for rec in records], dtype=float)
                    angle_efficiency_means[angle] = (
                        float(np.nanmean(eff_values)) if np.any(np.isfinite(eff_values)) else np.nan
                    )

                summary_path = os.path.join(design_dir, "OLED_postprocess_dipole_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as fp:
                    fp.write("method incoherent_single_dipole_source_plane_angular_spectrum\n")
                    fp.write(f"samples_per_polarization {len(positions)}\n")
                    fp.write(f"total_samples {len(records)}\n")
                    fp.write(f"source_plane_power_mean {float(np.mean(efficiencies)):.16e}\n")
                    fp.write(f"source_plane_power_std {float(np.std(efficiencies)):.16e}\n")
                    for pol in channel_polarizations:
                        fp.write(f"source_plane_power_{pol}_mean {float(np.mean(by_pol[pol])):.16e}\n")
                        fp.write(f"source_plane_power_{pol}_std {float(np.std(by_pol[pol])):.16e}\n")
                    fp.write("source_plane_angle_samples_mean\n")
                    fp.write("theta_deg mean_angular_power mean_efficiency ratio_to_zero\n")
                    zero = angle_means.get(0.0, np.nan)
                    for angle in angle_cols:
                        ratio = angle_means[angle] / zero if np.isfinite(zero) and abs(zero) > 1e-30 else np.nan
                        fp.write(
                            f"{angle:.8g} {angle_means[angle]:.16e} "
                            f"{angle_efficiency_means[angle]:.16e} {ratio:.16e}\n"
                        )
                print(f"[postprocess] saved dipole summary: {summary_path}")
                print(f"[postprocess] saved dipole samples: {csv_path}")
                if not angular_patterns:
                    print("[postprocess] skipped angular pattern plot: no source-plane angular patterns")
                    return records

                pattern_angles = angular_patterns[0][0]
                mean_pattern = np.nanmean(np.asarray([pattern for _, pattern in angular_patterns], dtype=float), axis=0)
                if not np.any(np.isfinite(mean_pattern)):
                    print("[postprocess] skipped angular pattern plot: all source-plane angular samples are NaN")
                    return records
                mean_pattern = np.nan_to_num(mean_pattern, nan=0.0, posinf=0.0, neginf=0.0)
                normalized = mean_pattern / max(float(np.max(mean_pattern)), 1e-30)

                txt_path = os.path.join(design_dir, "OLED_postprocess_angular_pattern.txt")
                np.savetxt(
                    txt_path,
                    np.column_stack([pattern_angles, normalized, mean_pattern]),
                    header="theta_deg normalized_intensity raw_mean_intensity",
                )
                fig = plt.figure(figsize=(6, 3.6))
                ax = fig.add_subplot(111, projection="polar")
                ax.plot(np.deg2rad(pattern_angles), normalized, linewidth=2.0)
                ax.set_thetamin(-90)
                ax.set_thetamax(90)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_rlim(0.0, 1.0)
                ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_title("OLED dipole postprocess angular emission")
                ax.grid(True, alpha=0.35)
                fig.tight_layout()
                png_path = os.path.join(design_dir, "OLED_postprocess_angular_pattern.png")
                fig.savefig(png_path, dpi=200)
                plt.close(fig)
                print(f"[postprocess] saved angular pattern data: {txt_path}")
                print(f"[postprocess] saved angular pattern plot: {png_path}")
                return records

            def save_optimized_report(channel_values, combined_fom, suffix="optimized"):
                vals = np.asarray(
                    [float(np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v)) for v in channel_values],
                    dtype=float,
                )
                summary = combined_oled_summary_from_values(vals)
                angle_powers = summary["angle_powers"]
                fractions = summary["fractions"]
                ratios_to_zero = summary["ratios_to_zero"]

                metrics_path = os.path.join(design_dir, f"OLED_optimized_{suffix}.txt")
                with open(metrics_path, "w", encoding="utf-8") as fp:
                    fp.write(f"combined_fom {float(np.real(combined_fom)):.16e}\n")
                    fp.write(f"distribution_weight {target_distribution_weight:.16e}\n")
                    fp.write(f"ez_component_weight {ez_component_weight:.16e}\n")
                    fp.write(f"effective_distribution_weight {current_distribution_weight:.16e}\n")
                    fp.write(f"penalty_binarization_fraction {current_binarization_fraction:.16e}\n")
                    fp.write(f"distribution_penalty {summary['distribution_penalty']:.16e}\n")
                    fp.write("channels\n")
                    fp.write("index name theta_deg source_polarization eml_components source_power_norm reciprocal_eml_proxy\n")
                    for idx, channel in enumerate(target_channels):
                        fp.write(
                            f"{idx} {channel['name']} {channel['theta_deg']:.8g} "
                            f"{channel['polarization']} {'/'.join(channel['eml_components'])} "
                            f"{channel['source_power_norm']:.16e} "
                            f"{vals[idx]:.16e}\n"
                        )
                    fp.write("angle_summary\n")
                    fp.write("theta_deg angle_power fraction ratio_to_zero target_ratio_min target_ratio_max\n")
                    for idx, theta_deg in enumerate(theta_channel_centers_deg):
                        fp.write(
                            f"{theta_deg:.8g} {angle_powers[idx]:.16e} "
                            f"{fractions[idx]:.16e} {ratios_to_zero[idx]:.16e} "
                            f"{target_angle_efficiency_ratio_min[idx]:.16e} "
                            f"{target_angle_efficiency_ratio_max[idx]:.16e}\n"
                        )
                np.savetxt(os.path.join(design_dir, f"OLED_optimized_channel_values_{suffix}.txt"), vals)
                np.savetxt(os.path.join(design_dir, f"OLED_optimized_angle_powers_{suffix}.txt"), angle_powers)
                np.savetxt(os.path.join(design_dir, f"OLED_optimized_angle_fractions_{suffix}.txt"), fractions)
                print(f"[optimized] saved reciprocal OLED metrics: {metrics_path}")

            def close_optimization_sessions(opt, keep_indices=()):
                keep_indices = set(keep_indices)
                for idx, problem in enumerate(opt):
                    if idx in keep_indices:
                        continue
                    fdtd = getattr(getattr(problem, "sim", None), "fdtd", None)
                    if fdtd is None:
                        continue
                    try:
                        fdtd.close()
                        print(f"[postprocess] closed reciprocal FDTD session {idx}")
                    except Exception as exc:
                        print(f"[postprocess] warning: failed to close reciprocal FDTD session {idx}: {exc}")

            def postprocess_final_design(opt):
                design_path = latest_local_best_design_path()
                print(f"[postprocess] requested final-design postprocess. design_path={design_path}")
                if not os.path.exists(design_path):
                    print(f"[postprocess] skipped: final design not found at {design_path}")
                    return None
                final_design = np.loadtxt(design_path)
                design_beta = design_beta_for_path(design_path)
                save_final_design_images(final_design, suffix="optimized", beta=design_beta)
                final_design = postprocess_design_array(final_design, beta=design_beta)
                adjoint_loop = make_adjoint_loop(opt)
                combined_fom, channel_values = adjoint_loop(final_design, N_fom, Case=False)
                save_optimized_report(channel_values, combined_fom, suffix="final")
                dipole_enabled = env_flag("MSOPT_OLED_DIPOLE_POSTPROCESS", "1")
                samples = int(os.environ.get("MSOPT_OLED_DIPOLE_SAMPLES_PER_POL", "20"))
                print(
                    "[postprocess] dipole postprocess "
                    f"enabled={dipole_enabled}, samples_per_polarization={samples}, "
                    f"radial_samples={os.environ.get('MSOPT_OLED_DIPOLE_RADIAL_SAMPLES', '0,0.5,0.85')}, "
                    f"angle_resolution={os.environ.get('MSOPT_OLED_POSTPROCESS_ANGLE_RES', '181')}"
                )
                if dipole_enabled:
                    close_optimization_sessions(opt, keep_indices=(0,))
                    print("[postprocess] starting dipole postprocess")
                    try:
                        run_dipole_postprocess(final_design, n_samples=samples, sim=opt[0].sim)
                        print("[postprocess] finished dipole postprocess")
                    except Exception as exc:
                        print(f"[postprocess] dipole postprocess failed: {type(exc).__name__}: {exc}")
                        raise
                    finally:
                        close_optimization_sessions(opt)
                else:
                    close_optimization_sessions(opt)
                    print("[postprocess] skipped dipole postprocess: MSOPT_OLED_DIPOLE_POSTPROCESS is disabled")
                return combined_fom, channel_values

        postprocess_final_design(opt)
    else:
        print("[postprocess] skipped all postprocess: MSOPT_OLED_POSTPROCESS is disabled")

    print(f"Runtime setup time: {time.time() - start:.2f} seconds")
