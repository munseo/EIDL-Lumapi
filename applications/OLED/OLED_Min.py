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
grating_design_h = 0.5
ito_h = 0.2
tcta_h = 0.2
eml_h = 0.2
tpbi_h = 0.2
ag_h = 0.2
air_bot_h = 0.40
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

channel_polarizations = ("x", "y")
polarization_angles = {"x": 0.0, "y": 90.0}
eml_component_by_polarization = {"x": "Ex", "y": "Ey"}

target_distribution_weight = float(os.environ.get("MSOPT_OLED_DISTRIBUTION_WEIGHT", "10.0"))
polarization_balance_weight = float(os.environ.get("MSOPT_OLED_POL_BALANCE_WEIGHT", "10.0"))

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
                "eml_component": eml_component_by_polarization[pol],
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

polarized_channel_indices = [
    [
        next(
            idx for idx, channel in enumerate(target_channels)
            if channel["angle_idx"] == angle_idx and channel["polarization"] == pol
        )
        for pol in channel_polarizations
    ]
    for angle_idx in range(len(theta_channel_centers_deg))
]


# FoM: each reciprocal linear-polarization channel couples to the matching EML
# dipole component, then orthogonal channels are incoherently summed per angle.
uniformity_power = 1.0


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


def angle_polarization_matrix(vals):
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), channel_power_floor)
    return npa.array([[vals[idx] for idx in row] for row in polarized_channel_indices])


def combine_oled_scalar_from_values(vals):
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), channel_power_floor)
    angle_powers = angle_powers_from_channel_values(vals)
    pol_matrix = angle_polarization_matrix(vals)
    total_power = npa.maximum(npa.sum(angle_powers), channel_power_floor)
    zero_power = npa.maximum(angle_powers[0], channel_power_floor)
    ratios_to_zero = angle_powers / zero_power
    low_violation = npa.maximum(target_angle_efficiency_ratio_min - ratios_to_zero, 0.0)
    high_violation = npa.maximum(ratios_to_zero - target_angle_efficiency_ratio_max, 0.0)
    distribution_penalty = npa.sum(
        (low_violation / (target_angle_efficiency_ratio_min + 1e-30)) ** 2
        + (high_violation / (target_angle_efficiency_ratio_max + 1e-30)) ** 2
    )
    pol_penalty = 0.0
    for row in pol_matrix:
        mean_pol = npa.maximum(npa.mean(row), channel_power_floor)
        pol_penalty += npa.mean(((row - mean_pol) / mean_pol) ** 2)
    pol_penalty /= max(len(theta_channel_centers_deg), 1)
    penalty = target_distribution_weight * distribution_penalty + polarization_balance_weight * pol_penalty
    penalty_score = 1.0 / (1.0 + penalty)
    return total_power * penalty_score


def combined_oled_summary_from_values(vals):
    vals = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, channel_power_floor)
    angle_powers = np.asarray(angle_powers_from_channel_values(vals), dtype=float)
    pol_matrix = np.asarray(angle_polarization_matrix(vals), dtype=float)
    fractions = angle_powers / max(float(np.sum(angle_powers)), 1e-30)
    ratios_to_zero = angle_powers / max(float(angle_powers[0]), 1e-30)
    low_violation = np.maximum(target_angle_efficiency_ratio_min - ratios_to_zero, 0.0)
    high_violation = np.maximum(ratios_to_zero - target_angle_efficiency_ratio_max, 0.0)
    distribution_penalty = np.sum(
        (low_violation / (target_angle_efficiency_ratio_min + 1e-30)) ** 2
        + (high_violation / (target_angle_efficiency_ratio_max + 1e-30)) ** 2
    )
    pol_penalty = 0.0
    for row in pol_matrix:
        mean_pol = np.mean(row) + 1e-30
        pol_penalty += np.mean(((row - mean_pol) / mean_pol) ** 2)
    pol_penalty /= max(len(theta_channel_centers_deg), 1)
    return {
        "angle_powers": angle_powers,
        "polarization_matrix": pol_matrix,
        "fractions": fractions,
        "ratios_to_zero": ratios_to_zero,
        "distribution_penalty": float(distribution_penalty),
        "polarization_balance_penalty": float(pol_penalty),
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
        component = channel["eml_component"]

        def J_oled(E_x, E_y, E_z, channel_idx=idx, channel=channel, component=component, source_norm=source_norm):
            raw_intensity = 0.0
            uniformity = 0.0
            if component == "Ex":
                E_component = E_x
            elif component == "Ey":
                E_component = E_y
            elif component == "Ez":
                E_component = E_z
            else:
                raise ValueError(f"Unknown EML component: {component}")

            n_freq = len(visible_wavelengths) if E_component.ndim == 4 else 1
            for fidx in range(n_freq):
                Ei = E_component[:, :, :, fidx] if E_component.ndim == 4 else E_component
                Ei = npa.where(npa.isfinite(Ei), Ei, 0.0)
                intensity = npa.where(npa.isfinite(npa.abs(Ei) ** 2), npa.abs(Ei) ** 2, 0.0)

                mean_intensity = npa.mean(intensity)
                mean_intensity_sq = npa.mean(intensity ** 2)

                raw_intensity += mean_intensity
                uniformity += mean_intensity ** 2 / (mean_intensity_sq + 1e-30)
            raw_fom = raw_intensity * (uniformity + 1e-30) ** uniformity_power
            intensity_gain = raw_intensity / source_norm
            fom = npa.maximum(raw_fom / source_norm, channel_power_floor)
            fom_value = real_scalar_or_none(fom)
            if fom_value is not None:
                fom_history[channel_idx].append(fom_value)
                print(
                    f"[{boundary_label} channel {channel_idx}] {channel['name']} "
                    f"reciprocal EML proxy: {fom} "
                    f"(component={component}, mean_absE2={raw_intensity}, "
                    f"uniformity={uniformity}, uniformity_power={uniformity_power}, "
                    f"intensity_gain={intensity_gain})"
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
    Is_slanted_grating=False,
)
x0 = grating_initial_density * np.ones(Nx * Ny)
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

        f0s = [0] * N_fom
        dJ_dus = [0] * N_fom
        for idx in range(N_fom):
            if isinstance(X, str):
                f0s[idx], dJ_dus[idx] = opt[idx](need_gradient=Case)
            else:
                rho = npa.clip(X, 0.0, 1.0)
                f0s[idx], dJ_dus[idx] = opt[idx](rho_vector=[rho], need_gradient=Case)

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
        pol_matrix = summary["polarization_matrix"]
        fractions = summary["fractions"]
        ratios_to_zero = summary["ratios_to_zero"]
        for idx, channel in enumerate(target_channels):
            print(
                f"[{boundary_label}] channel={channel['name']} reciprocal EML proxy={vals[idx]} "
                f"(theta={channel['theta_deg']:.1f}, pol={channel['polarization']}, "
                f"component={channel['eml_component']}, "
                f"source_power_norm={channel['source_power_norm']:.6g})"
            )
        for angle_idx, theta_deg in enumerate(theta_channel_centers_deg):
            print(
                f"[{boundary_label}] theta={theta_deg:.1f} deg angle_power={angle_powers[angle_idx]} "
                f"fraction={fractions[angle_idx] * 100:.3f}% "
                f"ratio_to_0={ratios_to_zero[angle_idx]:.4f} "
                f"x/y={pol_matrix[angle_idx, 0] / max(pol_matrix[angle_idx, 1], 1e-30):.4f} "
                f"(target_range={target_angle_efficiency_ratio_min[angle_idx]:.4f}-"
                f"{target_angle_efficiency_ratio_max[angle_idx]:.4f})"
            )
        print(
            f"combined {boundary_label} OLED FoM: {f0} "
            f"(distribution_weight={target_distribution_weight}, "
            f"polarization_balance_weight={polarization_balance_weight})"
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
            f"component={ch['eml_component']} source_power_norm={ch['source_power_norm']:.4f}"
            for ch in target_channels
        )
    )
    print(f"N_fom={N_fom}, design_grids={design_grids}, design_cells={design_cells}")
    print(f"boundary_mode={boundary_mode}, bc_x={bc_xy}, bc_y={bc_xy}, bc_z=PML")
    print(
        "FoM=sum of source-power-normalized matching-polarization EML coupling "
        "proxies, weighted by target angular distribution and per-angle x/y balance"
    )
    print(f"visible_wavelengths={visible_wavelengths}")
    print(f"EML FoM plane center={eml_c}, size={eml_s}")
    print(
        "Postprocess settings: "
        f"MSOPT_OLED_POSTPROCESS={env_flag('MSOPT_OLED_POSTPROCESS', '1')}, "
        f"MSOPT_OLED_DIPOLE_POSTPROCESS={env_flag('MSOPT_OLED_DIPOLE_POSTPROCESS', '1')}, "
        f"MSOPT_OLED_DIPOLE_SAMPLES_PER_POL={os.environ.get('MSOPT_OLED_DIPOLE_SAMPLES_PER_POL', '20')}, "
        f"MSOPT_OLED_POSTPROCESS_FARFIELD_RES={os.environ.get('MSOPT_OLED_POSTPROCESS_FARFIELD_RES', '181')}"
    )

    optimizer = ms.Opt_MS2.OPT_Ms(
        x0,
        dJ_0,
        Born_k=99,
        Initial_LR=0.2,
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
            def save_final_design_images(final_design, suffix="final"):
                rho = np.asarray(final_design, dtype=float)
                if rho.size != design_cells:
                    print(
                        f"[postprocess] skipped design image: expected {design_cells} values, "
                        f"got {rho.size}"
                    )
                    return
                rho = rho.reshape(design_grids)

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

            def run_dipole_postprocess(final_design, n_samples=20):
                rho = np.asarray(final_design, dtype=float)
                if rho.size == design_cells:
                    rho = rho.reshape(design_grids)
                if rho.shape != tuple(design_grids):
                    print(
                        f"[postprocess] skipped dipole validation: expected design shape {design_grids}, "
                        f"got {rho.shape}"
                    )
                    return None

                monitor_name = "postprocess_top_power"
                angular_resolution = int(os.environ.get("MSOPT_OLED_POSTPROCESS_FARFIELD_RES", "181"))
                nx = int(np.ceil(np.sqrt(n_samples)))
                ny = int(np.ceil(n_samples / nx))
                positions = []
                for y in np.linspace(-0.4 * active_y, 0.4 * active_y, ny):
                    for x in np.linspace(-0.4 * active_x, 0.4 * active_x, nx):
                        positions.append((float(x), float(y), float(eml_c[2])))
                        if len(positions) == n_samples:
                            break
                    if len(positions) == n_samples:
                        break
                records = []
                angular_patterns = []

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
                sim.add_monitor(name=monitor_name, center=out_c, size=out_s)

                try:
                    for polarization in ("x", "y"):
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
                            sim.fdtd.run()

                            try:
                                value = sim.fdtd.transmission(monitor_name)
                                transmission = float(np.real(np.asarray(value, dtype=float).reshape(-1)[0]))
                            except Exception:
                                sim.fdtd.eval(f'postprocess_transmission_value = transmission("{monitor_name}");')
                                transmission = float(
                                    np.real(np.asarray(sim.fdtd.getv("postprocess_transmission_value"), dtype=float).reshape(-1)[0])
                                )

                            target_pattern_angles = np.linspace(-90.0, 90.0, 181)
                            try:
                                e2 = np.squeeze(
                                    np.asarray(
                                        sim.fdtd.farfield3d(monitor_name, 1, angular_resolution, angular_resolution),
                                        dtype=float,
                                    )
                                )
                                try:
                                    ux = np.asarray(
                                        sim.fdtd.farfieldux(monitor_name, 1, angular_resolution, angular_resolution),
                                        dtype=float,
                                    )
                                    uy = np.asarray(
                                        sim.fdtd.farfielduy(monitor_name, 1, angular_resolution, angular_resolution),
                                        dtype=float,
                                    )
                                except TypeError:
                                    ux = np.asarray(sim.fdtd.farfieldux(monitor_name, 1), dtype=float)
                                    uy = np.asarray(sim.fdtd.farfielduy(monitor_name, 1), dtype=float)
                                if ux.ndim == 1 and uy.ndim == 1:
                                    ux, uy = np.meshgrid(np.ravel(ux), np.ravel(uy), indexing="ij")

                                theta_abs = np.rad2deg(np.arcsin(np.clip(np.sqrt(ux ** 2 + uy ** 2), 0.0, 1.0)))
                                phi_deg = np.rad2deg(np.arctan2(uy, ux))
                                angle_samples = {}
                                for angle in theta_channel_centers_deg:
                                    if abs(float(angle)) < 1e-12:
                                        metric = theta_abs
                                    else:
                                        metric = np.sqrt((theta_abs - float(angle)) ** 2 + phi_deg ** 2)
                                    sample_index = np.unravel_index(np.nanargmin(metric), metric.shape)
                                    angle_samples[float(angle)] = float(np.real(e2[sample_index]))

                                signed_theta = np.rad2deg(np.arcsin(np.clip(ux, -1.0, 1.0)))
                                pattern = []
                                for angle in target_pattern_angles:
                                    metric = np.abs(signed_theta - angle) + 1000.0 * np.abs(uy)
                                    sample_index = np.unravel_index(np.nanargmin(metric), metric.shape)
                                    pattern.append(float(np.real(e2[sample_index])))
                                pattern_angles = target_pattern_angles
                                pattern = np.asarray(pattern, dtype=float)
                                farfield_error = ""
                                pattern_error = ""
                            except Exception as exc:
                                angle_samples = {float(angle): np.nan for angle in theta_channel_centers_deg}
                                pattern_angles = target_pattern_angles
                                pattern = np.full(target_pattern_angles.shape, np.nan, dtype=float)
                                farfield_error = str(exc)
                                pattern_error = str(exc)

                            if not pattern_error:
                                angular_patterns.append((pattern_angles, pattern))
                            elif not farfield_error:
                                farfield_error = pattern_error
                            records.append(
                                {
                                    "polarization": polarization,
                                    "sample_idx": sample_idx,
                                    "x": position[0],
                                    "y": position[1],
                                    "z": position[2],
                                    "top_efficiency": abs(float(np.real(transmission))),
                                    "angle_samples": angle_samples,
                                    "farfield_error": farfield_error,
                                }
                            )
                finally:
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
                        "polarization,sample_idx,x_um,y_um,z_um,top_efficiency,"
                        + ",".join(f"farfield_{angle:g}deg" for angle in angle_cols)
                        + ",farfield_error\n"
                    )
                    for rec in records:
                        fp.write(
                            f"{rec['polarization']},{rec['sample_idx']},{rec['x']:.16e},"
                            f"{rec['y']:.16e},{rec['z']:.16e},{rec['top_efficiency']:.16e},"
                            + ",".join(f"{rec['angle_samples'][angle]:.16e}" for angle in angle_cols)
                            + f",{rec['farfield_error']!r}\n"
                        )

                efficiencies = np.asarray([rec["top_efficiency"] for rec in records], dtype=float)
                by_pol = {
                    pol: np.asarray([rec["top_efficiency"] for rec in records if rec["polarization"] == pol], dtype=float)
                    for pol in ("x", "y")
                }
                angle_means = {}
                for angle in angle_cols:
                    values = np.asarray([rec["angle_samples"][angle] for rec in records], dtype=float)
                    angle_means[angle] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan

                summary_path = os.path.join(design_dir, "OLED_postprocess_dipole_summary.txt")
                with open(summary_path, "w", encoding="utf-8") as fp:
                    fp.write("method incoherent_single_dipole_average\n")
                    fp.write(f"samples_per_polarization {len(positions)}\n")
                    fp.write(f"total_samples {len(records)}\n")
                    fp.write(f"top_efficiency_mean {float(np.mean(efficiencies)):.16e}\n")
                    fp.write(f"top_efficiency_std {float(np.std(efficiencies)):.16e}\n")
                    for pol in ("x", "y"):
                        fp.write(f"top_efficiency_{pol}_mean {float(np.mean(by_pol[pol])):.16e}\n")
                        fp.write(f"top_efficiency_{pol}_std {float(np.std(by_pol[pol])):.16e}\n")
                    fp.write("farfield_angle_samples_mean\n")
                    fp.write("theta_deg mean_relative_intensity ratio_to_zero\n")
                    zero = angle_means.get(0.0, np.nan)
                    for angle in angle_cols:
                        ratio = angle_means[angle] / zero if np.isfinite(zero) and abs(zero) > 1e-30 else np.nan
                        fp.write(f"{angle:.8g} {angle_means[angle]:.16e} {ratio:.16e}\n")
                print(f"[postprocess] saved dipole summary: {summary_path}")
                print(f"[postprocess] saved dipole samples: {csv_path}")
                if not angular_patterns:
                    print("[postprocess] skipped angular pattern plot: no far-field patterns")
                    return records

                pattern_angles = angular_patterns[0][0]
                mean_pattern = np.nanmean(np.asarray([pattern for _, pattern in angular_patterns], dtype=float), axis=0)
                if not np.any(np.isfinite(mean_pattern)):
                    print("[postprocess] skipped angular pattern plot: all far-field samples are NaN")
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
                pol_matrix = summary["polarization_matrix"]
                fractions = summary["fractions"]
                ratios_to_zero = summary["ratios_to_zero"]

                metrics_path = os.path.join(design_dir, f"OLED_optimized_{suffix}.txt")
                with open(metrics_path, "w", encoding="utf-8") as fp:
                    fp.write(f"combined_fom {float(np.real(combined_fom)):.16e}\n")
                    fp.write(f"distribution_weight {target_distribution_weight:.16e}\n")
                    fp.write(f"polarization_balance_weight {polarization_balance_weight:.16e}\n")
                    fp.write(f"distribution_penalty {summary['distribution_penalty']:.16e}\n")
                    fp.write(f"polarization_balance_penalty {summary['polarization_balance_penalty']:.16e}\n")
                    fp.write("channels\n")
                    fp.write("index name theta_deg polarization eml_component source_power_norm reciprocal_eml_proxy\n")
                    for idx, channel in enumerate(target_channels):
                        fp.write(
                            f"{idx} {channel['name']} {channel['theta_deg']:.8g} "
                            f"{channel['polarization']} {channel['eml_component']} "
                            f"{channel['source_power_norm']:.16e} "
                            f"{vals[idx]:.16e}\n"
                        )
                    fp.write("angle_summary\n")
                    fp.write("theta_deg angle_power fraction ratio_to_zero x_to_y target_ratio_min target_ratio_max\n")
                    for idx, theta_deg in enumerate(theta_channel_centers_deg):
                        x_to_y = pol_matrix[idx, 0] / max(float(pol_matrix[idx, 1]), 1e-30)
                        fp.write(
                            f"{theta_deg:.8g} {angle_powers[idx]:.16e} "
                            f"{fractions[idx]:.16e} {ratios_to_zero[idx]:.16e} "
                            f"{x_to_y:.16e} "
                            f"{target_angle_efficiency_ratio_min[idx]:.16e} "
                            f"{target_angle_efficiency_ratio_max[idx]:.16e}\n"
                        )
                np.savetxt(os.path.join(design_dir, f"OLED_optimized_channel_values_{suffix}.txt"), vals)
                np.savetxt(os.path.join(design_dir, f"OLED_optimized_angle_powers_{suffix}.txt"), angle_powers)
                np.savetxt(os.path.join(design_dir, f"OLED_optimized_angle_fractions_{suffix}.txt"), fractions)
                print(f"[optimized] saved reciprocal OLED metrics: {metrics_path}")

            def postprocess_final_design(opt):
                design_path = os.path.join(design_dir, "lastdesign.txt")
                print(f"[postprocess] requested final-design postprocess. design_path={design_path}")
                if not os.path.exists(design_path):
                    print(f"[postprocess] skipped: final design not found at {design_path}")
                    return None
                final_design = np.loadtxt(design_path)
                save_final_design_images(final_design, suffix="optimized")
                if final_design.size == design_cells:
                    final_design = final_design.reshape(design_grids)
                adjoint_loop = make_adjoint_loop(opt)
                combined_fom, channel_values = adjoint_loop(final_design, N_fom, Case=False)
                save_optimized_report(channel_values, combined_fom, suffix="final")
                dipole_enabled = env_flag("MSOPT_OLED_DIPOLE_POSTPROCESS", "1")
                samples = int(os.environ.get("MSOPT_OLED_DIPOLE_SAMPLES_PER_POL", "20"))
                print(
                    "[postprocess] dipole postprocess "
                    f"enabled={dipole_enabled}, samples_per_polarization={samples}, "
                    f"farfield_resolution={os.environ.get('MSOPT_OLED_POSTPROCESS_FARFIELD_RES', '181')}"
                )
                if dipole_enabled:
                    print("[postprocess] starting dipole postprocess")
                    try:
                        run_dipole_postprocess(final_design, n_samples=samples)
                        print("[postprocess] finished dipole postprocess")
                    except Exception as exc:
                        print(f"[postprocess] dipole postprocess failed: {type(exc).__name__}: {exc}")
                        raise
                else:
                    print("[postprocess] skipped dipole postprocess: MSOPT_OLED_DIPOLE_POSTPROCESS is disabled")
                return combined_fom, channel_values

        postprocess_final_design(opt)
    else:
        print("[postprocess] skipped all postprocess: MSOPT_OLED_POSTPROCESS is disabled")

    print(f"Runtime setup time: {time.time() - start:.2f} seconds")
