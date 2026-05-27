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

design_dir = "./A/"
os.makedirs(design_dir, exist_ok=True)
local_dir = "./Local_bests/"
os.makedirs(local_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Wavelength / channel setup
# -----------------------------------------------------------------------------
visible_wavelengths = np.array([0.55])
visible_weights = np.ones_like(visible_wavelengths) / len(visible_wavelengths)
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

window_x = 2.0
window_y = 2.0
active_x = window_x
active_y = window_y

ag_h = 0.20
tpbi_h = 0.10
eml_h = 0.10
tcta_h = 0.10
ito_h = 0.10
sio2_base_h = 0.00
grating_design_h = 0.50
sio2_cap_h = 0.10
air_top_h = 0.70
air_bot_h = 0.40
Sz = air_bot_h + ag_h + tpbi_h + eml_h + tcta_h + ito_h + sio2_base_h + grating_design_h + sio2_cap_h + air_top_h

Sx = window_x
Sy = window_y
Z_min = -0.5 * Sz
Z_max = 0.5 * Sz

grating_initial_density = 0.5
sio2_above_ito_h = sio2_base_h + grating_design_h + sio2_cap_h

background_index = 1.0


# -----------------------------------------------------------------------------
# Layer materials from the provided 550 nm stack. Wavelength unit: um.
# -----------------------------------------------------------------------------
air_index = [1.0]
sio2_index = {
    "name": "OLED_SiO2_sampled",
    "wavelength": [0.55],
    "n": [1.4516],
    "k": [0.0],
}
ito_index = {
    "name": "OLED_ITO_sampled",
    "wavelength": [0.55],
    "n": [1.8],
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
    "n": [1.753],
    "k": [0.0],
}
ag_index = {
    "name": "OLED_Ag_sampled",
    "wavelength": [0.55],
    "n": [0.76],
    "k": [5.9],
}
design_high_index = {
    "name": "OLED_grating_high_sampled",
    "wavelength": [0.55],
    "n": [1.8],
    "k": [0.0],
}
design_low_index = sio2_index


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------
z_cursor = Z_min + air_bot_h

ag_s = [Sx, Sy, ag_h]
ag_c = [0, 0, z_cursor + 0.5 * ag_h]
z_cursor += ag_h

tpbi_s = [Sx, Sy, tpbi_h]
tpbi_c = [0, 0, z_cursor + 0.5 * tpbi_h]
z_cursor += tpbi_h

eml_layer_s = [Sx, Sy, eml_h]
eml_layer_c = [0, 0, z_cursor + 0.5 * eml_h]
eml_s = [Sx, Sy, 0]
eml_c = [0, 0, eml_layer_c[2]]
z_cursor += eml_h

tcta_s = [Sx, Sy, tcta_h]
tcta_c = [0, 0, z_cursor + 0.5 * tcta_h]
z_cursor += tcta_h

ito_s = [Sx, Sy, ito_h]
ito_c = [0, 0, z_cursor + 0.5 * ito_h]
z_cursor += ito_h

design_s = [Sx, Sy, grating_design_h]
design_c = [0, 0, z_cursor + 0.5 * grating_design_h]
z_cursor += grating_design_h

sio2_base_s = [Sx, Sy, sio2_base_h]
sio2_base_c = [0, 0, z_cursor + 0.5 * sio2_base_h]
z_cursor += sio2_base_h

sio2_cap_s = [Sx, Sy, sio2_cap_h]
sio2_cap_c = [0, 0, z_cursor + 0.5 * sio2_cap_h]

src_s = [Sx, Sy, 0]
src_c = [0, 0, Z_max - 0.35]

out_s = [Sx, Sy, 0]
out_c = [0, 0, Z_max - 0.15]

Nx = int(round(design_s[0] * resolution)) + 1
Ny = int(round(design_s[1] * resolution)) + 1
Nz = int(round(design_s[2] * resolution)) + 1
design_grids = [Nx, Ny, Nz]
design_cells = Nx * Ny * Nz


def initial_grating_density():
    return grating_initial_density * np.ones(design_grids)


def initial_lens_density():
    return initial_grating_density()


def active_pixel_mask(shape):
    nx, ny = shape[:2]
    x = np.linspace(-0.5 * Sx, 0.5 * Sx, nx)
    y = np.linspace(-0.5 * Sy, 0.5 * Sy, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask2 = (np.abs(X) <= 0.5 * active_x) & (np.abs(Y) <= 0.5 * active_y)
    while mask2.ndim < len(shape):
        mask2 = mask2[..., None]
    return np.broadcast_to(mask2, shape).astype(float)


# -----------------------------------------------------------------------------
# Reciprocal radiation channels
# -----------------------------------------------------------------------------
theta_channel_centers_deg = np.array([0.0, 45.0])
target_angle_efficiency_ratio_min = np.array([1.0, 0.85], dtype=float)
target_angle_efficiency_ratio_max = np.array([1.0, 1.0], dtype=float)
channel_polarizations = ("s", "p")
polarization_angles = {"s": 0.0, "TE": 0.0, "p": 90.0, "TM": 90.0}
target_distribution_weight = float(os.environ.get("MSOPT_OLED_DISTRIBUTION_WEIGHT", "10.0"))


def make_angular_target_channels():
    channels = []
    if target_angle_efficiency_ratio_min.size != theta_channel_centers_deg.size:
        raise ValueError(
            "target_angle_efficiency_ratio_min length must match theta_channel_centers_deg."
        )
    if target_angle_efficiency_ratio_max.size != theta_channel_centers_deg.size:
        raise ValueError(
            "target_angle_efficiency_ratio_max length must match theta_channel_centers_deg."
        )
    for angle_idx, (center_deg, min_ratio, max_ratio) in enumerate(
        zip(theta_channel_centers_deg, target_angle_efficiency_ratio_min, target_angle_efficiency_ratio_max)
    ):
        for pol in channel_polarizations:
            theta_rad = np.deg2rad(center_deg)
            channels.append(
                {
                    "name": f"theta_{center_deg:.1f}deg_{pol}",
                    "angle_idx": angle_idx,
                    "theta_deg": float(center_deg),
                    "phi_deg": 0.0,
                    "polarization": pol,
                    "polarization_angle": polarization_angles[pol],
                    "target_ratio_to_zero_min": float(min_ratio),
                    "target_ratio_to_zero_max": float(max_ratio),
                    "source_power_norm": max(float(np.cos(theta_rad)), 1e-6),
                }
            )
    return channels


base_radiation_channels = make_angular_target_channels()


def make_target_channels():
    return [dict(base, wavelengths=np.asarray(visible_wavelengths, dtype=float)) for base in base_radiation_channels]


target_channels = make_target_channels()
N_fom = len(target_channels)


# FoM: EML thin-film volume mean |E|^2 with uniformity and polarization balance.
uniformity_power = 1.0
component_balance_power = float(os.environ.get("MSOPT_OLED_COMPONENT_BALANCE_POWER", "1.0"))


def _weighted_mean_abs_e2(E_x, E_y, E_z):
    score = 0.0
    for fidx, wl_weight in enumerate(visible_weights):
        Ex_i = E_x[:, :, :, fidx] if E_x.ndim == 4 else E_x
        Ey_i = E_y[:, :, :, fidx] if E_y.ndim == 4 else E_y
        Ez_i = E_z[:, :, :, fidx] if E_z.ndim == 4 else E_z
        intensity = npa.abs(Ex_i) ** 2 + npa.abs(Ey_i) ** 2 + npa.abs(Ez_i) ** 2
        score += wl_weight * npa.mean(intensity)
    return score


def eml_isotropic_stats(E_x, E_y, E_z, eps=1e-30):
    mean_score = 0.0
    uniformity_score = 0.0
    balance_score = 0.0
    for fidx, wl_weight in enumerate(visible_weights):
        Ex_i = E_x[:, :, :, fidx] if E_x.ndim == 4 else E_x
        Ey_i = E_y[:, :, :, fidx] if E_y.ndim == 4 else E_y
        Ez_i = E_z[:, :, :, fidx] if E_z.ndim == 4 else E_z
        mask = active_pixel_mask(Ex_i.shape)
        mask_sum = npa.maximum(npa.sum(mask), 1.0)

        Ix = npa.abs(Ex_i) ** 2 * mask
        Iy = npa.abs(Ey_i) ** 2 * mask
        Iz = npa.abs(Ez_i) ** 2 * mask
        intensity = (Ix + Iy + Iz) / 3.0

        mx = npa.sum(Ix) / mask_sum
        my = npa.sum(Iy) / mask_sum
        mz = npa.sum(Iz) / mask_sum
        arithmetic = (mx + my + mz) / 3.0
        geometric = (mx * my * mz + eps) ** (1.0 / 3.0)
        component_balance = geometric / (arithmetic + eps)

        mean_intensity = npa.sum(intensity) / mask_sum
        mean_intensity_sq = npa.sum(intensity ** 2) / mask_sum
        uniformity = mean_intensity ** 2 / (mean_intensity_sq + eps)
        mean_score += wl_weight * mean_intensity
        uniformity_score += wl_weight * uniformity
        balance_score += wl_weight * component_balance
    return mean_score, uniformity_score, balance_score


def monitor_mean_abs_e2(sim, monitor_name):
    Eres = sim.fdtd.getresult(monitor_name, "E")
    Eall = np.array(Eres["E"], dtype=np.complex128)
    return float(np.real(_weighted_mean_abs_e2(Eall[..., 0], Eall[..., 1], Eall[..., 2])))


def eml_isotropic_uniform_fom(E_x, E_y, E_z):
    mean_intensity, uniformity, component_balance = eml_isotropic_stats(E_x, E_y, E_z)
    return (
        mean_intensity
        * (uniformity + 1e-30) ** uniformity_power
        * (component_balance + 1e-30) ** component_balance_power
    )


def eml_abs_e2_intensity(E_x, E_y, E_z):
    mean_intensity, _, _ = eml_isotropic_stats(E_x, E_y, E_z)
    return mean_intensity


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
    powers = []
    for angle_idx in range(len(theta_channel_centers_deg)):
        indices = [
            idx for idx, channel in enumerate(target_channels)
            if channel["angle_idx"] == angle_idx
        ]
        powers.append(npa.mean(vals[indices]))
    return npa.array(powers)


def combine_oled_scalar_from_values(vals):
    angle_powers = angle_powers_from_channel_values(vals)
    total_power = npa.sum(angle_powers) + 1e-30
    zero_power = angle_powers[0] + 1e-30
    ratios_to_zero = angle_powers / zero_power
    low_violation = npa.maximum(target_angle_efficiency_ratio_min - ratios_to_zero, 0.0)
    high_violation = npa.maximum(ratios_to_zero - target_angle_efficiency_ratio_max, 0.0)
    distribution_penalty = npa.sum(
        (low_violation / (target_angle_efficiency_ratio_min + 1e-30)) ** 2
        + (high_violation / (target_angle_efficiency_ratio_max + 1e-30)) ** 2
    )
    distribution_score = 1.0 / (1.0 + target_distribution_weight * distribution_penalty)
    return total_power * distribution_score


def combine_oled_metrics(channel_values):
    vals = flatten_channel_values(channel_values)
    return combine_oled_scalar_from_values(vals)


def combine_oled_gradients(channel_values, channel_gradients):
    vals = flatten_channel_values(channel_values)
    grads = [npa.array(grad) for grad in channel_gradients]
    coeffs = ag_jacobian(combine_oled_scalar_from_values)(vals)
    combined = 0.0
    for coeff, grad in zip(coeffs, grads):
        combined += coeff * grad
    return combined


def print_oled_metric_summary(channel_values, label):
    vals = [float(np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v)) for v in channel_values]
    angle_powers = np.asarray(angle_powers_from_channel_values(np.asarray(vals, dtype=float)), dtype=float)
    fractions = angle_powers / max(float(np.sum(angle_powers)), 1e-30)
    ratios_to_zero = angle_powers / max(float(angle_powers[0]), 1e-30)
    for idx, channel in enumerate(target_channels):
        print(
            f"{label} channel={channel['name']} reciprocal EML proxy={vals[idx]} "
            f"(theta={channel['theta_deg']:.1f}, pol={channel['polarization']}, "
            f"source_power_norm={channel['source_power_norm']:.6g})"
        )
    for angle_idx, theta_deg in enumerate(theta_channel_centers_deg):
        print(
            f"{label} theta={theta_deg:.1f} deg angle_power={angle_powers[angle_idx]} "
            f"fraction={fractions[angle_idx] * 100:.3f}% "
            f"ratio_to_0={ratios_to_zero[angle_idx]:.4f} "
            f"(target_range={target_angle_efficiency_ratio_min[angle_idx]:.4f}-"
            f"{target_angle_efficiency_ratio_max[angle_idx]:.4f})"
        )


def make_oled_fom(channel_idx, fom_history, source_norms):
    source_norm = max(float(source_norms[channel_idx]), 1e-30)

    def J_oled(E_x, E_y, E_z):
        raw_fom = eml_isotropic_uniform_fom(E_x, E_y, E_z)
        raw_intensity, uniformity, component_balance = eml_isotropic_stats(E_x, E_y, E_z)
        intensity_gain = raw_intensity / source_norm
        fom = raw_fom / source_norm
        fom_value = real_scalar_or_none(fom)
        if fom_value is not None:
            channel = target_channels[channel_idx]
            fom_history[channel_idx].append(fom_value)
            print(
                f"[{boundary_label} channel {channel_idx}] {channel['name']} "
                f"reciprocal EML proxy: {fom} "
                f"(mean_isotropic_absE2={raw_intensity}, uniformity={uniformity}, "
                f"component_balance={component_balance}, uniformity_power={uniformity_power}, "
                f"component_balance_power={component_balance_power}, source_power_norm={source_norm}, "
                f"intensity_gain={intensity_gain})"
            )
        return fom

    return J_oled


def add_oled_stack(sim, wavelength):
    sim.add_geo(
        center=ag_c,
        size=ag_s,
        index=ag_index,
        name="Ag_reflector",
        wavelength=wavelength,
    )
    sim.add_geo(
        center=tpbi_c,
        size=tpbi_s,
        index=tpbi_index,
        name="TPBi",
        wavelength=wavelength,
    )
    sim.add_geo(
        center=eml_layer_c,
        size=eml_layer_s,
        index=eml_index,
        name="CBP_Irppy_EML",
        wavelength=wavelength,
    )
    sim.add_geo(
        center=tcta_c,
        size=tcta_s,
        index=tcta_index,
        name="TCTA",
        wavelength=wavelength,
    )
    sim.add_geo(
        center=ito_c,
        size=ito_s,
        index=ito_index,
        name="ITO",
        wavelength=wavelength,
    )
    sim.add_geo(
        center=design_c,
        size=design_s,
        index=sio2_index,
        name="SiO2_grating_region_background",
        wavelength=wavelength,
    )
    if sio2_base_h > 0:
        sim.add_geo(
            center=sio2_base_c,
            size=sio2_base_s,
            index=sio2_index,
            name="SiO2_base",
            wavelength=wavelength,
        )
    sim.add_geo(
        center=sio2_cap_c,
        size=sio2_cap_s,
        index=sio2_index,
        name="SiO2_cap",
        wavelength=wavelength,
    )


def build_optimization_problem():
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

        add_oled_stack(sim[idx], float(np.mean(visible_wavelengths)))

        initial_density = initial_grating_density()
        sim[idx].add_design_grid(
            name="design",
            center=design_c,
            size=design_s,
            index1=design_high_index,
            index2=design_low_index,
            design_grids=design_grids,
            density=initial_density,
            wavelength=float(np.mean(visible_wavelengths)),
        )
        sim[idx].add_design_monitor()
        sim[idx].add_monitor(name="eml_preview_monitor", center=eml_c, size=eml_s)

        opt[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim[idx],
            objective_functions=[make_oled_fom(idx, fom_history, source_norms)],
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
            grad = combine_oled_gradients(channel_values, dJ_dus)
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

        f0 = combine_oled_metrics(f0s)
        print_oled_metric_summary(f0s, f"[{boundary_label}]")
        print(
            f"combined {boundary_label} OLED FoM: {f0} "
            f"(distribution_weight={target_distribution_weight})"
        )

        if Case:
            if isinstance(X, str):
                return dJ_dus
            return f0, f0s, dJ_dus
        return f0, f0s

    return Adjoint_loop


def save_postprocess_report(channel_values, combined_fom, suffix="final"):
    vals = np.asarray(
        [float(np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v)) for v in channel_values],
        dtype=float,
    )
    angle_powers = np.asarray(angle_powers_from_channel_values(vals), dtype=float)
    fractions = angle_powers / max(float(np.sum(angle_powers)), 1e-30)
    ratios_to_zero = angle_powers / max(float(angle_powers[0]), 1e-30)

    metrics_path = os.path.join(design_dir, f"OLED_postprocess_{suffix}.txt")
    with open(metrics_path, "w", encoding="utf-8") as fp:
        fp.write(f"combined_fom {float(np.real(combined_fom)):.16e}\n")
        fp.write(f"distribution_weight {target_distribution_weight:.16e}\n")
        fp.write("channels\n")
        fp.write("index name theta_deg polarization source_power_norm reciprocal_eml_proxy\n")
        for idx, channel in enumerate(target_channels):
            fp.write(
                f"{idx} {channel['name']} {channel['theta_deg']:.8g} "
                f"{channel['polarization']} {channel['source_power_norm']:.16e} "
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
    np.savetxt(os.path.join(design_dir, f"OLED_postprocess_channel_values_{suffix}.txt"), vals)
    np.savetxt(os.path.join(design_dir, f"OLED_postprocess_angle_powers_{suffix}.txt"), angle_powers)
    np.savetxt(os.path.join(design_dir, f"OLED_postprocess_angle_fractions_{suffix}.txt"), fractions)
    print(f"[postprocess] saved reciprocal OLED metrics: {metrics_path}")


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


def postprocess_final_design(opt):
    design_path = os.path.join(design_dir, "lastdesign.txt")
    if not os.path.exists(design_path):
        print(f"[postprocess] skipped: final design not found at {design_path}")
        return None
    final_design = np.loadtxt(design_path)
    save_final_design_images(final_design, suffix="final")
    if final_design.size == design_cells:
        final_design = final_design.reshape(design_grids)
    adjoint_loop = make_adjoint_loop(opt)
    combined_fom, channel_values = adjoint_loop(final_design, N_fom, Case=False)
    save_postprocess_report(channel_values, combined_fom, suffix="final")
    return combined_fom, channel_values


def run_session_smoke_test():
    channel_idx = int(os.environ.get("MSOPT_OLED_SESSION_TEST_CHANNEL", "0"))
    channel = target_channels[channel_idx]
    print(f"[OLED session test] channel {channel_idx}: {channel['name']}")
    print(
        f"[OLED session test] theta={channel['theta_deg']}, pol={channel['polarization']}, "
        f"source_power_norm={channel['source_power_norm']}"
    )
    return channel["source_power_norm"]


if __name__ == "__main__":
    if os.environ.get("MSOPT_OLED_SESSION_TEST", "").lower() in ("1", "true", "yes"):
        run_session_smoke_test()
        raise SystemExit(0)

    run_optimization = True
    start = time.time()

    sim, opt, fom_history = build_optimization_problem()
    print(f"{boundary_label} OLED reciprocity scaffold built.")
    print(
        "OLED periodic 3D freeform setup: "
        f"period={window_x}x{window_y} um, active area={active_x}x{active_y} um, "
        f"air={air_top_h} um, SiO2={sio2_cap_h + sio2_base_h} um, "
        f"design={grating_design_h} um, ITO={ito_h} um, TCTA={tcta_h} um, "
        f"EML={eml_h} um, TPBi={tpbi_h} um, Ag={ag_h} um, "
        f"bottom_air_pad={air_bot_h} um, background_index={background_index}"
    )
    print(
        "Target theta channels: "
        + ", ".join(
            f"{ch['name']} target_ratio_to_zero="
            f"{ch['target_ratio_to_zero_min']:.4f}-{ch['target_ratio_to_zero_max']:.4f} "
            f"source_power_norm={ch['source_power_norm']:.4f}"
            for ch in target_channels
        )
    )
    print(f"N_fom={N_fom}, design_grids={design_grids}, design_cells={design_cells}")
    print(f"boundary_mode={boundary_mode}, bc_x={bc_xy}, bc_y={bc_xy}, bc_z=PML")
    print(
        "FoM=sum of source-power-normalized reciprocal EML coupling proxies, "
        "weighted by target angular distribution match"
    )
    print(f"visible_wavelengths={visible_wavelengths}")
    print(f"EML FoM plane center={eml_c}, size={eml_s}")

    if run_optimization:
        optimizer = ms.Opt_MS2.OPT_Ms(
            x0,
            dJ_0,
            Born_k=99,
            Initial_LR=0.2,
        )
        optimizer.flag=True
        optimizer(mapping, N_fom, make_adjoint_loop(opt))
        np.savetxt(os.path.join(design_dir, "FoM_history.txt"), np.array(fom_history, dtype=object), fmt="%s")
        if os.environ.get("MSOPT_OLED_POSTPROCESS", "1").lower() in ("1", "true", "yes"):
            postprocess_final_design(opt)
    else:
        print("Set run_optimization=True to start optimization.")

    print(f"Runtime setup time: {time.time() - start:.2f} seconds")
