import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import msopt as ms


RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
os.makedirs(design_dir, exist_ok=True)


def env_flag(name, default="1"):
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


# -----------------------------------------------------------------------------
# Image-matched OLED lens benchmark
# -----------------------------------------------------------------------------
wavelength_um = env_float("MSOPT_OLED_LENS_WAVELENGTH_UM", 0.55)
resolution = env_int("MSOPT_OLED_LENS_RESOLUTION", 80)

window_x = env_float("MSOPT_OLED_LENS_WINDOW_X_UM", 8.0)
window_y = env_float("MSOPT_OLED_LENS_WINDOW_Y_UM", 8.0)
lens_diameter = env_float("MSOPT_OLED_LENS_DIAMETER_UM", 4.0)
lens_radius = 0.5 * lens_diameter
lens_index = env_float("MSOPT_OLED_LENS_INDEX", 1.8)

air_bot_h = 0.20
al_h = 0.10
tpbi_h = 0.04
eml_h = 0.03
tcta_h = 0.035
ito_h = 0.10
sio2_h = 0.10
air_top_h = env_float("MSOPT_OLED_LENS_TOP_AIR_UM", 2.0)
max_lens_h = env_float("MSOPT_OLED_LENS_MAX_HEIGHT_UM", 1.5)

Sz = air_bot_h + al_h + tpbi_h + eml_h + tcta_h + ito_h + sio2_h + max_lens_h + air_top_h
Z_min = -0.5 * Sz
Z_max = 0.5 * Sz

z_cursor = Z_min + air_bot_h
al_s = [window_x, window_y, al_h]
al_c = [0.0, 0.0, z_cursor + 0.5 * al_h]
z_cursor += al_h

tpbi_s = [window_x, window_y, tpbi_h]
tpbi_c = [0.0, 0.0, z_cursor + 0.5 * tpbi_h]
z_cursor += tpbi_h

eml_s = [window_x, window_y, eml_h]
eml_c = [0.0, 0.0, z_cursor + 0.5 * eml_h]
z_cursor += eml_h

tcta_s = [window_x, window_y, tcta_h]
tcta_c = [0.0, 0.0, z_cursor + 0.5 * tcta_h]
z_cursor += tcta_h

ito_s = [window_x, window_y, ito_h]
ito_c = [0.0, 0.0, z_cursor + 0.5 * ito_h]
z_cursor += ito_h

sio2_s = [window_x, window_y, sio2_h]
sio2_c = [0.0, 0.0, z_cursor + 0.5 * sio2_h]
sio2_top_z = z_cursor + sio2_h
z_cursor += sio2_h

lens_base_z = sio2_top_z
monitor_z = min(Z_max - 0.15, lens_base_z + max_lens_h + 0.45)
monitor_s = [window_x, window_y, 0.0]
monitor_c = [0.0, 0.0, monitor_z]
src_s = [window_x, window_y, 0.0]
src_c = [0.0, 0.0, min(Z_max - 0.25, lens_base_z + max_lens_h + 0.65)]

theta_channel_centers_deg = np.array([0.0, 45.0])
target_angle_efficiency_ratio_min = np.array([1.0, 0.85], dtype=float)
target_angle_efficiency_ratio_max = np.array([1.0, 1.0], dtype=float)
channel_polarizations = ("x", "y")
polarization_angles = {"x": 0.0, "y": 90.0}
eml_component_by_polarization = {"x": "Ex", "y": "Ey"}
target_distribution_weight = env_float("MSOPT_OLED_DISTRIBUTION_WEIGHT", 30.0)
polarization_balance_weight = env_float("MSOPT_OLED_POL_BALANCE_WEIGHT", 10.0)
channel_power_floor = env_float("MSOPT_OLED_CHANNEL_POWER_FLOOR", 1e-12)
uniformity_power = env_float("MSOPT_OLED_LENS_UNIFORMITY_POWER", 1.0)


air_index = [1.0]
sio2_index = {
    "name": "OLED_lens_SiO2_sampled",
    "wavelength": [0.55],
    "n": [1.4516],
    "k": [0.0],
}
ito_index = {
    "name": "OLED_lens_ITO_sampled",
    "wavelength": [0.55],
    "n": [1.94735],
    "k": [0.0],
}
tcta_index = {
    "name": "OLED_lens_TCTA_sampled",
    "wavelength": [0.55],
    "n": [1.791923],
    "k": [0.0],
}
eml_index = {
    "name": "OLED_lens_CBP_Irppy_sampled",
    "wavelength": [0.55],
    "n": [1.80128],
    "k": [0.0],
}
tpbi_index = {
    "name": "OLED_lens_TPBi_sampled",
    "wavelength": [0.55],
    "n": [1.75417],
    "k": [0.0],
}
al_index = {
    "name": "OLED_lens_Al_sampled",
    "wavelength": [0.55],
    "n": [0.811317],
    "k": [5.79942],
}


def lens_height_grid():
    values = os.environ.get("MSOPT_OLED_LENS_HEIGHTS_UM", "").strip()
    if values:
        return np.asarray([float(v) for v in values.replace(",", " ").split()], dtype=float)
    h_min = env_float("MSOPT_OLED_LENS_HEIGHT_MIN_UM", 0.7)
    h_max = env_float("MSOPT_OLED_LENS_HEIGHT_MAX_UM", 1.5)
    n = env_int("MSOPT_OLED_LENS_HEIGHT_POINTS", 9)
    return np.linspace(h_min, h_max, n)


def make_target_channels():
    channels = []
    for angle_idx, center_deg in enumerate(theta_channel_centers_deg):
        for pol in channel_polarizations:
            channels.append(
                {
                    "name": f"theta_{center_deg:.1f}deg_{pol}",
                    "angle_idx": angle_idx,
                    "theta_deg": float(center_deg),
                    "phi_deg": 0.0,
                    "polarization": pol,
                    "polarization_angle": polarization_angles[pol],
                    "eml_component": eml_component_by_polarization[pol],
                    "source_power_norm": max(float(np.cos(np.deg2rad(center_deg))), 1e-6),
                }
            )
    return channels


target_channels = make_target_channels()


def active_pixel_mask(shape):
    return np.ones(shape, dtype=float)


def select_eml_component(E_x, E_y, E_z, component):
    if component == "Ex":
        return E_x
    if component == "Ey":
        return E_y
    if component == "Ez":
        return E_z
    raise ValueError(f"Unknown EML component: {component}")


def eml_component_stats(E_x, E_y, E_z, component, eps=1e-30):
    Ei = select_eml_component(E_x, E_y, E_z, component)
    if Ei.ndim == 4:
        Ei = Ei[:, :, :, 0]
    Ei = np.nan_to_num(np.asarray(Ei, dtype=np.complex128), nan=0.0, posinf=0.0, neginf=0.0)
    mask = active_pixel_mask(Ei.shape)
    mask_sum = max(float(np.sum(mask)), 1.0)
    intensity = np.nan_to_num(np.abs(Ei) ** 2, nan=0.0, posinf=0.0, neginf=0.0) * mask
    mean_intensity = float(np.sum(intensity) / mask_sum)
    mean_intensity_sq = float(np.sum(intensity ** 2) / mask_sum)
    uniformity = mean_intensity ** 2 / (mean_intensity_sq + eps)
    return mean_intensity, uniformity


def reciprocal_channel_fom(E_x, E_y, E_z, channel):
    raw_intensity, uniformity = eml_component_stats(E_x, E_y, E_z, channel["eml_component"])
    raw_fom = raw_intensity * (uniformity + 1e-30) ** uniformity_power
    return max(raw_fom / channel["source_power_norm"], channel_power_floor), raw_intensity, uniformity


def angle_powers_from_channel_values(vals):
    vals = np.maximum(np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), channel_power_floor)
    powers = []
    for angle_idx in range(len(theta_channel_centers_deg)):
        indices = [idx for idx, channel in enumerate(target_channels) if channel["angle_idx"] == angle_idx]
        powers.append(float(np.sum(vals[indices])))
    return np.asarray(powers, dtype=float)


def angle_polarization_matrix(vals):
    vals = np.maximum(np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), channel_power_floor)
    rows = []
    for angle_idx in range(len(theta_channel_centers_deg)):
        row = []
        for pol in channel_polarizations:
            indices = [
                idx for idx, channel in enumerate(target_channels)
                if channel["angle_idx"] == angle_idx and channel["polarization"] == pol
            ]
            row.append(float(vals[indices[0]]))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def combine_oled_scalar_from_values(vals):
    vals = np.maximum(np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), channel_power_floor)
    angle_powers = angle_powers_from_channel_values(vals)
    total_power = max(float(np.sum(angle_powers)), channel_power_floor)
    zero_power = max(float(angle_powers[0]), channel_power_floor)
    ratios_to_zero = angle_powers / zero_power
    low_violation = np.maximum(target_angle_efficiency_ratio_min - ratios_to_zero, 0.0)
    high_violation = np.maximum(ratios_to_zero - target_angle_efficiency_ratio_max, 0.0)
    distribution_penalty = np.sum(
        (low_violation / (target_angle_efficiency_ratio_min + 1e-30)) ** 2
        + (high_violation / (target_angle_efficiency_ratio_max + 1e-30)) ** 2
    )
    pol_matrix = angle_polarization_matrix(vals)
    pol_penalty = 0.0
    for row in pol_matrix:
        mean_pol = max(float(np.mean(row)), channel_power_floor)
        pol_penalty += float(np.mean(((row - mean_pol) / mean_pol) ** 2))
    pol_penalty /= max(len(theta_channel_centers_deg), 1)
    penalty = target_distribution_weight * distribution_penalty + polarization_balance_weight * pol_penalty
    return total_power / (1.0 + penalty)


def summarize_reciprocal_values(vals):
    vals = np.maximum(np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0), channel_power_floor)
    angle_powers = angle_powers_from_channel_values(vals)
    pol_matrix = angle_polarization_matrix(vals)
    fractions = angle_powers / max(float(np.sum(angle_powers)), channel_power_floor)
    ratios_to_zero = angle_powers / max(float(angle_powers[0]), channel_power_floor)
    return angle_powers, pol_matrix, fractions, ratios_to_zero


def spherical_cap_radius(aperture_radius, height):
    return (aperture_radius ** 2 + height ** 2) / max(2.0 * height, 1e-30)


def lens_profile_z(x, y, height):
    rr = np.sqrt(x[:, None] ** 2 + y[None, :] ** 2)
    R = spherical_cap_radius(lens_radius, height)
    z = height - (R - np.sqrt(np.maximum(R ** 2 - rr ** 2, 0.0)))
    return np.where(rr <= lens_radius, np.maximum(z, 0.0), 0.0)


def add_oled_stack(sim):
    sim.add_geo(center=al_c, size=al_s, index=al_index, name="Al_reflector", wavelength=wavelength_um)
    sim.add_geo(center=tpbi_c, size=tpbi_s, index=tpbi_index, name="TPBi", wavelength=wavelength_um)
    sim.add_geo(center=eml_c, size=eml_s, index=eml_index, name="CBP_Irppy_EML", wavelength=wavelength_um)
    sim.add_geo(center=tcta_c, size=tcta_s, index=tcta_index, name="TCTA", wavelength=wavelength_um)
    sim.add_geo(center=ito_c, size=ito_s, index=ito_index, name="ITO", wavelength=wavelength_um)
    sim.add_geo(center=sio2_c, size=sio2_s, index=sio2_index, name="SiO2_spacer", wavelength=wavelength_um)


def add_lens_import(sim, height, name="SiO2_lens"):
    dx = env_float("MSOPT_OLED_LENS_IMPORT_DX_UM", 1.0 / resolution)
    dz = env_float("MSOPT_OLED_LENS_IMPORT_DZ_UM", 1.0 / resolution)
    nx = int(round(lens_diameter / dx)) + 1
    ny = nx
    nz = int(round(max(height, dz) / dz)) + 1
    x = np.linspace(-lens_radius, lens_radius, nx) * sim.unit
    y = np.linspace(-lens_radius, lens_radius, ny) * sim.unit
    z = (lens_base_z + np.linspace(0.0, max(height, dz), nz)) * sim.unit

    profile = lens_profile_z(x / sim.unit, y / sim.unit, height)
    z_rel = np.linspace(0.0, max(height, dz), nz)
    inside = z_rel[None, None, :] <= profile[:, :, None]
    n_geo = np.ones((nx, ny, nz, 3), dtype=float)
    n_geo[inside, :] = lens_index

    sim.fdtd.putv("x_lens_geo", np.asarray(x, dtype=float))
    sim.fdtd.putv("y_lens_geo", np.asarray(y, dtype=float))
    sim.fdtd.putv("z_lens_geo", np.asarray(z, dtype=float))
    sim.fdtd.putv("n_lens_geo", np.ascontiguousarray(n_geo))
    sim.fdtd.eval(
        f'if (getnamednumber("{name}") > 0) {{select("{name}"); delete;}}'
        f'addimport; set("name","{name}");'
        f'importnk2(n_lens_geo, x_lens_geo, y_lens_geo, z_lens_geo);'
    )
    return x / sim.unit, y / sim.unit, z / sim.unit, inside


def add_dipole(fdtd, position, polarization):
    x, y, z = position
    theta, phi = {"x": (90.0, 0.0), "y": (90.0, 90.0)}[polarization]
    fdtd.eval('if (getnamednumber("sweep_dipole") > 0) {select("sweep_dipole"); delete;}')
    fdtd.adddipole()
    fdtd.set("name", "sweep_dipole")
    fdtd.set("x", x * 1e-6)
    fdtd.set("y", y * 1e-6)
    fdtd.set("z", z * 1e-6)
    fdtd.set("theta", theta)
    fdtd.set("phi", phi)
    fdtd.set("wavelength start", wavelength_um * 1e-6)
    fdtd.set("wavelength stop", wavelength_um * 1e-6)


def sample_positions(n_samples):
    nx = int(np.ceil(np.sqrt(n_samples)))
    ny = int(np.ceil(n_samples / nx))
    xs = np.linspace(-0.35 * lens_diameter, 0.35 * lens_diameter, nx)
    ys = np.linspace(-0.35 * lens_diameter, 0.35 * lens_diameter, ny)
    out = []
    for yy in ys:
        for xx in xs:
            out.append((float(xx), float(yy), float(eml_c[2])))
            if len(out) == n_samples:
                return out
    return out


def get_transmission(fdtd, monitor_name):
    try:
        value = fdtd.transmission(monitor_name)
        return float(np.real(np.asarray(value).reshape(-1)[0]))
    except Exception:
        fdtd.eval(f'_oled_lens_T = transmission("{monitor_name}");')
        return float(np.real(np.asarray(fdtd.getv("_oled_lens_T")).reshape(-1)[0]))


def farfield_samples(fdtd, monitor_name, angles_deg, resolution_samples):
    try:
        e2 = np.squeeze(np.asarray(fdtd.farfield3d(monitor_name, 1, resolution_samples, resolution_samples), dtype=float))
        try:
            ux = np.asarray(fdtd.farfieldux(monitor_name, 1, resolution_samples, resolution_samples), dtype=float)
            uy = np.asarray(fdtd.farfielduy(monitor_name, 1, resolution_samples, resolution_samples), dtype=float)
        except TypeError:
            ux = np.asarray(fdtd.farfieldux(monitor_name, 1), dtype=float)
            uy = np.asarray(fdtd.farfielduy(monitor_name, 1), dtype=float)
        if ux.ndim == 1 and uy.ndim == 1:
            ux, uy = np.meshgrid(np.ravel(ux), np.ravel(uy), indexing="ij")
        theta = np.rad2deg(np.arcsin(np.clip(np.sqrt(ux ** 2 + uy ** 2), 0.0, 1.0)))
        phi = np.rad2deg(np.arctan2(uy, ux))
        out = {}
        for angle in angles_deg:
            if abs(float(angle)) < 1e-12:
                metric = theta
            else:
                metric = np.sqrt((theta - float(angle)) ** 2 + phi ** 2)
            idx = np.unravel_index(np.nanargmin(metric), metric.shape)
            out[float(angle)] = float(np.real(e2[idx]))
        return out, ""
    except Exception as exc:
        return {float(angle): np.nan for angle in angles_deg}, str(exc)


def signed_theta_pattern(fdtd, monitor_name, resolution_samples):
    angles = np.linspace(-90.0, 90.0, 181)
    try:
        e2 = np.squeeze(np.asarray(fdtd.farfield3d(monitor_name, 1, resolution_samples, resolution_samples), dtype=float))
        try:
            ux = np.asarray(fdtd.farfieldux(monitor_name, 1, resolution_samples, resolution_samples), dtype=float)
            uy = np.asarray(fdtd.farfielduy(monitor_name, 1, resolution_samples, resolution_samples), dtype=float)
        except TypeError:
            ux = np.asarray(fdtd.farfieldux(monitor_name, 1), dtype=float)
            uy = np.asarray(fdtd.farfielduy(monitor_name, 1), dtype=float)
        if ux.ndim == 1 and uy.ndim == 1:
            ux, uy = np.meshgrid(np.ravel(ux), np.ravel(uy), indexing="ij")
        signed_theta = np.rad2deg(np.arcsin(np.clip(ux, -1.0, 1.0)))
        phi0_error = np.abs(uy)
        pattern = []
        for angle in angles:
            metric = np.abs(signed_theta - angle) + 1000.0 * phi0_error
            idx = np.unravel_index(np.nanargmin(metric), metric.shape)
            pattern.append(float(np.real(e2[idx])))
        return angles, np.asarray(pattern, dtype=float), ""
    except Exception as exc:
        return angles, np.full(angles.shape, np.nan, dtype=float), str(exc)


def build_sim(height, channel=None, top_monitor=True, eml_monitor=False):
    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[window_x, window_y, Sz],
        resolution=resolution,
        unit=1e-6,
        background_index=1.0,
        center_wl=wavelength_um,
        N_f=1,
        bc_x="PML",
        bc_y="PML",
        bc_z="PML",
    )
    sim.src_wl = np.asarray([wavelength_um], dtype=float) * sim.unit
    add_oled_stack(sim)
    lens_grid = add_lens_import(sim, height)
    if channel is not None:
        sim.add_source(
            mode="plane",
            name="reciprocal_source",
            center=src_c,
            size=src_s,
            direction="backward",
            src_wl=[wavelength_um],
            bandwidth=0.0,
            pol=channel["polarization_angle"],
            theta=channel["theta_deg"],
            phi=channel["phi_deg"],
            single=True,
        )
    if top_monitor:
        sim.add_monitor(name="top_power", center=monitor_c, size=monitor_s)
    if eml_monitor:
        sim.add_monitor(name="EML_monitor", center=[0.0, 0.0, eml_c[2]], size=[window_x, window_y, 0.0])
    try:
        sim.fdtd.setnamed("FDTD", "simulation time", env_float("MSOPT_OLED_LENS_SIM_TIME_FS", 1200.0) * 1e-15)
    except Exception:
        pass
    return sim, lens_grid


def evaluate_height(height, n_samples=None, farfield_resolution=None, save_patterns=False):
    print(f"[sweep] height={height:.6g} um")
    channel_values = []
    channel_records = []
    lens_grid = None
    for channel_idx, channel in enumerate(target_channels):
        print(
            f"[sweep] h={height:.4g} reciprocal channel {channel_idx + 1}/{len(target_channels)} "
            f"{channel['name']}"
        )
        sim, lens_grid = build_sim(height, channel=channel, top_monitor=False, eml_monitor=True)
        try:
            sim.run(name=f"OLED_lens_h{height:.4g}_{channel['name']}", save=False)
            Eres = sim.fdtd.getresult("EML_monitor", "E")
            Eall = np.asarray(Eres["E"], dtype=np.complex128)
            Ex = Eall[..., 0]
            Ey = Eall[..., 1]
            Ez = Eall[..., 2]
            value, raw_intensity, uniformity = reciprocal_channel_fom(Ex, Ey, Ez, channel)
        finally:
            try:
                sim.fdtd.close()
            except Exception:
                pass
        channel_values.append(value)
        channel_records.append(
            {
                "height_um": height,
                "channel": channel["name"],
                "theta_deg": channel["theta_deg"],
                "polarization": channel["polarization"],
                "eml_component": channel["eml_component"],
                "source_power_norm": channel["source_power_norm"],
                "reciprocal_eml_proxy": value,
                "mean_absE2": raw_intensity,
                "uniformity": uniformity,
            }
        )
    score = combine_oled_scalar_from_values(channel_values)
    angle_powers, pol_matrix, fractions, ratios_to_zero = summarize_reciprocal_values(channel_values)
    ff0_mean = float(angle_powers[0])
    ff45_mean = float(angle_powers[1]) if len(angle_powers) > 1 else 0.0
    ratio45 = float(ratios_to_zero[1]) if len(ratios_to_zero) > 1 else 0.0
    summary = {
        "height_um": height,
        "curvature_radius_um": spherical_cap_radius(lens_radius, height),
        "ff0_mean": ff0_mean,
        "ff45_mean": ff45_mean,
        "ratio45_to_0": ratio45,
        "top_efficiency_mean": 0.0,
        "score": score,
        "records": channel_records,
        "patterns": [],
        "lens_grid": lens_grid,
        "angle_fractions": fractions,
        "polarization_matrix": pol_matrix,
    }
    print(
        f"[sweep] h={height:.6g} reciprocal_score={score:.6e}, "
        f"angle_power_0={ff0_mean:.6e}, angle_power_45={ff45_mean:.6e}, "
        f"ratio45/0={ratio45:.4f}, fraction_0={fractions[0]:.4f}, "
        f"fraction_45={fractions[1]:.4f}"
    )
    return summary


def run_dipole_postprocess_for_height(height, n_samples, farfield_resolution):
    print(f"[postprocess] direct dipole validation for best height={height:.6g} um")
    sim, lens_grid = build_sim(height, channel=None, top_monitor=True, eml_monitor=False)
    positions = sample_positions(n_samples)
    records = []
    patterns = []
    try:
        for pol in ("x", "y"):
            for sample_idx, position in enumerate(positions):
                print(f"[postprocess] h={height:.4g} dipole {pol} {sample_idx + 1}/{len(positions)}")
                sim.fdtd.switchtolayout()
                add_dipole(sim.fdtd, position, pol)
                sim.run(name=f"OLED_lens_best_h{height:.4g}_{pol}_{sample_idx}", save=False)
                transmission = abs(get_transmission(sim.fdtd, "top_power"))
                angle_values, ff_error = farfield_samples(
                    sim.fdtd,
                    "top_power",
                    [0.0, 45.0],
                    farfield_resolution,
                )
                angles, pattern, pattern_error = signed_theta_pattern(sim.fdtd, "top_power", farfield_resolution)
                if not pattern_error:
                    patterns.append((angles, pattern))
                records.append(
                    {
                        "height_um": height,
                        "polarization": pol,
                        "sample_idx": sample_idx,
                        "x_um": position[0],
                        "y_um": position[1],
                        "z_um": position[2],
                        "top_efficiency": transmission,
                        "ff0": angle_values[0.0],
                        "ff45": angle_values[45.0],
                        "farfield_error": ff_error or pattern_error,
                    }
                )
    finally:
        try:
            sim.fdtd.close()
        except Exception:
            pass

    best = evaluate_height(height)
    best["records"] = records
    best["patterns"] = patterns
    best["lens_grid"] = lens_grid
    top = np.asarray([r["top_efficiency"] for r in records], dtype=float)
    best["top_efficiency_mean"] = float(np.nanmean(top)) if np.any(np.isfinite(top)) else 0.0
    return best


def save_sweep_results(summaries):
    path = os.path.join(design_dir, "OLED_lens_sweep_summary.csv")
    fields = [
        "height_um",
        "curvature_radius_um",
        "ff0_mean",
        "ff45_mean",
        "ratio45_to_0",
        "top_efficiency_mean",
        "score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({field: item[field] for field in fields})
    print(f"[sweep] saved summary: {path}")

    values = {field: np.asarray([item[field] for item in summaries], dtype=float) for field in fields}
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.plot(values["height_um"], values["score"], "o-", label="score")
    ax1.plot(values["height_um"], values["ff0_mean"], "s-", label="0 deg intensity")
    ax1.set_xlabel("lens height (um)")
    ax1.set_ylabel("score / intensity")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(values["height_um"], values["ratio45_to_0"], "^-", color="tab:red", label="45/0 ratio")
    ax2.axhline(env_float("MSOPT_OLED_LENS_TARGET_45_RATIO", 0.8), color="tab:red", linestyle="--", alpha=0.5)
    ax2.set_ylabel("45 deg / 0 deg")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    png = os.path.join(design_dir, "OLED_lens_sweep_curve.png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"[sweep] saved curve: {png}")


def save_best_records(best):
    records = best["records"]
    csv_path = os.path.join(design_dir, "OLED_lens_best_dipole_samples.csv")
    fields = [
        "height_um",
        "polarization",
        "sample_idx",
        "x_um",
        "y_um",
        "z_um",
        "top_efficiency",
        "ff0",
        "ff45",
        "farfield_error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"[postprocess] saved best dipole records: {csv_path}")

    txt_path = os.path.join(design_dir, "OLED_lens_best_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write("method lens_height_parameter_sweep\n")
        for key in (
            "height_um",
            "curvature_radius_um",
            "ff0_mean",
            "ff45_mean",
            "ratio45_to_0",
            "top_efficiency_mean",
            "score",
        ):
            fp.write(f"{key} {best[key]:.16e}\n")
        fp.write(f"lens_diameter_um {lens_diameter:.16e}\n")
        fp.write(f"lens_index {lens_index:.16e}\n")
        fp.write(f"resolution_grids_per_um {resolution}\n")
    print(f"[postprocess] saved best summary: {txt_path}")


def save_lens_images(best):
    x, y, z, inside = best["lens_grid"]
    rho_zavg = np.mean(inside.astype(float), axis=2).T
    mid_y = inside.shape[1] // 2
    section = inside[:, mid_y, :].T.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(
        section,
        origin="lower",
        extent=(x[0], x[-1], z[0], z[-1]),
        cmap="Blues",
        aspect="auto",
        interpolation="nearest",
    )
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("z (um)")
    axes[0].set_title("Best lens x-z section")
    axes[1].imshow(
        rho_zavg,
        origin="lower",
        extent=(x[0], x[-1], y[0], y[-1]),
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("y (um)")
    axes[1].set_title("Best lens top view")
    fig.tight_layout()
    path = os.path.join(design_dir, "OLED_lens_best_geometry.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[postprocess] saved best geometry image: {path}")


def save_best_angular_pattern(best):
    if not best["patterns"]:
        print("[postprocess] skipped angular pattern: no far-field patterns")
        return
    angles = best["patterns"][0][0]
    values = np.asarray([pattern for _, pattern in best["patterns"]], dtype=float)
    mean_pattern = np.nanmean(values, axis=0)
    if not np.any(np.isfinite(mean_pattern)):
        print("[postprocess] skipped angular pattern: all samples are NaN")
        return
    mean_pattern = np.nan_to_num(mean_pattern, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = mean_pattern / max(float(np.max(mean_pattern)), 1e-30)
    txt = os.path.join(design_dir, "OLED_lens_best_angular_pattern.txt")
    np.savetxt(txt, np.column_stack([angles, normalized, mean_pattern]), header="theta_deg normalized_intensity raw_mean_intensity")

    fig = plt.figure(figsize=(6, 3.6))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.deg2rad(angles), normalized, linewidth=2.0)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.set_title("OLED lens best angular emission")
    fig.tight_layout()
    png = os.path.join(design_dir, "OLED_lens_best_angular_pattern.png")
    fig.savefig(png, dpi=200)
    plt.close(fig)
    print(f"[postprocess] saved angular pattern data: {txt}")
    print(f"[postprocess] saved angular pattern plot: {png}")


def main():
    start = time.time()
    heights = lens_height_grid()
    post_samples = env_int("MSOPT_OLED_LENS_POSTPROCESS_SAMPLES_PER_POL", 20)
    farfield_resolution = env_int("MSOPT_OLED_LENS_FARFIELD_RES", 181)
    print("OLED lens parameter sweep")
    print(f"RUN_DIR={RUN_DIR}")
    print(f"wavelength_um={wavelength_um}, resolution={resolution} grids/um")
    print(f"window={window_x}x{window_y} um, lens_diameter={lens_diameter} um, lens_index={lens_index}")
    print(f"height sweep={heights}")
    print(
        f"sweep_eval=OLED.py reciprocal EML proxy, postprocess_samples_per_pol={post_samples}, "
        f"farfield_resolution={farfield_resolution}"
    )
    print(
        "target: same combined FoM as OLED.py "
        f"(distribution_weight={target_distribution_weight}, "
        f"polarization_balance_weight={polarization_balance_weight})"
    )

    summaries = [
        evaluate_height(float(height))
        for height in heights
    ]
    save_sweep_results(summaries)
    best = max(summaries, key=lambda item: item["score"])
    print(
        f"[sweep] best height={best['height_um']:.6g} um, "
        f"R={best['curvature_radius_um']:.6g} um, score={best['score']:.6e}"
    )

    if env_flag("MSOPT_OLED_LENS_POSTPROCESS", "1"):
        best = run_dipole_postprocess_for_height(
            float(best["height_um"]),
            post_samples,
            farfield_resolution,
        )
        save_best_records(best)
        save_lens_images(best)
        save_best_angular_pattern(best)
    else:
        print("[postprocess] skipped: MSOPT_OLED_LENS_POSTPROCESS is disabled")
    print(f"Runtime: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
