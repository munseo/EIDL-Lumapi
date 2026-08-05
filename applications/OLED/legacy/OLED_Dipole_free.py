import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from autograd import jacobian as ag_jacobian
from autograd import numpy as npa

import msopt as ms
# Dipole-based OLED outcoupling optimization.
# FoM = zero-order extraction efficiency * single angular shape score.


seed = 240
np.random.seed(seed)

RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
os.makedirs(design_dir, exist_ok=True)
local_dir = os.path.join(RUN_DIR, "Local_bests") + os.sep
os.makedirs(local_dir, exist_ok=True)

visible_wavelengths = np.array([0.55])
resolution = 50
bandwidth = 0.0


boundary_mode = os.environ.get("MSOPT_OLED_BOUNDARY_MODE", "Bloch")
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
        eml_c = [0, 0, center[2]]
    z_cursor += layer_h

design_s = [Sx, Sy, grating_design_h]
design_c = [0, 0, z_cursor + 0.5 * grating_design_h]

out_s = [Sx, Sy, 0]
out_c = [0, 0, Z_max - 0.15]

target_monitor_name = "FoM_monitor"
target_monitor_s = out_s
target_monitor_c = out_c

Nx = int(round(design_s[0] * resolution)) + 1
Ny = int(round(design_s[1] * resolution)) + 1
Nz = int(round(design_s[2] * resolution)) + 1
design_grids = [Nx, Ny, Nz]
design_cells = Nx * Ny * Nz


# Dipole sample configuration.
# Defaults keep the requested cylindrical-symmetry setup: three x-polarized
# dipoles on the x-axis at 0, 0.4R, and 0.8R. Every list can be overridden
# through environment variables:
# - MSOPT_OLED_DIPOLE_RADII_FRAC="0.0,0.4,0.8"
# - MSOPT_OLED_DIPOLE_AZIMUTHS_DEG="0" for the x-axis default
# - MSOPT_OLED_DIPOLE_POLARIZATIONS="x" or "x,y,z"
# - MSOPT_OLED_DIPOLE_WEIGHTS="1,1,1"
def _parse_float_list_env(name, default_values):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default_values)
    values = []
    for token in raw.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma- or space-separated list of floats.") from exc
    return values or list(default_values)


def _parse_str_list_env(name, default_values):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default_values)
    values = [
        token.strip().lower()
        for token in raw.replace(";", ",").replace(" ", ",").split(",")
        if token.strip()
    ]
    return values or list(default_values)


def _broadcast_list(values, target_len, name):
    if len(values) == target_len:
        return list(values)
    if len(values) == 1 and target_len > 1:
        return list(values) * target_len
    raise ValueError(f"{name} must have length 1 or {target_len}, got {len(values)}")


active_radius = 0.5 * min(active_x, active_y)
raw_dipole_radii_frac = _parse_float_list_env("MSOPT_OLED_DIPOLE_RADII_FRAC", [0.0])
raw_dipole_azimuths_deg = _parse_float_list_env("MSOPT_OLED_DIPOLE_AZIMUTHS_DEG", [0.0])
raw_dipole_polarizations = _parse_str_list_env("MSOPT_OLED_DIPOLE_POLARIZATIONS", ["x"])
raw_dipole_weights = _parse_float_list_env("MSOPT_OLED_DIPOLE_WEIGHTS", [1.0])

sample_count = max(
    len(raw_dipole_radii_frac),
    len(raw_dipole_azimuths_deg),
    len(raw_dipole_polarizations),
    len(raw_dipole_weights),
)

dipole_radii_frac = _broadcast_list(raw_dipole_radii_frac, sample_count, "MSOPT_OLED_DIPOLE_RADII_FRAC")
dipole_azimuths_deg = _broadcast_list(raw_dipole_azimuths_deg, sample_count, "MSOPT_OLED_DIPOLE_AZIMUTHS_DEG")
dipole_polarizations = _broadcast_list(
    raw_dipole_polarizations, sample_count, "MSOPT_OLED_DIPOLE_POLARIZATIONS"
)
dipole_weights = np.asarray(
    _broadcast_list(raw_dipole_weights, sample_count, "MSOPT_OLED_DIPOLE_WEIGHTS"),
    dtype=float,
)

dipole_positions = []
for frac, azimuth_deg in zip(dipole_radii_frac, dipole_azimuths_deg):
    azimuth_rad = np.deg2rad(azimuth_deg)
    dipole_positions.append(
        (
            float(frac * active_radius * np.cos(azimuth_rad)),
            float(frac * active_radius * np.sin(azimuth_rad)),
            float(eml_c[2]),
        )
    )
n_dipole_positions = len(dipole_positions)

# Keep the first polarization as the compatibility default for postprocess paths.
dipole_polarization = dipole_polarizations[0]

target_efficiency_curve_str = os.environ.get("MSOPT_OLED_TARGET_EFFICIENCY_CURVE", "0:1.0,45:0.85,60:0.0")


def parse_efficiency_curve(curve_str):
    curve_points = []
    for token in curve_str.split(","):
        parts = token.strip().split(":")
        if len(parts) == 2:
            try:
                theta_deg = float(parts[0].strip())
                efficiency = float(parts[1].strip())
                curve_points.append((theta_deg, efficiency))
            except ValueError:
                pass
    if not curve_points:
        curve_points = [(0.0, 1.0)]

    efficiency_map = {float(theta): float(eff) for theta, eff in curve_points}
    for theta, eff in list(efficiency_map.items()):
        if theta > 0.0 and -theta not in efficiency_map:
            efficiency_map[-theta] = eff
        if theta < 0.0 and -theta not in efficiency_map:
            efficiency_map[-theta] = eff

    curve_points = sorted((theta, efficiency_map[theta]) for theta in efficiency_map)
    return curve_points


def interpolate_efficiency_at_angle(theta_deg, efficiency_curve):
    if not efficiency_curve:
        return 1.0
    theta, efficiency = np.asarray(efficiency_curve, dtype=float).T
    return float(np.interp(theta_deg, theta, efficiency, left=efficiency[0], right=efficiency[-1]))


def symmetric_angle_series(angles_deg, values):
    angles = np.abs(np.asarray(angles_deg, dtype=float).reshape(-1))
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(angles)
    angles = angles[order]
    values = values[order]
    if angles.size:
        unique_angles, unique_inverse = np.unique(np.round(angles, 12), return_inverse=True)
        unique_values = np.array([
            float(np.mean(values[unique_inverse == idx]))
            for idx in range(unique_angles.size)
        ])
        angles = unique_angles
        values = unique_values
    positive = angles > 1e-12
    return (
        np.concatenate((-angles[positive][::-1], angles)),
        np.concatenate((values[positive][::-1], values)),
    )


def setup_semicircle_polar_axis(ax, title, rmax=1.0):
    rmax = max(float(rmax), 1e-12)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0.0, rmax)
    if rmax <= 1.2:
        ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    else:
        ax.set_rticks(np.linspace(0.0, rmax, 5)[1:])
    ax.set_title(title)
    ax.grid(True, alpha=0.35)


def dipole_orientation_angles(polarization):
    pol = str(polarization).strip().lower()
    if pol == "x":
        return 90.0, 0.0
    if pol == "y":
        return 90.0, 90.0
    if pol == "z":
        return 0.0, 0.0
    raise ValueError(f"Unsupported dipole polarization: {polarization!r}")


# Parse target efficiency curve
target_efficiency_curve = parse_efficiency_curve(target_efficiency_curve_str)
print(f"[ldos setup] target efficiency curve: {target_efficiency_curve}")


def env_flag(name, default="1"):
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


angular_use_hann_window = env_flag("MSOPT_OLED_ANGULAR_HANN_WINDOW", "0")
angular_diagnostic_bin_half_width_deg = float(
    os.environ.get("MSOPT_OLED_ANGULAR_BIN_HALF_WIDTH_DEG", "5.0")
)
angular_order_soft_sigma_deg = float(
    os.environ.get(
        "MSOPT_OLED_ANGULAR_ORDER_SIGMA_DEG",
        os.environ.get("MSOPT_OLED_ANGULAR_BIN_HALF_WIDTH_DEG", "5.0"),
    )
)
angular_leakage_start_deg = float(
    os.environ.get("MSOPT_OLED_LEAKAGE_START_DEG", "60.0")
)


def build_target_orders(wavelength_um, period_um, efficiency_curve, max_order=10):
    radial_order_map = {}
    for order_m in range(-max_order, max_order + 1):
        sin_theta = order_m * wavelength_um / period_um
        if abs(sin_theta) > 1.0:
            continue
        radial_order = abs(order_m)
        radial_angle = abs(float(np.rad2deg(np.arcsin(sin_theta))))
        efficiency = max(interpolate_efficiency_at_angle(radial_angle, efficiency_curve), 0.0)
        amplitude = np.sqrt(max(efficiency, 1e-10))
        entry = radial_order_map.get(radial_order)
        if entry is None or float(amplitude) > entry["amplitude"]:
            radial_order_map[radial_order] = {
                "angle_deg": radial_angle,
                "efficiency": float(efficiency),
                "amplitude": float(amplitude),
            }
    radial_orders = [
        (
            radial_order,
            radial_order_map[radial_order]["angle_deg"],
            radial_order_map[radial_order]["efficiency"],
            radial_order_map[radial_order]["amplitude"],
        )
        for radial_order in sorted(radial_order_map)
    ]
    max_amp = max((amp for _, _, _, amp in radial_orders), default=1.0)
    if max_amp > 0.0:
        radial_orders = [(order, angle, eff, amp / max_amp) for order, angle, eff, amp in radial_orders]

    return {
        "radial_orders": radial_orders,
        "wavelength_um": wavelength_um,
        "period_um": period_um,
    }


def save_target_order_info(target_field_info, design_dir):
    radial_orders = target_field_info["radial_orders"]
    wavelength = target_field_info["wavelength_um"]
    period = target_field_info["period_um"]

    radial_order_ids = np.asarray([order for order, _, _, _ in radial_orders], dtype=int)
    radial_angles = np.asarray([angle for _, angle, _, _ in radial_orders], dtype=float)
    radial_efficiencies = np.asarray([eff for _, _, eff, _ in radial_orders], dtype=float)
    angle_range = np.linspace(0.0, max(float(np.max(radial_angles)) + 5.0, 1.0), 181)
    efficiency_range = np.array([
        interpolate_efficiency_at_angle(ang, target_efficiency_curve)
        for ang in angle_range
    ])

    signed_angles, signed_efficiency = symmetric_angle_series(angle_range, efficiency_range)
    fig = plt.figure(figsize=(6.5, 3.9))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.deg2rad(signed_angles), signed_efficiency, "b-", linewidth=2, label="target relative power")
    scatter_angles = []
    scatter_efficiencies = []
    scatter_labels = []
    for order, angle, efficiency in zip(radial_order_ids, radial_angles, radial_efficiencies):
        order_angles = [0.0] if abs(angle) <= 1e-12 else [-angle, angle]
        scatter_angles.extend(order_angles)
        scatter_efficiencies.extend([efficiency] * len(order_angles))
        scatter_labels.append((angle, efficiency, order))
    ax.scatter(
        np.deg2rad(scatter_angles),
        scatter_efficiencies,
        c="tab:orange",
        s=80,
        label="available radial orders",
        zorder=5,
    )
    for angle, efficiency, order in scatter_labels:
        ax.text(
            np.deg2rad(angle),
            min(float(efficiency) + 0.05, 1.0),
            f"|m|={int(order)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    setup_semicircle_polar_axis(
        ax,
        f"Cylindrical angular target (lambda={wavelength}um, period={period}um)",
        rmax=1.0,
    )
    ax.legend()
    fig.tight_layout()
    path = os.path.join(design_dir, "LDOS_target_field_info.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ldos setup] saved target field info: {path}")

    summary_path = os.path.join(design_dir, "LDOS_target_field_summary.txt")
    with open(summary_path, "w") as fp:
        fp.write(f"Target field specification\n")
        fp.write(f"Wavelength: {wavelength} μm\n")
        fp.write(f"Grating period: {period} μm\n")
        fp.write(f"Efficiency curve: {target_efficiency_curve_str}\n")
        fp.write(f"\nCylindrical radial orders:\n")
        fp.write(f"|Order|\tPolar_Angle(deg)\tTarget_Eff\tRing_Amp\n")
        for order, angle, eff, amp in radial_orders:
            fp.write(f"{int(order)}\t{angle:8.3f}\t{eff:8.4f}\t{amp:8.4f}\n")
    print(f"[ldos setup] saved target field summary: {summary_path}")

    radial_profile_path = os.path.join(design_dir, "LDOS_target_field_radial_orders.csv")
    with open(radial_profile_path, "w", encoding="utf-8") as fp:
        fp.write("radial_order,polar_angle_deg,target_efficiency,ring_amplitude\n")
        for order, angle, eff, amp in radial_orders:
            fp.write(f"{int(order)},{angle:.6f},{eff:.6f},{amp:.6f}\n")
    print(f"[ldos setup] saved radial target field profile: {radial_profile_path}")


def nonnegative_efficiency_curve_arrays(efficiency_curve):
    efficiency_by_abs_angle = {}
    for theta_deg, efficiency in efficiency_curve:
        theta_key = abs(float(theta_deg))
        efficiency_by_abs_angle[theta_key] = max(
            float(efficiency_by_abs_angle.get(theta_key, -np.inf)),
            float(efficiency),
        )
    if not efficiency_by_abs_angle:
        return np.asarray([0.0]), np.asarray([1.0])
    angles = np.asarray(sorted(efficiency_by_abs_angle), dtype=float)
    efficiencies = np.asarray([efficiency_by_abs_angle[angle] for angle in angles], dtype=float)
    efficiencies = np.maximum(efficiencies, 0.0)
    if np.max(efficiencies) <= 0.0:
        efficiencies = np.ones_like(efficiencies)
    return angles, efficiencies


def build_dft_matrix(n_points):
    idx = np.arange(int(n_points), dtype=float)
    return np.exp(-2j * np.pi * np.outer(idx, idx) / float(n_points))


def angular_bin_mask(theta_grid_deg, center_deg, half_width_deg, propagating):
    lo = max(0.0, float(center_deg) - float(half_width_deg))
    hi = min(90.0, float(center_deg) + float(half_width_deg))
    return np.asarray((theta_grid_deg >= lo) & (theta_grid_deg <= hi) & propagating, dtype=float)


def soft_angular_ring_weight(theta_grid_deg, center_deg, sigma_deg, propagating):
    sigma = max(float(sigma_deg), 1e-9)
    normalized_delta = (theta_grid_deg - float(center_deg)) / sigma
    weight = np.exp(-0.5 * normalized_delta ** 2)
    return np.asarray(np.where(propagating, weight, 0.0), dtype=float)


def build_angular_power_target(x_axis, y_axis, target_field_info, efficiency_curve):
    x = np.ravel(np.asarray(x_axis, dtype=float))
    y = np.ravel(np.asarray(y_axis, dtype=float))
    if x.size < 2 or y.size < 2:
        raise ValueError("angular target requires at least two x/y monitor samples")

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    if dx == 0.0 or dy == 0.0:
        raise ValueError("angular target requires nonzero x/y monitor spacing")

    nx = x.size
    ny = y.size
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=abs(dx))
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=abs(dy))
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    wavelength_m = float(target_field_info["wavelength_um"]) * 1e-6
    k0 = 2.0 * np.pi / wavelength_m
    normalized_kr = np.sqrt(KX ** 2 + KY ** 2) / k0
    propagating = normalized_kr <= 1.0
    theta_grid_deg = np.rad2deg(np.arcsin(np.clip(normalized_kr, 0.0, 1.0)))

    angle_points, efficiency_points = nonnegative_efficiency_curve_arrays(efficiency_curve)
    target_map = np.interp(
        theta_grid_deg,
        angle_points,
        efficiency_points,
        left=efficiency_points[0],
        right=efficiency_points[-1],
    )
    max_target = float(np.max(np.abs(target_map)))
    if max_target > 0.0:
        target_map = target_map / max_target
    target_map = np.where(propagating, target_map, 0.0)

    zero_mask = soft_angular_ring_weight(
        theta_grid_deg,
        0.0,
        angular_order_soft_sigma_deg,
        propagating,
    )
    target0 = float(np.interp(0.0, angle_points, efficiency_points))
    target0 = max(target0, 1e-12)

    order_match_masks = []
    order_target_values = []
    ratio_match_masks = []
    ratio_target_values = []
    for radial_order, theta_deg, efficiency, _amp in target_field_info["radial_orders"]:
        theta_deg = float(theta_deg)
        target_value = max(float(efficiency), 0.0)
        if target_value <= ldos_order_target_min:
            continue
        order_entry = {
            "order": int(radial_order),
            "theta_deg": theta_deg,
            "mask": soft_angular_ring_weight(
                theta_grid_deg,
                theta_deg,
                angular_order_soft_sigma_deg,
                propagating,
            ),
        }
        order_match_masks.append(order_entry)
        order_target_values.append(target_value)

        target_ratio = target_value / target0
        if int(radial_order) == 0 or theta_deg <= 1e-12:
            continue
        ratio_match_masks.append(order_entry)
        ratio_target_values.append(target_ratio)

    if not order_match_masks:
        order_match_masks.append({"order": 0, "theta_deg": 0.0, "mask": zero_mask})
        order_target_values.append(1.0)

    order_target_values = np.asarray(order_target_values, dtype=float)
    order_target_shares = order_target_values / max(float(np.sum(order_target_values)), 1e-30)
    order_envelope = np.clip(
        np.sum([entry["mask"] for entry in order_match_masks], axis=0),
        0.0,
        1.0,
    )
    off_target_mask = np.asarray(propagating, dtype=float) * (1.0 - order_envelope)

    if angular_use_hann_window:
        window = np.outer(np.hanning(nx), np.hanning(ny))
    else:
        window = np.ones((nx, ny), dtype=float)

    leakage_angle_mask = np.asarray(
        (theta_grid_deg >= angular_leakage_start_deg) & propagating,
        dtype=float,
    )

    return {
        "x_size": nx,
        "y_size": ny,
        "dft_x": build_dft_matrix(nx),
        "dft_y": build_dft_matrix(ny),
        "monitor_dx": abs(dx),
        "monitor_dy": abs(dy),
        "monitor_cell_area": abs(dx * dy),
        "kx": kx,
        "ky": ky,
        "theta_grid_deg": theta_grid_deg,
        "target_map": np.asarray(target_map, dtype=float),
        "propagating": np.asarray(propagating, dtype=float),
        "window": np.asarray(window, dtype=float),
        "angle_points": angle_points,
        "efficiency_points": efficiency_points,
        "zero_angle_mask": zero_mask,
        "order_match_masks": order_match_masks,
        "order_target_values": order_target_values,
        "order_target_shares": order_target_shares,
        "ratio_match_masks": ratio_match_masks,
        "ratio_target_values": np.asarray(ratio_target_values, dtype=float),
        "off_target_mask": off_target_mask,
        "leakage_angle_mask": leakage_angle_mask,
    }


def save_angular_power_target_preview(angular_target, design_dir, file_prefix="LDOS_angular_target"):
    kx = np.fft.fftshift(np.asarray(angular_target["kx"], dtype=float))
    ky = np.fft.fftshift(np.asarray(angular_target["ky"], dtype=float))
    target_map = np.fft.fftshift(np.asarray(angular_target["target_map"], dtype=float))
    theta_grid = np.fft.fftshift(np.asarray(angular_target["theta_grid_deg"], dtype=float))
    extent = (kx[0], kx[-1], ky[0], ky[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    im0 = axes[0].imshow(
        target_map.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        vmin=0.0,
        vmax=max(float(np.max(target_map)), 1.0),
    )
    axes[0].set_title("Angular target weight")
    axes[0].set_xlabel("kx (1/m)")
    axes[0].set_ylabel("ky (1/m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        theta_grid.T,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="magma",
        vmin=0.0,
        vmax=90.0,
    )
    axes[1].set_title("Polar angle theta")
    axes[1].set_xlabel("kx (1/m)")
    axes[1].set_ylabel("ky (1/m)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = os.path.join(design_dir, f"{file_prefix}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    np.save(os.path.join(design_dir, f"{file_prefix}_map.npy"), angular_target["target_map"])
    print(f"[ldos setup] saved angular target map: {path}")


def expand_xy_weight(weight_xy, target_ndim):
    weight = npa.asarray(weight_xy)
    while weight.ndim < target_ndim:
        weight = weight[..., None]
    return weight


def dft2_xy(field, angular_target):
    dft_x = npa.asarray(angular_target["dft_x"])
    dft_y = npa.asarray(angular_target["dft_y"])
    return npa.einsum("ia,jb,ab...->ij...", dft_x, dft_y, field)


def compute_angular_power_metrics(Ex, Ey, Hx, Hy, angular_target, flux_sign=1.0):
    nx = int(angular_target["x_size"])
    ny = int(angular_target["y_size"])

    spectra = []
    for field in (Ex, Ey, Hx, Hy):
        if field.shape[0] != nx or field.shape[1] != ny:
            raise ValueError(
                f"angular target shape mismatch: field={field.shape}, target=({nx}, {ny})"
            )
        field = field * expand_xy_weight(angular_target["window"], field.ndim)
        spectra.append(dft2_xy(field, angular_target))

    Ex_k, Ey_k, Hx_k, Hy_k = spectra
    spectral_flux = 0.5 * float(flux_sign) * npa.real(
        Ex_k * npa.conj(Hy_k) - Ey_k * npa.conj(Hx_k)
    )
    if spectral_flux.ndim > 2:
        spectral_flux = npa.sum(spectral_flux, axis=tuple(range(2, spectral_flux.ndim)))
    weighted_spectrum = npa.maximum(
        npa.where(npa.isfinite(spectral_flux), spectral_flux, 0.0),
        0.0,
    ) * npa.asarray(angular_target["propagating"])

    top_power = npa.sum(weighted_spectrum)
    zero_power = npa.sum(weighted_spectrum * npa.asarray(angular_target["zero_angle_mask"]))
    order_powers = [
        npa.sum(weighted_spectrum * npa.asarray(entry["mask"]))
        for entry in angular_target["order_match_masks"]
    ]
    off_target_power = npa.sum(weighted_spectrum * npa.asarray(angular_target["off_target_mask"]))
    leakage_power = npa.sum(weighted_spectrum * npa.asarray(angular_target["leakage_angle_mask"]))
    return top_power, zero_power, order_powers, off_target_power, leakage_power


def compute_top_monitor_flux_power(Ex, Ey, Hx, Hy, angular_target, flux_sign=1.0):
    cell_area = float(angular_target.get("monitor_cell_area", 1.0))
    pz = 0.5 * npa.real(Ex * npa.conj(Hy) - Ey * npa.conj(Hx))
    flux_power = float(flux_sign) * npa.sum(pz) * cell_area
    return npa.maximum(npa.where(npa.isfinite(flux_power), flux_power, 0.0), 0.0)



# Setup dipole-based channels
target_channels = []
for dipole_idx, ((dipole_x, dipole_y, dipole_z), dipole_pol, dipole_weight) in enumerate(
    zip(dipole_positions, dipole_polarizations, dipole_weights)
):
    target_channels.append({
        "name": f"dipole_{dipole_idx}_pos_({dipole_x:.3f},{dipole_y:.3f})_{dipole_pol}",
        "dipole_idx": dipole_idx,
        "dipole_x": dipole_x,
        "dipole_y": dipole_y,
        "dipole_z": dipole_z,
        "polarization": dipole_pol,
        "weight": float(dipole_weight),
        "wavelengths": np.asarray(visible_wavelengths, dtype=float),
    })

N_fom = len(target_channels)
channel_weights = np.asarray([channel["weight"] for channel in target_channels], dtype=float)
combined_fom_history = []

# FoM control parameters
ldos_shape_weight = float(
    os.environ.get(
        "MSOPT_OLED_SHAPE_WEIGHT",
        os.environ.get(
            "MSOPT_OLED_RATIO_MATCH_WEIGHT",
            os.environ.get(
                "MSOPT_OLED_ANGULAR_SHAPE_WEIGHT",
                os.environ.get("MSOPT_OLED_LDOS_FIELD_MATCH_WEIGHT", "1.0"),
            ),
        ),
    )
)
ldos_score_cap = float(os.environ.get("MSOPT_OLED_LDOS_SCORE_CAP", "10.0"))
channel_power_floor = float(os.environ.get("MSOPT_OLED_CHANNEL_POWER_FLOOR", "1e-12"))
ldos_fom_floor = float(os.environ.get("MSOPT_OLED_FOM_FLOOR", "0.0"))
ldos_fom_scale = float(os.environ.get("MSOPT_OLED_FOM_SCALE", "1.0"))
unstable_candidate_fom = float(os.environ.get("MSOPT_OLED_UNSTABLE_CANDIDATE_FOM", "-1e30"))
ldos_order_share_error_weight = float(
    os.environ.get(
        "MSOPT_OLED_ORDER_SHARE_ERROR_WEIGHT",
        os.environ.get("MSOPT_OLED_RATIO_ERROR_WEIGHT", "4.0"),
    )
)
ldos_order_share_eps = float(os.environ.get("MSOPT_OLED_ORDER_SHARE_EPS", "1e-12"))
ldos_order_target_min = float(
    os.environ.get(
        "MSOPT_OLED_ORDER_TARGET_MIN",
        os.environ.get("MSOPT_OLED_RATIO_TARGET_MIN", "1e-3"),
    )
)
ldos_target_capture_weight = float(
    os.environ.get(
        "MSOPT_OLED_TARGET_CAPTURE_WEIGHT",
        os.environ.get(
            "MSOPT_OLED_OFF_TARGET_PENALTY_WEIGHT",
            os.environ.get("MSOPT_OLED_LEAKAGE_PENALTY_WEIGHT", "2.0"),
        ),
    )
)
ldos_extraction_efficiency_weight = float(
    os.environ.get(
        "MSOPT_OLED_EXTRACTION_EFFICIENCY_WEIGHT",
        os.environ.get("MSOPT_OLED_ZERO_EMISSION_WEIGHT", "1.0"),
    )
)

def real_scalar_or_none(value):
    try:
        return float(np.real(value))
    except (TypeError, ValueError):
        return None


def finite_positive_sum(values):
    values = np.real(np.asarray(values, dtype=np.complex128)).reshape(-1)
    values = values[np.isfinite(values)]
    values = values[values > 0.0]
    if values.size == 0:
        return None
    return float(np.sum(values))


def read_source_power(fdtd, freqs_hz):
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if freqs_hz.size == 0:
        return None
    try:
        return finite_positive_sum(fdtd.sourcepower(freqs_hz))
    except Exception:
        pass
    fdtd.putv("msopt_sourcepower_freqs", freqs_hz)
    fdtd.eval("msopt_sourcepower_values = sourcepower(msopt_sourcepower_freqs);")
    return finite_positive_sum(fdtd.getv("msopt_sourcepower_values"))


def read_dipole_total_power(fdtd, freqs_hz):
    # Lumerical's dipolepower API path used here is frequency-only.
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if freqs_hz.size == 0:
        return None
    try:
        return finite_positive_sum(fdtd.dipolepower(freqs_hz))
    except Exception:
        pass
    fdtd.putv("msopt_dipolepower_freqs", freqs_hz)
    fdtd.eval("msopt_dipolepower_values = dipolepower(msopt_dipolepower_freqs);")
    return finite_positive_sum(fdtd.getv("msopt_dipolepower_values"))


def read_monitor_transmission(fdtd, monitor_name):
    try:
        return float(np.real(np.asarray(fdtd.transmission(monitor_name)).reshape(-1)[0]))
    except Exception:
        safe_name = str(monitor_name).replace('"', '\\"')
        fdtd.eval(f'msopt_monitor_transmission = transmission("{safe_name}");')
        return float(np.real(np.asarray(fdtd.getv("msopt_monitor_transmission")).reshape(-1)[0]))


def monitor_flux_from_arrays(Ex, Ey, Hx, Hy, angular_target, flux_sign=1.0):
    cell_area = float(angular_target.get("monitor_cell_area", 1.0))
    pz = 0.5 * np.real(np.asarray(Ex) * np.conj(np.asarray(Hy)) - np.asarray(Ey) * np.conj(np.asarray(Hx)))
    flux_power = float(flux_sign) * float(np.sum(pz)) * cell_area
    if not np.isfinite(flux_power):
        return None
    return max(flux_power, 0.0)


def read_monitor_raw_flux(fdtd, monitor_name, angular_target, flux_sign=1.0):
    if angular_target is None:
        return None
    try:
        E = np.asarray(fdtd.getresult(monitor_name, "E")["E"], dtype=np.complex128)
        H = np.asarray(fdtd.getresult(monitor_name, "H")["H"], dtype=np.complex128)
    except Exception:
        return None
    if E.shape[-1] != 3 or H.shape[-1] != 3:
        return None
    return monitor_flux_from_arrays(E[..., 0], E[..., 1], H[..., 0], H[..., 1], angular_target, flux_sign)


def compute_angular_power_scores(Ex, Ey, Hx, Hy, channel):
    angular_target = channel.get("angular_target")
    if angular_target is None:
        raise ValueError("angular_target is missing from the channel configuration")

    flux_sign = float(channel.get("last_top_flux_sign", 1.0))
    top_power, zero_power, order_powers, off_target_power, leakage_power = compute_angular_power_metrics(
        Ex,
        Ey,
        Hx,
        Hy,
        angular_target,
        flux_sign=flux_sign,
    )
    top_power = npa.maximum(npa.where(npa.isfinite(top_power), top_power, 0.0), channel_power_floor)
    zero_power = npa.maximum(npa.where(npa.isfinite(zero_power), zero_power, 0.0), channel_power_floor)
    off_target_power = npa.maximum(npa.where(npa.isfinite(off_target_power), off_target_power, 0.0), 0.0)
    leakage_power = npa.maximum(npa.where(npa.isfinite(leakage_power), leakage_power, 0.0), 0.0)
    dipole_total_power = max(
        float(channel.get("last_dipole_total_power", channel_power_floor)),
        channel_power_floor,
    )
    raw_top_flux_power = compute_top_monitor_flux_power(Ex, Ey, Hx, Hy, angular_target, flux_sign=flux_sign)
    top_flux_power = raw_top_flux_power * float(channel.get("last_top_flux_calibration", 1.0))
    top_extraction_efficiency = top_flux_power / (dipole_total_power + 1e-30)
    zero_flux_fraction = zero_power / (top_power + 1e-30)
    zero_flux_power = top_flux_power * zero_flux_fraction
    zero_order_efficiency = zero_flux_power / (dipole_total_power + 1e-30)

    order_power_sum = 0.0
    clean_order_powers = []
    for order_power in order_powers:
        clean_power = npa.maximum(npa.where(npa.isfinite(order_power), order_power, 0.0), 0.0)
        clean_order_powers.append(clean_power)
        order_power_sum = order_power_sum + clean_power

    shape_error = 0.0
    order_share_dot = 0.0
    order_share_norm = 0.0
    target_share_norm = 0.0
    target_shares = np.asarray(angular_target["order_target_shares"], dtype=float)
    if clean_order_powers and target_shares.size == len(clean_order_powers):
        for order_power, target_share in zip(clean_order_powers, target_shares):
            order_share = order_power / (order_power_sum + 1e-30)
            target_share = max(float(target_share), ldos_order_share_eps)
            shape_error = shape_error + (
                npa.sqrt(order_share + ldos_order_share_eps)
                - np.sqrt(target_share + ldos_order_share_eps)
            ) ** 2
            order_share_dot = order_share_dot + order_share * target_share
            order_share_norm = order_share_norm + order_share ** 2
            target_share_norm = target_share_norm + target_share ** 2
        order_share_cosine = order_share_dot ** 2 / (order_share_norm * target_share_norm + 1e-30)
    else:
        order_share_cosine = 1.0

    shape_error = npa.maximum(shape_error, 0.0)
    distribution_score = 1.0 / (1.0 + ldos_order_share_error_weight * shape_error)
    order_share_cosine = npa.clip(order_share_cosine, 0.0, 1.0)
    off_target_fraction = off_target_power / (top_power + 1e-30)
    off_target_fraction = npa.clip(off_target_fraction, 0.0, 1.0)
    target_capture_fraction = 1.0 - off_target_fraction
    capture_score = target_capture_fraction / (
        target_capture_fraction + ldos_target_capture_weight * off_target_fraction + 1e-30
    )
    capture_score = npa.clip(capture_score, 0.0, 1.0)
    shape_score = distribution_score * capture_score
    leakage_fraction = leakage_power / (top_power + 1e-30)
    return (
        zero_order_efficiency,
        top_extraction_efficiency,
        top_flux_power,
        zero_flux_power,
        dipole_total_power,
        shape_score,
        distribution_score,
        capture_score,
        target_capture_fraction,
        shape_error,
        off_target_fraction,
        leakage_fraction,
        order_share_cosine,
    )



def combine_ldos_fom_from_values(vals):
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), ldos_fom_floor)
    weights = npa.asarray(channel_weights, dtype=float)
    weight_sum = npa.sum(weights)
    if float(weight_sum) > 0.0:
        weights = weights / weight_sum
    else:
        weights = npa.ones_like(weights) / max(float(weights.size), 1.0)

    combined_fom = npa.sum(vals * weights)
    
    return combined_fom


def ldos_summary_from_values(vals):
    vals = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, ldos_fom_floor)
    
    weight_sum = float(np.sum(channel_weights))
    if weight_sum > 0.0:
        normalized_weights = channel_weights / weight_sum
        weighted_mean = float(np.sum(vals * normalized_weights))
    else:
        weighted_mean = float(np.mean(vals))

    return {
        "dipole_foms": vals,
        "mean_fom": float(np.mean(vals)),
        "weighted_mean_fom": weighted_mean,
        "min_fom": float(np.min(vals)),
        "max_fom": float(np.max(vals)),
    }


def sanitize_fom_values(f0s):
    return np.asarray([
        max(
            float(np.nan_to_num(
                np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v),
                nan=ldos_fom_floor,
                posinf=ldos_score_cap,
                neginf=ldos_fom_floor,
            )),
            ldos_fom_floor,
        )
        for v in f0s
    ], dtype=float)


def finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def channel_metric_stats(metric_name):
    values = []
    for channel in target_channels:
        value = channel.get("last_fom_metrics", {}).get(metric_name)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    return finite_stats(values)


def efficiency_stat_line(label, stats):
    if stats is None:
        return None
    mean_value, min_value, max_value = stats
    return (
        f"{label} mean={mean_value:.3e} ({100.0 * mean_value:.3f}%), "
        f"min={min_value:.3e}, max={max_value:.3e}"
    )


def format_design_plot_status(f0_vals=None):
    lines = []
    fom_stats = finite_stats(f0_vals) if f0_vals is not None else None
    if fom_stats is not None:
        mean_value, min_value, max_value = fom_stats
        lines.append(f"FoM mean={mean_value:.3e}, min={min_value:.3e}, max={max_value:.3e}")

    for label, metric_name in (
        ("zero-order eff", "zero_order_efficiency"),
        ("top extraction", "top_extraction_efficiency"),
    ):
        line = efficiency_stat_line(label, channel_metric_stats(metric_name))
        if line is not None:
            lines.append(line)

    shape_stats = channel_metric_stats("shape_score")
    capture_stats = channel_metric_stats("target_capture_fraction")
    off_target_stats = channel_metric_stats("off_target_fraction")
    extras = []
    if shape_stats is not None:
        extras.append(f"shape score mean={shape_stats[0]:.3f}")
    if capture_stats is not None:
        extras.append(f"target capture mean={capture_stats[0]:.3f}")
    if off_target_stats is not None:
        extras.append(f"off-target mean={off_target_stats[0]:.3f}")
    if extras:
        lines.append(", ".join(extras))

    return "\n".join(lines)


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
    print("[ldos] Computing target angular orders...")
    target_field_info = build_target_orders(
        wavelength_um=float(np.mean(visible_wavelengths)),
        period_um=float(window_x),
        efficiency_curve=target_efficiency_curve,
    )
    save_target_order_info(target_field_info, design_dir)
    
    fom_history = [[] for _ in range(N_fom)]
    sim = [None] * N_fom
    opt = [None] * N_fom
    use_bulk_reference_grid = env_flag("MSOPT_OLED_BULK_NORMALIZATION", "1")

    for idx, channel in enumerate(target_channels):
        dipole_x = channel["dipole_x"]
        dipole_y = channel["dipole_y"]
        dipole_z = channel["dipole_z"]
        
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

        theta_deg, phi_deg = dipole_orientation_angles(channel["polarization"])
        sim[idx].fdtd.adddipole()
        sim[idx].fdtd.set("name", f"source")
        sim[idx].fdtd.set("x", dipole_x * 1e-6)
        sim[idx].fdtd.set("y", dipole_y * 1e-6)
        sim[idx].fdtd.set("z", dipole_z * 1e-6)
        sim[idx].fdtd.set("theta", theta_deg)
        sim[idx].fdtd.set("phi", phi_deg)
        sim[idx].fdtd.set("wavelength start", float(np.min(visible_wavelengths)) * 1e-6)
        sim[idx].fdtd.set("wavelength stop", float(np.max(visible_wavelengths)) * 1e-6)

        sim[idx].add_monitor(name=target_monitor_name, center=target_monitor_c, size=target_monitor_s)

        if use_bulk_reference_grid:
            sim[idx].run(name=f"bulk_reference_{idx}", save=True)
            bulk_result = sim[idx].fdtd.getresult(target_monitor_name, "E")
            bulk_x = np.ravel(np.asarray(bulk_result["x"], dtype=float))
            bulk_y = np.ravel(np.asarray(bulk_result["y"], dtype=float))
            channel["angular_target"] = build_angular_power_target(
                bulk_x,
                bulk_y,
                target_field_info,
                target_efficiency_curve,
            )
            if idx == 0:
                save_angular_power_target_preview(
                    channel["angular_target"],
                    design_dir,
                    file_prefix="LDOS_angular_target",
                )
            sim[idx].fdtd.switchtolayout()
            print(f"[ldos setup] channel {idx} angular target initialized from bulk monitor grid")
        else:
            channel["angular_target"] = None

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

        if not hasattr(sim[idx], "src_wl") or sim[idx].src_wl is None:
            sim[idx].src_wl = np.asarray(visible_wavelengths, dtype=float).reshape(-1) * sim[idx].unit
            sim[idx].src_bw = 0.0
        
        def J_ldos(E_x, E_y, H_x, H_y, channel_idx=idx, channel=channel):
            angular_target = channel.get("angular_target")
            if angular_target is None:
                x_axis = getattr(opt[channel_idx], "xg", None)
                y_axis = getattr(opt[channel_idx], "yg", None)
                if x_axis is None or y_axis is None:
                    raise ValueError("angular target is not initialized")
                angular_target = build_angular_power_target(
                    x_axis,
                    y_axis,
                    target_field_info,
                    target_efficiency_curve,
                )
                channel["angular_target"] = angular_target

            (
                zero_order_efficiency,
                top_extraction_efficiency,
                top_flux_power,
                zero_flux_power,
                dipole_total_power,
                shape_score,
                distribution_score,
                capture_score,
                target_capture_fraction,
                shape_error,
                off_target_fraction,
                leakage_fraction,
                order_share_cosine,
            ) = compute_angular_power_scores(
                E_x,
                E_y,
                H_x,
                H_y,
                channel,
            )
            
            zero_order_efficiency = npa.clip(zero_order_efficiency, 0.0, ldos_score_cap)
            top_extraction_efficiency = npa.clip(top_extraction_efficiency, 0.0, ldos_score_cap)
            shape_score = npa.clip(shape_score, 0.0, 1.0)
            fom = npa.power(
                zero_order_efficiency,
                ldos_extraction_efficiency_weight,
            ) * npa.power(
                shape_score,
                ldos_shape_weight,
            )
            physical_fom = fom
            fom = ldos_fom_scale * physical_fom
            fom = npa.clip(fom, 0.0, ldos_score_cap)
            
            fom_value = real_scalar_or_none(fom)
            if fom_value is not None:
                channel["last_fom_metrics"] = {
                    "fom": fom_value,
                    "physical_fom": real_scalar_or_none(physical_fom),
                    "zero_order_efficiency": real_scalar_or_none(zero_order_efficiency),
                    "top_extraction_efficiency": real_scalar_or_none(top_extraction_efficiency),
                    "shape_score": real_scalar_or_none(shape_score),
                    "distribution_score": real_scalar_or_none(distribution_score),
                    "capture_score": real_scalar_or_none(capture_score),
                    "target_capture_fraction": real_scalar_or_none(target_capture_fraction),
                    "shape_error": real_scalar_or_none(shape_error),
                    "off_target_fraction": real_scalar_or_none(off_target_fraction),
                    "leakage_fraction": real_scalar_or_none(leakage_fraction),
                    "order_share_cosine": real_scalar_or_none(order_share_cosine),
                }
                fom_history[channel_idx].append(fom_value)
                print(
                    f"[dipole {channel_idx}] {channel['name']} "
                    f"pos=({channel['dipole_x']:.3f},{channel['dipole_y']:.3f},{channel['dipole_z']:.3f}) "
                    f"pol={channel['polarization']} "
                    f"w={channel['weight']:.3f} "
                    f"FoM={fom:.6e} "
                    f"(physical_fom={physical_fom:.6e}, "
                    f"zero_order_efficiency={zero_order_efficiency:.6e}, "
                    f"top_extraction_efficiency={top_extraction_efficiency:.6e}, "
                    f"top_flux_power={top_flux_power:.6e}, "
                    f"zero_flux_power={zero_flux_power:.6e}, "
                    f"dipole_total_power={dipole_total_power:.6e}, "
                    f"source_power={channel.get('last_source_power', np.nan):.6e}, "
                    f"top_monitor_transmission={channel.get('last_top_monitor_transmission', np.nan):.6e}, "
                    f"top_flux_calibration={channel.get('last_top_flux_calibration', 1.0):.6e}, "
                    f"shape_score={shape_score:.6e}, "
                    f"distribution_score={distribution_score:.6e}, "
                    f"capture_score={capture_score:.6e}, "
                    f"target_capture_fraction={target_capture_fraction:.6e}, "
                    f"order_share_cosine={order_share_cosine:.6e}, "
                    f"shape_error={shape_error:.6e}, "
                    f"off_target_fraction={off_target_fraction:.6e}, "
                    f"leakage_fraction={leakage_fraction:.6e}, "
                    f"order_sigma_deg={angular_order_soft_sigma_deg:.3f})"
                )
            
            return fom

        opt[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim[idx],
            objective_functions=[J_ldos],
            objective_arguments=[0, 1, 3, 4],  # Ex, Ey, Hx, Hy
            FoM_size=target_monitor_s,
            FoM_center=target_monitor_c,
            adj_fwd=False,
            opt_idx=idx,
            broadband_adjoint=True,
        )

        def capture_current_dipole_power(problem, channel=channel):
            freqs_hz = np.asarray(getattr(problem, "src_freqs", []), dtype=float).reshape(-1)
            if freqs_hz.size == 0:
                src_wl = np.asarray(problem.sim.src_wl, dtype=float).reshape(-1)
                freqs_hz = problem.sim.c / src_wl

            dipole_total_power = read_dipole_total_power(problem.sim.fdtd, freqs_hz)
            source_power = read_source_power(problem.sim.fdtd, freqs_hz)
            power_warning = ""
            if dipole_total_power is None or dipole_total_power <= 0.0:
                power_warning = "dipolepower unavailable; using sourcepower fallback"
                dipole_total_power = source_power
            if dipole_total_power is None or dipole_total_power <= 0.0:
                power_warning = "dipolepower and sourcepower unavailable; using floor"
                dipole_total_power = channel_power_floor

            top_monitor_transmission = read_monitor_transmission(
                problem.sim.fdtd,
                target_monitor_name,
            )
            top_flux_sign = 1.0 if top_monitor_transmission >= 0.0 else -1.0
            source_power_value = (
                max(float(source_power), channel_power_floor)
                if source_power is not None
                else np.nan
            )
            reference_top_power = (
                abs(float(top_monitor_transmission)) * source_power_value
                if np.isfinite(source_power_value)
                else np.nan
            )
            raw_top_flux_power = read_monitor_raw_flux(
                problem.sim.fdtd,
                target_monitor_name,
                channel.get("angular_target"),
                flux_sign=top_flux_sign,
            )
            if (
                raw_top_flux_power is not None
                and raw_top_flux_power > 0.0
                and np.isfinite(reference_top_power)
                and reference_top_power > 0.0
            ):
                top_flux_calibration = reference_top_power / raw_top_flux_power
            else:
                top_flux_calibration = 1.0

            channel["last_dipole_total_power"] = max(float(dipole_total_power), channel_power_floor)
            channel["last_source_power"] = source_power_value
            channel["last_top_monitor_transmission"] = float(top_monitor_transmission)
            channel["last_top_flux_sign"] = top_flux_sign
            channel["last_reference_top_power"] = reference_top_power
            channel["last_raw_top_flux_power"] = raw_top_flux_power if raw_top_flux_power is not None else np.nan
            channel["last_top_flux_calibration"] = float(top_flux_calibration)
            channel["last_power_warning"] = power_warning
            if power_warning:
                print(f"[dipole power] channel {channel['dipole_idx']}: {power_warning}")

        opt[idx].forward_result_hook = capture_current_dipole_power

        print(
            f"[dipole setup] channel {idx}: {channel['name']} "
            f"pos=({dipole_x:.3f},{dipole_y:.3f},{dipole_z:.3f}) "
            f"pol={channel['polarization']} w={channel['weight']:.3f}"
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
        "apply_filter": False,
        "vertical_grating": False,
    },
    Is_slanted_grating=False,
)
design_parameters = mapping.parameter_count
x0 = grating_initial_density * np.ones(design_parameters)
dJ_0 = np.zeros(design_cells)


def design_to_grid(design, beta=1.0):
    rho = np.asarray(design, dtype=float)
    if rho.size == design_cells:
        return rho.reshape(design_grids)
    if rho.size == design_parameters:
        return np.asarray(mapping(rho, beta), dtype=float).reshape(design_grids)
    if rho.size == Nx * Ny:
        return np.repeat(rho.reshape(Nx, Ny)[:, :, None], Nz, axis=2)
    raise ValueError(f"expected {design_cells}, {design_parameters}, or {Nx * Ny} design values, got {rho.size}")


def save_current_design_sections(design, f0_vals=None):
    rho_temp = design_to_grid(npa.clip(design, 0.0, 1.0))
    x_axis = np.linspace(-0.5 * design_s[0], 0.5 * design_s[0], Nx)
    y_axis = np.linspace(-0.5 * design_s[1], 0.5 * design_s[1], Ny)
    z_axis = np.linspace(
        design_c[2] - 0.5 * design_s[2],
        design_c[2] + 0.5 * design_s[2],
        Nz,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
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
    axes[0].set_title("x-y section at z=center")

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

    status_text = format_design_plot_status(f0_vals)
    fig.suptitle("Current design sections")
    if status_text:
        fig.text(0.5, 0.02, status_text, ha="center", va="bottom", fontsize=8.5)
        fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.92))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    path = os.path.join(design_dir, "design_iter_temp.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_adjoint_loop(opt):
    def Adjoint_loop(X, N_cases, Case=True):
        if Case == 3:
            dJ_dus = X[0]
            vals = np.asarray([
                max(
                    float(np.nan_to_num(
                        np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v),
                        nan=ldos_fom_floor,
                        posinf=ldos_score_cap,
                        neginf=ldos_fom_floor,
                    )),
                    ldos_fom_floor,
                )
                for v in N_cases
            ], dtype=float)
            vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), ldos_fom_floor)
            coeffs = ag_jacobian(combine_ldos_fom_from_values)(vals)
            coeffs = npa.where(npa.isfinite(coeffs), coeffs, 0.0)

            grad = 0.0
            for coeff, channel_grad in zip(coeffs, dJ_dus):
                channel_grad = npa.where(npa.isfinite(npa.array(channel_grad)), npa.array(channel_grad), 0.0)
                grad += coeff * channel_grad
            grad = npa.where(npa.isfinite(grad), grad, 0.0)

            print(f"[ldos] combined grad mean: {np.mean(np.abs(grad)):.6e}")
            print(f"[ldos] combined grad max: {np.max(np.abs(grad)):.6e}")
            return grad

        f0s = [0] * N_fom
        dJ_dus = [0] * N_fom
        
        for idx in range(N_fom):
            if isinstance(X, str):
                f0s[idx], dJ_dus[idx] = opt[idx](need_gradient=Case)
            else:
                rho = npa.clip(X, 0.0, 1.0)
                f0s[idx], dJ_dus[idx] = opt[idx](rho_vector=[rho], need_gradient=Case)

        f0_vals = sanitize_fom_values(f0s)

        if not isinstance(X, str):
            try:
                path = save_current_design_sections(X, f0_vals)
                print(f"[ldos] saved temporary design section: {path}")
            except Exception as exc:
                print(f"[ldos] skipped temporary design section: {exc}")

        unstable_candidate = any(getattr(problem, "last_forward_had_nonfinite", False) for problem in opt)
        if unstable_candidate:
            print("[ldos] unstable candidate detected: non-finite Lumerical field/FoM. Rejecting through backtracking.")
            zero_grads = [
                np.zeros_like(grad, dtype=float)
                if not isinstance(grad, (int, float))
                else np.zeros(design_cells, dtype=float)
                for grad in dJ_dus
            ]
            f0s = [ldos_fom_floor for _ in range(N_fom)]
            if Case:
                if isinstance(X, str):
                    return zero_grads
                return unstable_candidate_fom, f0s, zero_grads
            return unstable_candidate_fom, f0s

        f0 = combine_ldos_fom_from_values(f0_vals)
        f0_value = real_scalar_or_none(f0)
        if f0_value is not None:
            combined_fom_history.append(f0_value)

        summary = ldos_summary_from_values(f0_vals)
        for idx, channel in enumerate(target_channels):
            print(
                f"[ldos] dipole_{idx} {channel['name']} "
                f"pos=({channel['dipole_x']:.3f},{channel['dipole_y']:.3f},{channel['dipole_z']:.3f}) "
                f"pol={channel['polarization']} "
                f"w={channel['weight']:.3f} "
                f"FoM={f0_vals[idx]:.6e}"
            )
        
        print(
            f"[ldos] combined LDOS FoM: {f0:.6e} "
            f"(mean={summary['mean_fom']:.6e}, weighted_mean={summary['weighted_mean_fom']:.6e}, "
            f"min={summary['min_fom']:.6e}, "
            f"max={summary['max_fom']:.6e})"
        )

        if Case:
            if isinstance(X, str):
                return dJ_dus
            return f0, f0s, dJ_dus
        
        return f0, f0s

    return Adjoint_loop


def print_setup_summary():
    lines = [
        "[ldos] Dipole-based LDOS optimization scaffold built.",
        (
            "OLED periodic 3D freeform setup: "
            f"period={window_x}x{window_y} um, active={active_x}x{active_y} um, "
            f"air={air_top_h} um, design={grating_design_h} um, SiO2={sio2_h} um, "
            f"ITO={ito_h} um, TCTA={tcta_h} um, EML={eml_h} um, TPBi={tpbi_h} um, "
            f"Ag={ag_h} um, bottom_air={air_bot_h} um, background_index={background_index}"
        ),
        (
            f"Dipole samples ({n_dipole_positions}): r_frac={dipole_radii_frac}, "
            f"azimuth_deg={dipole_azimuths_deg}, pol={dipole_polarizations}, "
            f"weights={dipole_weights.tolist()}"
        ),
        (
            f"N_fom={N_fom}, design_grids={design_grids}, design_cells={design_cells}, "
            f"radial_grating_shape=({radial_design_grids},), design_parameters={design_parameters}, "
            f"radial_radius={radial_design_radius}"
        ),
        (
            f"boundary_mode={boundary_label}, bc_x={bc_xy}, bc_y={bc_xy}, bc_z=PML, "
            f"visible_wavelengths={visible_wavelengths}"
        ),
        (
            "FoM = weighted dipole average of "
            "(zero-order emitted power / current dipole total power)^extraction_weight "
            "* angular shape score^shape_weight"
        ),
        f"Target relative angular power curve: {target_efficiency_curve_str}",
        f"Target monitor={target_monitor_name}, center={target_monitor_c}, size={target_monitor_s}",
        f"Dipole source plane center={eml_c}, sampled in EML",
        (
            f"FoM weights: extraction={ldos_extraction_efficiency_weight}, "
            f"shape={ldos_shape_weight}, share_error={ldos_order_share_error_weight}, "
            f"order_target_min={ldos_order_target_min}, "
            f"leakage_start={angular_leakage_start_deg}, "
            f"target_capture_weight={ldos_target_capture_weight}, score_cap={ldos_score_cap}, "
            f"fom_scale={ldos_fom_scale}, fom_floor={ldos_fom_floor}, "
            f"hann_window={angular_use_hann_window}, "
            f"order_sigma_deg={angular_order_soft_sigma_deg}"
        ),
        (
            f"Postprocess: enabled={env_flag('MSOPT_OLED_POSTPROCESS', '1')}, "
            f"only={env_flag('MSOPT_OLED_POSTPROCESS_ONLY', '0')}, "
            f"bulk_grid={env_flag('MSOPT_OLED_BULK_NORMALIZATION', '1')}"
        ),
    ]
    for line in lines:
        print(line)


def save_combined_fom_curve():
    if not combined_fom_history:
        print("[ldos] skipped FoM curve: no combined FoM history")
        return
    values = np.asarray(combined_fom_history, dtype=float)
    np.savetxt(os.path.join(design_dir, "LDOS_optimized_combined_fom_history.txt"), values)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, values.size + 1), values, linewidth=1.5)
    plt.xlabel("combined FoM evaluation")
    plt.ylabel("combined LDOS FoM")
    plt.title("Dipole-based LDOS optimized FoM curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(design_dir, "LDOS_optimized_fom_curve.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"[ldos] saved FoM curve: {path}")



if __name__ == "__main__":
    if os.environ.get("MSOPT_OLED_SESSION_TEST", "").lower() in ("1", "true", "yes"):
        channel_idx = int(os.environ.get("MSOPT_OLED_SESSION_TEST_CHANNEL", "0"))
        channel = target_channels[channel_idx]
        print(f"[ldos session test] channel {channel_idx}: {channel['name']}")
        print(
            f"[ldos session test] dipole_pos=({channel['dipole_x']:.3f},{channel['dipole_y']:.3f},{channel['dipole_z']:.3f}), "
            f"efficiency_curve={target_efficiency_curve_str}, "
            f"pol={channel['polarization']}"
        )
        raise SystemExit(0)

    start = time.time()

    sim, opt, fom_history = build_optimization_problem()
    print_setup_summary()

    postprocess_only = env_flag("MSOPT_OLED_POSTPROCESS_ONLY", "0")
    if postprocess_only:
        print("[ldos] skipped optimizer: MSOPT_OLED_POSTPROCESS_ONLY is enabled")
    else:
        optimizer = ms.Opt_MS2.OPT_Ms(
            x0,
            dJ_0,
            Born_k=50,
            Initial_LR=0.2,
            Raw=False,
        )
        optimizer.flag = True
        optimizer(mapping, N_fom, make_adjoint_loop(opt))
        np.savetxt(os.path.join(design_dir, "FoM_history.txt"), np.array(fom_history, dtype=object), fmt="%s")
        save_combined_fom_curve()


    if env_flag("MSOPT_OLED_POSTPROCESS", "1"):
        print("[postprocess] dipole-based LDOS incoherent postprocess")
        def get_angular_spectrum_from_monitor(sim, monitor_name, wavelength_um, flux_sign=1.0):
            e_result = sim.fdtd.getresult(monitor_name, "E")
            h_result = sim.fdtd.getresult(monitor_name, "H")
            E = np.asarray(e_result["E"], dtype=np.complex128)
            H = np.asarray(h_result["H"], dtype=np.complex128)
            x_axis_data = np.ravel(np.asarray(e_result["x"], dtype=float))
            y_axis_data = np.ravel(np.asarray(e_result["y"], dtype=float))
            
            if E.ndim < 3 or E.shape[-1] != 3:
                raise ValueError(f"unexpected monitor E shape {E.shape}")
            if H.ndim < 3 or H.shape[-1] != 3:
                raise ValueError(f"unexpected monitor H shape {H.shape}")

            spatial_shape = E.shape[:-1]
            x_candidates = [idx for idx, size in enumerate(spatial_shape) if size == x_axis_data.size]
            y_candidates = [idx for idx, size in enumerate(spatial_shape) if size == y_axis_data.size]
            if not x_candidates or not y_candidates:
                raise ValueError(f"could not map monitor axes: E shape={E.shape}")
            
            x_axis_idx = x_candidates[0]
            y_axis_idx = next((idx for idx in y_candidates if idx != x_axis_idx), y_candidates[0])
            E = np.moveaxis(E, [x_axis_idx, y_axis_idx], [0, 1])
            H = np.moveaxis(H, [x_axis_idx, y_axis_idx], [0, 1])

            while E.ndim > 3:
                E = np.mean(E, axis=2)
            while H.ndim > 3:
                H = np.mean(H, axis=2)

            dx = float(np.mean(np.diff(x_axis_data)))
            dy = float(np.mean(np.diff(y_axis_data)))
            kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(x_axis_data.size, d=dx))
            ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(y_axis_data.size, d=dy))

            Ex_k = np.fft.fftshift(np.fft.fft2(E[:, :, 0]))
            Ey_k = np.fft.fftshift(np.fft.fft2(E[:, :, 1]))
            Hx_k = np.fft.fftshift(np.fft.fft2(H[:, :, 0]))
            Hy_k = np.fft.fftshift(np.fft.fft2(H[:, :, 1]))
            spectrum = 0.5 * float(flux_sign) * np.real(Ex_k * np.conj(Hy_k) - Ey_k * np.conj(Hx_k))

            KX, KY = np.meshgrid(kx, ky, indexing="ij")
            wavelength_m = float(wavelength_um) * 1e-6
            k0 = 2.0 * np.pi / wavelength_m
            normalized_kr = np.sqrt(KX ** 2 + KY ** 2) / k0
            propagating = normalized_kr <= 1.0
            theta_deg = np.rad2deg(np.arcsin(np.clip(normalized_kr, 0.0, 1.0)))
            
            spectrum = np.maximum(np.where(propagating, spectrum, 0.0), 0.0)
            return theta_deg, spectrum

        def integrate_annular_angle_profile(theta_deg, spectrum, angle_centers_deg):
            centers = np.asarray(angle_centers_deg, dtype=float)
            if centers.size == 0:
                return np.asarray([], dtype=float)
            if centers.size == 1:
                edges = np.asarray([0.0, 90.0], dtype=float)
            else:
                mids = 0.5 * (centers[:-1] + centers[1:])
                edges = np.concatenate(([0.0], mids, [90.0]))
            powers = []
            for idx, center in enumerate(centers):
                lo = edges[idx]
                hi = edges[idx + 1]
                if idx == centers.size - 1:
                    mask = (theta_deg >= lo) & (theta_deg <= hi)
                else:
                    mask = (theta_deg >= lo) & (theta_deg < hi)
                powers.append(float(np.sum(np.asarray(spectrum)[mask])))
            return np.asarray(powers, dtype=float)

        def run_ldos_dipole_postprocess(final_design, n_samples=20):
            print(f"[postprocess] Loading final design and setting up simulator...")
            try:
                rho = design_to_grid(final_design)
            except ValueError as exc:
                print(f"[postprocess] skipped dipole postprocess: {exc}")
                return None

            n_dipoles = int(os.environ.get("MSOPT_OLED_LDOS_POSTPROCESS_N_DIPOLES", str(n_samples)))
            print(f"[postprocess] Generating {n_dipoles} random dipole positions in EML...")
            active_radius = 0.5 * min(active_x, active_y)
            fixed_postprocess_polarization = os.environ.get(
                "MSOPT_OLED_LDOS_POSTPROCESS_POLARIZATION",
                dipole_polarization,
            ).strip().lower()
            fixed_post_theta_deg, fixed_post_phi_deg = dipole_orientation_angles(fixed_postprocess_polarization)
            np.random.seed(240)
            dipole_positions = []
            for _ in range(n_dipoles):
                r = np.random.uniform(0, active_radius)
                phi = np.random.uniform(0, 2 * np.pi)
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                z = eml_c[2]
                dipole_positions.append((float(x), float(y), float(z)))

            print(f"[postprocess] Setting up FDTD simulator...")
            sim = ms.Lumerical_utill.LumericalFDTDSimulator(
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
            sim.add_monitor(name=target_monitor_name, center=target_monitor_c, size=target_monitor_s)

            print(f"[postprocess] Running {n_dipoles} dipole forward simulations...")
            incoherent_spectrum_sum = None
            records = []

            for dipole_idx, (x, y, z) in enumerate(dipole_positions):
                print(f"[postprocess] dipole {dipole_idx+1}/{n_dipoles} at ({x:.3f}, {y:.3f}, {z:.3f})")
                
                sim.fdtd.switchtolayout()
                delete_lumerical_object(sim.fdtd, "postprocess_dipole")
                sim.fdtd.adddipole()
                sim.fdtd.set("name", "postprocess_dipole")
                sim.fdtd.set("x", x * 1e-6)
                sim.fdtd.set("y", y * 1e-6)
                sim.fdtd.set("z", z * 1e-6)
                phi_deg = fixed_post_phi_deg
                sim.fdtd.set("theta", fixed_post_theta_deg)
                sim.fdtd.set("phi", fixed_post_phi_deg)
                sim.fdtd.set("wavelength start", float(np.min(visible_wavelengths)) * 1e-6)
                sim.fdtd.set("wavelength stop", float(np.max(visible_wavelengths)) * 1e-6)
                
                sim.run(name=f"postprocess_dipole_{dipole_idx:02d}", save=True)
                
                try:
                    top_monitor_transmission = read_monitor_transmission(
                        sim.fdtd,
                        target_monitor_name,
                    )
                    flux_sign = 1.0 if top_monitor_transmission >= 0.0 else -1.0
                    theta_deg, spectrum = get_angular_spectrum_from_monitor(
                        sim,
                        target_monitor_name,
                        float(np.mean(visible_wavelengths)),
                        flux_sign=flux_sign,
                    )
                    angular_spectrum_power = float(np.sum(spectrum))
                    freqs_hz = sim.c / np.asarray(sim.src_wl, dtype=float).reshape(-1)
                    dipole_total_power = read_dipole_total_power(sim.fdtd, freqs_hz)
                    source_power = read_source_power(sim.fdtd, freqs_hz)
                    power_warning = ""
                    if dipole_total_power is None or dipole_total_power <= 0.0:
                        power_warning = "dipolepower unavailable; using sourcepower fallback"
                        dipole_total_power = source_power
                    if dipole_total_power is None or dipole_total_power <= 0.0:
                        power_warning = "dipolepower and sourcepower unavailable; using floor"
                        dipole_total_power = channel_power_floor
                    extraction_efficiency = (
                        abs(float(top_monitor_transmission))
                        / max(float(dipole_total_power), channel_power_floor)
                    )
                    
                    if incoherent_spectrum_sum is None:
                        incoherent_spectrum_sum = spectrum.copy()
                    else:
                        incoherent_spectrum_sum += spectrum
                    
                    records.append({
                        "dipole_idx": dipole_idx,
                        "x": x,
                        "y": y,
                        "z": z,
                        "theta_deg": fixed_post_theta_deg,
                        "phi_deg": phi_deg,
                        "angular_spectrum_power": angular_spectrum_power,
                        "top_monitor_transmission": float(top_monitor_transmission),
                        "dipole_total_power": float(dipole_total_power),
                        "source_power": float(source_power) if source_power is not None else np.nan,
                        "extraction_efficiency": float(extraction_efficiency),
                        "power_warning": power_warning,
                        "spectrum": spectrum,
                    })
                except Exception as exc:
                    print(f"[postprocess] warning: dipole {dipole_idx} failed: {exc}")
                    records.append({
                        "dipole_idx": dipole_idx,
                        "x": x,
                        "y": y,
                        "z": z,
                        "theta_deg": fixed_post_theta_deg,
                        "phi_deg": phi_deg,
                        "angular_spectrum_power": np.nan,
                        "top_monitor_transmission": np.nan,
                        "dipole_total_power": np.nan,
                        "source_power": np.nan,
                        "extraction_efficiency": np.nan,
                        "power_warning": str(exc),
                        "spectrum": None,
                        "error": str(exc),
                    })

            try:
                sim.fdtd.close()
                print(f"[postprocess] closed FDTD simulator")
            except Exception as exc:
                print(f"[postprocess] warning: failed to close simulator: {exc}")

            if incoherent_spectrum_sum is None or np.all(incoherent_spectrum_sum == 0):
                print("[postprocess] skipped: incoherent spectrum sum is zero or empty")
                return None

            angle_resolution = int(os.environ.get("MSOPT_OLED_POSTPROCESS_ANGLE_RES", "181"))
            angles_deg = np.linspace(0.0, 90.0, angle_resolution)
            angle_powers = integrate_annular_angle_profile(
                theta_deg,
                incoherent_spectrum_sum,
                angles_deg,
            )

            if np.max(angle_powers) > 0:
                angle_powers_normalized = angle_powers / np.max(angle_powers)
            else:
                angle_powers_normalized = angle_powers

            target_efficiency_at_angles = np.array([
                interpolate_efficiency_at_angle(float(angle), target_efficiency_curve)
                for angle in angles_deg
            ])
            target_efficiency_zero = max(float(target_efficiency_at_angles[0]), 1e-12)
            target_ratio_at_angles = target_efficiency_at_angles / target_efficiency_zero
            zero_angle_power = max(float(angle_powers[0]), channel_power_floor)
            angle_ratios_to_zero = angle_powers / (zero_angle_power + 1e-30)
            total_angle_power = float(np.sum(angle_powers))
            leakage_mask = angles_deg >= angular_leakage_start_deg
            leakage_power = float(np.sum(angle_powers[leakage_mask]))
            leakage_fraction = (
                leakage_power / total_angle_power
                if total_angle_power > 0.0
                else 0.0
            )
            zero_angle_fraction = (
                float(angle_powers[0]) / total_angle_power
                if total_angle_power > 0.0
                else 0.0
            )
            extraction_values = np.asarray(
                [
                    rec.get("extraction_efficiency", np.nan)
                    for rec in records
                    if "spectrum" in rec and rec["spectrum"] is not None
                ],
                dtype=float,
            )
            extraction_values = extraction_values[np.isfinite(extraction_values)]
            mean_extraction_efficiency = (
                float(np.mean(extraction_values))
                if extraction_values.size
                else np.nan
            )
            std_extraction_efficiency = (
                float(np.std(extraction_values, ddof=1))
                if extraction_values.size > 1
                else 0.0
            )
            min_extraction_efficiency = (
                float(np.min(extraction_values))
                if extraction_values.size
                else np.nan
            )
            max_extraction_efficiency = (
                float(np.max(extraction_values))
                if extraction_values.size
                else np.nan
            )

            results_path = os.path.join(design_dir, "LDOS_postprocess_dipole_results.txt")
            with open(results_path, "w", encoding="utf-8") as fp:
                fp.write(f"method incoherent_ldos_dipole_sampling\n")
                fp.write(f"n_dipoles {len(records)}\n")
                fp.write(f"n_successful {sum(1 for r in records if 'spectrum' in r and r['spectrum'] is not None)}\n")
                fp.write(f"zero_angle_power {zero_angle_power:.16e}\n")
                fp.write(f"total_angle_power {total_angle_power:.16e}\n")
                fp.write(f"zero_angle_fraction {zero_angle_fraction:.16e}\n")
                fp.write(f"leakage_start_deg {angular_leakage_start_deg:.6e}\n")
                fp.write(f"leakage_power {leakage_power:.16e}\n")
                fp.write(f"leakage_fraction {leakage_fraction:.16e}\n")
                fp.write(f"mean_extraction_efficiency {mean_extraction_efficiency:.16e}\n")
                fp.write(f"std_extraction_efficiency {std_extraction_efficiency:.16e}\n")
                fp.write(f"min_extraction_efficiency {min_extraction_efficiency:.16e}\n")
                fp.write(f"max_extraction_efficiency {max_extraction_efficiency:.16e}\n")
                fp.write("angle_resolved_emission\n")
                fp.write("theta_deg emission_power normalized_power ratio_to_zero target_ratio\n")
                for angle, power, normalized, ratio, target_ratio in zip(
                    angles_deg,
                    angle_powers,
                    angle_powers_normalized,
                    angle_ratios_to_zero,
                    target_ratio_at_angles,
                ):
                    fp.write(
                        f"{angle:.2f} {power:.6e} {normalized:.6e} "
                        f"{ratio:.6e} {target_ratio:.6e}\n"
                    )
            print(f"[postprocess] saved dipole results: {results_path}")

            records_path = os.path.join(design_dir, "LDOS_postprocess_dipole_records.txt")
            with open(records_path, "w", encoding="utf-8") as fp:
                fp.write(
                    "dipole_idx x_um y_um z_um theta_deg phi_deg "
                    "angular_spectrum_power top_monitor_transmission "
                    "dipole_total_power source_power extraction_efficiency power_warning\n"
                )
                for rec in records:
                    if "spectrum" in rec and rec["spectrum"] is not None:
                        fp.write(
                            f"{rec['dipole_idx']} {rec['x']:.6e} {rec['y']:.6e} {rec['z']:.6e} "
                            f"{rec['theta_deg']:.6e} {rec['phi_deg']:.6e} "
                            f"{rec['angular_spectrum_power']:.6e} "
                            f"{rec['top_monitor_transmission']:.6e} "
                            f"{rec['dipole_total_power']:.6e} "
                            f"{rec['source_power']:.6e} "
                            f"{rec['extraction_efficiency']:.6e} "
                            f"{rec.get('power_warning', '')}\n"
                        )
            print(f"[postprocess] saved dipole records: {records_path}")

            fig = plt.figure(figsize=(12, 4.2))
            ax0 = fig.add_subplot(121, projection="polar")
            signed_angles, signed_emission = symmetric_angle_series(angles_deg, angle_powers_normalized)
            ax0.plot(np.deg2rad(signed_angles), signed_emission, "b-", linewidth=2, label="Emission")
            setup_semicircle_polar_axis(ax0, "LDOS incoherent dipole emission", rmax=1.0)
            ax0.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16))

            ax1 = fig.add_subplot(122, projection="polar")
            signed_target_angles, signed_target_ratio = symmetric_angle_series(angles_deg, target_ratio_at_angles)
            signed_ratio_angles, signed_achieved_ratio = symmetric_angle_series(angles_deg, angle_ratios_to_zero)
            ratio_values = np.concatenate((
                np.asarray(signed_target_ratio, dtype=float),
                np.asarray(signed_achieved_ratio, dtype=float),
            ))
            finite_ratio_values = ratio_values[np.isfinite(ratio_values)]
            ratio_rmax = max(1.1, float(np.max(finite_ratio_values)) * 1.05) if finite_ratio_values.size else 1.1
            ax1.plot(np.deg2rad(signed_target_angles), signed_target_ratio, "r-", linewidth=2, label="Target ratio")
            ax1.plot(np.deg2rad(signed_ratio_angles), signed_achieved_ratio, "b-", linewidth=2, label="Achieved ratio")
            setup_semicircle_polar_axis(ax1, "Target vs. achieved angular ratio", rmax=ratio_rmax)
            ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16))

            plot_path = os.path.join(design_dir, "LDOS_postprocess_emission_efficiency.png")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)
            print(f"[postprocess] saved emission efficiency plot: {plot_path}")

            return records

        def postprocess_final_design(opt):
            design_path = os.path.join(design_dir, "lastdesign.txt")
            if not os.path.exists(design_path):
                print(f"[postprocess] skipped: final design not found at {design_path}")
                return None

            print(f"[postprocess] loading final design from {design_path}")
            final_design = np.loadtxt(design_path)
            
            n_dipoles = int(os.environ.get("MSOPT_OLED_LDOS_POSTPROCESS_N_DIPOLES", "20"))
            print(f"[postprocess] starting {n_dipoles}-dipole incoherent postprocess")
            try:
                records = run_ldos_dipole_postprocess(final_design, n_samples=n_dipoles)
                if records is not None:
                    print(f"[postprocess] successfully completed with {len(records)} dipoles")
                else:
                    print(f"[postprocess] postprocess returned no results")
            except Exception as exc:
                print(f"[postprocess] postprocess failed: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()
                raise

        postprocess_final_design(opt)
    else:
        print("[postprocess] skipped all postprocess: MSOPT_OLED_POSTPROCESS is disabled")

    print(f"Runtime setup time: {time.time() - start:.2f} seconds")
