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
# Boundary: selectable in x/y and PML in z. Set
# MSOPT_OLED_BOUNDARY_MODE to "Bloch", "Periodic", or "PML".
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

target_monitor_name = "FoM_monitor"
target_monitor_s = out_s
target_monitor_c = out_c

Nx = int(round(design_s[0] * resolution)) + 1
Ny = int(round(design_s[1] * resolution)) + 1
Nz = int(round(design_s[2] * resolution)) + 1
design_grids = [Nx, Ny, Nz]
design_cells = Nx * Ny * Nz


# =============================================================================
# Dipole-based LDOS optimization setup
# =============================================================================

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
raw_dipole_radii_frac = _parse_float_list_env("MSOPT_OLED_DIPOLE_RADII_FRAC", [0.0, 0.4, 0.8])
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

# Target efficiency curve: angle (deg) -> efficiency ratio
# Format: "theta1:eff1,theta2:eff2,..." (linearly interpolated between points)
# Example: "0:1.0,45:0.85" means 0° has efficiency 1.0, 45° has efficiency 0.85
target_efficiency_curve_str = os.environ.get("MSOPT_OLED_TARGET_EFFICIENCY_CURVE", "0:1.0,45:0.85,60:0.0")


def parse_efficiency_curve(curve_str):
    """
    Parse efficiency curve specification into (theta, efficiency) pairs.
    
    Format: "theta1:eff1,theta2:eff2,..."
    Example: "0:1.0,45:0.85" -> [(-45.0, 0.85), (0.0, 1.0), (45.0, 0.85)]
    
    Args:
        curve_str: Efficiency curve specification string
        
    Returns:
        List of (theta_deg, efficiency) tuples, sorted by theta
    """
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

    # Build symmetric curve: mirror positive angles to negative if not already present
    efficiency_map = {float(theta): float(eff) for theta, eff in curve_points}
    for theta, eff in list(efficiency_map.items()):
        if theta > 0.0 and -theta not in efficiency_map:
            efficiency_map[-theta] = eff
        if theta < 0.0 and -theta not in efficiency_map:
            efficiency_map[-theta] = eff

    curve_points = sorted((theta, efficiency_map[theta]) for theta in efficiency_map)
    return curve_points


def interpolate_efficiency_at_angle(theta_deg, efficiency_curve):
    """
    Linearly interpolate efficiency value at given angle.
    
    Args:
        theta_deg: Angle in degrees
        efficiency_curve: List of (theta, efficiency) tuples
        
    Returns:
        Interpolated efficiency value
    """
    if not efficiency_curve:
        return 1.0
    
    # Find bracketing points
    for i in range(len(efficiency_curve) - 1):
        theta1, eff1 = efficiency_curve[i]
        theta2, eff2 = efficiency_curve[i + 1]
        
        if theta1 <= theta_deg <= theta2:
            # Linear interpolation
            if abs(theta2 - theta1) < 1e-10:
                return eff1
            frac = (theta_deg - theta1) / (theta2 - theta1)
            return eff1 + frac * (eff2 - eff1)
    
    # Outside range: use nearest boundary
    if theta_deg < efficiency_curve[0][0]:
        return efficiency_curve[0][1]
    else:
        return efficiency_curve[-1][1]


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


angular_flux_cosine_weight = env_flag("MSOPT_OLED_ANGULAR_FLUX_COSINE_WEIGHT", "1")
angular_use_hann_window = env_flag("MSOPT_OLED_ANGULAR_HANN_WINDOW", "0")
angular_diagnostic_bin_half_width_deg = float(
    os.environ.get("MSOPT_OLED_ANGULAR_BIN_HALF_WIDTH_DEG", "5.0")
)


def compute_diffraction_orders(wavelength_um, period_um, max_order=5):
    """
    Compute possible diffraction orders and their angles.
    
    Using grating equation: sin(theta_m) = m * lambda / period
    
    Args:
        wavelength_um: wavelength in micrometers
        period_um: grating period in micrometers
        max_order: maximum diffraction order to compute
        
    Returns:
        List of (order_m, theta_deg) tuples for valid propagating orders
    """
    orders = []
    for m in range(-max_order, max_order + 1):
        sin_theta = (m * wavelength_um) / period_um
        if abs(sin_theta) <= 1.0:
            theta_rad = np.arcsin(sin_theta)
            theta_deg = np.rad2deg(theta_rad)
            orders.append((m, float(theta_deg)))
    return orders


def compute_target_field_with_diffraction_orders(
    wavelength_um, period_um, efficiency_curve, 
    kx=None, ky=None, k0=None, resolution=None
):
    """
    Generate target field by combining diffraction orders with efficiency weighting.
    
    Args:
        wavelength_um: Operating wavelength
        period_um: Grating period (= window size typically)
        efficiency_curve: Efficiency vs angle curve
        kx, ky: k-space grids (optional for spectral info)
        k0: Free-space k-magnitude
        resolution: Optional, for fallback pure angle computation
        
    Returns:
        Dictionary with target field information including:
        - diffraction_orders: Possible orders and angles
        - angle_efficiencies: Efficiency at each order angle
        - target_spectrum: Relative amplitude at each angle
        - radial_orders: Unique |m| ring orders for cylindrical target synthesis
    """
    # Compute all possible diffraction orders
    diffraction_orders = compute_diffraction_orders(wavelength_um, period_um, max_order=10)
    
    # Evaluate efficiency at each diffraction angle
    angle_efficiencies = []
    target_spectrum = []
    
    for order_m, theta_deg in diffraction_orders:
        efficiency = interpolate_efficiency_at_angle(float(theta_deg), efficiency_curve)
        angle_efficiencies.append((order_m, theta_deg, efficiency))
        # Amplitude is square root of efficiency
        amplitude = np.sqrt(max(efficiency, 1e-10))
        target_spectrum.append(amplitude)
    
    target_spectrum = np.asarray(target_spectrum, dtype=float)
    if np.max(np.abs(target_spectrum)) > 0:
        target_spectrum = target_spectrum / np.max(np.abs(target_spectrum))

    radial_order_map = {}
    for (order_m, theta_deg, efficiency), amplitude in zip(angle_efficiencies, target_spectrum):
        radial_order = abs(int(order_m))
        radial_angle = abs(float(theta_deg))
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
    
    return {
        "diffraction_orders": diffraction_orders,
        "angle_efficiencies": angle_efficiencies,
        "target_spectrum": target_spectrum,
        "radial_orders": radial_orders,
        "wavelength_um": wavelength_um,
        "period_um": period_um,
    }


def visualize_target_field_info(target_field_info, design_dir):
    """
    Save visualization of target field properties.
    
    Args:
        target_field_info: Dictionary from compute_target_field_with_diffraction_orders
        design_dir: Directory to save visualizations
    """
    diffraction_orders = target_field_info["diffraction_orders"]
    angle_efficiencies = target_field_info["angle_efficiencies"]
    target_spectrum = target_field_info["target_spectrum"]
    radial_orders = target_field_info["radial_orders"]
    wavelength = target_field_info["wavelength_um"]
    period = target_field_info["period_um"]
    
    # Extract data
    orders = np.array([order for order, _, _ in angle_efficiencies])
    angles = np.array([angle for _, angle, _ in angle_efficiencies])
    efficiencies = np.array([eff for _, _, eff in angle_efficiencies])
    radial_order_ids = np.array([order for order, _, _, _ in radial_orders])
    radial_angles = np.array([angle for _, angle, _, _ in radial_orders])
    radial_efficiencies = np.array([eff for _, _, eff, _ in radial_orders])
    radial_spectrum = np.array([amp for _, _, _, amp in radial_orders])
    
    # Plot 1: Efficiency curve with diffraction orders
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Smooth efficiency curve for reference
    angle_range = np.linspace(0.0, max(float(np.max(np.abs(angles))) + 5.0, 1.0), 181)
    efficiency_range = np.array([
        interpolate_efficiency_at_angle(ang, target_efficiency_curve)
        for ang in angle_range
    ])
    
    ax1.plot(angle_range, efficiency_range, "b-", linewidth=2, label="Target efficiency curve")
    ax1.scatter(
        radial_angles,
        radial_efficiencies,
        c=radial_spectrum,
        cmap="viridis",
        s=100,
        label="Radial diffraction orders",
        zorder=5,
    )
    ax1.set_xlabel("Polar radiation angle (degrees)")
    ax1.set_ylabel("Target efficiency")
    ax1.set_title("Cylindrical target efficiency")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Cylindrical diffraction ring spectrum
    ax2.bar(radial_angles, radial_spectrum, width=2.0, alpha=0.7)
    for order, angle in zip(radial_order_ids, radial_angles):
        ax2.text(angle, 0.02, f"|m|={int(order)}", rotation=90, va="bottom", ha="center", fontsize=8)
    ax2.set_xlabel("Polar radiation angle (degrees)")
    ax2.set_ylabel("Normalized amplitude")
    ax2.set_title(f"Cylindrical ring spectrum (lambda={wavelength}um, period={period}um)")
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(design_dir, "LDOS_target_field_info.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[ldos setup] saved target field info: {path}")
    
    # Save text summary
    summary_path = os.path.join(design_dir, "LDOS_target_field_summary.txt")
    with open(summary_path, "w") as fp:
        fp.write(f"Target field specification\n")
        fp.write(f"Wavelength: {wavelength} μm\n")
        fp.write(f"Grating period: {period} μm\n")
        fp.write(f"Efficiency curve: {target_efficiency_curve_str}\n")
        fp.write(f"\nDiffraction orders:\n")
        fp.write(f"Order\tAngle(deg)\tTarget_Eff\tSpectrum_Amp\n")
        for order, angle, eff, amp in zip(orders, angles, efficiencies, target_spectrum):
            fp.write(f"{int(order)}\t{angle:8.3f}\t{eff:8.4f}\t{amp:8.4f}\n")
        fp.write(f"\nCylindrical radial orders:\n")
        fp.write(f"|Order|\tPolar_Angle(deg)\tTarget_Eff\tRing_Amp\n")
        for order, angle, eff, amp in radial_orders:
            fp.write(f"{int(order)}\t{angle:8.3f}\t{eff:8.4f}\t{amp:8.4f}\n")
    print(f"[ldos setup] saved target field summary: {summary_path}")

    profile_path = os.path.join(design_dir, "LDOS_target_field_profile.csv")
    with open(profile_path, "w", encoding="utf-8") as fp:
        fp.write("signed_order,angle_deg,target_efficiency,spectrum_amplitude\n")
        for order, angle, eff, amp in zip(orders, angles, efficiencies, target_spectrum):
            fp.write(f"{int(order)},{angle:.6f},{eff:.6f},{amp:.6f}\n")
    print(f"[ldos setup] saved target field profile: {profile_path}")

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


def build_angular_power_target(x_axis, y_axis, target_field_info, efficiency_curve):
    """
    Build a k-space target map for cylindrically symmetric angular emission.

    The target is a radial weighting function on the top monitor's angular
    spectrum. It integrates the full azimuthal direction for each theta instead
    of enforcing a coherent real-space beam template.
    """
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

    if angular_flux_cosine_weight:
        flux_weight = np.sqrt(np.clip(1.0 - normalized_kr ** 2, 0.0, 1.0))
    else:
        flux_weight = np.ones_like(normalized_kr)
    flux_weight = np.where(propagating, flux_weight, 0.0)

    zero_mask = angular_bin_mask(
        theta_grid_deg,
        0.0,
        angular_diagnostic_bin_half_width_deg,
        propagating,
    )
    target0 = float(np.interp(0.0, angle_points, efficiency_points))
    target0 = max(target0, 1e-12)
    ratio_match_masks = []
    ratio_target_values = []
    for angle_deg, efficiency in zip(angle_points, efficiency_points):
        angle_deg = float(angle_deg)
        if angle_deg <= 1e-12:
            continue
        ratio_match_masks.append(
            {
                "theta_deg": angle_deg,
                "mask": angular_bin_mask(
                    theta_grid_deg,
                    angle_deg,
                    angular_diagnostic_bin_half_width_deg,
                    propagating,
                ),
            }
        )
        ratio_target_values.append(float(efficiency) / target0)

    diagnostic_ring_masks = []
    for radial_order, theta_deg, _eff, _amp in target_field_info["radial_orders"]:
        diagnostic_ring_masks.append(
            {
                "order": int(radial_order),
                "theta_deg": float(theta_deg),
                "mask": angular_bin_mask(
                    theta_grid_deg,
                    theta_deg,
                    angular_diagnostic_bin_half_width_deg,
                    propagating,
                ),
            }
        )

    if angular_use_hann_window:
        window = np.outer(np.hanning(nx), np.hanning(ny))
    else:
        window = np.ones((nx, ny), dtype=float)

    return {
        "x_size": nx,
        "y_size": ny,
        "dft_x": build_dft_matrix(nx),
        "dft_y": build_dft_matrix(ny),
        "kx": kx,
        "ky": ky,
        "theta_grid_deg": theta_grid_deg,
        "target_map": np.asarray(target_map, dtype=float),
        "propagating": np.asarray(propagating, dtype=float),
        "flux_weight": np.asarray(flux_weight, dtype=float),
        "window": np.asarray(window, dtype=float),
        "angle_points": angle_points,
        "efficiency_points": efficiency_points,
        "zero_angle_mask": zero_mask,
        "ratio_match_masks": ratio_match_masks,
        "ratio_target_values": np.asarray(ratio_target_values, dtype=float),
        "diagnostic_ring_masks": diagnostic_ring_masks,
    }


def save_angular_power_target_preview(angular_target, design_dir, file_prefix="LDOS_angular_target"):
    kx = np.asarray(angular_target["kx"], dtype=float)
    ky = np.asarray(angular_target["ky"], dtype=float)
    target_map = np.asarray(angular_target["target_map"], dtype=float)
    theta_grid = np.asarray(angular_target["theta_grid_deg"], dtype=float)
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
    np.save(os.path.join(design_dir, f"{file_prefix}_map.npy"), target_map)
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


def compute_angular_power_metrics(Ex, Ey, Ez, angular_target):
    nx = int(angular_target["x_size"])
    ny = int(angular_target["y_size"])
    spectrum_xy = 0.0

    for field in (Ex, Ey, Ez):
        if field.shape[0] != nx or field.shape[1] != ny:
            raise ValueError(
                f"angular target shape mismatch: field={field.shape}, target=({nx}, {ny})"
            )
        field = field * expand_xy_weight(angular_target["window"], field.ndim)
        spectrum = npa.abs(dft2_xy(field, angular_target)) ** 2
        if spectrum.ndim > 2:
            spectrum = npa.sum(spectrum, axis=tuple(range(2, spectrum.ndim)))
        spectrum_xy = spectrum_xy + spectrum

    weighted_spectrum = spectrum_xy * npa.asarray(angular_target["flux_weight"])
    top_power = npa.sum(weighted_spectrum * npa.asarray(angular_target["propagating"]))
    zero_power = npa.sum(weighted_spectrum * npa.asarray(angular_target["zero_angle_mask"]))
    ratio_powers = [
        npa.sum(weighted_spectrum * npa.asarray(entry["mask"]))
        for entry in angular_target["ratio_match_masks"]
    ]
    ring_powers = [
        npa.sum(weighted_spectrum * npa.asarray(entry["mask"]))
        for entry in angular_target["diagnostic_ring_masks"]
    ]
    return top_power, zero_power, ratio_powers, ring_powers


def compute_monitor_angular_metrics(monitor_result, angular_target):
    E = np.asarray(monitor_result["E"], dtype=np.complex128)
    if E.shape[-1] != 3:
        raise ValueError(f"unexpected monitor E shape {E.shape}")
    top_power, zero_power, ratio_powers, ring_powers = compute_angular_power_metrics(
        E[..., 0],
        E[..., 1],
        E[..., 2],
        angular_target,
    )
    return (
        float(np.real(top_power)),
        float(np.real(zero_power)),
        [float(np.real(power)) for power in ratio_powers],
        [float(np.real(power)) for power in ring_powers],
    )



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
ldos_ratio_match_weight = float(
    os.environ.get(
        "MSOPT_OLED_RATIO_MATCH_WEIGHT",
        os.environ.get(
            "MSOPT_OLED_ANGULAR_SHAPE_WEIGHT",
            os.environ.get("MSOPT_OLED_LDOS_FIELD_MATCH_WEIGHT", "1.0"),
        ),
    )
)
ldos_zero_emission_weight = float(
    os.environ.get(
        "MSOPT_OLED_ZERO_EMISSION_WEIGHT",
        os.environ.get("MSOPT_OLED_LDOS_EFFICIENCY_WEIGHT", "1.0"),
    )
)
ldos_score_cap = float(os.environ.get("MSOPT_OLED_LDOS_SCORE_CAP", "10.0"))
channel_power_floor = float(os.environ.get("MSOPT_OLED_CHANNEL_POWER_FLOOR", "1e-12"))
unstable_candidate_fom = float(os.environ.get("MSOPT_OLED_UNSTABLE_CANDIDATE_FOM", "-1e30"))

current_binarization_fraction = 1.0
penalty_ramp_start = float(os.environ.get("MSOPT_OLED_PENALTY_RAMP_START", "0.20"))
penalty_ramp_end = float(os.environ.get("MSOPT_OLED_PENALTY_RAMP_END", "0.90"))
# Ez can be added as a weighted intensity term, but it is not balanced against Ex/Ey.
uniformity_power = float(os.environ.get("MSOPT_OLED_UNIFORMITY_POWER", "0.0"))

""" FoM subfunctions for dipole-based LDOS optimization """

def real_scalar_or_none(value):
    try:
        return float(np.real(value))
    except (TypeError, ValueError):
        return None


def binarization_fraction_from_design(X):
    if X.size == 0:
        return 1.0
    return float(np.mean((X <= 1e-3) | (X >= 1.0 - 1e-3)))


def penalty_ramp_fraction(binarization_fraction):
    if penalty_ramp_end <= penalty_ramp_start:
        return 1.0
    return float(np.clip(
        (binarization_fraction - penalty_ramp_start) / (penalty_ramp_end - penalty_ramp_start),
        0.0,
        1.0,
    ))


def update_ldos_penalty_weights(X):
    global current_binarization_fraction
    
    if isinstance(X, str):
        current_binarization_fraction = 1.0
        return current_binarization_fraction
    
    current_binarization_fraction = binarization_fraction_from_design(X)
    return current_binarization_fraction


def compute_angular_power_scores(Ex, Ey, Ez, channel):
    """
    Compute cylindrically averaged angular emission scores from the top monitor.

    Returns:
        zero_emission: 0-degree bin power normalized to the bulk 0-degree reference
        ratio_match: normalized overlap between measured and target P(theta)/P0 ratios
        ratio_error: diagnostic mean squared ratio mismatch
        ring_powers: diagnostic power in each configured radial diffraction ring
    """
    angular_target = channel.get("angular_target")
    if angular_target is None:
        raise ValueError("angular_target is missing from the channel configuration")

    _top_power, zero_power, ratio_powers, ring_powers = compute_angular_power_metrics(
        Ex,
        Ey,
        Ez,
        angular_target,
    )
    zero_power = npa.maximum(npa.where(npa.isfinite(zero_power), zero_power, 0.0), channel_power_floor)
    bulk_zero_ref = max(float(channel.get("bulk_zero_power_ref", 1.0)), channel_power_floor)

    ratio_error = 0.0
    ratio_count = 0
    ratio_dot = 1.0
    ratio_norm = 1.0
    target_norm = 1.0
    for ratio_power, target_ratio in zip(ratio_powers, angular_target["ratio_target_values"]):
        ratio_power = npa.maximum(npa.where(npa.isfinite(ratio_power), ratio_power, 0.0), 0.0)
        ratio = ratio_power / (zero_power + 1e-30)
        target_ratio = float(target_ratio)
        ratio_error = ratio_error + (ratio - target_ratio) ** 2
        ratio_dot = ratio_dot + ratio * target_ratio
        ratio_norm = ratio_norm + ratio ** 2
        target_norm = target_norm + target_ratio ** 2
        ratio_count += 1
    if ratio_count > 0:
        ratio_error = ratio_error / float(ratio_count)

    ratio_error = npa.maximum(ratio_error, 0.0)
    ratio_match = ratio_dot ** 2 / (ratio_norm * target_norm + 1e-30)
    zero_emission = npa.maximum(zero_power / (bulk_zero_ref + 1e-30), 0.0)
    return zero_emission, ratio_match, ratio_error, ring_powers



def combine_ldos_fom_from_values(vals):
    """
    Combine LDOS scores from all dipole positions into single FoM.
    
    Args:
        vals: List of FoM values from each channel/dipole position
        
    Returns:
        Combined scalar FoM value
    """
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), channel_power_floor)
    weights = npa.asarray(channel_weights, dtype=float)
    weight_sum = npa.sum(weights)
    if float(weight_sum) > 0.0:
        weights = weights / weight_sum
    else:
        weights = npa.ones_like(weights) / max(float(weights.size), 1.0)

    # Weighted average across the configured dipole samples.
    combined_fom = npa.sum(vals * weights)
    
    return combined_fom


def ldos_summary_from_values(vals):
    """
    Generate summary statistics from LDOS FoM values.
    
    Args:
        vals: List of FoM values from each channel
        
    Returns:
        Dictionary with summary metrics
    """
    vals = np.nan_to_num(np.asarray(vals, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, channel_power_floor)
    
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
    """
    Build dipole-based LDOS optimization problem with multiple dipole positions.
    
    Each dipole position (center, 0.4R, 0.8R) gets its own simulator and optimization instance.
    Forward simulations: EML dipole excitation -> collect field at the upper-air target plane.
    Adjoint: backpropagate from design region to maximize coupling
    """
    # Compute and visualize target field information
    print("[ldos] Computing target field with diffraction order analysis...")
    target_field_info = compute_target_field_with_diffraction_orders(
        wavelength_um=float(np.mean(visible_wavelengths)),
        period_um=float(window_x),
        efficiency_curve=target_efficiency_curve,
    )
    visualize_target_field_info(target_field_info, design_dir)
    
    fom_history = [[] for _ in range(N_fom)]
    sim = [None] * N_fom
    opt = [None] * N_fom
    use_bulk_normalization = env_flag("MSOPT_OLED_BULK_NORMALIZATION", "1")

    for idx, channel in enumerate(target_channels):
        dipole_x = channel["dipole_x"]
        dipole_y = channel["dipole_y"]
        dipole_z = channel["dipole_z"]
        
        # Create simulator with selected lateral BC and absorbing z boundaries.
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

        # Add dipole source at the configured position and polarization.
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

        # The objective is the emitted field in the upper air region, not the
        # near field on the EML source plane.
        sim[idx].add_monitor(name=target_monitor_name, center=target_monitor_c, size=target_monitor_s)

        # Measure a bulk-reference emission level before adding the OLED stack.
        # This acts as the LDOS normalization term requested by the user.
        if use_bulk_normalization:
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
            bulk_top_power, bulk_zero_power, bulk_ratio_powers, bulk_ring_powers = compute_monitor_angular_metrics(
                bulk_result,
                channel["angular_target"],
            )
            channel["bulk_top_power_ref"] = max(bulk_top_power, channel_power_floor)
            channel["bulk_zero_power_ref"] = max(bulk_zero_power, channel_power_floor)
            channel["bulk_ratio_powers"] = bulk_ratio_powers
            channel["bulk_ring_powers"] = bulk_ring_powers
            if idx == 0:
                save_angular_power_target_preview(
                    channel["angular_target"],
                    design_dir,
                    file_prefix="LDOS_angular_target",
                )
            sim[idx].fdtd.switchtolayout()
            print(
                f"[ldos setup] channel {idx} bulk P0 reference="
                f"{channel['bulk_zero_power_ref']:.6e}, "
                f"bulk top reference={channel['bulk_top_power_ref']:.6e}"
            )
        else:
            channel["bulk_top_power_ref"] = 1.0
            channel["bulk_zero_power_ref"] = 1.0
            channel["bulk_ratio_powers"] = []
            channel["angular_target"] = None

        # Add OLED stack
        add_oled_stack(sim[idx], float(np.mean(visible_wavelengths)))

        # Add design grating
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

        # Add design monitor for adjoint
        sim[idx].add_design_monitor()

        # Ensure source wavelengths are defined before adding the monitor
        if not hasattr(sim[idx], "src_wl") or sim[idx].src_wl is None:
            sim[idx].src_wl = np.asarray(visible_wavelengths, dtype=float).reshape(-1) * sim[idx].unit
            sim[idx].src_bw = 0.0
        
        # Define objective function: angular-power LDOS FoM
        def J_ldos(E_x, E_y, E_z, channel_idx=idx, channel=channel):
            """
            Objective function combining 0-degree emission and angular ratio matching.
            
            FoM = (bulk-normalized P0 ** zero_emission_weight)
                  * (P(theta)/P0 ratio match ** ratio_match_weight).
            """
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

            zero_emission, ratio_match, ratio_error, _ring_powers = compute_angular_power_scores(
                E_x,
                E_y,
                E_z,
                channel,
            )
            
            # Combined FoM
            zero_emission = npa.clip(zero_emission, 0.0, ldos_score_cap)
            ratio_match = npa.clip(ratio_match, 0.0, 1.0)
            fom = npa.power(zero_emission, ldos_zero_emission_weight) * npa.power(
                ratio_match,
                ldos_ratio_match_weight,
            )
            fom = npa.clip(fom, 0.0, ldos_score_cap)
            
            fom_value = real_scalar_or_none(fom)
            if fom_value is not None:
                fom_history[channel_idx].append(fom_value)
                print(
                    f"[dipole {channel_idx}] {channel['name']} "
                    f"pos=({channel['dipole_x']:.3f},{channel['dipole_y']:.3f},{channel['dipole_z']:.3f}) "
                    f"pol={channel['polarization']} "
                    f"w={channel['weight']:.3f} "
                    f"FoM={fom:.6e} "
                    f"(P0_norm={zero_emission:.6e}, "
                    f"ratio_match={ratio_match:.6e}, "
                    f"ratio_error={ratio_error:.6e})"
                )
            
            return fom

        # Create optimization problem
        opt[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim[idx],
            objective_functions=[J_ldos],
            objective_arguments=[0, 1, 2],  # Ex, Ey, Ez
            FoM_size=target_monitor_s,
            FoM_center=target_monitor_c,
            adj_fwd=False,
            opt_idx=idx,
            broadband_adjoint=True,
        )
        
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
        """
        Compute gradients and FoM for all dipole positions.
        
        Args:
            X: Design parameter vector
            N_cases: Number of FoM evaluations needed
            Case: True to include gradient computation, False for FoM only
            
        Returns:
            FoM values and gradients for all dipoles
        """
        if Case == 3:
            dJ_dus = X[0]
            vals = np.asarray([
                max(float(np.nan_to_num(np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v))), channel_power_floor)
                for v in N_cases
            ], dtype=float)
            vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), channel_power_floor)
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

        update_ldos_penalty_weights(X)
        
        f0s = [0] * N_fom
        dJ_dus = [0] * N_fom
        
        # Run forward and adjoint for each dipole position
        for idx in range(N_fom):
            if isinstance(X, str):
                f0s[idx], dJ_dus[idx] = opt[idx](need_gradient=Case)
            else:
                rho = npa.clip(X, 0.0, 1.0)
                f0s[idx], dJ_dus[idx] = opt[idx](rho_vector=[rho], need_gradient=Case)

        # Visualize current design
        if not isinstance(X, str):
            try:
                rho_temp = np.asarray(npa.clip(X, 0.0, 1.0), dtype=float)
                if rho_temp.size == design_cells:
                    rho_temp = rho_temp.reshape(design_grids)
                elif rho_temp.size == design_parameters:
                    rho_temp = np.asarray(mapping(rho_temp, 1.0), dtype=float).reshape(design_grids)
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
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
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
                
                fig.suptitle("Current design sections")
                fig.tight_layout()
                path = os.path.join(design_dir, "design_iter_temp.png")
                fig.savefig(path, dpi=200)
                plt.close(fig)
                print(f"[ldos] saved temporary design section: {path}")
            except Exception as exc:
                print(f"[ldos] skipped temporary design section: {exc}")

        # Check for numerical instability
        unstable_candidate = any(getattr(problem, "last_forward_had_nonfinite", False) for problem in opt)
        if unstable_candidate:
            print("[ldos] unstable candidate detected: non-finite Lumerical field/FoM. Rejecting through backtracking.")
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

        # Combine FoM across all dipole positions
        f0_vals = np.asarray([
            max(float(np.nan_to_num(np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v))), channel_power_floor)
            for v in f0s
        ], dtype=float)
        
        f0 = combine_ldos_fom_from_values(f0_vals)
        f0_value = real_scalar_or_none(f0)
        if f0_value is not None:
            combined_fom_history.append(f0_value)

        # Print status for each dipole
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
            f"max={summary['max_fom']:.6e}, binarization={current_binarization_fraction:.3f})"
        )

        # Compute combined gradient
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
        print(f"[ldos session test] channel {channel_idx}: {channel['name']}")
        print(
            f"[ldos session test] dipole_pos=({channel['dipole_x']:.3f},{channel['dipole_y']:.3f},{channel['dipole_z']:.3f}), "
            f"efficiency_curve={target_efficiency_curve_str}, "
            f"pol={channel['polarization']}"
        )
        raise SystemExit(0)

    start = time.time()

    sim, opt, fom_history = build_optimization_problem()
    print("[ldos] Dipole-based LDOS optimization scaffold built.")
    print(
        "OLED periodic 3D freeform setup: "
        f"period={window_x}x{window_y} um, active area={active_x}x{active_y} um, "
        f"air={air_top_h} um, design={grating_design_h} um, "
        f"SiO2={sio2_h} um, ITO={ito_h} um, TCTA={tcta_h} um, "
        f"EML={eml_h} um, TPBi={tpbi_h} um, Ag={ag_h} um, "
        f"bottom_air_pad={air_bot_h} um, background_index={background_index}"
    )
    print(
        f"Dipole sample config ({n_dipole_positions} samples): "
        f"r_frac={dipole_radii_frac}, azimuth_deg={dipole_azimuths_deg}, "
        f"pol={dipole_polarizations}, weights={dipole_weights.tolist()}"
    )
    print(
        "Dipole-based LDOS channels: "
        + ", ".join(
            f"{ch['name']} pos=({ch['dipole_x']:.3f},{ch['dipole_y']:.3f},{ch['dipole_z']:.3f}) "
            f"pol={ch['polarization']} w={ch['weight']:.3f}"
            for ch in target_channels
        )
    )
    print(f"Target efficiency curve: {target_efficiency_curve_str}")
    print(
        f"N_fom={N_fom}, design_grids={design_grids}, design_cells={design_cells}, "
        f"radial_grating_shape=({radial_design_grids},), design_parameters={design_parameters}, "
        f"radial_radius={radial_design_radius}"
    )
    print(
        f"boundary_mode={boundary_label}, bc_x={bc_xy}, bc_y={bc_xy}, bc_z=PML, "
        "dipole excitation, angular-power optimization"
    )
    print(
        "FoM = weighted average across dipoles of "
        "(bulk-normalized 0deg power ** zero_emission_weight) "
        "* (angular ratio match ** ratio_match_weight)"
    )
    print(f"visible_wavelengths={visible_wavelengths}")
    print(
        f"Target FoM monitor={target_monitor_name}, center={target_monitor_c}, "
        f"size={target_monitor_s}"
    )
    print(f"Dipole source plane center={eml_c}, sampled in EML")
    print(
        f"FoM control weights: zero_emission={ldos_zero_emission_weight}, "
        f"ratio_match={ldos_ratio_match_weight}, "
        f"score_cap={ldos_score_cap}, "
        f"cosine_flux_weight={angular_flux_cosine_weight}, "
        f"hann_window={angular_use_hann_window}"
    )
    print(
        "Postprocess settings: "
        f"MSOPT_OLED_POSTPROCESS={env_flag('MSOPT_OLED_POSTPROCESS', '1')}, "
        f"MSOPT_OLED_POSTPROCESS_ONLY={env_flag('MSOPT_OLED_POSTPROCESS_ONLY', '0')}, "
        f"MSOPT_OLED_BULK_NORMALIZATION={env_flag('MSOPT_OLED_BULK_NORMALIZATION', '1')}, "
        f"MSOPT_OLED_LDOS_POSTPROCESS_RANDOM_POLARIZATION={env_flag('MSOPT_OLED_LDOS_POSTPROCESS_RANDOM_POLARIZATION', '1')}"
    )

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

        if combined_fom_history:
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
        else:
            print("[ldos] skipped FoM curve: no combined FoM history")





    if env_flag("MSOPT_OLED_POSTPROCESS", "1"):
        print("[postprocess] dipole-based LDOS incoherent postprocess")
        if True: # post process with incoherent dipole sampling
            def postprocess_design_array(design, beta=1.0):
                """Convert design to density grid."""
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

            def get_angular_spectrum_from_monitor(sim, monitor_name, wavelength_um):
                """
                Extract angular spectrum (k-space) from the upper-air target monitor via FFT.
                Returns angles and power spectrum.
                """
                result = sim.fdtd.getresult(monitor_name, "E")
                E = np.asarray(result["E"])
                x_axis_data = np.ravel(np.asarray(result["x"], dtype=float))
                y_axis_data = np.ravel(np.asarray(result["y"], dtype=float))
                
                if E.ndim < 3 or E.shape[-1] != 3:
                    raise ValueError(f"unexpected monitor E shape {E.shape}")

                # Map axes
                spatial_shape = E.shape[:-1]
                x_candidates = [idx for idx, size in enumerate(spatial_shape) if size == x_axis_data.size]
                y_candidates = [idx for idx, size in enumerate(spatial_shape) if size == y_axis_data.size]
                if not x_candidates or not y_candidates:
                    raise ValueError(f"could not map monitor axes: E shape={E.shape}")
                
                x_axis_idx = x_candidates[0]
                y_axis_idx = next((idx for idx in y_candidates if idx != x_axis_idx), y_candidates[0])
                E = np.moveaxis(E, [x_axis_idx, y_axis_idx], [0, 1])

                # Collapse to 3D: (x, y, components)
                while E.ndim > 3:
                    E = np.mean(E, axis=2)

                # FFT to k-space
                dx = float(np.mean(np.diff(x_axis_data)))
                dy = float(np.mean(np.diff(y_axis_data)))
                kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(x_axis_data.size, d=dx))
                ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(y_axis_data.size, d=dy))
                
                spectrum = np.zeros((x_axis_data.size, y_axis_data.size), dtype=float)
                for component_idx in range(3):
                    fft_field = np.fft.fftshift(np.fft.fft2(E[:, :, component_idx]))
                    spectrum += np.abs(fft_field) ** 2

                # Convert to angles
                KX, KY = np.meshgrid(kx, ky, indexing="ij")
                wavelength_m = float(wavelength_um) * 1e-6
                k0 = 2.0 * np.pi / wavelength_m
                normalized_kr = np.sqrt(KX ** 2 + KY ** 2) / k0
                propagating = normalized_kr <= 1.0
                theta_deg = np.rad2deg(np.arcsin(np.clip(normalized_kr, 0.0, 1.0)))
                
                # Extract angle-resolved power
                if angular_flux_cosine_weight:
                    spectrum = spectrum * np.sqrt(np.clip(1.0 - normalized_kr ** 2, 0.0, 1.0))
                spectrum = np.where(propagating, spectrum, 0.0)
                return theta_deg, spectrum

            def integrate_annular_angle_profile(theta_deg, spectrum, angle_centers_deg):
                """Integrate all azimuthal k-space pixels in each theta bin."""
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
                """
                Run postprocess with n_samples random dipoles.
                Compute the incoherent, azimuth-integrated angular emission pattern.
                """
                print(f"[postprocess] Loading final design and setting up simulator...")
                try:
                    rho = postprocess_design_array(final_design)
                except ValueError as exc:
                    print(f"[postprocess] skipped dipole postprocess: {exc}")
                    return None

                # Compute target field information
                print(f"[postprocess] Computing target field spectrum...")
                target_field_info = compute_target_field_with_diffraction_orders(
                    wavelength_um=float(np.mean(visible_wavelengths)),
                    period_um=float(window_x),
                    efficiency_curve=target_efficiency_curve,
                )
                # Generate random dipole positions in the EML.
                n_dipoles = int(os.environ.get("MSOPT_OLED_LDOS_POSTPROCESS_N_DIPOLES", str(n_samples)))
                print(f"[postprocess] Generating {n_dipoles} random dipole positions in EML...")
                active_radius = 0.5 * min(active_x, active_y)
                random_postprocess_polarization = env_flag("MSOPT_OLED_LDOS_POSTPROCESS_RANDOM_POLARIZATION", "1")
                fixed_postprocess_polarization = os.environ.get(
                    "MSOPT_OLED_LDOS_POSTPROCESS_POLARIZATION",
                    dipole_polarization,
                ).strip().lower()
                fixed_post_theta_deg, fixed_post_phi_deg = dipole_orientation_angles(fixed_postprocess_polarization)
                np.random.seed(240)  # Reproducible
                dipole_positions = []
                for _ in range(n_dipoles):
                    r = np.random.uniform(0, active_radius)
                    phi = np.random.uniform(0, 2 * np.pi)
                    x = r * np.cos(phi)
                    y = r * np.sin(phi)
                    z = eml_c[2]
                    dipole_positions.append((float(x), float(y), float(z)))

                # Setup simulator once
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

                # Run simulations for all dipoles
                print(f"[postprocess] Running {n_dipoles} dipole forward simulations...")
                incoherent_spectrum_sum = None
                records = []

                for dipole_idx, (x, y, z) in enumerate(dipole_positions):
                    print(f"[postprocess] dipole {dipole_idx+1}/{n_dipoles} at ({x:.3f}, {y:.3f}, {z:.3f})")
                    
                    # Reset layout
                    sim.fdtd.switchtolayout()
                    
                    # Delete old dipole
                    delete_lumerical_object(sim.fdtd, "postprocess_dipole")
                    
                    # Add the configured dipole orientation.
                    sim.fdtd.adddipole()
                    sim.fdtd.set("name", "postprocess_dipole")
                    sim.fdtd.set("x", x * 1e-6)
                    sim.fdtd.set("y", y * 1e-6)
                    sim.fdtd.set("z", z * 1e-6)
                    if random_postprocess_polarization:
                        sim.fdtd.set("theta", 90.0)
                        phi_deg = float(np.random.uniform(0.0, 360.0))
                        sim.fdtd.set("phi", phi_deg)
                    else:
                        phi_deg = fixed_post_phi_deg
                        sim.fdtd.set("theta", fixed_post_theta_deg)
                        sim.fdtd.set("phi", fixed_post_phi_deg)
                    sim.fdtd.set("wavelength start", float(np.min(visible_wavelengths)) * 1e-6)
                    sim.fdtd.set("wavelength stop", float(np.max(visible_wavelengths)) * 1e-6)
                    
                    # Run simulation
                    sim.run(name=f"postprocess_dipole_{dipole_idx:02d}", save=True)
                    
                    try:
                        # Extract angular spectrum
                        theta_deg, spectrum = get_angular_spectrum_from_monitor(
                            sim, target_monitor_name, float(np.mean(visible_wavelengths))
                        )
                        total_power = float(np.sum(spectrum))
                        
                        # Accumulate incoherent (power-based) sum
                        if incoherent_spectrum_sum is None:
                            incoherent_spectrum_sum = spectrum.copy()
                        else:
                            incoherent_spectrum_sum += spectrum
                        
                        records.append({
                            "dipole_idx": dipole_idx,
                            "x": x,
                            "y": y,
                            "z": z,
                            "theta_deg": 90.0 if random_postprocess_polarization else fixed_post_theta_deg,
                            "phi_deg": phi_deg,
                            "total_power": total_power,
                            "spectrum": spectrum,
                        })
                    except Exception as exc:
                        print(f"[postprocess] warning: dipole {dipole_idx} failed: {exc}")
                        records.append({
                            "dipole_idx": dipole_idx,
                            "x": x,
                            "y": y,
                            "z": z,
                            "theta_deg": 90.0 if random_postprocess_polarization else fixed_post_theta_deg,
                            "phi_deg": phi_deg,
                            "total_power": np.nan,
                            "spectrum": None,
                            "error": str(exc),
                        })

                # Close simulator
                try:
                    sim.fdtd.close()
                    print(f"[postprocess] closed FDTD simulator")
                except Exception as exc:
                    print(f"[postprocess] warning: failed to close simulator: {exc}")

                if incoherent_spectrum_sum is None or np.all(incoherent_spectrum_sum == 0):
                    print("[postprocess] skipped: incoherent spectrum sum is zero or empty")
                    return None

                # Extract azimuth-integrated angle-resolved power.
                angle_resolution = int(os.environ.get("MSOPT_OLED_POSTPROCESS_ANGLE_RES", "181"))
                angles_deg = np.linspace(0.0, 90.0, angle_resolution)
                angle_powers = integrate_annular_angle_profile(
                    theta_deg,
                    incoherent_spectrum_sum,
                    angles_deg,
                )

                # Normalize to max
                if np.max(angle_powers) > 0:
                    angle_powers_normalized = angle_powers / np.max(angle_powers)
                else:
                    angle_powers_normalized = angle_powers

                # Compute efficiency curve interpolation at sample angles
                target_efficiency_at_angles = np.array([
                    interpolate_efficiency_at_angle(float(angle), target_efficiency_curve)
                    for angle in angles_deg
                ])
                target_efficiency_zero = max(float(target_efficiency_at_angles[0]), 1e-12)
                target_ratio_at_angles = target_efficiency_at_angles / target_efficiency_zero
                zero_angle_power = max(float(angle_powers[0]), channel_power_floor)
                angle_ratios_to_zero = angle_powers / (zero_angle_power + 1e-30)

                # Save postprocess results
                results_path = os.path.join(design_dir, "LDOS_postprocess_dipole_results.txt")
                with open(results_path, "w", encoding="utf-8") as fp:
                    fp.write(f"method incoherent_ldos_dipole_sampling\n")
                    fp.write(f"n_dipoles {len(records)}\n")
                    fp.write(f"n_successful {sum(1 for r in records if 'spectrum' in r and r['spectrum'] is not None)}\n")
                    fp.write(f"zero_angle_power {zero_angle_power:.16e}\n")
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

                # Save records
                records_path = os.path.join(design_dir, "LDOS_postprocess_dipole_records.txt")
                with open(records_path, "w", encoding="utf-8") as fp:
                    fp.write("dipole_idx x_um y_um z_um theta_deg phi_deg total_power\n")
                    for rec in records:
                        if "spectrum" in rec and rec["spectrum"] is not None:
                            fp.write(
                                f"{rec['dipole_idx']} {rec['x']:.6e} {rec['y']:.6e} {rec['z']:.6e} "
                                f"{rec['theta_deg']:.6e} {rec['phi_deg']:.6e} {rec['total_power']:.6e}\n"
                            )
                print(f"[postprocess] saved dipole records: {records_path}")

                # Plot angle-resolved efficiency
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Left: Emission pattern
                axes[0].plot(angles_deg, angle_powers_normalized, "b-", linewidth=2, label="Emission (incoherent)")
                axes[0].fill_between(angles_deg, 0, angle_powers_normalized, alpha=0.3)
                axes[0].axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
                axes[0].set_xlabel("Emission angle (degrees)")
                axes[0].set_ylabel("Normalized power")
                axes[0].set_title("LDOS Incoherent Dipole Emission Pattern")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()

                # Right: ratio-to-zero comparison
                axes[1].plot(angles_deg, target_ratio_at_angles, "r-", linewidth=2, label="Target ratio")
                axes[1].plot(angles_deg, angle_ratios_to_zero, "b-", linewidth=2, label="Achieved ratio")
                axes[1].fill_between(angles_deg, 0, target_ratio_at_angles, alpha=0.2, color="red")
                axes[1].fill_between(angles_deg, 0, angle_ratios_to_zero, alpha=0.2, color="blue")
                axes[1].set_xlabel("Angle (degrees)")
                axes[1].set_ylabel("Power ratio to 0 deg")
                axes[1].set_title("Target vs. Achieved Angular Ratio")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend()
                axes[1].set_ylim([0, max(1.1, float(np.nanmax(angle_ratios_to_zero)) * 1.05)])

                plot_path = os.path.join(design_dir, "LDOS_postprocess_emission_efficiency.png")
                fig.tight_layout()
                fig.savefig(plot_path, dpi=200)
                plt.close(fig)
                print(f"[postprocess] saved emission efficiency plot: {plot_path}")

                return records

            def postprocess_final_design(opt):
                """Main postprocess entry point."""
                design_path = os.path.join(design_dir, "lastdesign.txt")
                if not os.path.exists(design_path):
                    print(f"[postprocess] skipped: final design not found at {design_path}")
                    return None

                print(f"[postprocess] loading final design from {design_path}")
                final_design = np.loadtxt(design_path)
                
                # Run dipole postprocess
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

            # Execute postprocess on final design
            postprocess_final_design(opt)
    else:
        print("[postprocess] skipped all postprocess: MSOPT_OLED_POSTPROCESS is disabled")

    print(f"Runtime setup time: {time.time() - start:.2f} seconds")
