import os
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


# Bulk-normalized dipole outcoupling optimization.
# FoM = target-order extracted power / bulk dipole power * angular order score.

seed = 240
np.random.seed(seed)

RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
os.makedirs(design_dir, exist_ok=True)
local_dir = os.path.join(RUN_DIR, "Local_bests") + os.sep
os.makedirs(local_dir, exist_ok=True)


def env_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_list_float(name, default):
    raw = os.environ.get(name, "").replace(";", ",").replace(" ", ",")
    vals = [v for v in raw.split(",") if v.strip()]
    return [float(v) for v in vals] if vals else list(default)


def env_list_str(name, default):
    raw = os.environ.get(name, "").replace(";", ",").replace(" ", ",")
    vals = [v.strip().lower() for v in raw.split(",") if v.strip()]
    return vals if vals else list(default)


def broadcast(values, size, name):
    if len(values) == size:
        return list(values)
    if len(values) == 1:
        return list(values) * size
    raise ValueError(f"{name} must have length 1 or {size}, got {len(values)}")


visible_wavelengths = np.asarray(env_list_float("MSOPT_OLED_WAVELENGTHS", [0.55]), dtype=float)
resolution = env_int("MSOPT_OLED_RESOLUTION", 50)
background_index = env_float("MSOPT_OLED_BACKGROUND_INDEX", 1.0)
grating_initial_density = env_float("MSOPT_OLED_INITIAL_DENSITY", 0.5)

boundary_mode = os.environ.get("MSOPT_OLED_BOUNDARY_MODE", "Bloch").strip().upper()
if boundary_mode not in ("PML", "BLOCH", "PERIODIC"):
    raise ValueError("MSOPT_OLED_BOUNDARY_MODE must be PML, Bloch, or Periodic.")
bc_xy = {"PML": "PML", "BLOCH": "Bloch", "PERIODIC": "Periodic"}[boundary_mode]

window_x = env_float("MSOPT_OLED_PERIOD_X_UM", 2.5)
window_y = env_float("MSOPT_OLED_PERIOD_Y_UM", 2.5)
active_x = env_float("MSOPT_OLED_ACTIVE_X_UM", window_x)
active_y = env_float("MSOPT_OLED_ACTIVE_Y_UM", window_y)

air_top_h = env_float("MSOPT_OLED_AIR_TOP_UM", 0.7)
sio2_h = env_float("MSOPT_OLED_SIO2_UM", 0.3)
grating_design_h = env_float("MSOPT_OLED_DESIGN_H_UM", 0.25)
ito_h = env_float("MSOPT_OLED_ITO_UM", 0.2)
tcta_h = env_float("MSOPT_OLED_TCTA_UM", 0.2)
eml_h = env_float("MSOPT_OLED_EML_UM", 0.2)
tpbi_h = env_float("MSOPT_OLED_TPBI_UM", 0.2)
ag_h = env_float("MSOPT_OLED_AG_UM", 0.2)
air_bot_h = env_float("MSOPT_OLED_AIR_BOT_UM", 0.10)

Sx, Sy = window_x, window_y
Sz = air_bot_h + ag_h + tpbi_h + eml_h + tcta_h + ito_h + grating_design_h + sio2_h + air_top_h
Z_min, Z_max = -0.5 * Sz, 0.5 * Sz

air_index = [1.0]
design_high_index = {"name": "OLED_grating_high_sampled", "wavelength": [0.55], "n": [1.45], "k": [0.0]}
design_low_index = air_index
sio2_index = {"name": "OLED_SiO2_sampled", "wavelength": [0.55], "n": [1.45], "k": [0.0]}
ito_index = {"name": "OLED_ITO_sampled", "wavelength": [0.55], "n": [1.7], "k": [0.0]}
tcta_index = {"name": "OLED_TCTA_sampled", "wavelength": [0.55], "n": [1.82], "k": [0.0]}
eml_index = {"name": "OLED_CBP_Irppy_sampled", "wavelength": [0.55], "n": [1.77], "k": [0.0]}
tpbi_index = {"name": "OLED_TPBi_sampled", "wavelength": [0.55], "n": [1.75], "k": [0.0]}
ag_index = {"name": "OLED_Ag_sampled", "wavelength": [0.55], "n": [0.76], "k": [5.9]}

layer_specs = [
    ("Ag_reflector", ag_h, ag_index),
    ("TPBi", tpbi_h, tpbi_index),
    ("CBP_Irppy_EML", eml_h, eml_index),
    ("TCTA", tcta_h, tcta_index),
    ("ITO", ito_h, ito_index),
    ("SiO2", sio2_h, sio2_index),
]

stack_layers = []
z = Z_min + air_bot_h
for name, height, index in layer_specs:
    center = [0.0, 0.0, z + 0.5 * height]
    stack_layers.append({"name": name, "center": center, "size": [Sx, Sy, height], "index": index})
    if name == "CBP_Irppy_EML":
        eml_c = [0.0, 0.0, center[2]]
    z += height

design_s = [Sx, Sy, grating_design_h]
design_c = [0.0, 0.0, z + 0.5 * grating_design_h]
target_monitor_name = "FoM_monitor"
target_monitor_s = [Sx, Sy, 0.0]
target_monitor_c = [0.0, 0.0, Z_max - 0.15]

# Flux monitor on the lower face of the design region: net upward power here is
# the light incident on the design region from the dipole, used to normalize the
# extraction efficiency (fraction of incident light outcoupled to the target).
design_incident_monitor_name = "design_incident_monitor"
design_incident_monitor_s = [Sx, Sy, 0.0]
design_incident_monitor_c = [0.0, 0.0, design_c[2] - 0.5 * grating_design_h]

Nx = int(round(design_s[0] * resolution)) + 1
Ny = int(round(design_s[1] * resolution)) + 1
Nz = int(round(design_s[2] * resolution)) + 1
design_grids = [Nx, Ny, Nz]
design_cells = Nx * Ny * Nz


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


target_efficiency_curve_str = os.environ.get("MSOPT_OLED_TARGET_EFFICIENCY_CURVE", "0:1.0,45:0.85,50:0.0")
target_efficiency_curve = parse_curve(target_efficiency_curve_str)


def parse_target_angles(text):
    # "angle:ratio,angle:ratio" -> [(angle_deg, ratio), ...], ratios > 0 only.
    pairs = []
    for token in text.split(","):
        parts = token.strip().split(":")
        if len(parts) == 2:
            try:
                a, r = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if r > 0.0:
                pairs.append((abs(a), r))
    return pairs or [(0.0, 1.0)]


# Discrete FoM target: emit into these angles (snapped to the nearest propagating
# diffraction order) with the given power ratios; everything else is suppressed.
target_angles_str = os.environ.get("MSOPT_OLED_TARGET_ANGLES", "0:1.0,45:0.85")
target_angle_pairs = parse_target_angles(target_angles_str)


def interp_curve(theta_deg, curve=target_efficiency_curve):
    theta, value = np.asarray(curve, dtype=float).T
    return float(np.interp(float(theta_deg), theta, value, left=value[0], right=value[-1]))


def material_real_index(index, wavelength_um):
    if isinstance(index, dict):
        wl = np.asarray(index["wavelength"], dtype=float)
        n = np.asarray(index["n"], dtype=float)
        return float(np.interp(float(wavelength_um), wl, n[:, 0] if n.ndim > 1 else n))
    return float(np.real(np.asarray(index, dtype=np.complex128).reshape(-1)[0]))


bulk_reference_index = env_float(
    "MSOPT_OLED_BULK_REFERENCE_INDEX",
    material_real_index(eml_index, float(np.mean(visible_wavelengths))),
)

score_cap = env_float("MSOPT_OLED_SCORE_CAP", env_float("MSOPT_OLED_LDOS_SCORE_CAP", 1.0))
fom_floor = env_float("MSOPT_OLED_FOM_FLOOR", 0.0)
channel_power_floor = env_float("MSOPT_OLED_CHANNEL_POWER_FLOOR", 1e-20)
unstable_candidate_fom = env_float("MSOPT_OLED_UNSTABLE_CANDIDATE_FOM", -1e30)
# FoM = throughput * match**ratio_emphasis. >1 prioritizes the angle ratio over raw
# throughput (the user's stated priority); 1.0 weights them equally.
ratio_emphasis = env_float("MSOPT_OLED_RATIO_EMPHASIS", 1.0)
# Only used by the postprocess order-share readout (the FoM itself is bin-exact).
angular_order_soft_sigma_deg = env_float("MSOPT_OLED_ANGULAR_ORDER_SIGMA_DEG", 5.0)
opt_emission_plot = env_flag("MSOPT_OLED_OPT_EMISSION_PLOT", "1")

# Postprocess far-field geometry: extra air above the stack so the monitor sits
# further out, and a lateral extent large enough that emission up to
# pp_max_angle_deg from the CENTRAL cell still reaches the monitor
# (half-width >= h * tan(angle)). Cell count scales with it -- watch the cost.
pp_far_z_um = env_float("MSOPT_OLED_PP_FAR_Z_UM", 2.0)
pp_max_angle_deg = env_float("MSOPT_OLED_PP_MAX_ANGLE_DEG", 60.0)
pp_min_tiles = env_int("MSOPT_OLED_PP_MIN_TILES", 3)
pp_resolution = env_int("MSOPT_OLED_PP_RESOLUTION", resolution)


def dipole_angles(pol):
    pol = str(pol).strip().lower()
    if pol == "x":
        return 90.0, 0.0
    if pol == "y":
        return 90.0, 90.0
    if pol == "z":
        return 0.0, 0.0
    raise ValueError(f"Unsupported dipole polarization {pol!r}.")


active_radius = 0.5 * min(active_x, active_y)
# Multiple dipole radii by default: a single on-axis dipole (r=0) barely excites the
# high-angle (off-normal) diffraction orders, so the FoM was blind to the leakage that
# off-axis dipoles dominate in the real (postprocess) device. Sampling several radii —
# with areal weights (~r) so the incoherent average matches a uniform dipole sheet —
# makes the FoM see and suppress that high-angle emission.
raw_r = env_list_float("MSOPT_OLED_DIPOLE_RADII_FRAC", [0.1, 0.4, 0.8])
raw_phi = env_list_float("MSOPT_OLED_DIPOLE_AZIMUTHS_DEG", [0.0])
raw_pol = env_list_str("MSOPT_OLED_DIPOLE_POLARIZATIONS", ["x"])
raw_w = env_list_float("MSOPT_OLED_DIPOLE_WEIGHTS", [1.0, 4.0, 8.0])
N_fom = max(len(raw_r), len(raw_phi), len(raw_pol), len(raw_w))
dipole_radii_frac = broadcast(raw_r, N_fom, "MSOPT_OLED_DIPOLE_RADII_FRAC")
dipole_azimuths_deg = broadcast(raw_phi, N_fom, "MSOPT_OLED_DIPOLE_AZIMUTHS_DEG")
dipole_polarizations = broadcast(raw_pol, N_fom, "MSOPT_OLED_DIPOLE_POLARIZATIONS")
channel_weights = np.asarray(broadcast(raw_w, N_fom, "MSOPT_OLED_DIPOLE_WEIGHTS"), dtype=float)

target_channels = []
for i, (rf, az, pol, weight) in enumerate(zip(dipole_radii_frac, dipole_azimuths_deg, dipole_polarizations, channel_weights)):
    a = np.deg2rad(az)
    x = float(rf * active_radius * np.cos(a))
    y = float(rf * active_radius * np.sin(a))
    target_channels.append(
        {
            "name": f"dipole_{i}_{pol}",
            "dipole_idx": i,
            "dipole_x": x,
            "dipole_y": y,
            "dipole_z": float(eml_c[2]),
            "polarization": pol,
            "weight": float(weight),
        }
    )


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


def save_target_orders(info):
    path = os.path.join(design_dir, "OLED_target_orders.csv")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("m,n,kx_over_k0,ky_over_k0,theta_deg,phi_deg,target_efficiency\n")
        for o in info["orders"]:
            fp.write(
                f"{o['m']},{o['n']},{o['ux']:.8f},{o['uy']:.8f},"
                f"{o['theta_deg']:.8f},{o['phi_deg']:.8f},{o['efficiency']:.8f}\n"
            )
    print(f"[target] saved {len(info['orders'])} propagating orders: {path}")
    save_target_orders_figure(info)


def save_target_orders_figure(info):
    orders = info["orders"]
    if not orders:
        return
    theta_max = max((o["theta_deg"] for o in orders), default=0.0)
    angle_range = np.linspace(0.0, max(theta_max + 5.0, 1.0), 181)
    eff_range = np.asarray([interp_curve(a) for a in angle_range], dtype=float)
    signed_angles = np.concatenate((-angle_range[1:][::-1], angle_range))
    signed_eff = np.concatenate((eff_range[1:][::-1], eff_range))

    fig = plt.figure(figsize=(6.5, 3.9))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.deg2rad(signed_angles), signed_eff, "b-", linewidth=2, label="target efficiency")
    for o in orders:
        if o["efficiency"] <= 0.0:
            continue
        angle = o["theta_deg"]
        order_angles = [0.0] if abs(angle) <= 1e-12 else [-angle, angle]
        ax.scatter(np.deg2rad(order_angles), [o["efficiency"]] * len(order_angles), c="tab:orange", s=60, zorder=5)
        ax.text(np.deg2rad(angle), min(float(o["efficiency"]) + 0.05, 1.0), f"({o['m']},{o['n']})", ha="center", va="bottom", fontsize=7)
    ax.scatter([], [], c="tab:orange", s=60, label="propagating orders")
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0.0, 1.0)
    ax.set_title(f"Angular target (lambda={info['wavelength_um']}um, period={info['period_x_um']}x{info['period_y_um']}um)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    path = os.path.join(design_dir, "OLED_target_field_info.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[target] saved target field info figure: {path}")


def save_angular_target_preview(angular_target, file_prefix="OLED_angular_target"):
    # The FoM basis: the continuous linear-ramp target across the in-range emission
    # angles (control points marked), normalized to sum 1 in-range.
    thetas = np.asarray(angular_target["angle_thetas"], dtype=float)
    target = np.asarray(angular_target["target_profile"], dtype=float)
    inr = np.asarray(angular_target["in_range"], dtype=float) > 0.5
    ctrl_a = np.asarray(angular_target["spec_snapped"], dtype=float)
    ctrl_r = np.asarray(angular_target["ctrl_ratios"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(thetas[inr], target[inr], width=1.8, color="tab:red", alpha=0.75, label="target profile (in-range)")
    ax.plot(thetas, np.zeros_like(thetas), "k|", markersize=14, label="available angles")
    # Show the underlying ramp (control ratios) on a twin axis for reference.
    axr = ax.twinx()
    axr.plot(ctrl_a, ctrl_r, "o--", color="tab:blue", lw=1.5, label="control ratio ramp")
    axr.set_ylabel("control ratio (relative)", color="tab:blue")
    axr.set_ylim(0.0, 1.1 * max(float(np.max(ctrl_r)), 1e-9))
    for th, t in zip(thetas[inr], target[inr]):
        ax.annotate(f"{th:.1f}\n{t:.2f}", (th, t), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xlabel("emission angle theta (deg)")
    ax.set_ylabel("target power fraction (in-range, sums to 1)")
    ax.set_title("Angular target profile (phi = 0 line): linear ratio ramp")
    ax.set_xlim(-3.0, 92.0)
    ax.set_ylim(0.0, max(1.35 * float(np.max(target)) if target.size else 1.0, 0.05))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    path = os.path.join(design_dir, f"{file_prefix}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    np.savez(os.path.join(design_dir, f"{file_prefix}.npz"), angle_thetas=thetas, target_profile=target, in_range=inr)
    print(f"[target] saved angular target profile: {path}")


def dft_matrix(n):
    i = np.arange(int(n), dtype=float)
    return np.exp(-2j * np.pi * np.outer(i, i) / float(n))


def order_mask(ukx, uky, ux, uy, sigma_deg, propagating):
    sigma = max(float(np.sin(np.deg2rad(max(float(sigma_deg), 1e-9)))), 1e-9)
    w = np.exp(-0.5 * (((ukx - ux) ** 2 + (uky - uy) ** 2) ** 0.5 / sigma) ** 2)
    w = np.where(propagating, w, 0.0)
    peak = float(np.max(w))
    return w / peak if peak > 0.0 else w


def angular_target_from_monitor_grid(x_axis, y_axis, target_info):
    """Angle basis for the FoM, defined on the phi = 0 radial line of the FoM monitor.

    The design is cylindrically symmetric, so the emission depends only on the polar
    angle theta. We therefore reduce the monitor to its y = 0 line (the phi = 0 radius),
    1D-DFT it to kx, and read the propagating bins as the angles this periodic cell can
    emit into:  kx = m * 2pi/P  ->  sin(theta) = kx/k0 = m * lambda/P.
    +kx and -kx are the same polar angle and are grouped into one mode.
    The requested target angles snap onto that angle set.
    """
    x = np.ravel(np.asarray(x_axis, dtype=float))
    y = np.ravel(np.asarray(y_axis, dtype=float))
    dx, dy = abs(float(np.mean(np.diff(x)))), abs(float(np.mean(np.diff(y))))
    iy0 = int(np.argmin(np.abs(y)))                             # phi = 0 radial line
    kx = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    k0 = 2.0 * np.pi / (float(target_info["wavelength_um"]) * 1e-6)
    u = kx / k0                                                 # = sin(theta) in air
    propagating = np.abs(u) <= 1.0 + 1e-9
    theta = np.round(np.rad2deg(np.arcsin(np.clip(np.abs(u), 0.0, 1.0))), 3)

    # One emission mode per distinct polar angle; angle_select picks its kx bins.
    angle_thetas = np.asarray(sorted({float(t) for t, p in zip(theta, propagating) if p}), dtype=float)
    if angle_thetas.size == 0:
        angle_thetas = np.zeros(1, dtype=float)
    angle_select = np.stack([((np.abs(theta - th) < 1e-3) & propagating).astype(float) for th in angle_thetas])

    # --- Continuous (piecewise-linear) angular target --------------------------------
    # The specified (angle, ratio) pairs are control points of a target ratio CURVE:
    # between them the target ratio varies linearly, so the intermediate emission angles
    # also follow the ratio (e.g. 0:1, 45:0.85  ->  1.0 at 0 deg ramping to 0.85 at the
    # 45-deg mode, with every mode in between interpolated). The goal is to pack all the
    # energy into [min control angle, max control angle] with that shape; angles outside
    # the range are leakage (target 0). Control angles snap to the nearest emission mode.
    ctrl = sorted(((float(a), float(r)) for a, r in target_angle_pairs), key=lambda ar: ar[0])
    ctrl_a = np.asarray([float(angle_thetas[int(np.argmin(np.abs(angle_thetas - a)))]) for a, _ in ctrl], dtype=float)
    ctrl_r = np.asarray([r for _, r in ctrl], dtype=float)
    order = np.argsort(ctrl_a)
    ctrl_a, ctrl_r = ctrl_a[order], ctrl_r[order]
    a_lo, a_hi = float(ctrl_a[0]), float(ctrl_a[-1])
    in_range = (angle_thetas >= a_lo - 1e-6) & (angle_thetas <= a_hi + 1e-6)
    target_curve = np.where(in_range, np.interp(angle_thetas, ctrl_a, ctrl_r), 0.0)
    target_profile = target_curve / max(float(np.sum(target_curve)), 1e-30)   # sums to 1 in-range
    spec_idx = [int(np.argmin(np.abs(angle_thetas - a))) for a in ctrl_a]      # control-point modes
    print(f"[target] phi=0 emission angles {np.round(angle_thetas, 1).tolist()} deg; "
          f"control {[(a, r) for a, r in ctrl]} -> snapped {np.round(ctrl_a, 1).tolist()}; "
          f"in-range {np.round(angle_thetas[in_range], 1).tolist()}, "
          f"target {np.round(target_profile, 3).tolist()}")

    return {
        "x_size": x.size,
        "y_size": y.size,
        "iy0": iy0,
        "dft_x": dft_matrix(x.size),
        "monitor_cell_area": dx * dy,
        "angle_select": angle_select,
        "angle_thetas": angle_thetas,
        "in_range": in_range.astype(float),
        "target_profile": target_profile,
        "spec_idx": spec_idx,
        "spec_snapped": ctrl_a.tolist(),
        "ctrl_ratios": ctrl_r.tolist(),
    }


def angular_powers(Ex, Ey, Hx, Hy, target, flux_sign=1.0):
    """1) phi=0 radial field line  ->  2) mode power per angle  ->  3) ratio FoM."""
    iy0 = int(target["iy0"])
    dft = npa.asarray(target["dft_x"])

    def line_spectrum(f):
        if f.shape[0] != target["x_size"] or f.shape[1] != target["y_size"]:
            raise ValueError(f"monitor shape {f.shape} does not match angular target")
        return npa.einsum("ia,a...->i...", dft, f[:, iy0])      # 1) radial line -> kx

    Ex_k, Ey_k, Hx_k, Hy_k = [line_spectrum(f) for f in (Ex, Ey, Hx, Hy)]
    flux = 0.5 * float(flux_sign) * npa.real(Ex_k * npa.conj(Hy_k) - Ey_k * npa.conj(Hx_k))
    if flux.ndim > 1:
        flux = npa.sum(flux, axis=tuple(range(1, flux.ndim)))   # sum over wavelengths
    flux = npa.maximum(npa.where(npa.isfinite(flux), flux, 0.0), 0.0)

    # 2) mode power per emission angle (+kx and -kx bins of the same |theta| summed)
    power = npa.einsum("ki,i->k", npa.asarray(target["angle_select"]), flux)
    profile = power / (npa.sum(power) + 1e-30)

    # 3) FoM = T * M against the continuous target profile t_k (a linear ramp over the
    # in-range emission angles, 0 outside; sums to 1 in-range):
    #   T = sum_k profile_k * in_range_k          -> energy packed into [0, max angle]
    #   q_k = profile_k / T                       -> emission shape within that cone
    #   M = sum_k min(q_k, t_k)                   -> match to the ramp over ALL in-range
    #                                                angles (intermediate angles included),
    #                                                unique max at q = t on the simplex
    # FoM = T * M**ratio_emphasis in [0, 1]; = 1 only when all energy is in-range AND
    # follows the requested linear ratio across every intermediate angle.
    in_range = npa.asarray(target["in_range"])
    tgt = npa.asarray(target["target_profile"])
    throughput = npa.sum(profile * in_range)
    q = profile / (throughput + 1e-30)
    match = npa.sum(npa.minimum(q * in_range, tgt))
    fom = throughput * (match ** ratio_emphasis)
    fracs = [profile[int(k)] for k in target["spec_idx"]]
    return npa.sum(power), fom, fracs, profile, throughput, match


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
    # After a GPU session run the live session is left in layout mode, so
    # dipolepower/sourcepower/getresult raise "please run the simulation". Reload
    # the saved project (which holds the results) to switch into analysis mode.
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


def choose_power(bulk_power, source_power, current_power):
    for label, value in (("bulk_reference_power", bulk_power), ("source_power", source_power), ("current_dipole_power", current_power)):
        value = valid_power(value)
        if value is not None:
            return max(value, channel_power_floor), label
    return channel_power_floor, "floor"


def choose_norm_power(incident_power, bulk_power, source_power, current_power):
    for label, value in (
        ("design_incident_power", incident_power),
        ("bulk_reference_power", bulk_power),
        ("source_power", source_power),
        ("current_dipole_power", current_power),
    ):
        value = valid_power(value)
        if value is not None:
            return max(value, channel_power_floor), label
    return channel_power_floor, "floor"


def set_fdtd_background_index(fdtd, index_value):
    try:
        fdtd.setnamed("FDTD", "index", float(index_value))
    except Exception:
        fdtd.eval(f'select("FDTD"); set("index", {float(index_value):.16g});')


def delete_object(fdtd, name):
    fdtd.eval(f'if (getnamednumber("{name}") > 0) {{ select("{name}"); delete; }}')


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


def add_dipole(sim, x, y, z, pol, name="source"):
    theta, phi = dipole_angles(pol)
    sim.fdtd.adddipole()
    sim.fdtd.set("name", name)
    sim.fdtd.set("x", x * 1e-6)
    sim.fdtd.set("y", y * 1e-6)
    sim.fdtd.set("z", z * 1e-6)
    sim.fdtd.set("theta", theta)
    sim.fdtd.set("phi", phi)
    sim.fdtd.set("wavelength start", float(np.min(visible_wavelengths)) * 1e-6)
    sim.fdtd.set("wavelength stop", float(np.max(visible_wavelengths)) * 1e-6)
    sim.src_wl = visible_wavelengths.reshape(-1) * sim.unit
    sim.src_bw = 0.0


def measure_design_incident_reference(channel):
    # Reference power incident on the design region from the dipole, measured with
    # the design region AND everything above it filled with design material. That
    # material (n=1.45) is index-matched to the SiO2 directly below, so there is no
    # reflecting interface at or above the design's lower face: the net upward flux
    # there is the pure incident power (no back-reflection from the design/superstrate).
    ref = make_sim([Sx, Sy, Sz])
    add_dipole(ref, channel["dipole_x"], channel["dipole_y"], channel["dipole_z"], channel["polarization"])
    add_stack(ref)
    fill_bottom = design_incident_monitor_c[2]
    fill_h = Z_max - fill_bottom
    fill_c = [0.0, 0.0, fill_bottom + 0.5 * fill_h]
    ref.add_geo(fill_c, [Sx, Sy, fill_h], design_high_index, "design_incident_fill", float(np.mean(visible_wavelengths)))
    ref.add_monitor(design_incident_monitor_name, design_incident_monitor_c, design_incident_monitor_s)
    ref.run(name=f"design_incident_reference_{channel['dipole_idx']}", save=True)
    load_run_results(ref)
    freqs = source_freqs(ref)
    source_power = read_source_power(ref.fdtd, freqs)
    T_incident = read_transmission(ref.fdtd, design_incident_monitor_name)
    incident_power = abs(T_incident) * source_power if valid_power(source_power) is not None else None
    try:
        ref.fdtd.close()
    except Exception:
        pass
    incident_power = valid_power(incident_power)
    print(
        f"[incident-ref] channel {channel['dipole_idx']}: T={T_incident:.6e}, "
        f"source_power={valid_power(source_power) or float('nan'):.6e}, "
        f"incident_power={incident_power if incident_power is not None else float('nan'):.6e}"
    )
    return incident_power


def raw_monitor_flux(fdtd, monitor_name, target, flux_sign):
    try:
        E = np.asarray(fdtd.getresult(monitor_name, "E")["E"], dtype=np.complex128)
        H = np.asarray(fdtd.getresult(monitor_name, "H")["H"], dtype=np.complex128)
        pz = 0.5 * np.real(E[..., 0] * np.conj(H[..., 1]) - E[..., 1] * np.conj(H[..., 0]))
        return max(float(flux_sign) * float(np.sum(pz)) * float(target["monitor_cell_area"]), 0.0)
    except Exception:
        return None


def channel_fom_terms(Ex, Ey, Hx, Hy, channel):
    target = channel["angular_target"]
    flux_sign = float(channel.get("last_top_flux_sign", 1.0))
    total, fom, spec_p, profile, throughput, match = angular_powers(Ex, Ey, Hx, Hy, target, flux_sign)
    return {"fom": fom, "spec_fracs": spec_p, "profile": profile, "total": total,
            "throughput": throughput, "match": match}


def combine_fom(vals):
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), fom_floor)
    weights = npa.asarray(channel_weights, dtype=float)
    weights = weights / npa.sum(weights) if float(np.sum(channel_weights)) > 0.0 else npa.ones_like(weights) / max(float(weights.size), 1.0)
    return npa.sum(vals * weights)


def clean_fom_values(values):
    return np.asarray(
        [
            max(float(np.nan_to_num(np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v), nan=fom_floor, posinf=score_cap, neginf=fom_floor)), fom_floor)
            for v in values
        ],
        dtype=float,
    )


def build_optimization_problem():
    target_info = build_target_orders(float(np.mean(visible_wavelengths)), window_x, window_y)
    save_target_orders(target_info)
    sims, opts, histories = [None] * N_fom, [None] * N_fom, [[] for _ in range(N_fom)]

    for idx, channel in enumerate(target_channels):
        sim = sims[idx] = make_sim([Sx, Sy, Sz])
        add_dipole(sim, channel["dipole_x"], channel["dipole_y"], channel["dipole_z"], channel["polarization"])
        sim.add_monitor(target_monitor_name, target_monitor_c, target_monitor_s)

        if env_flag("MSOPT_OLED_BULK_NORMALIZATION", "1"):
            set_fdtd_background_index(sim.fdtd, bulk_reference_index)
            sim.run(name=f"bulk_reference_{idx}", save=True)
            load_run_results(sim)
            bulk_power = read_dipole_power(sim.fdtd, source_freqs(sim)) or read_source_power(sim.fdtd, source_freqs(sim)) or channel_power_floor
            channel["bulk_reference_power"] = max(float(bulk_power), channel_power_floor)
            grid = sim.fdtd.getresult(target_monitor_name, "E")
            channel["angular_target"] = angular_target_from_monitor_grid(grid["x"], grid["y"], target_info)
            if idx == 0:
                save_angular_target_preview(channel["angular_target"])
            sim.fdtd.switchtolayout()
            set_fdtd_background_index(sim.fdtd, background_index)
            print(f"[bulk] channel {idx}: n={bulk_reference_index:.4g}, power={channel['bulk_reference_power']:.6e}")
        else:
            channel["bulk_reference_power"] = np.nan
            channel["angular_target"] = None

        add_stack(sim)
        sim.add_design_grid("design", design_c, design_s, design_high_index, design_low_index, design_grids, grating_initial_density * np.ones(design_grids), float(np.mean(visible_wavelengths)))
        sim.add_design_monitor()
        channel["design_incident_reference_power"] = measure_design_incident_reference(channel)

        def J(Ex, Ey, Hx, Hy, channel_idx=idx, channel=channel):
            if channel["angular_target"] is None:
                channel["angular_target"] = angular_target_from_monitor_grid(opts[channel_idx].xg, opts[channel_idx].yg, target_info)
            terms = channel_fom_terms(Ex, Ey, Hx, Hy, channel)
            # Objective: FoM = throughput * match**ratio_emphasis (angular_powers).
            fom = npa.clip(terms["fom"], 0.0, score_cap)
            try:
                at = channel["angular_target"]
                thetas = np.asarray(at["angle_thetas"], dtype=float)
                tgt = np.asarray(at["target_profile"], dtype=float)
                profile = np.real(np.asarray(terms["profile"], dtype=np.complex128))
                thru = float(np.real(terms["throughput"]))
                match = float(np.real(terms["match"]))
                q = profile / max(thru, 1e-30)                    # emission shape within cone
                channel["last_fom_metrics"] = {
                    "fom": float(np.real(fom)),
                    "throughput": thru,
                    "match": match,
                    "angle_thetas": thetas.tolist(),
                    "target_profile": tgt.tolist(),
                    "angle_profile": profile.tolist(),
                    "off_target_fraction": float(max(0.0, 1.0 - thru)),
                }
                histories[channel_idx].append(float(np.real(fom)))
                if opt_emission_plot:
                    channel["last_angle_profile"] = profile.tolist()
                # Achieved shape q vs the linear target t at every in-range emission angle.
                inr = np.asarray(at["in_range"], dtype=float) > 0.5
                prof_str = ", ".join(f"{t:.0f}:{qi:.2f}/{ti:.2f}" for t, qi, ti in zip(thetas[inr], q[inr], tgt[inr]))
                print(
                    f"[dipole {channel_idx}] FoM={float(np.real(fom)):.4f}  "
                    f"thru={thru:.3f} match={match:.3f}  q/t[{prof_str}]"
                )
            except Exception:
                pass
            return fom

        opt = opts[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim,
            objective_functions=[J],
            objective_arguments=[0, 1, 3, 4],
            FoM_size=target_monitor_s,
            FoM_center=target_monitor_c,
            adj_fwd=False,
            opt_idx=idx,
            broadband_adjoint=True,
        )

        def hook(problem, channel=channel):
            freqs = np.asarray(getattr(problem, "src_freqs", []), dtype=float).reshape(-1)
            freqs = freqs if freqs.size else source_freqs(problem.sim)
            source_power = read_source_power(problem.sim.fdtd, freqs)
            current_power = read_dipole_power(problem.sim.fdtd, freqs)
            # Normalize by the fixed reference power incident on the design region
            # (measured once with the design + superstrate index-matched, so no
            # back-reflection). Falls back to bulk/source/dipole power if unavailable.
            norm_power, source = choose_norm_power(channel.get("design_incident_reference_power"), channel.get("bulk_reference_power"), source_power, current_power)
            T = read_transmission(problem.sim.fdtd, target_monitor_name)
            sign = 1.0 if T >= 0.0 else -1.0
            raw_flux = raw_monitor_flux(problem.sim.fdtd, target_monitor_name, channel.get("angular_target"), sign)
            ref_top = abs(T) * source_power if valid_power(source_power) is not None else np.nan
            channel["last_normalization_power"] = norm_power
            channel["last_normalization_power_source"] = source
            channel["last_source_power"] = valid_power(source_power) or np.nan
            channel["last_current_dipole_power"] = valid_power(current_power) or np.nan
            channel["last_top_flux_sign"] = sign
            channel["last_top_monitor_transmission"] = T
            channel["last_top_flux_calibration"] = ref_top / raw_flux if raw_flux and np.isfinite(ref_top) and ref_top > 0.0 else 1.0

        opt.forward_result_hook = hook
        print(f"[setup] channel {idx}: pos=({channel['dipole_x']:.3f},{channel['dipole_y']:.3f},{channel['dipole_z']:.3f}), pol={channel['polarization']}")

    return sims, opts, histories


DR_info = [design_s[0], design_s[1], design_s[2], 0, 1, 2]
DR_N_info = [Nx, Ny, Nz, resolution]
radial_design_radius = env_float("MSOPT_OLED_RADIAL_RADIUS", 0.5 * min(design_s[0], design_s[1]))
radial_design_grids = env_int("MSOPT_OLED_RADIAL_GRIDS", int(round(radial_design_radius * resolution)) + 1)
mapping = None
design_parameters = None
x0 = None
dJ_0 = np.zeros(design_cells)
combined_fom_history = []


def ensure_mapping():
    global mapping, design_parameters, x0
    if mapping is None:
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
    return mapping


def design_to_grid(design, beta=1.0):
    rho = np.asarray(design, dtype=float)
    if rho.size == design_cells:
        return rho.reshape(design_grids)
    local_mapping = ensure_mapping()
    if rho.size == design_parameters:
        return np.asarray(local_mapping(rho, beta), dtype=float).reshape(design_grids)
    if rho.size == Nx * Ny:
        return np.repeat(rho.reshape(Nx, Ny)[:, :, None], Nz, axis=2)
    raise ValueError(f"expected {design_cells}, {design_parameters}, or {Nx * Ny} design values, got {rho.size}")


def format_design_plot_status(f0_vals=None):
    lines = []
    if f0_vals is not None:
        vals = np.asarray(f0_vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lines.append(f"FoM mean={np.mean(vals):.3e}, min={np.min(vals):.3e}, max={np.max(vals):.3e}")
    for label, key in (("target extraction", "target_extraction_efficiency"), ("zero-order eff", "zero_order_efficiency"), ("shape score", "shape_score")):
        vals = [c.get("last_fom_metrics", {}).get(key) for c in target_channels]
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if vals:
            lines.append(f"{label} mean={np.mean(vals):.3e}")
    return "\n".join(lines)


def save_current_design_sections(design, f0_vals=None):
    rho = design_to_grid(npa.clip(design, 0.0, 1.0))
    x_axis = np.linspace(-0.5 * design_s[0], 0.5 * design_s[0], Nx)
    y_axis = np.linspace(-0.5 * design_s[1], 0.5 * design_s[1], Ny)
    z_axis = np.linspace(design_c[2] - 0.5 * design_s[2], design_c[2] + 0.5 * design_s[2], Nz)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].imshow(rho[:, :, Nz // 2].T, origin="lower", extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]), cmap="binary", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[0].set_title("x-y section at z=center")

    axes[1].imshow(rho[:, Ny // 2, :].T, origin="lower", extent=(x_axis[0], x_axis[-1], z_axis[0], z_axis[-1]), cmap="binary", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
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


def adjoint_loop(opts):
    def loop(X, N_cases, Case=True):
        if Case == 3:
            vals = npa.maximum(npa.asarray(clean_fom_values(N_cases)), fom_floor)
            coeffs = npa.where(npa.isfinite(ag_jacobian(combine_fom)(vals)), ag_jacobian(combine_fom)(vals), 0.0)
            grad = 0.0
            for c, g in zip(coeffs, X[0]):
                grad = grad + c * npa.where(npa.isfinite(npa.array(g)), npa.array(g), 0.0)
            print(f"[outcoupling] combined grad max={np.max(np.abs(grad)):.6e}")
            return npa.where(npa.isfinite(grad), grad, 0.0)

        f0s, grads = [0] * N_fom, [0] * N_fom
        for i, opt in enumerate(opts):
            if isinstance(X, str):
                f0s[i], grads[i] = opt(need_gradient=Case)
            else:
                f0s[i], grads[i] = opt(rho_vector=[npa.clip(X, 0.0, 1.0)], need_gradient=Case)

        if any(getattr(opt, "last_forward_had_nonfinite", False) for opt in opts):
            zero_grads = [np.zeros_like(g, dtype=float) if not isinstance(g, (int, float)) else np.zeros(design_cells) for g in grads]
            return (unstable_candidate_fom, [fom_floor] * N_fom, zero_grads) if Case and not isinstance(X, str) else zero_grads

        vals = clean_fom_values(f0s)

        if not isinstance(X, str):
            try:
                path = save_current_design_sections(X, vals)
                print(f"[outcoupling] saved temporary design section: {path}")
            except Exception as exc:
                print(f"[outcoupling] skipped temporary design section: {exc}")
            if opt_emission_plot:
                try:
                    epath = save_optimization_emission_plot()
                    if epath:
                        print(f"[outcoupling] saved current emission plot: {epath}")
                except Exception as exc:
                    print(f"[outcoupling] skipped emission plot: {exc}")

        combined = combine_fom(vals)
        combined_fom_history.append(float(np.real(combined)))
        print(f"[outcoupling] combined FoM={combined:.6e}, mean={np.mean(vals):.6e}, min={np.min(vals):.6e}, max={np.max(vals):.6e}")
        if Case:
            return grads if isinstance(X, str) else (combined, f0s, grads)
        return combined, f0s

    return loop



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


def render_emission_figure(angles, radiance_norm, target_norm, order_thetas, achieved_share, target_share, path, label):
    # Shared 2-panel emission figure: (left) per-direction radiance vs target,
    # (right) diffraction-order power share achieved vs target.
    fig = plt.figure(figsize=(12, 4.6))
    ax0 = fig.add_subplot(121, projection="polar")
    signed = np.r_[-angles[1:][::-1], angles]
    ax0.plot(np.deg2rad(signed), np.r_[radiance_norm[1:][::-1], radiance_norm], label=label)
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
    signed = np.r_[-angles[1:][::-1], angles]
    mean_eff = np.mean(np.stack([d["eff"] for d in per_dipole]), axis=0)
    mean_rad = np.mean(np.stack([d["rad"] for d in per_dipole]), axis=0)

    fig = plt.figure(figsize=(14.5, 5.2))
    fig.subplots_adjust(wspace=0.32)
    ax0 = fig.add_subplot(121, projection="polar")
    ax1 = fig.add_subplot(122)
    for d in per_dipole:
        c = cmap(norm(d["r"]))
        shape = d["rad"] / max(float(np.max(d["rad"])), 1e-30)
        ax0.plot(np.deg2rad(signed), np.r_[shape[1:][::-1], shape], color=c, lw=1.0, alpha=0.75)
        ax1.plot(angles, d["eff"], color=c, lw=1.0, alpha=0.75)
    ax0.plot(np.deg2rad(signed), np.r_[(mean_rad / max(float(np.max(mean_rad)), 1e-30))[1:][::-1],
                                       mean_rad / max(float(np.max(mean_rad)), 1e-30)], "k--", lw=2.0, label="mean")
    ax1.plot(angles, mean_eff, "k--", lw=2.0, label="mean over dipoles")

    ax0.set_thetamin(-90)
    ax0.set_thetamax(90)
    ax0.set_theta_zero_location("N")
    ax0.set_theta_direction(-1)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Per-dipole radiance (each self-normalized): shape")
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


def save_optimization_emission_plot():
    # Weight-weighted mean of the per-dipole angular mode-power profiles from the last
    # optimization forward pass, plotted against the requested ratio.
    total, at, wsum = None, None, 0.0
    for ch, w in zip(target_channels, channel_weights):
        prof = ch.get("last_angle_profile")
        if prof is None:
            continue
        at = at or ch.get("angular_target")
        contrib = float(w) * np.asarray(prof, dtype=float)
        total = contrib if total is None else total + contrib
        wsum += float(w)
    if total is None or at is None:
        return None

    achieved = total / max(wsum, 1e-30)                       # full profile (all angles, sum=1)
    thetas = np.asarray(at["angle_thetas"], dtype=float)
    inr = np.asarray(at["in_range"], dtype=float)
    tgt = np.asarray(at["target_profile"], dtype=float)       # linear ramp, sums to 1 in-range
    thru = float(np.sum(achieved * inr))                      # T = energy in [0, max angle]
    # Target drawn on the SAME normalization as the achieved bars: t_k * T. Then the bars
    # are directly comparable (achieved_k vs t_k*T  <=>  q_k vs t_k, the log's numbers) and
    # any out-of-range bar shows the leakage that steals from T.
    target = tgt * thru
    q = achieved / max(thru, 1e-30)                           # emission shape within cone
    match = float(np.sum(np.minimum(q * inr, tgt)))
    fom = thru * match ** ratio_emphasis

    fig, ax = plt.subplots(figsize=(8.5, 4))
    xpos = np.arange(thetas.size)
    bw = 0.4
    ax.bar(xpos - 0.5 * bw, target, width=bw, label=r"target $t_k\cdot T$ (linear ramp)", color="tab:orange")
    ax.bar(xpos + 0.5 * bw, achieved, width=bw, label="achieved (of total)", color="tab:blue")
    for k in np.where(inr > 0.5)[0]:
        ax.annotate(f"q={q[int(k)]:.2f}", (xpos[int(k)] + 0.5 * bw, achieved[int(k)]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color="tab:blue")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{t:.1f}" for t in thetas])
    ax.set_xlabel("emission angle theta (deg)")
    ax.set_ylabel("angular mode-power fraction (of total)")
    ax.set_title(f"Angular mode power (phi = 0), opt dipoles - FoM={fom:.3f}  thru={thru:.3f}  match={match:.3f}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(design_dir, "OLED_opt_emission.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    np.savez(os.path.join(design_dir, "OLED_opt_emission.npz"),
             angle_thetas=thetas, achieved=achieved, target=target, q=q, throughput=thru, in_range=inr)
    return path


def central_cell_dipoles(n_samples):
    pol = os.environ.get("MSOPT_OLED_POSTPROCESS_POLARIZATION", dipole_polarizations[0]).strip().lower()
    # MSOPT_OLED_PP_DIPOLE_GRID=N (N>0) replaces the seeded random draw with the
    # deterministic NxN endpoint grid (2-pixel edge inset) used by the Meep coherence
    # reference scripts (step1_trace_comparison / step2b case1): the incoherent
    # reference is the source-wise average over a fixed spatial grid, not a Monte
    # Carlo sample. n_samples is ignored in grid mode (count = N*N).
    grid_n = env_int("MSOPT_OLED_PP_DIPOLE_GRID", 0)
    if grid_n > 0:
        inset = 2.0 / resolution
        xs = np.linspace(-0.5 * active_x + inset, 0.5 * active_x - inset, grid_n)
        ys = np.linspace(-0.5 * active_y + inset, 0.5 * active_y - inset, grid_n)
        return [(float(x), float(y), float(eml_c[2]), pol) for x in xs for y in ys]
    rng = np.random.default_rng(seed)
    return [
        (float(rng.uniform(-0.5 * active_x, 0.5 * active_x)), float(rng.uniform(-0.5 * active_y, 0.5 * active_y)), float(eml_c[2]), pol)
        for _ in range(n_samples)
    ]


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

    angles = np.linspace(0.0, 90.0, env_int("MSOPT_OLED_POSTPROCESS_ANGLE_RES", 181))
    records, spectrum_sum, per_dipole = [], None, []
    ukx = uky = None
    keep_fsp = env_flag("MSOPT_OLED_PP_KEEP_FSP", "1")

    def _drop_last_fsp():
        # MSOPT_OLED_PP_KEEP_FSP=0: delete the just-analyzed run's .fsp so a long
        # dipole/phase sweep does not fill the disk; results are already extracted.
        if keep_fsp:
            return
        fsp = getattr(sim, "_last_run_fsp_path", None)
        if fsp and os.path.exists(fsp):
            try:
                os.remove(fsp)
            except Exception:
                pass

    for i, (x, y, z, pol) in enumerate(central_cell_dipoles(env_int("MSOPT_OLED_POSTPROCESS_N_DIPOLES", 20))):
        print(f"[postprocess] {tile_n}x{tile_n}/PML dipole {i + 1}: x={x:.3f}, y={y:.3f}, pol={pol}")
        sim.fdtd.switchtolayout()
        delete_object(sim.fdtd, "postprocess_dipole")
        add_dipole(sim, x, y, z + z_shift, pol, "postprocess_dipole")   # same shift as the stack
        sim.run(name=f"postprocess_pml_{i:02d}", save=True)
        load_run_results(sim)
        try:
            T = read_transmission(sim.fdtd, target_monitor_name)
            theta, spectrum, ukx, uky = monitor_spectrum(sim, target_monitor_name, float(np.mean(visible_wavelengths)), 1.0 if T >= 0 else -1.0)
            spectrum_sum = spectrum.copy() if spectrum_sum is None else spectrum_sum + spectrum
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
            records.append((i, x, y, z, pol, float(T), src or np.nan, dip or np.nan, top_power_abs, lee))
            # Per-dipole angular breakdown: ring flux split into absolute extraction
            # efficiency per angle bin (sums to this dipole's LEE), plus the
            # per-direction radiance (solid-angle Jacobian removed) for the shape.
            ring_flux = angle_profile(theta, spectrum, angles)
            ring_count = angle_profile(theta, np.ones_like(spectrum), angles)
            per_dipole.append({
                "idx": i, "x": float(x), "y": float(y), "r": float(np.hypot(x, y)), "lee": float(lee),
                "eff": ring_flux / max(float(np.sum(ring_flux)), 1e-30) * float(lee),
                "rad": ring_flux / np.maximum(ring_count, 1.0),
            })
            _drop_last_fsp()
        except Exception as exc:
            print(f"[postprocess] warning: dipole {i} failed: {exc}")

    # --- Coherence cases (Meep step2b analog) ---------------------------------
    # Case 2: the same dipole set emitting SIMULTANEOUSLY with identical phase --
    # this realization contains the dipole-dipole cross terms that a coherent-grid
    # FoM sees. Case 3: the same simultaneous ensemble with random per-dipole
    # phases, ensemble-averaged over MSOPT_OLED_PP_RANDOM_PHASE_TRIALS trials; the
    # running average should converge toward the incoherent per-dipole sum above.
    coh_cases = []                                   # (tag, theta, spectrum, ukx, uky)
    rand_trial_profiles = []                         # (trial_no, theta, spectrum)
    if env_flag("MSOPT_OLED_PP_COHERENT_CHECK", "0"):
        pts = central_cell_dipoles(env_int("MSOPT_OLED_POSTPROCESS_N_DIPOLES", 20))
        sim.fdtd.switchtolayout()
        delete_object(sim.fdtd, "postprocess_dipole")
        for k, (cx, cy, cz, cpol) in enumerate(pts):
            add_dipole(sim, cx, cy, cz + z_shift, cpol, f"coherent_dipole_{k:03d}")
        try:
            print(f"[postprocess] coherence case2: {len(pts)} simultaneous dipoles, identical phase")
            sim.run(name="postprocess_coherent_same_phase", save=True)
            load_run_results(sim)
            T_c = read_transmission(sim.fdtd, target_monitor_name)
            th_c, sp_c, ux_c, uy_c = monitor_spectrum(sim, target_monitor_name, float(np.mean(visible_wavelengths)), 1.0 if T_c >= 0 else -1.0)
            coh_cases.append(("coherent_same_phase", th_c, sp_c, ux_c, uy_c))
            _drop_last_fsp()
        except Exception as exc:
            print(f"[postprocess] warning: coherence case2 failed: {exc}")
        n_rand_trials = env_int("MSOPT_OLED_PP_RANDOM_PHASE_TRIALS", 0)
        if n_rand_trials > 0:
            rng_ph = np.random.default_rng(env_int("MSOPT_OLED_PP_RANDOM_PHASE_SEED", 1234))
            all_ph = rng_ph.uniform(0.0, 360.0, size=(n_rand_trials, len(pts)))
            np.savetxt(os.path.join(design_dir, "OLED_postprocess_randphase_phases_deg.txt"), all_ph)
            sp_cum, n_ok = None, 0
            for t_no in range(1, n_rand_trials + 1):
                try:
                    sim.fdtd.switchtolayout()
                    for k in range(len(pts)):
                        sim.fdtd.setnamed(f"coherent_dipole_{k:03d}", "phase", float(all_ph[t_no - 1, k]))
                    print(f"[postprocess] coherence case3 trial {t_no}/{n_rand_trials} (random phases)")
                    sim.run(name=f"postprocess_randphase_{t_no:03d}", save=True)
                    load_run_results(sim)
                    T_c = read_transmission(sim.fdtd, target_monitor_name)
                    th_c, sp_c, ux_c, uy_c = monitor_spectrum(sim, target_monitor_name, float(np.mean(visible_wavelengths)), 1.0 if T_c >= 0 else -1.0)
                    _drop_last_fsp()
                except Exception as exc:
                    print(f"[postprocess] warning: coherence case3 trial {t_no} failed: {exc}")
                    continue
                n_ok += 1
                rand_trial_profiles.append((t_no, th_c, sp_c))
                sp_cum = sp_c.copy() if sp_cum is None else sp_cum + sp_c
                if n_ok in (1, 2, 5, 10, 20) or t_no == n_rand_trials:
                    coh_cases.append((f"randphase_avg{n_ok:03d}", th_c, sp_cum / n_ok, ux_c, uy_c))

    try:
        sim.fdtd.close()
    except Exception:
        pass
    if spectrum_sum is None:
        print("[postprocess] skipped: no valid spectra")
        return

    # Per-direction radiance (solid-angle Jacobian removed), same as the in-loop plot.
    ring_flux = angle_profile(theta, spectrum_sum, angles)
    radiance_norm = radiance_from_spectrum(theta, spectrum_sum, angles)
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
    render_emission_figure(angles, radiance_norm, target_norm, order_thetas, achieved_share, target_share, path, "3x3 PML radiance")
    print(f"[postprocess] saved emission (radiance + order-share) plot: {path}")

    per_path = save_per_dipole_emission_plot(angles, per_dipole, os.path.join(design_dir, "OLED_postprocess_per_dipole_emission.png"))
    if per_path:
        print(f"[postprocess] saved per-dipole angular emission plot ({len(per_dipole)} dipoles): {per_path}")

    # Coherence-case products: same angle-profile / order-share formats as the
    # incoherent reference, tagged per case, so the cases can be overlaid directly.
    for tag, th_c, sp_c, ux_c, uy_c in coh_cases:
        np.savetxt(
            os.path.join(design_dir, f"OLED_postprocess_{tag}_angle_profile.txt"),
            np.column_stack([angles, angle_profile(th_c, sp_c, angles), radiance_from_spectrum(th_c, sp_c, angles)]),
            header="theta_deg ring_flux radiance_norm",
        )
        c_prop = np.sqrt(ux_c ** 2 + uy_c ** 2) <= 1.0 + 1e-12
        c_rows = {}
        for o in order_info["orders"]:
            mask = order_mask(ux_c, uy_c, o["ux"], o["uy"], angular_order_soft_sigma_deg, c_prop)
            key = round(float(o["theta_deg"]), 1)
            row = c_rows.setdefault(key, {"achieved": 0.0, "target": 0.0})
            row["achieved"] += float(np.sum(sp_c * mask))
            row["target"] += max(float(o["efficiency"]), 0.0)
        c_thetas = sorted(c_rows)
        c_ach = np.asarray([c_rows[t]["achieved"] for t in c_thetas], dtype=float)
        c_te = np.asarray([c_rows[t]["target"] for t in c_thetas], dtype=float)
        c_ash = c_ach / max(float(np.sum(c_ach)), 1e-30)
        c_tsh = c_te / max(float(np.sum(c_te)), 1e-30)
        with open(os.path.join(design_dir, f"OLED_postprocess_{tag}_order_shares.txt"), "w", encoding="utf-8") as fp:
            fp.write("theta_deg achieved_power achieved_share target_efficiency target_share\n")
            for t, a, ash, te, tsh in zip(c_thetas, c_ach, c_ash, c_te, c_tsh):
                fp.write(f"{t:.3f} {a:.6e} {ash:.6e} {te:.6e} {tsh:.6e}\n")
        print(f"[postprocess] {tag}: order shares " + ", ".join(f"{t:g}deg={s * 100:.2f}%" for t, s in zip(c_thetas, c_ash)))
    for t_no, th_c, sp_c in rand_trial_profiles:
        np.savetxt(
            os.path.join(design_dir, f"OLED_postprocess_randphase_trial{t_no:03d}_angle_profile.txt"),
            np.column_stack([angles, angle_profile(th_c, sp_c, angles), radiance_from_spectrum(th_c, sp_c, angles)]),
            header="theta_deg ring_flux radiance_norm",
        )


def main():
    if env_flag("MSOPT_OLED_SESSION_TEST", "0"):
        print(f"[session] N_fom={N_fom}, design_grid={design_grids}, boundary={bc_xy}, postprocess=PML 3x3")
        return

    start = time.time()
    post_only = env_flag("MSOPT_OLED_POSTPROCESS_ONLY", "0")
    if not post_only:
        if ag_jacobian is None:
            raise RuntimeError("autograd is required for optimization. Install autograd or run with MSOPT_OLED_POSTPROCESS_ONLY=1.")
        ensure_mapping()
        sims, opts, histories = build_optimization_problem()
        print(
            f"[setup] period={Sx:g}x{Sy:g} um, design={design_grids}, N_fom={N_fom}, "
            f"bulk_n={bulk_reference_index:g}, opt_bc={bc_xy}"
        )
        optimizer = ms.Opt_MS2.OPT_Ms(x0, dJ_0, Born_k=99, Initial_LR=0.2, Raw=False)
        optimizer.flag = True
        optimizer(mapping, N_fom, adjoint_loop(opts))

        plt.figure()
        for i in range(optimizer.bt_tol):
            plt.plot(optimizer.wrong_evaluation_history[i], 'r-')
        for i in range(optimizer.bt_tol):
            plt.plot(optimizer.wrong_evaluation_history2[i], 'b-')
        plt.plot(optimizer.evaluation_history, 'k-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('FoM')
        plt.savefig(os.path.join(design_dir,  'result1.png'))
        plt.close()

        plt.figure()
        plt.plot(optimizer.evaluation_history, 'k-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('FoM')
        plt.savefig(os.path.join(design_dir,  'result0.png'))
        plt.close()

        plt.figure()
        plt.plot(optimizer.binarization_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Binarized')
        plt.savefig(os.path.join(design_dir,  'result2.png'))
        plt.close()

        plt.figure()
        plt.plot(optimizer.learning_rate_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.savefig(os.path.join(design_dir, 'result4.png'))
        plt.close()

        plt.figure()
        plt.plot(optimizer.grad_mean_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Mean grad')
        plt.savefig(os.path.join(design_dir, 'result5.png'))
        plt.close()

        plt.figure()
        plt.plot(optimizer.grad_max_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Max grad')
        plt.savefig(os.path.join(design_dir,  'result6.png'))
        plt.close()

        plt.figure()
        plt.plot(optimizer.beta_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Beta')
        plt.savefig(os.path.join(design_dir,  'result3.png'))
        plt.close()

    if env_flag("MSOPT_OLED_POSTPROCESS", "1"):
        # MSOPT_OLED_POSTPROCESS_DESIGN lets the postprocess re-run on a design from a
        # previous run (e.g. a Done/ result) instead of this run's lastdesign.txt.
        design_path = os.environ.get("MSOPT_OLED_POSTPROCESS_DESIGN", "").strip()
        if not (design_path and os.path.exists(design_path)):
            design_path = os.path.join(design_dir, "lastdesign.txt")
        if os.path.exists(design_path):
            print(f"[postprocess] using design: {design_path}")
            run_postprocess(np.loadtxt(design_path))
        else:
            print(f"[postprocess] skipped: {design_path} not found")
    print(f"Runtime setup time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
