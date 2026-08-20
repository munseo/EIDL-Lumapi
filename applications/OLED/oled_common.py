"""Shared OLED outcoupling-optimization utilities.

Extracted from the four sibling scripts in this directory (OLED_new.py,
OLED_opt.py, OLED_lens.py, OLED_Min.py), which are ~80% copy-pasted.
OLED_new.py is the newest script and the primary source for every shared
function body; sections whose only home is another script are marked.

Section -> source map:
  * env helpers ................. OLED_new.py (env_flag/env_float/env_int/
                                  env_list_float) + OLED_lens.py (env_list_str,
                                  broadcast).  env_flag keeps an explicit
                                  default parameter (OLED_Min-style); every
                                  call site in this module passes it.
  * build_config ................ the module-level config block of OLED_new.py
                                  (~lines 18-225): wavelengths, resolution,
                                  boundary mode, periods, layer thicknesses,
                                  material tables, stack/monitor geometry,
                                  target curve/angles, normalization floors,
                                  postprocess geometry -- returned as one
                                  explicit types.SimpleNamespace `G` instead
                                  of module globals.
  * sim/geometry helpers ........ OLED_new.py (make_sim, add_stack, add_dipole
                                  with enabled/group_name, dipole_angles,
                                  delete_object, set_fdtd_background_index)
  * result readers .............. OLED_new.py (load_run_results with the GPU
                                  layout-mode guard, read_source_power,
                                  read_dipole_power, read_transmission,
                                  source_freqs, valid_power, finite_sum,
                                  choose_norm_power)
  * targets / orders ............ OLED_new.py (parse_curve, parse_target_angles,
                                  material_real_index, build_target_orders,
                                  save_target_orders(_figure), order_mask,
                                  dft_matrix)
  * k-space FoM basis ........... OLED_new.py (build_ramp_target,
                                  _angular_target_1d, _angular_target_2d,
                                  angular_target_from_monitor_grid, _ramp_fom,
                                  angular_powers; autograd-safe through npa)
  * coherent-array helpers ...... OLED_new.py (optimization_polarizations,
                                  build_evenly_spaced_dipoles)
  * channel glue ................ OLED_new.py (channel_fom_terms, combine_fom
                                  [mean version], clean_fom_values,
                                  measure_design_incident_reference,
                                  raw_monitor_flux, design_to_grid,
                                  format_design_plot_status,
                                  save_current_design_sections, adjoint_loop
                                  with the per-iteration plotting hoisted into
                                  the optional iter_plot_fn callback)
  * spectrum / analysis ......... OLED_new.py (monitor_spectrum, n2f_spectrum,
                                  angle_profile, radiance_from_spectrum,
                                  signed_angle_axis, directional_radiance)
  * plotting .................... OLED_new.py (render_xz_field_image,
                                  render_emission_figure,
                                  save_per_dipole_emission_plot,
                                  save_angular_target_preview,
                                  save_xz_monitor_field_snapshot,
                                  save_fom_monitor_field_snapshot,
                                  save_optimization_emission_plot)
  * postprocess ................. OLED_new.py base, upgraded to the validated
                                  step1/step2 protocol: source-wise incoherent
                                  sum, EML flux-box normalization, farfield3d
                                  solid-angle integration, and optional
                                  case2a/2b/3 coherence audit. Supercell is the
                                  default; single/n2f is qualitative only.
                                  The dipole layout has since moved on from the
                                  reference's 6x6 endpoint grid: a mirror-
                                  symmetric design is sampled on a 12x12
                                  cell_center grid folded onto its C4v orbits --
                                  the same 36 FDTD runs, four times the
                                  independent samples. See resolve_dipole_grid;
                                  MSOPT_OLED_PP_DIPOLE_GRID and
                                  MSOPT_OLED_PP_SOURCE_LAYOUT still select the
                                  reference protocol explicitly.
  * optimizer harness ........... OLED_new.py main(): the result0..result6
                                  optimizer-history figures
                                  (save_result_plots) and the
                                  MSOPT_OLED_SESSION_TEST early-exit banner
                                  (session_test_banner)

    NOTE (2026-08-19): the inventory above is the provenance of everything this
    module ever absorbed, and is kept as such. The application has since been
    reduced to the OLED_rec chain, and the 28 functions only the retired scripts
    used now live in legacy/oled_common_legacy.py -- so some names listed above
    are no longer defined here. The split was computed as a reachability closure
    from every `oc.NAME` the live files reference; oled_constrained_score stayed
    because test_oled_common still exercises it.

Design pattern: every function that read module globals in the originals now
takes the config namespace `G` (from build_config) as its first argument;
functions that only used their own arguments are unchanged.  Importing this
module starts no Lumerical session and has no filesystem side effects;
build_config performs the only side effects (makedirs of design_dir/local_dir
plus np.random.seed(seed)), exactly like the original module-level code did.
"""

import hashlib
import json
import os
import shutil
import time
import types

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


# =============================================================================
# Environment helpers  (OLED_new.py + env_list_str/broadcast from OLED_lens.py)
# =============================================================================


def env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def env_str(name, default):
    """Trimmed string override; an empty/whitespace value falls back to default
    so `VAR= python ...` behaves like "not set" rather than selecting ""."""
    return (os.environ.get(name, "") or "").strip() or str(default)


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_list_float(name, default):
    raw = os.environ.get(name, "").replace(";", ",").replace(" ", ",")
    vals = [v for v in raw.split(",") if v.strip()]
    return [float(v) for v in vals] if vals else list(default)






# =============================================================================
# Targets: curve/angle parsing and material index  (OLED_new.py)
# =============================================================================


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


def make_ratio_performance_spec(angles_deg, target_ratios, tolerance=0.05):
    """Build the canonical angular-ratio contract shared by optimization and PP.

    A zero target is a suppression channel and therefore keeps an exact [0, 0]
    window.  Positive targets use a symmetric tolerance, except the normal
    reference channel which is fixed to ratio 1 by definition.
    """
    angles = np.asarray(angles_deg, dtype=float).reshape(-1)
    targets = np.asarray(target_ratios, dtype=float).reshape(-1)
    if angles.size == 0 or angles.size != targets.size:
        raise ValueError("angles_deg and target_ratios must have the same non-zero length.")
    if not np.isclose(angles[0], 0.0):
        raise ValueError("The first OLED performance channel must be the 0-degree reference.")
    tol = max(float(tolerance), 0.0)
    suppressed = targets <= 0.0
    ratio_min = np.maximum(targets - tol, 0.0)
    ratio_max = np.minimum(targets + tol, 1.0)
    ratio_min[0] = ratio_max[0] = 1.0
    ratio_min[suppressed] = 0.0
    ratio_max[suppressed] = 0.0
    return {
        "angles_deg": angles,
        "target_ratios": targets,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "tolerance": tol,
    }


def oled_constrained_score(
    angle_powers,
    ratio_min,
    ratio_max,
    *,
    power_floor=1e-20,
    violation_scale=0.05,
    throughput_reference=1.0,
    throughput_weight=0.10,
):
    """Bounded, autograd-safe OLED score with angular compliance as priority.

    The old ``P0 / (1 + penalty)`` score could be made arbitrarily large by
    increasing only P0, even when every requested ratio was wrong.  This score
    is bounded to [0, 1]: angular-window compliance receives the dominant
    weight and a saturated normal-channel throughput is only a tie-breaker.
    """
    powers = npa.maximum(
        npa.where(npa.isfinite(angle_powers), angle_powers, 0.0),
        float(power_floor),
    )
    lo = npa.asarray(ratio_min)
    hi = npa.asarray(ratio_max)
    zero_power = npa.maximum(powers[0], float(power_floor))
    ratios = powers / zero_power
    violations = npa.maximum(lo - ratios, 0.0) + npa.maximum(ratios - hi, 0.0)
    scale = max(float(violation_scale), 1e-6)
    shape_score = 1.0 / (1.0 + npa.mean((violations / scale) ** 2))
    reference = max(float(throughput_reference), float(power_floor))
    throughput_score = zero_power / (zero_power + reference)
    weight = float(np.clip(throughput_weight, 0.0, 1.0))
    return shape_score * ((1.0 - weight) + weight * throughput_score)


def oled_performance_metrics(
    angle_powers,
    performance_spec,
    *,
    power_floor=1e-20,
    violation_scale=None,
    throughput_reference=1.0,
    throughput_weight=0.10,
):
    """NumPy readout matching :func:`oled_constrained_score` exactly."""
    powers = np.maximum(
        np.nan_to_num(np.asarray(angle_powers, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
        float(power_floor),
    )
    lo = np.asarray(performance_spec["ratio_min"], dtype=float)
    hi = np.asarray(performance_spec["ratio_max"], dtype=float)
    if powers.size != lo.size or lo.size != hi.size:
        raise ValueError("angle_powers and performance-spec channel counts differ.")
    zero_power = max(float(powers[0]), float(power_floor))
    ratios = powers / zero_power
    violations = np.maximum(lo - ratios, 0.0) + np.maximum(ratios - hi, 0.0)
    scale = max(
        float(performance_spec["tolerance"] if violation_scale is None else violation_scale),
        1e-6,
    )
    shape_score = float(1.0 / (1.0 + np.mean((violations / scale) ** 2)))
    reference = max(float(throughput_reference), float(power_floor))
    throughput_score = float(zero_power / (zero_power + reference))
    weight = float(np.clip(throughput_weight, 0.0, 1.0))
    score = shape_score * ((1.0 - weight) + weight * throughput_score)
    return {
        "angle_powers": powers,
        "ratios_to_zero": ratios,
        "violations": violations,
        "shape_score": shape_score,
        "throughput_score": throughput_score,
        "score": float(score),
        "all_ratio_windows_met": bool(np.all(violations <= 1e-12)),
    }


def material_real_index(index, wavelength_um):
    if isinstance(index, dict):
        wl = np.asarray(index["wavelength"], dtype=float)
        n = np.asarray(index["n"], dtype=float)
        return float(np.interp(float(wavelength_um), wl, n[:, 0] if n.ndim > 1 else n))
    return float(np.real(np.asarray(index, dtype=np.complex128).reshape(-1)[0]))


# =============================================================================
# Configuration  (module-level block of OLED_new.py, ~lines 18-225)
# =============================================================================


def build_config(period_x_default=2.5, period_y_default=None, seed=240,
                 layer_specs=None, eml_layer_name="CBP_Irppy_EML"):
    """Resolved OLED configuration.

    layer_specs, when given, REPLACES the built-in bottom-emission stack with a
    caller-supplied one: a list of (name, thickness_um, index) ordered bottom to
    top, sitting on the bottom air pad and carrying the design region on top.
    Sz, all z coordinates, the EML plane, the design region and every monitor are
    then derived from that stack inside this function, so callers do not have to
    patch the returned config (which is error prone -- see microcavity_layers).
    eml_layer_name selects which layer is the emission plane.
    """
    """Read every geometry/env config the four scripts shared at module level
    and return it as one explicit namespace `G`.

    period_x_default / period_y_default are the script-specific defaults for
    MSOPT_OLED_PERIOD_X_UM / MSOPT_OLED_PERIOD_Y_UM (OLED_new/OLED_lens used
    2.5 um, OLED_opt used 1.1 um); period_y_default=None means "same as x".
    Side effects (identical to the original module-level code): seeds numpy
    with `seed` and creates design_dir ("A") and local_dir ("Local_bests")
    under EIDL_RUN_DIR.
    """
    if period_y_default is None:
        period_y_default = period_x_default

    np.random.seed(seed)

    RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
    design_dir = os.path.join(RUN_DIR, "A") + os.sep
    os.makedirs(design_dir, exist_ok=True)
    local_dir = os.path.join(RUN_DIR, "Local_bests") + os.sep
    os.makedirs(local_dir, exist_ok=True)

    visible_wavelengths = np.asarray(env_list_float("MSOPT_OLED_WAVELENGTHS", [0.55]), dtype=float)
    resolution = env_int("MSOPT_OLED_RESOLUTION", 50)
    background_index = env_float("MSOPT_OLED_BACKGROUND_INDEX", 1.0)
    grating_initial_density = env_float("MSOPT_OLED_INITIAL_DENSITY", 0.5)

    boundary_mode = os.environ.get("MSOPT_OLED_BOUNDARY_MODE", "Bloch").strip().upper()
    if boundary_mode not in ("PML", "BLOCH", "PERIODIC"):
        raise ValueError("MSOPT_OLED_BOUNDARY_MODE must be PML, Bloch, or Periodic.")
    bc_xy = {"PML": "PML", "BLOCH": "Bloch", "PERIODIC": "Periodic"}[boundary_mode]

    window_x = env_float("MSOPT_OLED_PERIOD_X_UM", period_x_default)
    window_y = env_float("MSOPT_OLED_PERIOD_Y_UM", period_y_default)
    active_x = env_float("MSOPT_OLED_ACTIVE_X_UM", window_x)
    active_y = env_float("MSOPT_OLED_ACTIVE_Y_UM", window_y)

    # --- probe plane -------------------------------------------------------
    # One plane serves both formulations, which is what makes them comparable:
    #   reciprocal runs (OLED_opt, OLED_Min) LAUNCH the symmetric plane wave there;
    #   dipole runs (OLED_lens, every postprocess) MEASURE there.
    # It sits PROBE_GAP above the top of the design region, and the domain is then
    # only as tall as it has to be: structure + design + gap + a small top margin.
    probe_gap_um = env_float("MSOPT_OLED_PROBE_GAP_UM", 0.7)
    top_margin_req = env_float("MSOPT_OLED_TOP_MARGIN_UM", 0.1)
    # Lumerical's PML lives INSIDE the simulation span (msopt leaves the default
    # 8 layers), so a margin thinner than the PML would put the probe plane
    # inside the absorber and quietly corrupt it. Raise it and say so.
    pml_um = env_int("MSOPT_OLED_PML_LAYERS", 8) / float(resolution)
    top_margin_um = max(top_margin_req, pml_um + 2.0 / resolution)
    if top_margin_um > top_margin_req + 1e-12:
        print(f"[geometry] top margin raised {top_margin_req * 1000:.0f} -> "
              f"{top_margin_um * 1000:.0f} nm so the probe plane clears the PML "
              f"({pml_um * 1000:.0f} nm at resolution {resolution}/um)")
    air_top_h = env_float("MSOPT_OLED_AIR_TOP_UM", probe_gap_um + top_margin_um)
    sio2_h = env_float("MSOPT_OLED_SIO2_UM", 0.3)
    grating_design_h = env_float("MSOPT_OLED_DESIGN_H_UM", 0.3)
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
    # Design-region materials. rho = 1 is DESIGN_N, rho = 0 is DESIGN_LOW_N, and
    # the sampled table is labelled at THIS run's wavelength -- the old hardcoded
    # 0.55 um label disagreed with the microcavity colors (0.46/0.53/0.62).
    # The default 1.45 (index-matched to the SiO2 under it in the legacy stack) is
    # rarely what you want on a different stack: on the microcavity the design
    # sits on an n=2.2 CPL, so patterning in 1.45-vs-air there is a weak contrast.
    design_wl = float(np.mean(visible_wavelengths))
    design_n = env_float("MSOPT_OLED_DESIGN_N", 1.45)
    design_k = env_float("MSOPT_OLED_DESIGN_K", 0.0)
    design_low_n = env_float("MSOPT_OLED_DESIGN_LOW_N", 1.0)
    design_low_k = env_float("MSOPT_OLED_DESIGN_LOW_K", 0.0)
    design_high_index = {"name": "OLED_design_high_sampled", "wavelength": [design_wl],
                         "n": [design_n], "k": [design_k]}
    design_low_index = ([1.0] if (design_low_n == 1.0 and design_low_k == 0.0)
                        else {"name": "OLED_design_low_sampled", "wavelength": [design_wl],
                              "n": [design_low_n], "k": [design_low_k]})
    sio2_index = {"name": "OLED_SiO2_sampled", "wavelength": [0.55], "n": [1.45], "k": [0.0]}
    ito_index = {"name": "OLED_ITO_sampled", "wavelength": [0.55], "n": [1.7], "k": [0.0]}
    tcta_index = {"name": "OLED_TCTA_sampled", "wavelength": [0.55], "n": [1.82], "k": [0.0]}
    eml_index = {"name": "OLED_CBP_Irppy_sampled", "wavelength": [0.55], "n": [1.77], "k": [0.0]}
    tpbi_index = {"name": "OLED_TPBi_sampled", "wavelength": [0.55], "n": [1.75], "k": [0.0]}
    ag_index = {"name": "OLED_Ag_sampled", "wavelength": [0.55], "n": [0.76], "k": [5.9]}

    if layer_specs is None:
        layer_specs = [
            ("Ag_reflector", ag_h, ag_index),
            ("TPBi", tpbi_h, tpbi_index),
            ("CBP_Irppy_EML", eml_h, eml_index),
            ("TCTA", tcta_h, tcta_index),
            ("ITO", ito_h, ito_index),
            ("SiO2", sio2_h, sio2_index),
        ]
    else:
        # A caller-supplied stack replaces the built-in one. Sz and every z
        # coordinate below are then derived from IT, so the caller never has to
        # patch G afterwards -- doing that by hand is how the microcavity runs
        # first ended up with a mis-placed far-field monitor.
        layer_specs = [(str(n), float(h), idx) for n, h, idx in layer_specs]
        Sz = air_bot_h + sum(h for _n, h, _i in layer_specs) + grating_design_h + air_top_h
        Z_min, Z_max = -0.5 * Sz, 0.5 * Sz

    stack_layers = []
    eml_c = None
    eml_s = None
    z = Z_min + air_bot_h
    for name, height, index in layer_specs:
        center = [0.0, 0.0, z + 0.5 * height]
        stack_layers.append({"name": name, "center": center, "size": [Sx, Sy, height], "index": index})
        if name == eml_layer_name:
            eml_c = [0.0, 0.0, center[2]]
            eml_s = [active_x, active_y, 0.0]
        z += height
    if eml_c is None:
        raise ValueError(
            f"no layer named {eml_layer_name!r} in the stack "
            f"({[n for n, _h, _i in layer_specs]}); pass eml_layer_name."
        )

    # Design region footprint: the full unit cell by default, but a smaller
    # patch is allowed so a design can be confined inside the period.
    design_x = env_float("MSOPT_OLED_DESIGN_X_UM", Sx)
    design_y = env_float("MSOPT_OLED_DESIGN_Y_UM", Sy)
    if design_x > Sx + 1e-12 or design_y > Sy + 1e-12:
        raise ValueError(
            f"design footprint {design_x:g}x{design_y:g} um exceeds the period "
            f"{Sx:g}x{Sy:g} um; neighbouring cells would overlap."
        )
    design_s = [design_x, design_y, grating_design_h]
    design_c = [0.0, 0.0, z + 0.5 * grating_design_h]
    target_monitor_name = "FoM_monitor"
    xyz_monitor_name = "xz_field_monitor"
    target_monitor_s = [Sx, Sy, 0.0]
    design_top_z = design_c[2] + 0.5 * grating_design_h
    probe_plane_z = design_top_z + probe_gap_um
    target_monitor_c = [0.0, 0.0, probe_plane_z]

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

    target_efficiency_curve_str = os.environ.get("MSOPT_OLED_TARGET_EFFICIENCY_CURVE", "0:1.0,45:0.85,50:0.0")
    target_efficiency_curve = parse_curve(target_efficiency_curve_str)

    # Discrete FoM target: emit into these angles (snapped to the nearest propagating
    # diffraction order) with the given power ratios; everything else is suppressed.
    target_angles_str = os.environ.get("MSOPT_OLED_TARGET_ANGLES", "0:1.0,45:0.85")
    target_angle_pairs = parse_target_angles(target_angles_str)

    def interp_curve(theta_deg, curve=target_efficiency_curve):
        theta, value = np.asarray(curve, dtype=float).T
        return float(np.interp(float(theta_deg), theta, value, left=value[0], right=value[-1]))

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
    # FoM angle basis:
    #   radial_1d  - phi=0 line, (m,0) orders only. Valid ONLY for a cylindrically symmetric
    #                (radial) design + single polarization.
    #   kspace_2d  - full 2D monitor DFT, azimuthally integrate Sz(kx,ky) into a polar-angle
    #                theta profile, so ALL (m,n) diffraction orders are captured. Use this for
    #                the non-cylindrical / dual-polarization case where the 1D line is blind
    #                to the off-axis (m,n!=0) orders.
    fom_mode = os.environ.get("MSOPT_OLED_FOM_MODE", "kspace_2d").strip().lower()
    if fom_mode not in ("radial_1d", "kspace_2d"):
        raise ValueError("MSOPT_OLED_FOM_MODE must be radial_1d or kspace_2d.")
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

    active_radius = 0.5 * min(active_x, active_y)

    return types.SimpleNamespace(
        seed=seed,
        RUN_DIR=RUN_DIR,
        design_dir=design_dir,
        local_dir=local_dir,
        visible_wavelengths=visible_wavelengths,
        resolution=resolution,
        background_index=background_index,
        grating_initial_density=grating_initial_density,
        boundary_mode=boundary_mode,
        bc_xy=bc_xy,
        window_x=window_x,
        window_y=window_y,
        active_x=active_x,
        active_y=active_y,
        air_top_h=air_top_h,
        sio2_h=sio2_h,
        grating_design_h=grating_design_h,
        ito_h=ito_h,
        tcta_h=tcta_h,
        eml_h=eml_h,
        tpbi_h=tpbi_h,
        ag_h=ag_h,
        air_bot_h=air_bot_h,
        Sx=Sx,
        Sy=Sy,
        Sz=Sz,
        Z_min=Z_min,
        Z_max=Z_max,
        air_index=air_index,
        design_high_index=design_high_index,
        design_low_index=design_low_index,
        sio2_index=sio2_index,
        ito_index=ito_index,
        tcta_index=tcta_index,
        eml_index=eml_index,
        tpbi_index=tpbi_index,
        ag_index=ag_index,
        layer_specs=layer_specs,
        stack_layers=stack_layers,
        eml_c=eml_c,
        eml_s=eml_s,
        design_s=design_s,
        design_c=design_c,
        target_monitor_name=target_monitor_name,
        xyz_monitor_name=xyz_monitor_name,
        target_monitor_s=target_monitor_s,
        target_monitor_c=target_monitor_c,
        probe_plane_z=probe_plane_z,
        probe_gap_um=probe_gap_um,
        top_margin_um=top_margin_um,
        design_top_z=design_top_z,
        design_incident_monitor_name=design_incident_monitor_name,
        design_incident_monitor_s=design_incident_monitor_s,
        design_incident_monitor_c=design_incident_monitor_c,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        design_grids=design_grids,
        design_cells=design_cells,
        target_efficiency_curve_str=target_efficiency_curve_str,
        target_efficiency_curve=target_efficiency_curve,
        target_angles_str=target_angles_str,
        target_angle_pairs=target_angle_pairs,
        interp_curve=interp_curve,
        bulk_reference_index=bulk_reference_index,
        score_cap=score_cap,
        fom_floor=fom_floor,
        channel_power_floor=channel_power_floor,
        unstable_candidate_fom=unstable_candidate_fom,
        ratio_emphasis=ratio_emphasis,
        fom_mode=fom_mode,
        angular_order_soft_sigma_deg=angular_order_soft_sigma_deg,
        opt_emission_plot=opt_emission_plot,
        pp_far_z_um=pp_far_z_um,
        pp_max_angle_deg=pp_max_angle_deg,
        pp_min_tiles=pp_min_tiles,
        pp_resolution=pp_resolution,
        active_radius=active_radius,
    )


# =============================================================================
# Simulation / geometry helpers  (OLED_new.py)
# =============================================================================


def dipole_angles(pol):
    pol = str(pol).strip().lower()
    if pol == "x":
        return 90.0, 0.0
    if pol == "y":
        return 90.0, 90.0
    if pol == "z":
        return 0.0, 0.0
    raise ValueError(f"Unsupported dipole polarization {pol!r}.")




def delete_object(fdtd, name):
    fdtd.eval(f'if (getnamednumber("{name}") > 0) {{ select("{name}"); delete; }}')


def make_sim(G, size, bc_x=None, bc_y=None, res=None):
    return ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=size,
        resolution=int(res or G.resolution),
        unit=1e-6,
        background_index=G.background_index,
        center_wl=float(np.mean(G.visible_wavelengths)),
        N_f=len(G.visible_wavelengths),
        bc_x=G.bc_xy if bc_x is None else bc_x,
        bc_y=G.bc_xy if bc_y is None else bc_y,
        bc_z="PML",
    )


def add_stack_z_mesh_override(G, sim, span_x=None, span_y=None, z_offset=0.0):
    """Refine the mesh in Z ONLY over the thin part of the stack.

    msopt installs a single uniform mesh over the whole domain, so a stack with
    10 nm layers would need that global mesh at ~2 nm -- unaffordable in 3D,
    since x and y would be refined too even though nothing in the plane needs
    it.  Lumerical mesh-override regions can refine one axis at a time, so this
    adds a region spanning the thin layers with "override z mesh" only: x and y
    keep the global step, z gets dz_fine.

    dz_fine = thinnest layer / MSOPT_OLED_STACK_MIN_CELLS_PER_LAYER (default 4),
    or MSOPT_OLED_STACK_MESH_DZ_NM when set explicitly.  The override is skipped
    when the global mesh is already fine enough, and when
    MSOPT_OLED_STACK_Z_MESH=0.
    """
    if not env_flag("MSOPT_OLED_STACK_Z_MESH", "1") or not G.stack_layers:
        return None
    span_x = G.Sx if span_x is None else span_x
    span_y = G.Sy if span_y is None else span_y
    global_dz_um = 1.0 / float(getattr(sim, "resolution", G.resolution))

    min_cells = env_float("MSOPT_OLED_STACK_MIN_CELLS_PER_LAYER", 4.0)
    thinnest = min(l["size"][2] for l in G.stack_layers)
    dz_nm = env_float("MSOPT_OLED_STACK_MESH_DZ_NM", 0.0)
    dz_fine_um = dz_nm * 1e-3 if dz_nm > 0 else thinnest / max(min_cells, 1.0)
    if dz_fine_um >= global_dz_um:
        print(f"[stack mesh] global dz={global_dz_um * 1000:.1f} nm already resolves the "
              f"thinnest layer ({thinnest * 1000:.1f} nm); no z override added")
        return None

    # Cover only the layers that the global mesh under-resolves, plus one
    # neighbour on each side so the refinement does not start mid-interface.
    thin_idx = [i for i, l in enumerate(G.stack_layers)
                if l["size"][2] < min_cells * global_dz_um]
    if not thin_idx:
        return None
    lo_i, hi_i = max(min(thin_idx) - 1, 0), min(max(thin_idx) + 1, len(G.stack_layers) - 1)
    z_lo = G.stack_layers[lo_i]["center"][2] - 0.5 * G.stack_layers[lo_i]["size"][2] + z_offset
    z_hi = G.stack_layers[hi_i]["center"][2] + 0.5 * G.stack_layers[hi_i]["size"][2] + z_offset
    # Never reach into the design region: it carries its own mesh override tied
    # to the import grid, and two overrides fighting there would break the
    # design-field alignment the adjoint gradient depends on.
    design_bottom = G.design_c[2] - 0.5 * G.design_s[2] + z_offset
    if z_hi > design_bottom:
        z_hi = design_bottom
    if z_hi <= z_lo:
        print("[stack mesh] thin layers overlap the design region; z override skipped")
        return None

    name = "oled_stack_z_mesh"
    fdtd = sim.fdtd
    if fdtd.getnamednumber(name) == 0:
        fdtd.addmesh()
        fdtd.set("name", name)
    else:
        fdtd.eval(f'select("{name}");')
    fdtd.set("x", 0.0)
    fdtd.set("x span", float(span_x) * sim.unit)
    fdtd.set("y", 0.0)
    fdtd.set("y span", float(span_y) * sim.unit)
    fdtd.set("z min", float(z_lo) * sim.unit)
    fdtd.set("z max", float(z_hi) * sim.unit)
    fdtd.set("override x mesh", 0)          # keep the global in-plane step
    fdtd.set("override y mesh", 0)
    fdtd.set("override z mesh", 1)
    fdtd.set("dz", float(dz_fine_um) * sim.unit)
    print(f"[stack mesh] z-only override {z_lo:+.4f}..{z_hi:+.4f} um: dz "
          f"{global_dz_um * 1000:.1f} -> {dz_fine_um * 1000:.2f} nm "
          f"(thinnest layer {thinnest * 1000:.1f} nm, >= {min_cells:g} cells); x/y unchanged")
    return {"name": name, "z_min_um": z_lo, "z_max_um": z_hi, "dz_um": dz_fine_um,
            "global_dz_um": global_dz_um, "thinnest_layer_um": thinnest}


def add_stack(G, sim, span_x=None, span_y=None, z_offset=0.0):
    # z_offset shifts the whole stack (used by the postprocess, which grows the
    # domain upward for far field and must keep the stack at the same height
    # above the bottom boundary).
    if span_x is None:
        span_x = G.Sx
    if span_y is None:
        span_y = G.Sy
    for layer in G.stack_layers:
        c = layer["center"]
        center = [c[0], c[1], c[2] + z_offset]
        size = [span_x, span_y, layer["size"][2]]
        sim.add_geo(center, size, layer["index"], layer["name"], float(np.mean(G.visible_wavelengths)))
    # Refine z only where the layers are thinner than the global mesh resolves;
    # no-op when the global mesh is already fine enough.
    try:
        G.stack_z_mesh = add_stack_z_mesh_override(G, sim, span_x, span_y, z_offset)
    except Exception as exc:
        G.stack_z_mesh = None
        print(f"[stack mesh] warning: z-only mesh override failed: {exc}")


def add_dipole(G, sim, x, y, z, pol, name="source", enabled=True, group_name=None):
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
    sim.fdtd.set("wavelength start", float(np.min(G.visible_wavelengths)) * 1e-6)
    sim.fdtd.set("wavelength stop", float(np.max(G.visible_wavelengths)) * 1e-6)
    sim.src_wl = G.visible_wavelengths.reshape(-1) * sim.unit
    sim.src_bw = 0.0


# =============================================================================
# Result readers  (OLED_new.py)
# =============================================================================


def finite_sum(values):
    vals = np.real(np.asarray(values, dtype=np.complex128)).reshape(-1)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    return float(np.sum(vals)) if vals.size else None


def signed_finite_sum(values):
    """Algebraic sum of the finite entries, negatives INCLUDED.

    dipolepower() is per-dipole net emission: in a coherent array a dipole
    driven against its neighbours' field ABSORBS, and that entry is legitimately
    negative.  finite_sum() drops those, which inflates the total emitted power
    and therefore deflates every efficiency computed from it.  Single-dipole
    runs are unaffected (one non-negative entry), so this only changes the
    coherent-array cases -- where dropping them was simply wrong.
    """
    vals = np.real(np.asarray(values, dtype=np.complex128)).reshape(-1)
    vals = vals[np.isfinite(vals)]
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
    # Signed: see signed_finite_sum -- a dipole in a coherent array may absorb.
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    try:
        return signed_finite_sum(fdtd.dipolepower(freqs_hz))
    except Exception:
        fdtd.putv("msopt_dipolepower_freqs", freqs_hz)
        fdtd.eval("msopt_dipolepower_values = dipolepower(msopt_dipolepower_freqs);")
        return signed_finite_sum(fdtd.getv("msopt_dipolepower_values"))


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


def source_freqs(G, sim):
    wl = np.asarray(getattr(sim, "src_wl", []), dtype=float).reshape(-1)
    if wl.size == 0:
        wl = G.visible_wavelengths * sim.unit
    return sim.c / wl


def valid_power(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > 0.0 else None




# =============================================================================
# Diffraction-order targets  (OLED_new.py)
# =============================================================================


def build_target_orders(G, wavelength_um=None, period_x_um=None, period_y_um=None):
    if wavelength_um is None:
        wavelength_um = float(np.mean(G.visible_wavelengths))
    if period_x_um is None:
        period_x_um = G.window_x
    if period_y_um is None:
        period_y_um = G.window_y
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
                        "efficiency": max(G.interp_curve(theta), 0.0),
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










def nearest_order_labels(ukx, uky, orders, propagating):
    """Assign each propagating far-field sample to exactly one diffraction order.

    Gaussian masks overlap and therefore count the same power more than once.
    A nearest-order (Voronoi) partition conserves total power by construction.
    """
    if not orders:
        return -np.ones(np.shape(ukx), dtype=int)
    distances = np.stack(
        [(ukx - float(o["ux"])) ** 2 + (uky - float(o["uy"])) ** 2 for o in orders],
        axis=0,
    )
    return np.where(propagating, np.argmin(distances, axis=0), -1)


# =============================================================================
# k-space FoM basis  (OLED_new.py; autograd-safe -- keep npa)
# =============================================================================














# =============================================================================
# Coherent-array optimization dipoles  (OLED_new.py)
# =============================================================================






# =============================================================================
# Channel glue: FoM terms, normalization references, design mapping  (OLED_new.py)
# =============================================================================












def design_to_grid(G, design, mapping=None, beta=1.0):
    """Expand a design vector to the full [Nx, Ny, Nz] grid.

    Accepts the same three input sizes the originals did: the full design_cells
    voxel vector, the mapped-parameter vector (via the `mapping` argument's
    parameter_count -- pass the script's ms.Opt_MS2.Mapping instance), or an
    Nx*Ny sheet that is repeated over z.
    """
    rho = np.asarray(design, dtype=float)
    Nx, Ny, Nz = G.design_grids
    if rho.size == G.design_cells:
        return rho.reshape(G.design_grids)
    if mapping is not None:
        design_parameters = getattr(mapping, "parameter_count", None)
        if design_parameters is not None and rho.size == int(design_parameters):
            return np.asarray(mapping(rho, beta), dtype=float).reshape(G.design_grids)
    if rho.size == Nx * Ny:
        return np.repeat(rho.reshape(Nx, Ny)[:, :, None], Nz, axis=2)
    raise ValueError(f"expected {G.design_cells}, mapped-parameter, or {Nx * Ny} design values, got {rho.size}")






def save_final_structure(G, design, mapping=None, prefix="OLED_final_structure"):
    """Write the postprocessed design's ACTUAL structure next to its results.

    The postprocess reads a design vector from somewhere else (an archived run,
    lastdesign.txt, ...), so without this the result folder does not record
    WHICH structure produced the numbers.  Writes, into G.design_dir:
      <prefix>.txt        flat rho on the (Nx, Ny, Nz) design grid (np.loadtxt-able,
                          same layout as lastdesign.txt), with the grid, cell size
                          and index endpoints in the header
      <prefix>_layers.txt per-z-layer rho as Nz blocks of an Nx x Ny matrix
      <prefix>.npz        rho + physical axes (x, y, z in um) + indices
      <prefix>.png        x-y sections for every z layer + an x-z section
    Returns the PNG path.
    """
    rho = np.asarray(design_to_grid(G, np.clip(np.asarray(design, dtype=float), 0.0, 1.0), mapping), dtype=float)
    Nx, Ny, Nz = G.design_grids
    x_axis = np.linspace(-0.5 * G.design_s[0], 0.5 * G.design_s[0], Nx)
    y_axis = np.linspace(-0.5 * G.design_s[1], 0.5 * G.design_s[1], Ny)
    z_axis = np.linspace(G.design_c[2] - 0.5 * G.design_s[2], G.design_c[2] + 0.5 * G.design_s[2], Nz)
    def _index_value(spec):
        # A design index is either a sampled-material dict or a plain [n] list.
        if isinstance(spec, dict):
            spec = spec.get("n", [np.nan])
        return float(np.mean(np.asarray(spec, dtype=float)))

    n_high = _index_value(G.design_high_index)
    n_low = _index_value(G.design_low_index)
    binarization = float(np.mean((rho < 1e-3) | (rho > 1.0 - 1e-3)))

    header = (
        f"OLED design density rho in [0,1] (n = {n_low:g} + rho*({n_high:g} - {n_low:g}))\n"
        f"design_grids Nx={Nx} Ny={Ny} Nz={Nz} (flat order: rho.reshape(Nx,Ny,Nz))\n"
        f"design_size_um {G.design_s[0]:g} x {G.design_s[1]:g} x {G.design_s[2]:g}\n"
        f"design_center_um {G.design_c[0]:g} {G.design_c[1]:g} {G.design_c[2]:g}\n"
        f"period_um {G.Sx:g} x {G.Sy:g}   resolution {G.resolution} /um\n"
        f"mean_rho {float(np.mean(rho)):.6f}   binarized_fraction {binarization:.4f}"
    )
    txt_path = os.path.join(G.design_dir, f"{prefix}.txt")
    np.savetxt(txt_path, rho.reshape(-1), header=header)

    layers_path = os.path.join(G.design_dir, f"{prefix}_layers.txt")
    with open(layers_path, "w", encoding="utf-8") as fp:
        for line in header.splitlines():
            fp.write(f"# {line}\n")
        fp.write(f"# {Nz} blocks of an {Nx} x {Ny} matrix, one per z layer (rows: x, cols: y)\n")
        for iz in range(Nz):
            fp.write(f"# z_layer {iz} z_um {z_axis[iz]:.6f}\n")
            np.savetxt(fp, rho[:, :, iz], fmt="%.6f")
    np.savez(
        os.path.join(G.design_dir, f"{prefix}.npz"),
        rho=rho, x_um=x_axis, y_um=y_axis, z_um=z_axis,
        n_high=n_high, n_low=n_low, period_um=np.asarray([G.Sx, G.Sy], dtype=float),
    )

    ncols = min(Nz, 7)
    layer_ids = np.unique(np.linspace(0, Nz - 1, ncols).round().astype(int))
    fig, axes = plt.subplots(1, len(layer_ids) + 1, figsize=(2.5 * (len(layer_ids) + 1), 3.0))
    axes = np.atleast_1d(axes)
    for ax, iz in zip(axes[:-1], layer_ids):
        ax.imshow(rho[:, :, iz].T, origin="lower", extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]),
                  cmap="binary", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
        ax.set_title(f"z={z_axis[iz]:.3f} um", fontsize=8)
        ax.set_xlabel("x (um)", fontsize=8)
        ax.set_ylabel("y (um)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].imshow(rho[:, Ny // 2, :].T, origin="lower", extent=(x_axis[0], x_axis[-1], z_axis[0], z_axis[-1]),
                    cmap="binary", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest")
    axes[-1].set_title("x-z at y=0", fontsize=8)
    axes[-1].set_xlabel("x (um)", fontsize=8)
    axes[-1].set_ylabel("z (um)", fontsize=8)
    axes[-1].tick_params(labelsize=7)
    fig.suptitle(
        f"Postprocessed structure: rho on {Nx}x{Ny}x{Nz}, n={n_low:g}/{n_high:g}, "
        f"period {G.Sx:g}x{G.Sy:g} um, binarized {binarization * 100:.1f}%",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    png_path = os.path.join(G.design_dir, f"{prefix}.png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"[postprocess] saved final structure: {png_path} (+ .txt, _layers.txt, .npz)")
    return png_path




# =============================================================================
# Far-field spectra and angle statistics  (OLED_new.py)
# =============================================================================


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


def direction_cosine_power_spectrum(E2, ux, uy):
    """Convert radial intensity on a uniform (ux,uy) grid to power samples.

    This is the direction-cosine form of the validated spherical integral
    ``Sr * R**2 * sin(theta) dtheta dphi`` because
    ``dOmega = dux duy / cos(theta)`` on the upper hemisphere.
    """
    E2 = np.squeeze(np.asarray(E2, dtype=float))
    ux = np.ravel(np.asarray(ux, dtype=float))
    uy = np.ravel(np.asarray(uy, dtype=float))
    if E2.shape != (ux.size, uy.size):
        E2 = E2.reshape(ux.size, uy.size)
    UX, UY = np.meshgrid(ux, uy, indexing="ij")
    r2 = UX ** 2 + UY ** 2
    du = abs(float(np.mean(np.diff(ux)))) if ux.size > 1 else 1.0
    dv = abs(float(np.mean(np.diff(uy)))) if uy.size > 1 else 1.0
    cos_theta = np.sqrt(np.maximum(1.0 - r2, 0.0))
    cos_floor = max(0.5 * max(du, dv), 1e-6)
    spectrum = np.where(
        r2 <= 1.0,
        np.maximum(E2, 0.0) * du * dv / np.maximum(cos_theta, cos_floor),
        0.0,
    )
    theta = np.rad2deg(np.arcsin(np.clip(np.sqrt(r2), 0.0, 1.0)))
    return theta, spectrum, UX, UY


def n2f_spectrum(sim, monitor_name, wavelength_um=None, na=None):
    """Lumerical top-aperture N2F in the validated radial-power convention."""
    na = int(na or env_int("MSOPT_OLED_PP_N2F_POINTS", 181))
    fdtd = sim.fdtd
    fdtd.eval(
        f'n2f_E2 = farfield3d("{monitor_name}", 1, {na}, {na});'
        f'n2f_ux = farfieldux("{monitor_name}", 1, {na}, {na});'
        f'n2f_uy = farfielduy("{monitor_name}", 1, {na}, {na});'
    )
    return direction_cosine_power_spectrum(
        fdtd.getv("n2f_E2"),
        fdtd.getv("n2f_ux"),
        fdtd.getv("n2f_uy"),
    )


def angle_profile(theta, spectrum, angles):
    centers = np.asarray(angles, dtype=float)
    edges = np.r_[0.0, 0.5 * (centers[:-1] + centers[1:]), 90.0] if centers.size > 1 else np.asarray([0.0, 90.0])
    out = []
    for i in range(centers.size):
        hi_cmp = theta <= edges[i + 1] if i == centers.size - 1 else theta < edges[i + 1]
        out.append(float(np.sum(spectrum[(theta >= edges[i]) & hi_cmp])))
    return np.asarray(out, dtype=float)


def angular_radiance_value(theta_grid, spectrum, center_deg, half_width_deg=None):
    """Cone/annulus-averaged per-direction radiance.

    An exact theta=0 power bin has zero solid-angle measure and usually contains
    only one far-field pixel. Averaging radiance inside a small normal cone is
    the stable quantity to compare with a 0-degree directional target.
    """
    half_width_deg = float(
        half_width_deg
        if half_width_deg is not None
        else env_float("MSOPT_OLED_PP_RADIANCE_HALF_WIDTH_DEG", 2.0)
    )
    theta_grid = np.asarray(theta_grid, dtype=float)
    radiance_samples = np.asarray(spectrum) * np.cos(np.deg2rad(theta_grid))
    center_deg = abs(float(center_deg))
    if center_deg <= 1e-12:
        positive = theta_grid[theta_grid > 1e-9]
        if positive.size:
            half_width_deg = max(half_width_deg, 1.01 * float(np.min(positive)))
        mask = theta_grid <= half_width_deg
    else:
        mask = np.abs(theta_grid - center_deg) <= half_width_deg
    vals = radiance_samples[mask & np.isfinite(radiance_samples)]
    return float(np.mean(vals)) if vals.size else 0.0


def radiance_from_spectrum(theta_grid, spectrum, angles):
    # spectrum contains power per uniform direction-cosine cell. Since
    # dOmega=dux*duy/cos(theta), multiply by cos(theta) before averaging to
    # recover a per-solid-angle (radiance-like) quantity.
    radiance_samples = np.asarray(spectrum) * np.cos(np.deg2rad(theta_grid))
    ring_flux = angle_profile(theta_grid, radiance_samples, angles)
    ring_count = angle_profile(theta_grid, np.ones_like(spectrum), angles)
    radiance = ring_flux / np.maximum(ring_count, 1.0)
    if np.asarray(angles).size and abs(float(np.asarray(angles)[0])) <= 1e-12:
        radiance[0] = angular_radiance_value(theta_grid, spectrum, 0.0)
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
    radiance_samples = np.asarray(spectrum) * np.cos(np.deg2rad(theta_grid))
    centers = np.asarray(signed_angles, dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    edges = np.r_[centers[0] - 90.0, mids, centers[-1] + 90.0]
    flux = np.empty(centers.size)
    cnt = np.empty(centers.size)
    for i in range(centers.size):
        if abs(float(centers[i])) <= 1e-12:
            normal_half_width = env_float("MSOPT_OLED_PP_RADIANCE_HALF_WIDTH_DEG", 2.0)
            positive = np.asarray(theta_grid)[np.asarray(theta_grid) > 1e-9]
            if positive.size:
                normal_half_width = max(normal_half_width, 1.01 * float(np.min(positive)))
            m = theta_grid <= normal_half_width
        else:
            m = (signed >= edges[i]) & (signed < edges[i + 1])
        flux[i] = float(np.sum(radiance_samples[m]))
        cnt[i] = float(np.sum(m))
    return flux, flux / np.maximum(cnt, 1.0)


# =============================================================================
# Plotting  (OLED_new.py)
# =============================================================================


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








def load_planar_reference_lee(path):
    """Mean LEE of an unpatterned (planar) postprocess run, for the 'no design'
    reference printed on the emission figure.

    `path` may be the run directory, its A/ directory, or either product file.
    The run is REJECTED unless its manifest says planar_baseline is true, so a
    patterned run cannot silently be presented as the bare-stack reference.
    Returns None (with a printed reason) when unavailable.
    """
    if not path:
        return None
    path = os.path.abspath(path)
    if os.path.isdir(path):
        a_dir = path if os.path.basename(path) == "A" else os.path.join(path, "A")
    else:
        a_dir = os.path.dirname(path)
    manifest_path = os.path.join(a_dir, "OLED_postprocess_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as fp:
                manifest = json.load(fp)
        except Exception as exc:
            print(f"[postprocess] warning: unreadable planar manifest {manifest_path}: {exc}")
            return None
        if not manifest.get("planar_baseline", False):
            print(f"[postprocess] warning: {manifest_path} is NOT a planar-baseline run "
                  "(planar_baseline is false); refusing to use it as the no-design reference.")
            return None
        lee = manifest.get("ensemble_lee")
        if lee is not None and np.isfinite(float(lee)):
            return float(lee)
    records = os.path.join(a_dir, "OLED_postprocess_3x3_records.txt")
    if os.path.exists(records):
        if not os.path.exists(manifest_path):
            print(f"[postprocess] warning: no manifest beside {records}; cannot verify it is a "
                  "planar run. Using its mean_LEE as the no-design reference anyway.")
        try:
            for line in open(records, encoding="utf-8"):
                if line.startswith("mean_LEE"):
                    return float(line.split()[1])
        except Exception as exc:
            print(f"[postprocess] warning: could not read {records}: {exc}")
    print(f"[postprocess] warning: no planar reference LEE found at {path}")
    return None


def annotate_emission_lee(ax, lee=None, planar_lee=None, x=0.98, y=0.98, va="top", ha="right"):
    """Put the LEE read-out on the emission figure: with the design, without it
    (the planar stack), and the enhancement factor between them."""
    lines = []
    if lee is not None:
        lines.append(f"LEE with design:  {lee * 100:.2f}%")
    if planar_lee is not None:
        lines.append(f"LEE no design:    {planar_lee * 100:.2f}%")
    if lee is not None and planar_lee:
        lines.append(f"enhancement:      {lee / planar_lee:.2f}x")
    elif lee is not None and planar_lee is None:
        lines.append("LEE no design:    (planar run not provided)")
    if not lines:
        return
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, va=va, ha=ha,
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))


def render_emission_figure(angles, radiance_signed_norm, target_norm, order_thetas, achieved_share, target_share, path, label, lee=None, planar_lee=None, cumulative=None):
    # Shared 2-panel emission figure: (left) SIGNED per-direction radiance (real +/-
    # asymmetry, not a mirror) vs the symmetric target, (right) order power share
    # plus the LEE read-out (with design / without design / enhancement).
    ncol = 3 if cumulative is not None else 2
    fig = plt.figure(figsize=(6.0 * ncol, 4.6))
    ax0 = fig.add_subplot(1, ncol, 1, projection="polar")
    signed = signed_angle_axis(angles)
    ax0.plot(np.deg2rad(signed), radiance_signed_norm, label=label)   # already over signed axis
    ax0.plot(np.deg2rad(signed), np.r_[target_norm[1:][::-1], target_norm], label="target")
    ax0.set_thetamin(-90)
    ax0.set_thetamax(90)
    ax0.set_theta_zero_location("N")
    ax0.set_theta_direction(-1)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("RADIANCE  (power per unit SOLID ANGLE)")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18))

    ax1 = fig.add_subplot(1, ncol, 2)
    xpos = np.arange(len(order_thetas))
    bw = 0.4
    ax1.bar(xpos - 0.5 * bw, target_share, width=bw, label="target share", color="tab:orange")
    ax1.bar(xpos + 0.5 * bw, achieved_share, width=bw, label="achieved share", color="tab:blue")
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([f"{t:.0f}" for t in order_thetas])
    ax1.set_xlabel("diffraction order angle (deg)")
    ax1.set_ylabel("power share")
    ax1.set_title("POWER share per order  (radiance x solid angle)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper left")
    if lee is not None or planar_lee is not None:
        top = max(float(np.max(achieved_share)), float(np.max(target_share)))
        ax1.set_ylim(0.0, top * 1.38)          # headroom for the LEE box
        annotate_emission_lee(ax1, lee, planar_lee)

    if cumulative is not None:
        # Third panel: the question the other two do NOT answer. Radiance is
        # per-direction and the bars are per-order, so neither can be integrated
        # by eye into "how much light comes out within theta" -- that needs the
        # per-bin cell count and a 1/cos(theta), which is exactly what this curve
        # already carries.
        cum_ang, cum_share, cum_abs = cumulative
        ax2 = fig.add_subplot(1, ncol, 3)
        ax2.plot(cum_ang, np.asarray(cum_abs) * 100.0, "o-", lw=2, color="tab:blue",
                 label="of emitted power")
        ax2.plot(cum_ang, np.asarray(cum_share) * 100.0, "s--", lw=1.6, color="tab:green",
                 label="of extracted light")
        lam = np.asarray(cum_ang, dtype=float)
        ax2.plot(lam, np.sin(np.deg2rad(lam)) ** 2 * 100.0, ":", lw=1.4, color="0.5",
                 label="Lambertian reference")
        ax2.set_xlabel("acceptance half-angle theta (deg)")
        ax2.set_ylabel("cumulative [%]")
        ax2.set_title("EXTRACTION within theta  (the efficiency number)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

    # The two panels are DIFFERENT quantities and routinely look contradictory
    # at normal incidence: radiance peaks there while the power share nearly
    # vanishes, because the solid angle of a ring goes to zero as sin(theta).
    # Say so on the figure instead of letting the reader infer a bug.
    fig.text(0.5, 0.005,
             "left = per-direction radiance; right = power integrated over each order. "
             "A ring's solid angle ~ sin(theta), so normal incidence peaks in radiance "
             "yet carries little power. Neither panel is the extraction efficiency: "
             "see OLED_postprocess_cumulative_extraction.png",
             ha="center", va="bottom", fontsize=7.5, color="0.30")
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    fig.savefig(path, dpi=200)
    plt.close(fig)


PLANAR_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Planer_data.txt")


def planar_reference_identity(G=None, grid_n=None):
    """What the cached planar curve has to agree with to be comparable.

    One global file at a fixed path, read by every later run, is exactly the
    shape of cache that goes stale silently: change the stack colour, the
    wavelength, the pitch or the dipole grid and the black overlay in
    PP_summary.png is still drawn, just no longer describing this device. So the
    identity is written INTO the file and checked on load -- the same discipline
    pp_cache_key uses, which this file predates.
    """
    if G is None:
        return None
    return "|".join(str(v) for v in (
        # FORMAT TAG. "abs1" = the curve is radiance per unit emitted power.
        # Files written before that carried a peak-normalized curve instead, which
        # the overlay would now divide by an absolute 0 deg value and put six
        # orders of magnitude off the axis. Bump this whenever the stored quantity
        # changes, so an old file is refused rather than misread.
        "abs1",
        os.environ.get("MSOPT_MC_COLOR", os.environ.get("MSOPT_OLED_MC_COLOR", "green")),
        os.environ.get("MSOPT_MC_STACK_KIND", "optimized"),
        os.environ.get("MSOPT_OLED_STACK", "microcavity"),
        round(float(np.mean(G.visible_wavelengths)), 6),
        round(float(G.Sx), 6), round(float(G.Sy), 6),
        round(float(G.target_monitor_c[2]), 6),
        int(G.resolution),
        int(grid_n) if grid_n is not None else "?",
        os.environ.get("MSOPT_OLED_PP_SOURCE_LAYOUT", "cell_center").strip().lower(),
    ))


def save_planar_reference_curve(angles, radiance_signed_abs, lee=None,
                                path=PLANAR_DATA_PATH, identity=None):
    """Persist the PLANAR device's angular radiance next to the scripts.

    Stored in ABSOLUTE units -- radiance per unit emitted power -- because every
    future run divides it by ITS OWN 0 deg radiance to place it, and a curve
    already divided by its own peak carries no level information to place.
    (It used to be handed radiance_signed_norm, which IS peak-normalized, so the
    overlay silently compared shapes only; see render_pp_summary_figure.)
    Written once, on the first planar postprocess, and read back by every run
    after that, so the reference costs one simulation ever -- which is why it
    also carries the identity it was measured under.
    """
    ang = np.asarray(angles, dtype=float)
    arr = np.asarray(radiance_signed_abs, dtype=float)
    pos = arr[ang.size - 1:]
    neg = arr[:ang.size][::-1]
    curve = 0.5 * (pos + neg)                     # +/-kx averaged, still raw
    header = ("planar reference radiance, ABSOLUTE: per unit emitted power.\n"
              "NOT divided by its own peak or its own 0 deg -- the overlay divides\n"
              "it by THIS design's 0 deg value, so the dashed line's height is real.\n"
              f"lee {'' if lee is None else float(lee)}\n"
              f"identity {identity if identity else 'unknown'}\n"
              "theta_deg  radiance")
    np.savetxt(path, np.column_stack([ang, curve]), header=header)
    print(f"[postprocess] saved planar reference curve -> {path}")
    return path


def load_planar_reference_curve(path=PLANAR_DATA_PATH, identity=None):
    """The stored planar curve, or None when it is missing or was measured under
    a different configuration.

    Refusing beats overlaying: a missing black line is obviously missing, while a
    wrong one is read as truth. Files written before the identity existed are
    also refused -- they cannot be shown to match.
    """
    if not os.path.exists(path):
        return None, None
    if identity is not None:
        stored = None
        try:
            with open(path) as fh:
                for line in fh:
                    if not line.startswith("#"):
                        break
                    body = line.lstrip("# ").rstrip()
                    if body.startswith("identity "):
                        stored = body.split(" ", 1)[1]
        except Exception:
            stored = None
        if stored != identity:
            print(f"[postprocess] planar reference SKIPPED: measured under "
                  f"{stored!r}, this run is {identity!r}. Re-measure with "
                  f"MSOPT_OLED_PP_PLANAR=low to restore the overlay.")
            return None, None
    try:
        d = np.atleast_2d(np.loadtxt(path))
        if d.shape[1] < 2 or d.shape[0] < 2:
            return None, None
        lee = None
        with open(path) as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if line.lstrip("# ").startswith("lee "):
                    tok = line.lstrip("# ").split()
                    if len(tok) > 1:
                        try:
                            lee = float(tok[1])
                        except ValueError:
                            lee = None
        return d[:, 0], (d[:, 1], lee)
    except Exception as exc:
        print(f"[postprocess] planar reference unreadable ({exc}); ignoring")
        return None, None


def render_pp_summary_figure(angles, radiance_signed_norm, target_norm, path,
                             lee=None, planar_lee=None, planar_identity=None,
                             radiance_signed_abs=None):
    """PP_summary.png -- the radiance, twice: as the polar lobe and as the same
    curve unrolled onto a linear theta axis.

    left  : signed polar radiance vs target (the familiar lobe).
    right : identical data, x = theta 0..90 deg, y = radiance AS A PERCENT OF THE
            0 deg value. This is the read-out the polar plot makes hard to do by
            eye -- "how bright is 45 deg compared to normal" -- and it is the
            same quantity the angular-metric ratios_to_zero report.

    Both kx branches are drawn so a real left/right asymmetry stays visible.
    Lambertian is a flat 100% line here, because Lambertian means CONSTANT
    radiance; it is not a quality target, only the neutral shape.
    """
    fig = plt.figure(figsize=(12.6, 4.8))

    ax0 = fig.add_subplot(1, 2, 1, projection="polar")
    signed = signed_angle_axis(angles)
    ax0.plot(np.deg2rad(signed), radiance_signed_norm, lw=1.8, label="radiance")
    if target_norm is not None:
        ax0.plot(np.deg2rad(signed), np.r_[target_norm[1:][::-1], target_norm],
                 lw=1.3, ls="--", label="target")
    ax0.set_thetamin(-90)
    ax0.set_thetamax(90)
    ax0.set_theta_zero_location("N")
    ax0.set_theta_direction(-1)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("RADIANCE  (polar, normalized to peak)")
    ax0.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)

    ang = np.asarray(angles, dtype=float)
    n = ang.size
    arr = np.asarray(radiance_signed_norm, dtype=float)
    pos = arr[n - 1:]                 # +kx half, theta 0..90
    neg = arr[:n][::-1]               # -kx half, theta 0..90
    r0 = float(arr[n - 1])
    scale = 100.0 / r0 if abs(r0) > 1e-30 else 0.0

    ax1 = fig.add_subplot(1, 2, 2)
    asymmetric = float(np.max(np.abs(pos - neg))) > 0.02 * max(abs(r0), 1e-30)
    if asymmetric:
        ax1.plot(ang, pos * scale, lw=1.8, color="tab:blue", label="radiance (+kx)")
        ax1.plot(ang, neg * scale, lw=1.4, color="tab:cyan", ls="-.", label="radiance (-kx)")
    else:
        ax1.plot(ang, 0.5 * (pos + neg) * scale, lw=1.8, color="tab:blue", label="radiance")
    if target_norm is not None:
        t = np.asarray(target_norm, dtype=float)
        if abs(float(t[0])) > 1e-30:
            ax1.plot(ang, t / float(t[0]) * 100.0, lw=1.4, ls="--", color="tab:orange",
                     label="target")
    # PLANAR overlay, divided by THIS DESIGN's 0 deg radiance -- both curves in
    # ABSOLUTE per-emitted-power units, so the dashed line's HEIGHT is meaningful:
    # above 100% means the flat stack is brighter in that direction than this
    # design is on axis.
    #
    # It used to be scaled by the design's 0 deg value too, but the quantity being
    # scaled was radiance_signed_norm -- already divided by its OWN peak. Both
    # curves therefore landed at ~100% at 0 deg and only their SHAPES could be
    # compared, which reads as "the design beats planar everywhere past 25 deg"
    # while the design actually extracts LESS light in total (LEE 41.2% against
    # the planar 44.7%). Absolute units are what make that visible.
    p_ang, p_pack = load_planar_reference_curve(identity=planar_identity)
    abs_arr = None if radiance_signed_abs is None else np.asarray(radiance_signed_abs, dtype=float)
    r0_abs = float(abs_arr[n - 1]) if abs_arr is not None and abs_arr.size >= n else 0.0
    if p_ang is not None and abs(r0_abs) > 1e-30:
        p_curve, _p_lee = p_pack
        p_rel = np.asarray(p_curve, dtype=float) / r0_abs * 100.0
        ax1.plot(p_ang, p_rel, lw=1.5, ls="--", color="black",
                 label="planar reference (absolute)")
        planar_on_axis = float(p_rel[0]) if p_rel.size else float("nan")
    elif p_ang is not None:
        print("[postprocess] planar overlay skipped: no absolute radiance available "
              "(radiance_signed_abs not supplied)")
        planar_on_axis = float("nan")
    else:
        planar_on_axis = float("nan")
    ax1.axhline(100.0, lw=1.0, ls=":", color="0.55", label="Lambertian (constant)")
    ax1.set_xlim(0.0, 90.0)
    ax1.set_xticks(np.arange(0.0, 91.0, 10.0))
    ax1.set_xlabel("emission angle theta (deg)")
    ax1.set_ylabel("radiance relative to 0 deg  [%]")
    ax1.set_title("RADIANCE vs theta   (0 deg = 100%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # Numeric read-out at the angles the performance spec is written against,
    # so the percentages can be quoted off the figure without measuring pixels.
    marks = [a for a in (30.0, 45.0, 60.0) if a <= float(ang[-1])]
    rows = [f"0deg = 100.0%  (reference)"]
    if planar_on_axis == planar_on_axis:      # not NaN
        rows.append(f"planar 0deg = {planar_on_axis:5.1f}%")
    for a in marks:
        v = float(np.interp(a, ang, 0.5 * (pos + neg) * scale))
        ax1.plot([a], [v], "o", ms=4, color="tab:blue")
        rows.append(f"{a:.0f}deg = {v:5.1f}%")
    ax1.text(0.02, 0.02, "\n".join(rows), transform=ax1.transAxes, va="bottom",
             ha="left", fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))

    top = float(np.nanmax([np.max(pos * scale), 100.0]))
    ax1.set_ylim(0.0, top * 1.55)
    if lee is not None or planar_lee is not None:
        annotate_emission_lee(ax1, lee, planar_lee, x=0.02, y=0.98, va="top", ha="left")

    fig.text(0.5, 0.005,
             "Both panels are the SAME per-direction radiance; the right one is the polar lobe\n"
             "unrolled onto a linear theta axis and referenced to normal incidence.\n"
             "Radiance is per unit solid angle -- it is not extraction efficiency: "
             "see OLED_postprocess_cumulative_extraction.png",
             ha="center", va="bottom", fontsize=7.5, color="0.30")
    fig.tight_layout(rect=(0.0, 0.115, 1.0, 1.0))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_per_dipole_emission_plot(G, angles, per_dipole, path):
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

    for a, _ in G.target_angle_pairs:
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




# =============================================================================
# Postprocess: incoherent-dipole far-field verification  (OLED_new.py)
# =============================================================================


def central_cell_dipoles(G, n_samples, pol="x", grid_n=None):
    # Same seeded positions regardless of pol, so a polarization sweep re-samples the
    # identical dipole locations and only the orientation changes.
    pol = str(pol).strip().lower()
    # The validated Meep reference uses a 6x6 endpoint grid inset by two mesh
    # pixels from the emission-region boundary, and runs every point. Its
    # 9-run/multiplicity shortcut is valid only when geometric mirror symmetry is
    # guaranteed -- which resolve_dipole_grid now decides from the density grid
    # itself, and passes in as grid_n. None keeps the standalone default.
    grid_n = env_int("MSOPT_OLED_PP_DIPOLE_GRID", 6) if grid_n is None else int(grid_n)
    if grid_n > 0:
        protocol = os.environ.get("MSOPT_OLED_PP_SOURCE_LAYOUT", "cell_center").strip().lower()
        if protocol not in ("validated_endpoint", "cell_center"):
            raise ValueError(
                "MSOPT_OLED_PP_SOURCE_LAYOUT must be 'validated_endpoint' or 'cell_center'."
            )
        if protocol == "validated_endpoint":
            inset_pixels = env_float("MSOPT_OLED_PP_SOURCE_INSET_PIXELS", 2.0)
            inset = inset_pixels / max(float(G.pp_resolution), 1.0)
            half_x = 0.5 * G.active_x - inset
            half_y = 0.5 * G.active_y - inset
            if half_x < 0.0 or half_y < 0.0:
                raise ValueError("Postprocess source inset is larger than the active emission area.")
            xs = np.asarray([0.0]) if grid_n == 1 else np.linspace(-half_x, half_x, grid_n)
            ys = np.asarray([0.0]) if grid_n == 1 else np.linspace(-half_y, half_y, grid_n)
        else:
            xs = (np.arange(grid_n, dtype=float) + 0.5) * G.active_x / grid_n - 0.5 * G.active_x
            ys = (np.arange(grid_n, dtype=float) + 0.5) * G.active_y / grid_n - 0.5 * G.active_y
        return [(float(x), float(y), float(G.eml_c[2]), pol) for x in xs for y in ys]
    rng = np.random.default_rng(G.seed)
    return [
        (float(rng.uniform(-0.5 * G.active_x, 0.5 * G.active_x)), float(rng.uniform(-0.5 * G.active_y, 0.5 * G.active_y)), float(G.eml_c[2]), pol)
        for _ in range(n_samples)
    ]


def mirror_orbits(grid_n):
    """Group the dipole grid into orbits of the two mirror planes x->-x, y->-y.

    Both source layouts central_cell_dipoles can build are symmetric about the
    cell centre, so grid index (ix, iy) -- flattened as ix*n + iy, the order
    central_cell_dipoles emits -- pairs with (n-1-ix, iy), (ix, n-1-iy) and
    (n-1-ix, n-1-iy).  For a design whose geometry carries those mirrors, all
    four are the SAME simulation: an electric dipole is a true vector, so a
    reflection that maps the structure onto itself maps an x-oriented source at
    (x, y) onto an x-oriented source at (-x, y) (the sign of the moment flips
    with the field, and |E|^2 does not see it).  The far field of the image is
    the reflected far field, which is why the caller has to flip the spectrum.

    Returns [(rep, [(member, flip_x, flip_y), ...]), ...], every index covered
    exactly once, rep first.  With an odd grid_n the centre row/column is its
    own image and the orbit is correspondingly smaller -- membership is built
    from the index set, not assumed to be 4.
    """
    n = int(grid_n)
    seen, out = set(), []
    for ix in range(n):
        for iy in range(n):
            rep = ix * n + iy
            if rep in seen:
                continue
            members = []
            for jx, fx in ((ix, False), (n - 1 - ix, True)):
                for jy, fy in ((iy, False), (n - 1 - iy, True)):
                    m = jx * n + jy
                    if m not in {mm for mm, _a, _b in members}:
                        members.append((m, fx, fy))
            seen.update(m for m, _a, _b in members)
            out.append((rep, members))
    return out


def resolve_dipole_grid(rho):
    """Decide the dipole grid AND whether the C4v fold applies, in one place.

    THE POINT OF THE DEFAULT: a mirror-symmetric design makes each orbit of four
    grid points one simulation, so the 6x6 grid's 36 runs only ever held 9
    independent numbers -- three quarters of the sweep was paid for and
    discarded. Rather than bank that as a speedup, the default spends it on
    RESOLUTION: 12x12 folded is the same 36 FDTD runs as 6x6 unfolded, with 36
    independent samples instead of 9 and a sampling Nyquist of 6 per period
    instead of 3. The measured driver is that the per-dipole LEE varies ~26%
    peak-to-peak across the cell while its second difference exceeds its first,
    i.e. the 6-point grid sits at or past its own resolution limit.

    An asymmetric design cannot fold, and there the grid stays 6x6 -- 12x12
    unfolded would be 144 runs, quadrupling the cost of every postprocess.
    So the FDTD count is 36 either way and only the sampling changes.

    MSOPT_OLED_PP_SYMMETRY_FOLD: "auto" (default) folds when the density grid
    actually carries both mirrors, "1" demands it and fails if it does not, "0"
    disables it. MSOPT_OLED_PP_DIPOLE_GRID still overrides the grid explicitly.

    Returns (grid_n, fold_on, asymmetry).
    """
    asym = design_mirror_asymmetry(rho)
    symmetric = asym <= env_float("MSOPT_OLED_PP_SYMMETRY_TOL", 1e-9)
    mode = os.environ.get("MSOPT_OLED_PP_SYMMETRY_FOLD", "auto").strip().lower() or "auto"
    if mode in ("1", "true", "yes", "on"):
        if not symmetric:
            raise ValueError(
                f"MSOPT_OLED_PP_SYMMETRY_FOLD=1 needs a mirror-symmetric design, but the "
                f"density grid deviates from its own mirror image by {asym:.3e}. "
                f"Use 'auto' to fall back to the full grid, or fix the design symmetry."
            )
        fold_on = True
    elif mode in ("0", "false", "no", "off"):
        fold_on = False
    elif mode == "auto":
        fold_on = symmetric
    else:
        raise ValueError("MSOPT_OLED_PP_SYMMETRY_FOLD must be auto, 0 or 1, "
                         f"got {mode!r}")
    grid_n = env_int("MSOPT_OLED_PP_DIPOLE_GRID", 12 if fold_on else 6)
    # grid_n <= 0 selects the random-sample layout, which has no mirror pairing.
    if grid_n <= 0:
        fold_on = False
    return grid_n, fold_on, asym


def design_mirror_asymmetry(rho):
    """Max relative deviation of the design from its own x and y mirror images.

    The postprocess symmetry fold is only valid if the STRUCTURE actually has
    those mirrors.  Rather than trust a symmetry flag set in another file, this
    measures the density grid that is about to be built, so a design that lost
    its symmetry (or was never constrained to it) cannot silently be scored
    from a quarter of the dipoles.
    """
    r = np.asarray(rho, dtype=float)
    scale = max(float(np.max(np.abs(r))), 1e-30)
    return max(float(np.max(np.abs(r - r[::-1, ...]))),
               float(np.max(np.abs(r - r[:, ::-1, ...])))) / scale


def minimum_source_box_clearance(sources, center, size):
    """Minimum axis-aligned clearance of point sources from a flux-box face."""
    center = np.asarray(center, dtype=float).reshape(3)
    half_size = 0.5 * np.asarray(size, dtype=float).reshape(3)
    points = list(sources)
    if not points:
        return None
    clearances = []
    for source in points:
        position = np.asarray(source[:3], dtype=float)
        clearances.append(float(np.min(half_size - np.abs(position - center))))
    return float(min(clearances))


def postprocess_polarizations():
    # "x,y" (or add "z") -> route that runs the incoherent single-dipole sweep once per
    # polarization; spectra add incoherently (independent emitter orientations).
    raw = os.environ.get(
        "MSOPT_OLED_POSTPROCESS_POLARIZATIONS",
        os.environ.get("MSOPT_OLED_POSTPROCESS_POLARIZATION", "x,y"),
    )
    pols = [p.strip().lower() for p in raw.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    for p in pols:
        if p not in ("x", "y", "z"):
            raise ValueError(f"Unsupported postprocess polarization {p!r}.")
    return pols or ["x"]


def add_flux_box_monitors(sim, prefix, center, size):
    """Add the six signed faces of the validated source-power flux box."""
    cx, cy, cz = [float(v) for v in center]
    sx, sy, sz = [float(v) for v in size]
    if min(sx, sy, sz) <= 0.0:
        raise ValueError(f"Flux-box spans must be positive, got {size}.")
    faces = [
        (f"{prefix}_xp", [cx + 0.5 * sx, cy, cz], [0.0, sy, sz], +1.0),
        (f"{prefix}_xm", [cx - 0.5 * sx, cy, cz], [0.0, sy, sz], -1.0),
        (f"{prefix}_yp", [cx, cy + 0.5 * sy, cz], [sx, 0.0, sz], +1.0),
        (f"{prefix}_ym", [cx, cy - 0.5 * sy, cz], [sx, 0.0, sz], -1.0),
        (f"{prefix}_zp", [cx, cy, cz + 0.5 * sz], [sx, sy, 0.0], +1.0),
        (f"{prefix}_zm", [cx, cy, cz - 0.5 * sz], [sx, sy, 0.0], -1.0),
    ]
    for name, face_center, face_size, _sign in faces:
        sim.add_monitor(name, face_center, face_size)
    return [(name, sign) for name, _center, _size, sign in faces]


def read_flux_box_power(fdtd, signed_monitor_names, source_power):
    """Outward power through a six-face box using transmission sign convention."""
    src = valid_power(source_power)
    if src is None:
        return None
    signed_transmission = 0.0
    for name, sign in signed_monitor_names:
        signed_transmission += float(sign) * read_transmission(fdtd, name)
    return valid_power(signed_transmission * src)


def _theta_bin_edges(angles):
    mid = 0.5 * (angles[:-1] + angles[1:])
    return np.concatenate([[angles[0]], mid, [angles[-1] + 1e-9]])


def theta_phi_map(theta2d, ukx, uky, spectrum, n_theta=91, n_phi=36):
    """Regrid the (ukx, uky) spectrum onto a (theta, phi) map of per-direction MEAN
    intensity (radiance-like) -- the Sr(theta, phi) heatmap the Meep step2b reference
    script reports. The map preserves the azimuthal asymmetry of off-center dipoles
    that the |k|-ring-binned profiles average away. Empty bins stay 0."""
    prop = np.sqrt(ukx ** 2 + uky ** 2) <= 1.0 + 1e-12
    th = np.asarray(theta2d)[prop]
    ph = (np.degrees(np.arctan2(uky, ukx)) + 360.0) % 360.0
    ph = ph[prop]
    sp = (
        np.asarray(spectrum)
        * np.cos(np.deg2rad(np.asarray(theta2d)))
    )[prop]
    th_edges = np.linspace(0.0, 90.0, n_theta + 1)
    ph_edges = np.linspace(0.0, 360.0, n_phi + 1)
    ti = np.clip(np.digitize(th, th_edges) - 1, 0, n_theta - 1)
    pi = np.clip(np.digitize(ph, ph_edges) - 1, 0, n_phi - 1)
    acc = np.zeros((n_theta, n_phi))
    cnt = np.zeros((n_theta, n_phi))
    np.add.at(acc, (ti, pi), sp)
    np.add.at(cnt, (ti, pi), 1.0)
    Sr = acc / np.maximum(cnt, 1.0)
    # Near theta=0 the direction-cosine grid provides only a handful of samples
    # (only ~9 k-points inside theta<1 deg) but n_phi azimuth sectors, so most
    # small-theta bins receive no sample; leaving them 0 punches a fake hole at
    # normal incidence and biases the azimuth-mean cut low by filled/n_phi.
    # phi is degenerate at theta=0 anyway, so fill EMPTY bins with the ring
    # (constant-theta) mean of the filled ones; theta rows with no samples at
    # all stay 0.
    ring_mean = acc.sum(axis=1) / np.maximum(cnt.sum(axis=1), 1.0)
    Sr = np.where(cnt > 0, Sr, ring_mean[:, None])
    th_axis = 0.5 * (th_edges[:-1] + th_edges[1:])
    ph_axis = 0.5 * (ph_edges[:-1] + ph_edges[1:])
    return th_axis, ph_axis, Sr


def save_radiation_map_figure(path, th_axis, ph_axis, Sr, title):
    """Meep step2b-style radiation figure: Sr(theta, phi) heatmap + azimuth-averaged
    polar cut. The cut resolves theta=0 directly (radiance, not ring power)."""
    I_theta = Sr.mean(axis=1)
    I_norm = I_theta / max(float(np.max(np.abs(I_theta))), 1e-30)
    fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1)
    im = ax0.imshow(
        Sr, origin="lower", aspect="auto",
        extent=[ph_axis[0], ph_axis[-1], th_axis[0], th_axis[-1]],
    )
    ax0.set_title("per-direction intensity")
    ax0.set_xlabel("phi [deg]")
    ax0.set_ylabel("theta [deg]")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax1 = fig.add_subplot(1, 2, 2, projection="polar")
    th = np.deg2rad(th_axis)
    th_full = np.concatenate([-th[::-1], th[1:]])
    I_full = np.concatenate([I_norm[::-1], I_norm[1:]])
    ax1.plot(th_full, I_full, lw=2.0)
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_thetamin(-90)
    ax1.set_thetamax(90)
    ax1.set_title("azimuth-averaged radiance cut")
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Top-emission microcavity stack (optimized by OLED_stack_design.py)
# =============================================================================
# Thicknesses found with the validated analytic layered-dipole solver on
# Johnson & Christy silver and literature organic/ITO indices. Analytic
# horizontal-dipole outcoupling: R 61.8 %, G 57.3 %, B 50.3 %. The originally
# reconstructed layer table sat at cavity ANTI-resonance (green 0.89 %), so it
# is kept only for comparison, under kind="table".
MICROCAVITY_OPTIMIZED = {
    "red":   dict(wavelength_um=0.620, htl_nm=205.0, eml_nm=25.0, cpl_nm=75.0,
                  etl_nm=50.0, cath_nm=10.0, n_cpl=2.45, ag_n=0.135, ag_k=3.990,
                  eml_n=1.770, n_ito=1.900, k_ito=0.010, n_hil=1.830,
                  n_htl=1.790, n_ebl=1.800, n_hbl=1.740, n_etl=1.730),
    "green": dict(wavelength_um=0.530, htl_nm=155.0, eml_nm=25.0, cpl_nm=70.0,
                  etl_nm=40.0, cath_nm=13.0, n_cpl=2.20, ag_n=0.121, ag_k=3.090,
                  eml_n=1.790, n_ito=1.950, k_ito=0.010, n_hil=1.850,
                  n_htl=1.810, n_ebl=1.820, n_hbl=1.760, n_etl=1.750),
    "blue":  dict(wavelength_um=0.460, htl_nm=125.0, eml_nm=25.0, cpl_nm=70.0,
                  etl_nm=20.0, cath_nm=11.0, n_cpl=2.00, ag_n=0.140, ag_k=2.560,
                  eml_n=1.830, n_ito=2.020, k_ito=0.030, n_hil=1.880,
                  n_htl=1.850, n_ebl=1.860, n_hbl=1.790, n_etl=1.790),
}
MICROCAVITY_TABLE = {
    "red":   dict(wavelength_um=0.620, htl_nm=140.0, eml_nm=30.0, cpl_nm=70.0,
                  etl_nm=30.0, cath_nm=12.0, n_cpl=1.800, ag_n=0.12, ag_k=4.05,
                  eml_n=1.81, cath_n=0.250, cath_k=3.60),
    "green": dict(wavelength_um=0.530, htl_nm=95.0, eml_nm=25.0, cpl_nm=60.0,
                  etl_nm=30.0, cath_nm=12.0, n_cpl=1.800, ag_n=0.09, ag_k=3.32,
                  eml_n=1.79, cath_n=0.250, cath_k=3.10),
    "blue":  dict(wavelength_um=0.460, htl_nm=60.0, eml_nm=20.0, cpl_nm=50.0,
                  etl_nm=30.0, cath_nm=12.0, n_cpl=1.800, ag_n=0.06, ag_k=2.66,
                  eml_n=1.78, cath_n=0.250, cath_k=2.70),
}
MICROCAVITY_EML_LAYER = "EML"
MICROCAVITY_FIXED_NM = dict(ag_anode=100.0, ito=10.0, hil=10.0, ebl=10.0, hbl=10.0)


def export_run_knobs(**knobs):
    """Publish the script-top run switches as environment DEFAULTS.

    Every optimizer carries a small block of plain Python variables at the top so
    a run can be configured by editing the file instead of prefixing a command
    with a dozen MSOPT_OLED_* assignments.  They are exported with setdefault, so
    an explicitly set environment variable still wins and sweeps keep working
    without touching the file.  None means "leave unset".

    Booleans map to the "1"/"0" the readers expect; planar accepts True/False or
    the "low"/"high" variants that also choose WHICH bare stack to build.
    """
    def put(name, value, cast=str):
        if value is None:
            return
        os.environ.setdefault(name, cast(value))

    flag = lambda v: "1" if v else "0"
    put("MSOPT_OLED_RESOLUTION", knobs.get("resolution"))
    put("MSOPT_OLED_DESIGN_H_UM", knobs.get("design_h_um"))
    put("MSOPT_OLED_DESIGN_X_UM", knobs.get("design_x_um"))
    put("MSOPT_OLED_DESIGN_Y_UM", knobs.get("design_y_um"))
    put("MSOPT_OLED_DESIGN_N", knobs.get("design_n"))
    put("MSOPT_OLED_DESIGN_K", knobs.get("design_k"))
    put("MSOPT_OLED_DESIGN_LOW_N", knobs.get("design_low_n"))
    put("MSOPT_OLED_PROBE_GAP_UM", knobs.get("probe_gap_um"))
    put("MSOPT_OLED_TOP_MARGIN_UM", knobs.get("top_margin_um"))
    put("MSOPT_OLED_STACK", knobs.get("stack"))
    put("MSOPT_MC_COLOR", knobs.get("mc_color"))
    put("MSOPT_MC_STACK_KIND", knobs.get("mc_stack_kind"))

    run_opt = knobs.get("run_optimization")
    if run_opt is not None:
        os.environ.setdefault("MSOPT_OLED_POSTPROCESS_ONLY", flag(not run_opt))
    put("MSOPT_OLED_POSTPROCESS", knobs.get("run_postprocess"), flag)

    planar = knobs.get("planar_pp")
    if planar is not None:
        if isinstance(planar, str):
            os.environ.setdefault("MSOPT_OLED_PP_PLANAR", planar)
        else:
            os.environ.setdefault("MSOPT_OLED_PP_PLANAR", "low" if planar else "0")

    put("MSOPT_OLED_PP_MODE", knobs.get("pp_mode"))
    put("MSOPT_OLED_PP_DIPOLE_GRID", knobs.get("pp_dipole_grid"))
    put("MSOPT_OLED_PP_CAPTURE_ANGLE_DEG", knobs.get("pp_capture_deg"))
    put("MSOPT_OLED_PP_KEEP_FSP", knobs.get("pp_keep_fsp"), flag)
    put("MSOPT_OLED_PP_PLANAR_REFERENCE", knobs.get("pp_planar_reference"))
    put("MSOPT_OLED_POSTPROCESS_DESIGN", knobs.get("pp_design_file"))


def planar_request():
    """(mode, enabled) for MSOPT_OLED_PP_PLANAR.

    Single source of truth, because this variable is NOT a plain flag: it also
    selects WHICH unpatterned stack to build ("low" = design region filled with
    air, "high" = a flat slab of design material).  env_flag only accepts
    1/true/yes/on, so asking it about "low" silently answers False -- which is
    exactly how a planar request once got dropped and the run reported
    "no design file" instead of characterizing the stack.
    """
    mode = os.environ.get("MSOPT_OLED_PP_PLANAR", "0").strip().lower()
    return mode, mode not in ("", "0", "false", "no", "off")


def planar_requested():
    return planar_request()[1]


def select_stack(stack="microcavity", color="green", kind="optimized",
                 period_legacy=2.5, period_mc=2.0):
    """The single place a run picks its layer stack, shared by every optimizer.

    Besides building the config it PINS THE SIMULATION WAVELENGTH to that stack's
    own design line. A microcavity is tuned to one wavelength (0.46 / 0.53 /
    0.62 um); evaluating it at the 0.55 um default would put the cavity off its
    own resonance, which is exactly the anti-resonant case that cost 64x before.
    Env vars still win, so sweeps need no file edit.
    """
    which = os.environ.get("MSOPT_OLED_STACK", stack).strip().lower()
    if which == "legacy":
        print("[stack] legacy built-in stack @ "
              f"{os.environ.get('MSOPT_OLED_WAVELENGTHS', '0.55')} um")
        return build_config(period_x_default=period_legacy), None
    if which != "microcavity":
        raise ValueError("MSOPT_OLED_STACK must be 'legacy' or 'microcavity'")
    c = os.environ.get("MSOPT_MC_COLOR", color).strip().lower()
    k = os.environ.get("MSOPT_MC_STACK_KIND", kind).strip().lower()
    layers, spec = microcavity_layers(c, k)
    os.environ.setdefault("MSOPT_OLED_WAVELENGTHS", str(spec["wavelength_um"]))
    G = build_config(period_x_default=period_mc, layer_specs=layers,
                     eml_layer_name=MICROCAVITY_EML_LAYER)
    print(f"[stack] microcavity/{k} {c} @ {spec['wavelength_um']:g} um, "
          f"{len(layers)} layers, thinnest "
          f"{min(h for _n, h, _i in layers) * 1000:.0f} nm")
    return G, spec


def microcavity_layers(color="green", kind="optimized"):
    """(layer_specs, spec) for the top-emission microcavity stack.

    Feed layer_specs straight into build_config(layer_specs=...,
    eml_layer_name=MICROCAVITY_EML_LAYER); every z coordinate is then derived
    there. spec carries the wavelength and the thicknesses actually used.
    NOTE the thinnest layers are 10 nm, so a mesh finer than ~5 nm
    (MSOPT_OLED_RESOLUTION >= 200) is required for the cavity to be represented
    at all -- msopt applies ONE uniform mesh to the whole domain.
    """
    kind = str(kind).strip().lower()
    table = {"optimized": MICROCAVITY_OPTIMIZED, "table": MICROCAVITY_TABLE}.get(kind)
    if table is None:
        raise ValueError("kind must be 'optimized' or 'table'")
    color = str(color).strip().lower()
    if color not in table:
        raise ValueError(f"color must be one of {sorted(table)}")
    spec = dict(table[color])
    spec["kind"] = kind
    wl = spec["wavelength_um"]
    um = 1e-3
    fx = MICROCAVITY_FIXED_NM

    def mat(name, n, k=0.0):
        return {"name": f"MC_{name}_sampled", "wavelength": [wl],
                "n": [float(n)], "k": [float(k)]}

    # Optimized stack uses an Ag cathode (Johnson & Christy, traceable); the
    # table stack keeps its Mg:Ag, whose constants are a published-range estimate.
    cath = (("Ag_cathode", spec["ag_n"], spec["ag_k"]) if kind == "optimized"
            else ("MgAg_cathode", spec["cath_n"], spec["cath_k"]))
    g = lambda k, d: float(spec.get(k, d))
    layer_specs = [
        ("Ag_anode_mirror", fx["ag_anode"] * um, mat("Ag", spec["ag_n"], spec["ag_k"])),
        ("ITO_contact",     fx["ito"] * um,      mat("ITO", g("n_ito", 1.82), g("k_ito", 0.0))),
        ("HATCN_HIL",       fx["hil"] * um,      mat("HATCN", g("n_hil", 1.85))),
        ("NPB_HTL",         spec["htl_nm"] * um, mat("NPB_HTL", g("n_htl", 1.80))),
        ("TCBTA_EBL",       fx["ebl"] * um,      mat("TCBTA", g("n_ebl", 1.76))),
        (MICROCAVITY_EML_LAYER, spec["eml_nm"] * um, mat("EML", spec["eml_n"])),
        ("TSPO1_HBL",       fx["hbl"] * um,      mat("TSPO1", g("n_hbl", 1.75))),
        ("TPBi_ETL",        spec["etl_nm"] * um, mat("TPBi", g("n_etl", 1.74))),
        (cath[0],           spec["cath_nm"] * um, mat(cath[0], cath[1], cath[2])),
        ("CPL",             spec["cpl_nm"] * um, mat("CPL", spec["n_cpl"])),
    ]
    return layer_specs, spec


def load_optimization_angular_target(design_path):
    """Load the angular target the design was ACTUALLY optimized against.

    Every optimization run writes A/OLED_angular_target.npz next to its
    lastdesign.txt, holding the FoM's own ring angles, normalized target power
    profile and in-range mask.  Without it the postprocess can only compare
    against whatever generic MSOPT_OLED_TARGET_ANGLES default the driver script
    happens to carry, which is NOT this design's goal -- an apples-to-oranges
    comparison that silently looks like a real target match.

    MSOPT_OLED_PP_TARGET_NPZ overrides the path; "" disables the lookup.
    Returns None when no target is available.
    """
    override = os.environ.get("MSOPT_OLED_PP_TARGET_NPZ")
    if override is not None:
        if not override.strip():
            return None
        candidates = [override.strip()]
    elif design_path:
        base = os.path.dirname(os.path.abspath(design_path))
        candidates = [os.path.join(base, "OLED_angular_target.npz")]
    else:
        return None
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            z = np.load(path)
            thetas = np.asarray(z["angle_thetas"], dtype=float).reshape(-1)
            profile = np.asarray(z["target_profile"], dtype=float).reshape(-1)
            in_range = np.asarray(z["in_range"]).reshape(-1).astype(bool)
        except Exception as exc:
            print(f"[postprocess] warning: could not read optimization target {path}: {exc}")
            continue
        if not (thetas.size == profile.size == in_range.size) or thetas.size == 0:
            print(f"[postprocess] warning: malformed optimization target in {path}")
            continue
        print(f"[postprocess] optimization target loaded from {path}: "
              + ", ".join(f"{t:.3f}deg->{p:.4f}" for t, p in zip(thetas, profile)))
        return {"path": path, "angle_thetas": thetas,
                "target_profile": profile, "in_range": in_range}
    return None


def optimization_target_match(theta2d, ukx, uky, spectrum, target):
    """Re-evaluate the OPTIMIZATION's own FoM from the postprocess far field.

    The FoM bins propagating power into rings at fixed polar angles and scores
    throughput * match, where
        profile_k   = ring power / total propagating power
        throughput  = sum of profile over the in-range rings
        q_k         = profile_k / throughput
        match       = sum_k min(q_k, target_k)      (<= 1)
    Here each propagating direction is assigned to its NEAREST target ring
    (a Voronoi partition in theta), which is total-power preserving; the FoM's
    own exact-bin selection is not reproducible on the far-field projection's
    direction-cosine grid.
    """
    thetas = np.asarray(target["angle_thetas"], dtype=float)
    tgt = np.asarray(target["target_profile"], dtype=float)
    in_range = np.asarray(target["in_range"], dtype=bool)
    prop = np.sqrt(np.asarray(ukx) ** 2 + np.asarray(uky) ** 2) <= 1.0 + 1e-12
    th = np.asarray(theta2d)[prop]
    sp = np.asarray(spectrum)[prop]
    if th.size == 0 or float(np.sum(sp)) <= 0.0:
        return None
    ring_idx = np.argmin(np.abs(th[:, None] - thetas[None, :]), axis=1)
    power = np.zeros(thetas.size)
    np.add.at(power, ring_idx, sp)
    total = float(np.sum(power))
    profile = power / max(total, 1e-30)
    throughput = float(np.sum(profile[in_range]))
    q = profile / max(throughput, 1e-30)
    match = float(np.sum(np.minimum(q[in_range], tgt[in_range])))
    return {
        "angle_thetas": thetas,
        "target_profile": tgt,
        "in_range": in_range,
        "achieved_profile": profile,
        "throughput": throughput,
        "match": match,
        "fom": throughput * match,
        "total_variation": float(0.5 * np.sum(np.abs(profile - tgt))),
    }


def save_phi_distribution_figure(path, th_axis, ph_axis, Sr, title, theta_rings=(15.0, 30.0, 45.0, 60.0)):
    """Azimuthal (phi = 0..360) view of the far field.

    Left  : true circular map -- polar pcolormesh with phi as the angular
            coordinate and theta as the radius, so the emission hemisphere is
            drawn as the disc a detector above the device would see.
    Right : phi distribution curves on a 0..360 polar axis -- the theta-summed
            azimuthal distribution (thick) plus one curve per constant-theta
            ring, each self-normalized, which is what reveals whether the
            design's azimuthal symmetry (C4v for a square lattice) is intact.
    """
    ph_closed = np.concatenate([ph_axis, [ph_axis[0] + 360.0]])
    Sr_closed = np.concatenate([Sr, Sr[:, :1]], axis=1)
    fig = plt.figure(figsize=(11.5, 5.0), constrained_layout=True)

    ax0 = fig.add_subplot(1, 2, 1, projection="polar")
    PH, TH = np.meshgrid(np.deg2rad(ph_closed), th_axis)
    mesh = ax0.pcolormesh(PH, TH, Sr_closed, shading="auto")
    ax0.set_theta_zero_location("E")
    ax0.set_theta_direction(1)
    ax0.set_rlabel_position(112.5)
    ax0.set_rticks([r for r in (15, 30, 45, 60, 75, 90) if r <= th_axis[-1]])
    ax0.set_title("emission disc: radius = theta, angle = phi")
    fig.colorbar(mesh, ax=ax0, fraction=0.046, pad=0.10, label="per-direction intensity")

    ax1 = fig.add_subplot(1, 2, 2, projection="polar")
    total = Sr.sum(axis=0)
    total_closed = np.concatenate([total, total[:1]])
    ax1.plot(np.deg2rad(ph_closed), total_closed / max(float(np.max(total_closed)), 1e-30),
             lw=2.5, color="k", label="all theta")
    for tr in theta_rings:
        if tr > th_axis[-1]:
            continue
        row = Sr[int(np.argmin(np.abs(th_axis - tr)))]
        row_closed = np.concatenate([row, row[:1]])
        peak = float(np.max(row_closed))
        if peak <= 0:
            continue
        ax1.plot(np.deg2rad(ph_closed), row_closed / peak, lw=1.2, alpha=0.85, label=f"theta={tr:g}deg")
    ax1.set_theta_zero_location("E")
    ax1.set_theta_direction(1)
    ax1.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax1.set_title("phi distribution (each curve self-normalized)")
    ax1.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.22, 1.10))

    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def phi_distribution(th_axis, ph_axis, Sr):
    """(phi_deg, theta-summed intensity, anisotropy) for the phi map.

    anisotropy = (max - min) / (max + min) over phi of the theta-summed
    distribution: 0 for a perfectly azimuthally uniform far field.
    """
    total = Sr.sum(axis=0)
    hi, lo = float(np.max(total)), float(np.min(total))
    aniso = (hi - lo) / max(hi + lo, 1e-30)
    return np.asarray(ph_axis, dtype=float), total, float(aniso)


def plane_cut_signed(theta2d, ukx, uky, spectrum, angles, plane="x", half_width_deg=7.5):
    """Signed +/-theta radiance cut through ONE azimuth plane ("x": phi=0/180,
    "y": phi=90/270). Unlike the kx-sign-only signed reduction, this resolves a
    dipole offset along EITHER axis: the +theta side is the phi0 wedge, the
    -theta side the phi0+180 wedge."""
    prop = np.sqrt(ukx ** 2 + uky ** 2) <= 1.0 + 1e-12
    phi = (np.degrees(np.arctan2(uky, ukx)) + 360.0) % 360.0
    p0 = 0.0 if str(plane).lower() == "x" else 90.0
    edges = _theta_bin_edges(angles)

    def wedge_profile(center):
        d = np.abs((phi - center + 180.0) % 360.0 - 180.0)
        m = prop & (d <= half_width_deg)
        acc = np.zeros(angles.size)
        cnt = np.zeros(angles.size)
        ti = np.clip(np.digitize(np.asarray(theta2d)[m], edges) - 1, 0, angles.size - 1)
        radiance_samples = np.asarray(spectrum) * np.cos(np.deg2rad(np.asarray(theta2d)))
        np.add.at(acc, ti, radiance_samples[m])
        np.add.at(cnt, ti, 1.0)
        return acc / np.maximum(cnt, 1.0)

    plus, minus = wedge_profile(p0), wedge_profile(p0 + 180.0)
    signed = np.concatenate([-angles[::-1], angles[1:]])
    prof = np.concatenate([minus[::-1], plus[1:]])
    return signed, prof


CUMULATIVE_CONE_ANGLES_DEG = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)


def save_cumulative_extraction_figure(path, angles_deg, shares, lee, title, planar=None):
    """Cumulative extraction vs polar half-angle.

    Two curves because they answer different questions:
      * ABSOLUTE (lee * share): the fraction of the dipole's emitted power that
        leaves within theta -- directly comparable to the analytic layered-media
        cumulative, and the number that says how much light a detector with that
        acceptance angle collects.
      * SHARE: the same curve normalized to the extracted light only, i.e. how
        the OUTCOUPLED beam is distributed in angle.
    A finite FDTD domain only captures up to its own geometric acceptance angle,
    so the absolute curve saturating (or not) is also the truncation diagnostic.
    """
    angles = np.asarray(angles_deg, dtype=float)
    shares = np.asarray(shares, dtype=float)
    absolute = shares * float(lee)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(angles, absolute * 100.0, "o-", lw=2, color="tab:blue", label="this run")
    if planar:
        p_ang, p_abs = planar
        ax.plot(np.asarray(p_ang, dtype=float), np.asarray(p_abs, dtype=float) * 100.0,
                "s--", lw=1.6, color="0.45", label="planar reference")
    for a, v in zip(angles, absolute):
        ax.annotate(f"{v * 100:.2f}", (a, v * 100), fontsize=7,
                    textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_xlabel("polar half-angle theta [deg]")
    ax.set_ylabel("power emitted within theta [% of emitted]")
    ax.set_title("Cumulative extraction (absolute)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(angles, shares * 100.0, "o-", lw=2, color="tab:green")
    ax.set_xlabel("polar half-angle theta [deg]")
    ax.set_ylabel("share of EXTRACTED light within theta [%]")
    ax.set_ylim(0, 105)
    ax.set_title("Cumulative share of the outcoupled beam")
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cone_shares(theta2d, ukx, uky, spectrum, cones=(5.0, 10.0, 20.0, 30.0)):
    """Fraction of total propagating power inside theta <= cone, for a fair
    'near-normal' metric: the exact-0-deg delta-order share is structurally ~1%
    for ANY smooth incoherent pattern (sin(theta) solid-angle weight), so cone
    integrals are the honest way to quantify near-normal performance."""
    prop = np.sqrt(ukx ** 2 + uky ** 2) <= 1.0 + 1e-12
    th = np.asarray(theta2d)[prop]
    sp = np.asarray(spectrum)[prop]
    total = float(np.sum(sp))
    total = total if total > 0 else 1e-30
    return [(float(c), float(np.sum(sp[th <= c])) / total) for c in cones]


def pp_case_path(cache_dir, pol, idx):
    return os.path.join(cache_dir, f"case_{pol}_{int(idx):03d}.npz")


def save_pp_case(cache_dir, pol, idx, key, theta, spectrum, ukx, uky, T, src, dip, box):
    """Persist one finished postprocess case so a re-run need not repeat it."""
    if not cache_dir:
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # np.savez_compressed appends .npz unless the name already ends in it, so the
        # temp name has to carry the suffix or os.replace below chases a missing file.
        tmp = pp_case_path(cache_dir, pol, idx) + ".tmp.npz"
        np.savez_compressed(
            tmp, key=np.array(key), theta=theta, spectrum=spectrum, ukx=ukx, uky=uky,
            T=float(T),
            src=np.nan if src is None else float(src),
            dip=np.nan if dip is None else float(dip),
            box=np.nan if box is None else float(box),
        )
        os.replace(tmp, pp_case_path(cache_dir, pol, idx))   # atomic: no half files
    except Exception as exc:
        print(f"[postprocess] cache write failed for pol={pol} dipole {idx}: {exc}")


def load_pp_case(cache_dir, pol, idx, key):
    """Return a cached case, or None. `key` pins the cache to the design and the
    angular settings it was produced with, so a different design silently reusing
    someone else's spectra is impossible."""
    if not cache_dir:
        return None
    f = pp_case_path(cache_dir, pol, idx)
    if not os.path.isfile(f):
        return None
    try:
        with np.load(f, allow_pickle=False) as d:
            if str(d["key"]) != str(key):
                return None
            unpack = lambda v: (None if not np.isfinite(v) else float(v))
            return (d["theta"], d["spectrum"], d["ukx"], d["uky"], float(d["T"]),
                    unpack(d["src"]), unpack(d["dip"]), unpack(d["box"]))
    except Exception as exc:
        print(f"[postprocess] cache read failed for {f}: {exc}")
        return None


def _postprocess_coherence_audit(
        *, G,
        sim,
        manifest,
        records,
        pols,
        n_dipoles,
        pp_grid_n,
        z_shift,
        source_flux_box_faces,
        coh_cases,
        rand_trial_profiles,
        _case_spectrum,
        _drop_last_fsp,
        _write_manifest):
    """The optional step2b coherence audit, lifted out of run_postprocess.

    Gated by MSOPT_OLED_PP_COHERENT_CHECK, which defaults OFF, so this is 167
    lines the main postprocess path never executes -- the clearest seam in a
    function that had grown to 1375 lines.

    Everything it touches is passed in by keyword under its ORIGINAL name, so
    the body below is byte-identical to what lived inline. That is deliberate:
    the point of the move is to shorten run_postprocess, and it must not be
    able to change a published number. coh_cases and rand_trial_profiles are
    appended to IN PLACE, so the caller sees the results; coherence_summary is
    written into manifest here and never read outside.
    """
    # The coherence cases fire EVERY grid point at once, so they get the full
    # layout regardless of the fold -- there is no per-source result to
    # reconstruct from a mirror image here, only one simultaneous run.
    pts = central_cell_dipoles(G, n_dipoles, pols[0], grid_n=pp_grid_n)
    sim.fdtd.switchtolayout()
    delete_object(sim.fdtd, "postprocess_dipole")
    for k, (cx, cy, cz, cpol) in enumerate(pts):
        add_dipole(G, sim, cx, cy, cz + z_shift, cpol, f"coherent_dipole_{k:03d}")
    reference_records = [rec for rec in records if rec[4] == pols[0]]
    reference_power = float(np.sum([rec[9] for rec in reference_records]))
    reference_top = float(np.sum([rec[10] for rec in reference_records]))
    reference_eta = reference_top / max(reference_power, G.channel_power_floor)

    def _run_coherent_case(tag, phases_deg):
        sim.fdtd.switchtolayout()
        for source_idx, phase_deg in enumerate(phases_deg):
            sim.fdtd.setnamed(
                f"coherent_dipole_{source_idx:03d}",
                "phase",
                float(phase_deg),
            )
        sim.run(name=f"postprocess_{tag}", save=True)
        load_run_results(sim)
        T_c = read_transmission(sim.fdtd, G.target_monitor_name)
        freqs_c = source_freqs(G, sim)
        src_c = valid_power(read_source_power(sim.fdtd, freqs_c))
        box_c = read_flux_box_power(sim.fdtd, source_flux_box_faces, src_c)
        if src_c is None or box_c is None:
            raise RuntimeError("coherence case has no valid source/source-box power")
        th_c, raw_c, ux_c, uy_c = _case_spectrum()
        top_c = max(float(T_c), 0.0) * src_c
        raw_sum_c = float(np.sum(np.maximum(raw_c, 0.0)))
        if not np.isfinite(raw_sum_c) or raw_sum_c <= 0.0:
            raise RuntimeError("coherence case has zero/non-finite far-field power")
        spectrum_c = np.maximum(raw_c, 0.0) * top_c / raw_sum_c
        eta_c = top_c / max(box_c, G.channel_power_floor)
        _drop_last_fsp()
        return {
            "tag": tag,
            "P_norm": float(box_c),
            "P_up": float(top_c),
            "eta_ff": float(eta_c),
            "delta_eta_vs_incoherent": float(eta_c - reference_eta),
            "rel_eta_error_vs_incoherent": float(
                abs(eta_c - reference_eta)
                / max(abs(reference_eta), G.channel_power_floor)
            ),
            "theta": th_c,
            "spectrum": spectrum_c,
            "ukx": ux_c,
            "uky": uy_c,
        }

    coherence_summary = {
        "enabled": True,
        "source_layout": f"{pp_grid_n}x{pp_grid_n} "
                         + os.environ.get("MSOPT_OLED_PP_SOURCE_LAYOUT",
                                          "cell_center").strip().lower()
                         + " grid",
        "polarization": pols[0],
        "case1_incoherent": {
            "P_norm_sum": reference_power,
            "P_up_sum": reference_top,
            "eta_ff": reference_eta,
            "n_runs": len(reference_records),
        },
        "case3_checkpoints": [],
        "failures": [],
    }

    # Case 2a
    try:
        print(f"[postprocess] coherence case2a: {len(pts)} simultaneous dipoles, phase=0")
        case2a = _run_coherent_case("case2a_same_phase", np.zeros(len(pts)))
        coh_cases.append((
            "case2a_same_phase",
            case2a["theta"], case2a["spectrum"], case2a["ukx"], case2a["uky"],
        ))
        coherence_summary["case2a_same_phase"] = {
            key: value for key, value in case2a.items()
            if key not in ("theta", "spectrum", "ukx", "uky")
        }
    except Exception as exc:
        coherence_summary["failures"].append({"case": "case2a", "error": str(exc)})
        print(f"[postprocess] warning: coherence case2a failed: {exc}")

    # Case 2b
    try:
        case2b_rng = np.random.default_rng(0)
        case2b_phases = case2b_rng.uniform(0.0, 360.0, size=len(pts))
        print("[postprocess] coherence case2b: one random-phase realization, seed=0")
        case2b = _run_coherent_case("case2b_random_single", case2b_phases)
        coh_cases.append((
            "case2b_random_single",
            case2b["theta"], case2b["spectrum"], case2b["ukx"], case2b["uky"],
        ))
        coherence_summary["case2b_random_single"] = {
            key: value for key, value in case2b.items()
            if key not in ("theta", "spectrum", "ukx", "uky")
        }
    except Exception as exc:
        coherence_summary["failures"].append({"case": "case2b", "error": str(exc)})
        print(f"[postprocess] warning: coherence case2b failed: {exc}")

    # Case 3
    n_rand_trials = env_int("MSOPT_OLED_PP_RANDOM_PHASE_TRIALS", 20)
    rng_ph = np.random.default_rng(env_int("MSOPT_OLED_PP_RANDOM_PHASE_SEED", 1234))
    all_ph = rng_ph.uniform(0.0, 360.0, size=(n_rand_trials, len(pts)))
    np.savetxt(os.path.join(G.design_dir, "OLED_postprocess_randphase_phases_deg.txt"), all_ph)
    sp_cum, norm_cum, top_cum, n_ok = None, 0.0, 0.0, 0
    for t_no in range(1, n_rand_trials + 1):
        try:
            print(f"[postprocess] coherence case3 trial {t_no}/{n_rand_trials}")
            case3_trial = _run_coherent_case(
                f"case3_trial_{t_no:03d}",
                all_ph[t_no - 1],
            )
        except Exception as exc:
            coherence_summary["failures"].append({
                "case": f"case3_trial_{t_no:03d}",
                "error": str(exc),
            })
            print(f"[postprocess] warning: coherence case3 trial {t_no} failed: {exc}")
            continue
        n_ok += 1
        norm_cum += case3_trial["P_norm"]
        top_cum += case3_trial["P_up"]
        rand_trial_profiles.append((
            t_no,
            case3_trial["theta"],
            case3_trial["spectrum"],
        ))
        sp_cum = (
            case3_trial["spectrum"].copy()
            if sp_cum is None
            else sp_cum + case3_trial["spectrum"]
        )
        if n_ok in (1, 2, 5, 10, 20) or t_no == n_rand_trials:
            eta_avg = top_cum / max(norm_cum, G.channel_power_floor)
            coherence_summary["case3_checkpoints"].append({
                "n_trials": n_ok,
                "P_norm_avg": norm_cum / n_ok,
                "P_up_avg": top_cum / n_ok,
                "eta_ff": eta_avg,
                "delta_eta_vs_incoherent": eta_avg - reference_eta,
                "rel_eta_error_vs_incoherent": (
                    abs(eta_avg - reference_eta)
                    / max(abs(reference_eta), G.channel_power_floor)
                ),
            })
            coh_cases.append((
                f"case3_avg{n_ok:03d}",
                case3_trial["theta"],
                sp_cum / n_ok,
                case3_trial["ukx"],
                case3_trial["uky"],
            ))
    manifest["coherence_validation"] = coherence_summary
    with open(
        os.path.join(G.design_dir, "OLED_postprocess_coherence_validation.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(coherence_summary, fp, indent=2, sort_keys=True)
        fp.write("\n")
    _write_manifest()


def _postprocess_ensemble_products(
        *, G,
        angles,
        angular_projection,
        complete,
        design_hash,
        emitted_power_sum,
        manifest,
        mean_lee,
        performance_spec,
        planar_baseline,
        pp_mode,
        spectrum_sum,
        theta,
        ukx,
        uky,
        use_source_flux_box,
        _write_manifest):
    """The step2b ensemble products: theta-phi map, cone shares, cumulative
    extraction, channel ratios, and the metrics they feed into the manifest.

    Lifted out of run_postprocess to shorten it. The body is byte-identical and
    every name it uses is passed in under its original name.

    RETURNS the cumulative-extraction triple, which the caller needs for the
    emission figure. Inline, that relied on `if "cum_ang" in dir()` -- a test
    for whether the try block got far enough to bind the name. The idiom cannot
    survive the move, because the names would no longer be in the scope of the
    caller, and it degrades SILENTLY: render_emission_figure draws 2 panels
    instead of 3 when cumulative is None, so the panel would vanish rather than
    raise. Returning the triple, pre-seeded to None, keeps the behaviour and
    makes the dependency visible.
    """
    cum_ang = cum_share = cum_abs = None
    try:
        th_ax, ph_ax, Sr_map = theta_phi_map(theta, ukx, uky, spectrum_sum)
        save_radiation_map_figure(
            os.path.join(G.design_dir, "OLED_postprocess_thetaphi_map.png"),
            th_ax, ph_ax, Sr_map, "incoherent sum: per-direction intensity",
        )
        np.savez(
            os.path.join(G.design_dir, "OLED_postprocess_thetaphi_map.npz"),
            theta_deg=th_ax, phi_deg=ph_ax, Sr=Sr_map,
        )
        # Azimuthal view: circular (phi 0..360) emission disc + phi distribution
        # curves. A square lattice on a radial design should be C4v-symmetric, so
        # the reported anisotropy is a direct symmetry-breaking check.
        save_phi_distribution_figure(
            os.path.join(G.design_dir, "OLED_postprocess_phi_map.png"),
            th_ax, ph_ax, Sr_map, "incoherent sum: azimuthal distribution",
        )
        ph_deg, ph_total, ph_aniso = phi_distribution(th_ax, ph_ax, Sr_map)
        np.savetxt(
            os.path.join(G.design_dir, "OLED_postprocess_phi_distribution.txt"),
            np.column_stack([ph_deg, ph_total]),
            header=f"phi_deg theta_summed_intensity  (anisotropy=(max-min)/(max+min)={ph_aniso:.6f})",
        )
        print(f"[postprocess] phi anisotropy (0=azimuthally uniform): {ph_aniso:.4f}")
        signed_axis, cut_x = plane_cut_signed(theta, ukx, uky, spectrum_sum, angles, "x")
        _, cut_y = plane_cut_signed(theta, ukx, uky, spectrum_sum, angles, "y")
        np.savetxt(
            os.path.join(G.design_dir, "OLED_postprocess_plane_cuts.txt"),
            np.column_stack([signed_axis, cut_x, cut_y]),
            header="theta_signed_deg I_phi0_plane I_phi90_plane",
        )
        cs = cone_shares(theta, ukx, uky, spectrum_sum)
        with open(os.path.join(G.design_dir, "OLED_postprocess_cone_shares.txt"), "w", encoding="utf-8") as fp:
            fp.write("cone_half_angle_deg power_share\n")
            for c, s in cs:
                fp.write(f"{c:.1f} {s:.6e}\n")
        print("[postprocess] cone shares: " + ", ".join(f"<={c:g}deg: {s * 100:.2f}%" for c, s in cs))

        # Cumulative extraction from <=10 deg to <=80 deg, absolute (fraction of
        # emitted power) as well as share of the outcoupled beam. The absolute
        # curve is what the analytic layered-media solver reports, so the two
        # methods can be compared angle by angle instead of on one truncated
        # number.
        cum = cone_shares(theta, ukx, uky, spectrum_sum, cones=CUMULATIVE_CONE_ANGLES_DEG)
        cum_ang = [c for c, _s in cum]
        cum_share = [s for _c, s in cum]
        cum_abs = [s * mean_lee for s in cum_share]
        np.savetxt(
            os.path.join(G.design_dir, "OLED_postprocess_cumulative_extraction.txt"),
            np.column_stack([cum_ang, cum_abs, cum_share]),
            header=("theta_deg cumulative_extraction_fraction_of_emitted "
                    "cumulative_share_of_extracted"),
        )
        # Overlay the planar (no-design) curve when a reference run is given.
        # Resolved from the env var directly: the planar LEE read-out further
        # down runs later, and this plot should not depend on that ordering.
        planar_curve = None
        ref_root = os.environ.get("MSOPT_OLED_PP_PLANAR_REFERENCE", "").strip()
        if ref_root and not planar_baseline:
            ref_root = os.path.abspath(ref_root)
            a_dir = (ref_root if os.path.basename(ref_root) == "A"
                     else os.path.join(ref_root, "A")) if os.path.isdir(ref_root) \
                else os.path.dirname(ref_root)
            ref = os.path.join(a_dir, "OLED_postprocess_cumulative_extraction.txt")
            if os.path.exists(ref):
                try:
                    arr = np.loadtxt(ref)
                    planar_curve = (arr[:, 0], arr[:, 1])
                except Exception as exc:
                    print(f"[postprocess] warning: could not read planar cumulative curve: {exc}")
        save_cumulative_extraction_figure(
            os.path.join(G.design_dir, "OLED_postprocess_cumulative_extraction.png"),
            cum_ang, cum_share, mean_lee,
            f"Cumulative extraction vs acceptance angle (LEE {mean_lee * 100:.2f} %)",
            planar=planar_curve,
        )
        print("[postprocess] cumulative extraction (% of emitted): "
              + ", ".join(f"<={a:g}deg {v * 100:.3f}%" for a, v in zip(cum_ang, cum_abs)))
        manifest["cumulative_extraction"] = {
            "theta_deg": [float(a) for a in cum_ang],
            "fraction_of_emitted": [float(v) for v in cum_abs],
            "share_of_extracted": [float(v) for v in cum_share],
        }

        # Optimization <-> PP consistency readout: radiance ratios to the normal
        # direction at the FoM's target angles, in the same language as the
        # reciprocal ramp targets (ratio-to-0deg windows). If the in-opt FoM is
        # honest, these achieved ratios should sit inside the ramp windows.
        ch_angles = np.asarray(performance_spec["angles_deg"], dtype=float)
        radiance_half_width = env_float("MSOPT_OLED_PP_RADIANCE_HALF_WIDTH_DEG", 2.0)
        positive_theta_samples = np.asarray(theta)[np.asarray(theta) > 1e-9]
        normal_half_width = (
            max(radiance_half_width, 1.01 * float(np.min(positive_theta_samples)))
            if positive_theta_samples.size
            else radiance_half_width
        )
        channel_radiance = np.asarray([
            angular_radiance_value(
                theta,
                spectrum_sum,
                a,
                normal_half_width if abs(float(a)) <= 1e-12 else radiance_half_width,
            )
            for a in ch_angles
        ])
        normalized_channel_radiance = channel_radiance / max(emitted_power_sum, G.channel_power_floor)
        metric = oled_performance_metrics(
            normalized_channel_radiance,
            performance_spec,
            power_floor=G.channel_power_floor,
            violation_scale=performance_spec["tolerance"],
            throughput_reference=env_float("MSOPT_OLED_PP_NORMAL_RADIANCE_REFERENCE", 1.0),
            throughput_weight=env_float("MSOPT_OLED_THROUGHPUT_WEIGHT", 0.10),
        )
        with open(os.path.join(G.design_dir, "OLED_postprocess_channel_ratios.txt"), "w", encoding="utf-8") as fp:
            fp.write(f"# requested radiance annulus half-width: {radiance_half_width:.3f} deg\n")
            fp.write(f"# effective normal-radiance cone half-width: {normal_half_width:.3f} deg\n")
            fp.write(f"# angular_shape_score: {metric['shape_score']:.8e}\n")
            fp.write(f"# all_ratio_windows_met: {int(metric['all_ratio_windows_met'])}\n")
            fp.write("theta_deg radiance_per_emitted_power ratio_to_0 target_ratio ratio_min ratio_max violation\n")
            for idx, a in enumerate(ch_angles):
                fp.write(
                    f"{a:.3f} {normalized_channel_radiance[idx]:.6e} "
                    f"{metric['ratios_to_zero'][idx]:.6e} "
                    f"{performance_spec['target_ratios'][idx]:.6e} "
                    f"{performance_spec['ratio_min'][idx]:.6e} "
                    f"{performance_spec['ratio_max'][idx]:.6e} "
                    f"{metric['violations'][idx]:.6e}\n"
                )
        ratio_msg = ", ".join(
            f"{a:g}deg={ratio:.3f}" for a, ratio in zip(ch_angles, metric["ratios_to_zero"])
        )
        throughput_weight = env_float("MSOPT_OLED_THROUGHPUT_WEIGHT", 0.10)
        actual_performance_score = metric["shape_score"] * (
            (1.0 - throughput_weight) + throughput_weight * float(np.clip(mean_lee, 0.0, 1.0))
        )
        manifest["angular_metrics"] = {
            "angles_deg": [float(v) for v in ch_angles],
            "radiance_per_emitted_power": [float(v) for v in normalized_channel_radiance],
            "ratios_to_zero": [float(v) for v in metric["ratios_to_zero"]],
            "violations": [float(v) for v in metric["violations"]],
            "shape_score": float(metric["shape_score"]),
            "all_ratio_windows_met": bool(metric["all_ratio_windows_met"]),
        }
        manifest["ensemble_lee"] = float(mean_lee)
        manifest["actual_performance_score"] = float(actual_performance_score)
        # Filled in below once the design's own optimization target is loaded;
        # None means the comparison could not be made (no target file), which is
        # NOT the same as "the design missed its target".
        manifest["optimization_target_match"] = None
        manifest["authoritative"] = bool(
            complete
            and not planar_baseline          # a planar run characterizes no design
            and pp_mode == "supercell"
            and manifest["convergence_confirmed"]
            and use_source_flux_box
            and angular_projection == "farfield3d"
            and manifest["source_layout"] == "validated_endpoint"
            and manifest["dipole_grid"] >= 6
        )
        optimization_report_path = os.path.join(
            G.design_dir,
            "OLED_optimization_latest_metrics.json",
        )
        comparison = {"status": "optimization_metrics_not_available"}
        if os.path.exists(optimization_report_path):
            with open(optimization_report_path, encoding="utf-8") as fp:
                optimization_report = json.load(fp)
            same_design = optimization_report.get("design_sha256") == design_hash
            opt_ratios = np.asarray(optimization_report.get("ratios_to_zero", []), dtype=float)
            pp_ratios = np.asarray(metric["ratios_to_zero"], dtype=float)
            comparable = same_design and opt_ratios.shape == pp_ratios.shape
            ratio_rmse = (
                float(np.sqrt(np.mean((opt_ratios - pp_ratios) ** 2)))
                if comparable
                else None
            )
            shape_delta = (
                float(metric["shape_score"] - float(optimization_report["shape_score"]))
                if comparable and "shape_score" in optimization_report
                else None
            )
            agreement_tol = env_float("MSOPT_OLED_OPT_PP_AGREEMENT_TOL", 0.10)
            comparison = {
                "status": "compared" if comparable else "design_or_channel_mismatch",
                "same_design": bool(same_design),
                "optimization_score": optimization_report.get("optimization_score"),
                "optimization_shape_score": optimization_report.get("shape_score"),
                "postprocess_actual_performance_score": float(actual_performance_score),
                "postprocess_shape_score": float(metric["shape_score"]),
                "ratio_rmse": ratio_rmse,
                "shape_score_delta": shape_delta,
                "agreement_tolerance": agreement_tol,
                "agreement_pass": bool(
                    comparable
                    and ratio_rmse is not None
                    and shape_delta is not None
                    and np.isfinite(ratio_rmse)
                    and np.isfinite(shape_delta)
                    and ratio_rmse <= agreement_tol
                    and abs(shape_delta) <= agreement_tol
                ),
            }
        manifest["optimization_postprocess_comparison"] = comparison
        with open(
            os.path.join(G.design_dir, "OLED_optimization_postprocess_comparison.json"),
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(comparison, fp, indent=2, sort_keys=True)
            fp.write("\n")
        _write_manifest()
        print(
            f"[postprocess] radiance ratios to 0deg (shared FoM language): {ratio_msg}; "
            f"shape_score={metric['shape_score']:.6f}, "
            f"actual_performance_score={actual_performance_score:.6f}"
        )
        if comparison["status"] != "optimization_metrics_not_available":
            print(
                "[postprocess] optimization agreement: "
                f"status={comparison['status']}, "
                f"ratio_rmse={comparison['ratio_rmse']}, "
                f"shape_delta={comparison['shape_score_delta']}, "
                f"pass={comparison['agreement_pass']}"
            )
    except Exception as exc:
        manifest["status"] = "metrics_failed"
        manifest["authoritative"] = False
        manifest["metrics_error"] = str(exc)
        _write_manifest()
        if env_flag("MSOPT_OLED_PP_REQUIRE_METRICS", "1"):
            raise RuntimeError(f"OLED postprocess angular metrics failed: {exc}") from exc
        print(f"[postprocess] warning: step2b-style products failed: {exc}")
    return cum_ang, cum_share, cum_abs


def run_postprocess(G, final_design, mapping=None, performance_spec=None):
    pp_t0 = time.time()
    # MSOPT_OLED_PP_PLANAR discards the design and characterizes an UNPATTERNED
    # stack through the exact same pipeline, so its outputs are directly
    # comparable folder-to-folder and give the reference every enhancement
    # factor needs.  Two readings of "no design structure", both supported:
    #   low  (= "1", default) : design region = low index (air) -- nothing at
    #                           all on top of the stack
    #   high                  : design region = high index -- a flat slab of the
    #                           design material (isolates PATTERNING from the
    #                           mere presence of the layer)
    planar_mode, planar_baseline = planar_request()
    if planar_baseline:
        if planar_mode not in ("1", "true", "yes", "on", "low", "high"):
            raise ValueError("MSOPT_OLED_PP_PLANAR must be 0, 1/low, or high.")
        fill = 1.0 if planar_mode == "high" else 0.0
        final_design = np.full(int(np.prod(G.design_grids)), fill, dtype=float)
        mapping = None
        which = "high index (flat slab of design material)" if fill else "low index (bare stack, nothing on top)"
        print(f"[postprocess] PLANAR BASELINE: design discarded, design region = {which}")
    # Case cache. Defaults to this run's own folder; MSOPT_OLED_PP_CACHE_DIR can point
    # at a previous run's cache to finish a postprocess that died part-way.
    pp_cache_dir = os.environ.get(
        "MSOPT_OLED_PP_CACHE_DIR", os.path.join(G.design_dir, "pp_cache")).strip()
    rho = design_to_grid(G, final_design, mapping)
    if np.asarray(G.visible_wavelengths).size != 1:
        raise ValueError(
            "OLED postprocess currently requires one wavelength; transmission, "
            "dipole power, and angular spectra must be accumulated per wavelength."
        )

    # Record WHICH structure these results belong to, before any simulation runs:
    # the design usually comes from another run's lastdesign.txt, so the result
    # folder would otherwise not contain the geometry it characterizes.
    try:
        save_final_structure(G, final_design, mapping)
    except Exception as exc:
        print(f"[postprocess] warning: could not save final structure: {exc}")

    # --- Far-field geometry ----------------------------------------------------
    # Two modes (MSOPT_OLED_PP_MODE):
    #   single (alias "n2f"): SINGLE cell + PML; the angular
    #        spectrum comes from the near-to-far projection of the top monitor,
    #        so no tiling and no tall air gap are needed. Caveat: the periodic
    #        array is truncated at the cell edge -- guided light reaching the
    #        PML is absorbed instead of scattering at neighbour cells, so LEE
    #        and order sharpness are approximations of the tiled reference.
    #   supercell (alias "tile", the default): NxN tiling with the monitor
    #        geometrically sized to capture pp_max_angle_deg from the central
    #        cell (exact but expensive).
    pp_mode = os.environ.get("MSOPT_OLED_PP_MODE", "supercell").strip().lower()
    if pp_mode not in ("single", "n2f", "supercell", "tile"):
        raise ValueError("MSOPT_OLED_PP_MODE must be 'single' (alias 'n2f') or 'supercell' (alias 'tile').")
    pp_mode = {"n2f": "single", "tile": "supercell"}.get(pp_mode, pp_mode)
    # Match the validated scripts: monitors are separated from the PML by the
    # PML thickness plus a two-pixel gap, rather than sitting 0.15 um from the
    # outer boundary where they may overlap the absorbing layer.
    monitor_boundary_inset = env_float(
        "MSOPT_OLED_PP_MONITOR_BOUNDARY_INSET_UM",
        0.50 + 2.0 / max(float(G.pp_resolution), 1.0),
    )
    # Lateral size is set by the CAPTURE ANGLE, in both modes.
    #
    # near2far projects the field ON the monitor, so light that leaves through
    # the lateral PML before reaching the monitor plane is simply absent from
    # its input -- the projection cannot recover it. Measured directly: the same
    # stack and the same farfield3d gave LEE 47.1 % at 2.72 um lateral and
    # 58.3 % at 5.65 um. So the domain must be wide enough that
    #     monitor half-width >= h * tan(capture) ,
    # where h is measured from the LAST SOLID INTERFACE the light leaves -- the
    # top of the stack, not the top of the design region. Using the design top
    # under-counts h by the design thickness and silently narrows the capture
    # (60 deg instead of the intended 69 deg in one run).
    capture_deg = env_float("MSOPT_OLED_PP_CAPTURE_ANGLE_DEG", G.pp_max_angle_deg)
    stack_top = max(l["center"][2] + 0.5 * l["size"][2] for l in G.stack_layers)

    def _needed_half_width(monitor_z_, z_shift_):
        h = max(monitor_z_ - (stack_top + z_shift_), 1e-6)
        return h * float(np.tan(np.deg2rad(capture_deg))) + monitor_boundary_inset, h

    if pp_mode == "single":
        # n2f is a projection, so the monitor only has to clear the evanescent
        # near field; keeping it LOW also keeps the escape cone narrow, which is
        # what makes a modest domain sufficient.
        pp_far = env_float("MSOPT_OLED_PP_N2F_FAR_Z_UM", 0.4)
        post_sz = G.Sz + pp_far
        z_shift = -0.5 * pp_far
        monitor_z = 0.5 * post_sz - monitor_boundary_inset
        tile_n = 1
        half_needed, monitor_h = _needed_half_width(monitor_z, z_shift)
        pad_env = os.environ.get("MSOPT_OLED_PP_N2F_PAD_UM")
        pp_pad = (float(pad_env) if pad_env not in (None, "")
                  else max(half_needed - 0.5 * G.Sx, 0.1))
        post_sx, post_sy = G.Sx + 2.0 * pp_pad, G.Sy + 2.0 * pp_pad
    else:
        post_sz = G.Sz + G.pp_far_z_um
        z_shift = -0.5 * G.pp_far_z_um              # keep the stack at the same height above the bottom
        monitor_z = 0.5 * post_sz - monitor_boundary_inset
        half_needed, monitor_h = _needed_half_width(monitor_z, z_shift)
        tile_n = max(int(G.pp_min_tiles), int(np.ceil(2.0 * half_needed / G.Sx)))
        if tile_n % 2 == 0:
            tile_n += 1                              # odd -> there is a central cell
        post_sx, post_sy = tile_n * G.Sx, tile_n * G.Sy

    achieved_deg = float(np.degrees(np.arctan(
        max(0.5 * post_sx - monitor_boundary_inset, 0.0) / max(monitor_h, 1e-9))))
    print(f"[postprocess] capture: target {capture_deg:g} deg, monitor {monitor_h:.3f} um "
          f"({monitor_h / float(np.mean(G.visible_wavelengths)):.2f} lambda) above the stack top, "
          f"lateral {post_sx:g} um -> ACHIEVED {achieved_deg:.1f} deg")
    if achieved_deg < capture_deg - 1.0:
        print(f"[postprocess] WARNING: the domain only captures {achieved_deg:.1f} deg; "
              "emission beyond that is absorbed by the lateral PML before it reaches "
              "the monitor and is MISSING from the near2far input (LEE is a lower bound).")
    post_monitor_s = [
        post_sx - 2.0 * monitor_boundary_inset,
        post_sy - 2.0 * monitor_boundary_inset,
        0.0,
    ]
    if min(post_monitor_s[:2]) <= 0.0:
        raise ValueError(
            "Postprocess monitor span is non-positive; increase domain padding or "
            "reduce MSOPT_OLED_PP_MONITOR_BOUNDARY_INSET_UM."
        )
    post_monitor_c = [0.0, 0.0, monitor_z]

    cells = (post_sx * G.pp_resolution) * (post_sy * G.pp_resolution) * (post_sz * G.pp_resolution)
    print(
        f"[postprocess] mode={pp_mode}: {tile_n}x{tile_n} cell(s), domain "
        f"{post_sx:g}x{post_sy:g}x{post_sz:g}um, monitor z={monitor_z:.2f}um, "
        f"res={G.pp_resolution}, ~{cells/1e6:.0f}M cells"
    )

    sim = make_sim(G, [post_sx, post_sy, post_sz], bc_x="PML", bc_y="PML", res=G.pp_resolution)
    add_stack(G, sim, span_x=post_sx, span_y=post_sy, z_offset=z_shift)
    half = tile_n // 2
    for ix in range(-half, half + 1):
        for iy in range(-half, half + 1):
            sim.add_design_grid(
                f"design_{ix}_{iy}",
                [ix * G.Sx, iy * G.Sy, G.design_c[2] + z_shift],
                G.design_s,
                G.design_high_index,
                G.design_low_index,
                G.design_grids,
                rho,
                float(np.mean(G.visible_wavelengths)),
            )
    sim.add_monitor(G.target_monitor_name, post_monitor_c, post_monitor_s)

    # Validated step1/step2 normalization: total emitted power is the outward
    # Poynting flux through a six-face box just inside the EML. dipolepower is
    # retained as an independent cross-check, not silently treated as truth.
    use_source_flux_box = env_flag("MSOPT_OLED_PP_SOURCE_FLUX_BOX", "1")
    source_flux_box_faces = []
    source_box_size = None
    source_box_inset = None
    if use_source_flux_box:
        source_box_inset = (
            # Meep tolerates a point source exactly on a flux-region face, but
            # Lumerical does not: edge/corner sources then lose roughly 1/2 or
            # 3/4 of their box-normalization power.  Keep the validated source
            # grid at a two-pixel inset and place the box one pixel from the EML
            # boundary so every source is strictly enclosed.
            env_float("MSOPT_OLED_PP_SOURCE_BOX_INSET_PIXELS", 1.0)
            / max(float(G.pp_resolution), 1.0)
        )
        source_box_size = [
            G.active_x - 2.0 * source_box_inset,
            G.active_y - 2.0 * source_box_inset,
            G.eml_h - 2.0 * source_box_inset,
        ]
        source_flux_box_faces = add_flux_box_monitors(
            sim,
            "pp_source_box",
            [0.0, 0.0, G.eml_c[2] + z_shift],
            source_box_size,
        )

    # Full-domain XZ field monitor so every dipole's emission can be saved as an |E|
    # cross-section image (dipole_emission_N.png). The monitor plane is re-centered on
    # each dipole's y so the slice passes through that dipole.
    pp_field_images = env_flag("MSOPT_OLED_PP_FIELD_IMAGES", "1")
    # Per-dipole Sr(theta,phi) radiation maps (Meep step2b style). Cheap (reuses the
    # already-extracted spectrum), so on by default.
    pp_radiation_images = env_flag("MSOPT_OLED_PP_RADIATION_IMAGES", "1")
    pp_xz_monitor_name = "pp_xz_field"
    if pp_field_images:
        sim.add_monitor(pp_xz_monitor_name, [0.0, 0.0, 0.0], [post_sx, 0.0, post_sz])

    angles = np.linspace(0.0, 90.0, env_int("MSOPT_OLED_POSTPROCESS_ANGLE_RES", 181))
    signed_angles = signed_angle_axis(angles)
    n_dipoles = env_int("MSOPT_OLED_POSTPROCESS_N_DIPOLES", 20)
    pols = postprocess_polarizations()
    if performance_spec is None:
        target_pairs = sorted({0.0: 1.0, **{float(a): float(r) for a, r in G.target_angle_pairs}}.items())
        performance_spec = make_ratio_performance_spec(
            [a for a, _r in target_pairs],
            [r for _a, r in target_pairs],
            env_float("MSOPT_OLED_RATIO_TOL", 0.05),
        )
    else:
        performance_spec = make_ratio_performance_spec(
            performance_spec["angles_deg"],
            performance_spec["target_ratios"],
            performance_spec.get("tolerance", env_float("MSOPT_OLED_RATIO_TOL", 0.05)),
        )
    # Resolved once, from the density grid, and threaded everywhere else -- the
    # grid size and the fold decision are coupled (see resolve_dipole_grid), so
    # re-reading the env var independently in each caller would let them disagree.
    pp_grid_n, pp_fold_on, pp_asymmetry = resolve_dipole_grid(rho)
    post_sources_by_pol = {
        pol: central_cell_dipoles(G, n_dipoles, pol, grid_n=pp_grid_n)
        for pol in pols
    }
    source_box_min_clearance = None
    if use_source_flux_box:
        all_sources = [
            point
            for points in post_sources_by_pol.values()
            for point in points
        ]
        source_box_min_clearance = minimum_source_box_clearance(
            all_sources,
            [0.0, 0.0, float(G.eml_c[2])],
            source_box_size,
        )
        required_clearance = 0.25 / max(float(G.pp_resolution), 1.0)
        if (
            source_box_min_clearance is None
            or source_box_min_clearance < required_clearance
        ):
            raise ValueError(
                "Every postprocess dipole must be strictly inside the source flux "
                "box. Current minimum clearance is "
                f"{source_box_min_clearance!r} um; require >= "
                f"{required_clearance:.6g} um. Reduce "
                "MSOPT_OLED_PP_SOURCE_BOX_INSET_PIXELS or increase "
                "MSOPT_OLED_PP_SOURCE_INSET_PIXELS."
            )
    requested_runs = int(sum(len(points) for points in post_sources_by_pol.values()))
    failed_cases = []
    records, spectrum_sum, per_dipole = [], None, []
    pol_spectra = {}                                   # per-polarization incoherent sum
    ukx = uky = None
    run_idx = 0
    keep_fsp = env_flag("MSOPT_OLED_PP_KEEP_FSP", "1")
    angular_projection = os.environ.get(
        "MSOPT_OLED_PP_ANGULAR_PROJECTION",
        "farfield3d",
    ).strip().lower()
    if angular_projection not in ("farfield3d", "monitor_fft"):
        raise ValueError(
            "MSOPT_OLED_PP_ANGULAR_PROJECTION must be 'farfield3d' or 'monitor_fft'."
        )

    # Cache identity: the design, plus everything that changes what a single case
    # computes. A cached spectrum is reused only when all of it matches, so a
    # different design can never quietly inherit another one's cases.
    #
    # THE GRID AND LAYOUT ARE PART OF THE IDENTITY. A case is addressed on disk by
    # its flat grid INDEX (case_<pol>_<i>.npz) and nothing else, so two runs that
    # agree on the design but not on the grid would map index i to two different
    # dipole POSITIONS -- and the second would silently inherit the first's far
    # field. That was only ever latent while every run used the 6x6 default; with
    # the fold spending its saving on a 12x12 grid it is a live hazard, and a
    # stale-cache mix-up is invisible in the output. Including them here demotes
    # it to a cache miss.
    pp_cache_key = "|".join(str(v) for v in (
        hashlib.sha256(np.ascontiguousarray(rho, dtype=float).tobytes()).hexdigest()[:16],
        G.resolution, angular_projection, float(np.mean(G.visible_wavelengths)),
        G.target_monitor_c[2], G.Sx, G.Sy, G.Sz,
        pp_grid_n,
        os.environ.get("MSOPT_OLED_PP_SOURCE_LAYOUT", "cell_center").strip().lower(),
        env_float("MSOPT_OLED_PP_SOURCE_INSET_PIXELS", 2.0),
    ))
    manifest_path = os.path.join(G.design_dir, "OLED_postprocess_manifest.json")
    design_hash = hashlib.sha256(np.ascontiguousarray(rho, dtype=np.float64).tobytes()).hexdigest()
    manifest = {
        "schema_version": 2,
        "validation_protocol": {
            "name": "step1_step2b_sourcewise_incoherent_v1",
            "references": [
                "step1_trace_comparison.py",
                "step2b_coherence_case2abc_36src.py",
            ],
            "authoritative_estimator": "independent_sourcewise_incoherent_sum",
            "coherent_random_phase_role": "trend_check_only",
        },
        "status": "running",
        "authoritative": False,
        "mode": pp_mode,
        "convergence_confirmed": env_flag("MSOPT_OLED_PP_CONVERGENCE_CONFIRMED", "0"),
        "tile_count": [int(tile_n), int(tile_n)],
        "resolution": int(G.pp_resolution),
        "monitor_boundary_inset_um": float(monitor_boundary_inset),
        "monitor_span_um": [float(post_monitor_s[0]), float(post_monitor_s[1])],
        "wavelength_um": [float(v) for v in np.asarray(G.visible_wavelengths).reshape(-1)],
        "period_um": [float(G.window_x), float(G.window_y)],
        "polarizations": list(pols),
        "dipole_grid": int(pp_grid_n),
        "symmetry_fold": bool(pp_fold_on),
        "design_mirror_asymmetry": float(pp_asymmetry),
        "source_layout": os.environ.get(
            "MSOPT_OLED_PP_SOURCE_LAYOUT",
            "cell_center",
        ),
        "source_flux_box": bool(use_source_flux_box),
        "source_flux_box_inset_um": (
            float(source_box_inset) if source_box_inset is not None else None
        ),
        "source_flux_box_size_um": source_box_size,
        "source_flux_box_min_source_clearance_um": source_box_min_clearance,
        "angular_projection": angular_projection,
        "requested_runs": requested_runs,
        "successful_runs": 0,
        "failed_runs": 0,
        "completion_fraction": 0.0,
        "design_sha256": design_hash,
        # Planar runs characterize the unpatterned stack, so they are a
        # REFERENCE, never an authoritative readout of a design.
        "planar_baseline": bool(planar_baseline),
        "capture": {"target_deg": float(capture_deg),
                    "achieved_deg": float(achieved_deg),
                    "monitor_above_stack_um": float(monitor_h),
                    "lateral_um": float(post_sx)},
        "performance_spec": {
            key: (
                [float(v) for v in np.asarray(value).reshape(-1)]
                if isinstance(value, (np.ndarray, list, tuple))
                else float(value)
            )
            for key, value in performance_spec.items()
        },
    }

    def _write_manifest():
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2, sort_keys=True)
            fp.write("\n")

    def _write_records(path):
        with open(path, "w", encoding="utf-8") as fp:
            fp.write(f"status {manifest['status']}\n")
            fp.write(f"requested_runs {requested_runs}\n")
            fp.write(f"successful_runs {len(records)}\n")
            fp.write(f"failed_runs {len(failed_cases)}\n")
            fp.write(
                "dipole_idx x_um y_um z_um pol top_monitor_transmission "
                "source_power dipole_power source_box_power normalization_power "
                "extracted_top_power LEE dipole_box_rel_diff\n"
            )
            for rec in records:
                fp.write(
                    "%d %.8e %.8e %.8e %s %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n"
                    % rec
                )

    _write_manifest()

    def _drop_last_fsp():
        # MSOPT_OLED_PP_KEEP_FSP=0: delete the just-analyzed run's .fsp so a long
        # dipole/phase sweep does not fill the disk; results are already extracted.
        # Lumerical writes a SIBLING DIRECTORY and a _p0.log next to every .fsp
        # (~200 MB per run at postprocess sizes); removing only the .fsp still
        # filled the disk and crashed later runs, so drop all three.
        if keep_fsp:
            return
        fsp = getattr(sim, "_last_run_fsp_path", None)
        if not fsp:
            return
        stem = fsp[:-4] if fsp.endswith(".fsp") else fsp
        for path in (fsp, f"{stem}_p0.log"):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        try:
            if os.path.isdir(stem):
                shutil.rmtree(stem, ignore_errors=True)
        except Exception:
            pass

    def _case_spectrum():
        # farfield3d on the top aperture is the Lumerical equivalent of the
        # validated radial-Poynting N2F readout. A large supercell is required
        # so high-angle power crosses this monitor before reaching lateral PML.
        if angular_projection == "farfield3d":
            return n2f_spectrum(sim, G.target_monitor_name)
        T_c = read_transmission(sim.fdtd, G.target_monitor_name)
        return monitor_spectrum(sim, G.target_monitor_name, float(np.mean(G.visible_wavelengths)), 1.0 if T_c >= 0 else -1.0)

    # A GPU shared with other jobs occasionally leaves a run with NO session
    # results ("no d-card ... with the necessary data"); re-running just that
    # dipole recovers it, so each dipole gets up to MSOPT_OLED_PP_RETRIES
    # re-runs. Accumulators are only committed after EVERY session read
    # succeeded, so a failed attempt can never double-count.
    pp_retries = env_int("MSOPT_OLED_PP_RETRIES", 2)

    # ---- C4v symmetry fold ---------------------------------------------------
    # A mirror-symmetric design makes each dipole orbit ONE simulation instead of
    # four: 36 runs of the 6x6 grid can only ever hold 9 distinct numbers, so the
    # other 27 are paid for and discarded. Folding them away is NOT a coarser
    # ensemble -- every orbit member still gets its own record, its own position
    # and its own (reflected) far field, so the published LEE, angle profile and
    # per-dipole plots are the same products over the same 36 emitters. Only the
    # FDTD count changes.
    #
    # It is exact in the continuum and very nearly exact on the mesh: measured on
    # run 20260812_035518, the four members of every orbit agreed to <= 6e-5
    # relative, so replacing three of them by the fourth moves the ensemble by at
    # most that. Summation order also changes, which is another ~1e-16. Neither
    # is a modelling approximation, but "identical" means to those tolerances,
    # not bitwise.
    #
    # OFF by default: it is exact for a C4v design and WRONG for anything else,
    # so it is opted into, and even then refused unless the density grid actually
    # carries both mirrors (see design_mirror_asymmetry -- the design flag lives
    # in another file and can be stale, the grid cannot).
    n_points = len(post_sources_by_pol[pols[0]])
    fold_members = {i: [(i, False, False)] for i in range(n_points)}
    if pp_fold_on:
        if pp_grid_n * pp_grid_n != n_points:
            raise ValueError("the symmetry fold requires the square "
                             "MSOPT_OLED_PP_DIPOLE_GRID layout.")
        orbits = mirror_orbits(pp_grid_n)
        fold_members = {rep: members for rep, members in orbits}
        print(f"[postprocess] symmetry fold ON (design mirror asymmetry {pp_asymmetry:.2e}): "
              f"{pp_grid_n}x{pp_grid_n} grid, {len(orbits)} orbit(s) cover {n_points} point(s) "
              f"-> {len(orbits)} FDTD run(s) per polarization instead of {n_points}")
    else:
        print(f"[postprocess] symmetry fold OFF (design mirror asymmetry "
              f"{pp_asymmetry:.2e}): {pp_grid_n}x{pp_grid_n} grid, "
              f"{n_points} FDTD run(s) per polarization")

    for pol in pols:
        post_sources = post_sources_by_pol[pol]
        pol_base = pols.index(pol) * len(post_sources)
        for i, (x, y, z, _p) in enumerate(post_sources):
            if i not in fold_members:
                continue          # mirror image of an orbit representative; see fold_members
            print(f"[postprocess] {tile_n}x{tile_n}/PML pol={pol} dipole {i + 1}/{len(post_sources)}: x={x:.3f}, y={y:.3f}"
                  + (f"  (+{len(fold_members[i]) - 1} mirror image(s))" if len(fold_members[i]) > 1 else ""))
            case_succeeded = False
            # One flaky case at the end used to discard every other case's ANGULAR
            # SPECTRUM, because the spectra only ever lived in memory -- 71 good runs
            # thrown away for one failure. Each finished case is now written next to
            # the run, so a re-run repeats only what is actually missing.
            cached = load_pp_case(pp_cache_dir, pol, i, pp_cache_key)
            last_error = None
            for attempt in range(pp_retries + 1):
                if cached is not None:
                    (theta_i, spectrum_i, ukx_i, uky_i,
                     T, src_power, dip_power, box_power) = cached
                    if attempt == 0:
                        print(f"[postprocess] cache hit pol={pol} dipole {i} -- FDTD skipped")
                else:
                    try:
                        sim.fdtd.switchtolayout()
                        delete_object(sim.fdtd, "postprocess_dipole")
                        add_dipole(G, sim, x, y, z + z_shift, pol, "postprocess_dipole")
                        if pp_field_images:
                            sim.fdtd.setnamed(pp_xz_monitor_name, "y", y * 1e-6)
                        sim.run(name=f"postprocess_pml_{run_idx:03d}", save=True)
                        load_run_results(sim)
                        T = read_transmission(sim.fdtd, G.target_monitor_name)
                        if angular_projection == "farfield3d":
                            theta_i, spectrum_i, ukx_i, uky_i = n2f_spectrum(sim, G.target_monitor_name)
                        else:
                            theta_i, spectrum_i, ukx_i, uky_i = monitor_spectrum(sim, G.target_monitor_name, float(np.mean(G.visible_wavelengths)), 1.0 if T >= 0 else -1.0)
                        freqs = source_freqs(G, sim)
                        src_power = read_source_power(sim.fdtd, freqs)
                        dip_power = read_dipole_power(sim.fdtd, freqs)
                        box_power = (
                            read_flux_box_power(sim.fdtd, source_flux_box_faces, src_power)
                            if source_flux_box_faces
                            else None
                        )
                    except Exception as exc:
                        last_error = str(exc)
                        if attempt < pp_retries:
                            print(f"[postprocess] warning: pol={pol} dipole {i} attempt {attempt + 1} failed ({exc}); re-running")
                            continue
                        print(f"[postprocess] warning: pol={pol} dipole {i} failed after {pp_retries + 1} attempts: {exc}")
                        break
                # ---- all session reads done; commit (no fdtd calls below) ----
                # Full LEE = extracted top power / validated source-box emission.
                # is normalized to source power, so absolute top power = |T|*source_power;
                # total emission = dipolepower (falls back to source power if unavailable).
                src = valid_power(src_power)
                dip = valid_power(dip_power)
                box = valid_power(box_power)
                total_emitted = box or dip or src or G.channel_power_floor
                if use_source_flux_box and box is None:
                    last_error = "source flux-box power is missing/non-positive"
                    if attempt < pp_retries:
                        print(
                            f"[postprocess] warning: pol={pol} dipole {i} attempt "
                            f"{attempt + 1} has no valid source-box power; re-running"
                        )
                        continue
                    print(
                        f"[postprocess] warning: pol={pol} dipole {i} failed after "
                        f"{pp_retries + 1} attempts: {last_error}"
                    )
                    break
                top_power_abs = max(float(T), 0.0) * (
                    src if src is not None else total_emitted
                )
                # The floor is a divide-by-zero guard only. If it is anywhere
                # near the real emitted power (~1e-15 W here) it silently
                # REPLACES the denominator and every LEE is wrong, so refuse
                # rather than publish a corrupted number.
                if total_emitted <= 100.0 * G.channel_power_floor:
                    raise RuntimeError(
                        f"channel_power_floor ({G.channel_power_floor:.3e}) is not negligible "
                        f"versus the emitted power ({total_emitted:.3e} W); it would replace the "
                        "LEE denominator. Lower MSOPT_OLED_CHANNEL_POWER_FLOOR (it guards "
                        "absolute watts, not FoM scores)."
                    )
                lee = top_power_abs / total_emitted
                dipole_box_rel_diff = (
                    abs(dip - box) / max(abs(dip), abs(box), G.channel_power_floor)
                    if dip is not None and box is not None
                    else np.nan
                )
                # Calibrate every angular spectrum to the independently measured
                # absolute top-monitor power. This makes incoherent sums valid even
                # when FFT/n2f normalization constants or source normalizations differ.
                raw_spectrum_sum = float(np.sum(np.maximum(spectrum_i, 0.0)))
                if not np.isfinite(raw_spectrum_sum) or raw_spectrum_sum <= 0.0:
                    last_error = "zero/non-finite far-field power"
                    if attempt < pp_retries:
                        print(
                            f"[postprocess] warning: pol={pol} dipole {i} attempt "
                            f"{attempt + 1} produced zero/non-finite far-field power; re-running"
                        )
                        continue
                    print(
                        f"[postprocess] warning: pol={pol} dipole {i} failed after "
                        f"{pp_retries + 1} attempts: zero/non-finite far-field power"
                    )
                    break
                spectrum = np.maximum(spectrum_i, 0.0) * top_power_abs / raw_spectrum_sum
                theta, ukx, uky = theta_i, ukx_i, uky_i
                case_succeeded = True
                if cached is None:
                    save_pp_case(pp_cache_dir, pol, i, pp_cache_key,
                                 theta_i, spectrum_i, ukx_i, uky_i,
                                 T, src_power, dip_power, box_power)
                # One pass per orbit member. Without the fold that is exactly this
                # one case and the loop is what it always was; with it, the mirror
                # images are reconstructed rather than simulated. Every scalar is
                # mirror-invariant (same structure, same |E|^2), so only the far
                # field has to be transformed: ukx runs along axis 0 and uky along
                # axis 1, so x->-x is a flip of axis 0 and y->-y of axis 1. theta
                # is invariant under both, which is why the theta-binned products
                # below come out identical for every member -- but the 2-D
                # spectrum_sum, the theta-phi map and the SIGNED radiance are not,
                # and those are what the flips are for.
                for m_i, flip_x, flip_y in fold_members[i]:
                    m_spectrum = spectrum
                    if flip_x:
                        m_spectrum = np.flip(m_spectrum, axis=0)
                    if flip_y:
                        m_spectrum = np.flip(m_spectrum, axis=1)
                    m_x, m_y, m_z, _mp = post_sources[m_i]
                    m_idx = pol_base + m_i
                    # Incoherent sum over BOTH position and polarization.
                    spectrum_sum = m_spectrum.copy() if spectrum_sum is None else spectrum_sum + m_spectrum
                    pol_spectra[pol] = (m_spectrum.copy() if pol not in pol_spectra
                                        else pol_spectra[pol] + m_spectrum)
                    records.append((
                        m_idx, m_x, m_y, m_z, pol, float(T),
                        src or np.nan, dip or np.nan, box or np.nan,
                        total_emitted, top_power_abs, lee, dipole_box_rel_diff,
                    ))
                    # Per-dipole angular breakdown: ring flux split into absolute extraction
                    # efficiency per angle bin (sums to this dipole's LEE), plus the
                    # per-direction radiance (solid-angle Jacobian removed) for the shape.
                    ring_flux = angle_profile(theta, m_spectrum, angles)
                    ring_count = angle_profile(theta, np.ones_like(m_spectrum), angles)
                    _, rad_signed = directional_radiance(theta, ukx, m_spectrum, signed_angles)
                    per_dipole.append({
                        "idx": m_idx, "x": float(m_x), "y": float(m_y),
                        "r": float(np.hypot(m_x, m_y)), "pol": pol, "lee": float(lee),
                        "eff": ring_flux / max(total_emitted, G.channel_power_floor),
                        "rad": ring_flux / np.maximum(ring_count, 1.0),
                        "rad_signed": rad_signed,     # signed (+/-) radiance, real left/right shape
                    })
                if pp_field_images:
                    try:
                        render_xz_field_image(
                            sim.fdtd.getresult(pp_xz_monitor_name, "E"),
                            os.path.join(G.design_dir, f"dipole_emission_{run_idx}.png"),
                            f"dipole {run_idx}  pol={pol}  x={x:.2f} y={y:.2f} um  |E|",
                        )
                    except Exception as exc:
                        print(f"[postprocess] warning: field image {run_idx} failed: {exc}")
                if pp_radiation_images:
                    try:
                        th_ax, ph_ax, Sr_map = theta_phi_map(theta, ukx, uky, spectrum)
                        save_radiation_map_figure(
                            os.path.join(G.design_dir, f"dipole_radiation_{run_idx:03d}.png"),
                            th_ax, ph_ax, Sr_map,
                            f"dipole {run_idx}  pol={pol}  x={x:.2f} y={y:.2f} um",
                        )
                    except Exception as exc:
                        print(f"[postprocess] warning: radiation map {run_idx} failed: {exc}")
                _drop_last_fsp()
                break
            if not case_succeeded:
                failed_cases.append({
                    "dipole_idx": int(run_idx),
                    "position_um": [float(x), float(y), float(z)],
                    "polarization": pol,
                    "error": last_error or "unknown postprocess failure",
                })
            manifest["successful_runs"] = len(records)
            manifest["failed_runs"] = len(failed_cases)
            manifest["completion_fraction"] = len(records) / max(requested_runs, 1)
            manifest["failed_cases"] = failed_cases
            _write_manifest()
            run_idx += 1
    sweep_seconds = time.time() - pp_t0
    print(f"[postprocess] polarizations {pols}: {len(records)} total dipole runs (incoherent sum)")
    print(f"[postprocess] dipole sweep wall time: {sweep_seconds / 60.0:.1f} min "
          f"({sweep_seconds / max(run_idx, 1):.1f} s per simulation over {run_idx} runs)")

    complete = len(records) == requested_runs
    manifest["status"] = "complete" if complete else "incomplete"
    manifest["successful_runs"] = len(records)
    manifest["failed_runs"] = len(failed_cases)
    manifest["completion_fraction"] = len(records) / max(requested_runs, 1)
    manifest["failed_cases"] = failed_cases
    _write_manifest()
    _write_records(os.path.join(G.design_dir, "OLED_postprocess_records.txt"))
    if not complete and env_flag("MSOPT_OLED_PP_REQUIRE_COMPLETE", "1"):
        try:
            sim.fdtd.close()
        except Exception:
            pass
        raise RuntimeError(
            f"OLED postprocess incomplete: {len(records)}/{requested_runs} dipole/polarization "
            f"cases succeeded. Aggregate performance was not published; rerun the failed cases."
        )

    # --- Coherence cases (validated step2b definitions) ------------------------
    # Case 1: source-wise incoherent reference (the main loop above).
    # Case 2a: all sources simultaneous, phase=0.
    # Case 2b: one simultaneous random-phase draw, seed=0.
    # Case 3: simultaneous random-phase draws, seed=1234, cumulative averages at
    #         1/2/5/10/20 trials. Case 3 is a trend check, never a replacement
    #         for the source-wise incoherent reference.
    coh_cases = []                                   # (tag, theta, spectrum, ukx, uky)
    rand_trial_profiles = []                         # (trial_no, theta, spectrum)
    coherence_summary = {"enabled": False}
    # Optional coherence audit (step2b cases 2a/2b/3). Body in
    # _postprocess_coherence_audit; it is off by default and self-contained.
    if env_flag("MSOPT_OLED_PP_COHERENT_CHECK", "0"):
        _postprocess_coherence_audit(
            G=G, sim=sim, manifest=manifest, records=records, pols=pols,
            n_dipoles=n_dipoles, pp_grid_n=pp_grid_n, z_shift=z_shift,
            source_flux_box_faces=source_flux_box_faces,
            coh_cases=coh_cases, rand_trial_profiles=rand_trial_profiles,
            _case_spectrum=_case_spectrum, _drop_last_fsp=_drop_last_fsp,
            _write_manifest=_write_manifest)

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
    target = np.asarray([G.interp_curve(a) for a in angles], dtype=float)
    target_norm = target / max(float(np.max(target)), 1e-30)

    profile_data = np.column_stack([angles, ring_flux, radiance_norm, target_norm])
    for profile_name in ("OLED_postprocess_angle_profile.txt", "OLED_postprocess_3x3_angle_profile.txt"):
        np.savetxt(
            os.path.join(G.design_dir, profile_name),
            profile_data,
            header="theta_deg ring_flux radiance_norm target_norm",
        )

    lee_values = np.asarray([rec[11] for rec in records], dtype=float)
    lee_values = lee_values[np.isfinite(lee_values)]
    valid_records = [
        rec for rec in records
        if np.isfinite(rec[10]) and np.isfinite(rec[9]) and rec[9] > 0.0
    ]
    emitted_power_sum = float(np.sum([rec[9] for rec in valid_records])) if valid_records else 0.0
    # ABSOLUTE per-direction radiance: per unit EMITTED power, so it is comparable
    # BETWEEN devices and independent of how many dipoles were summed.
    # radiance_signed_norm above is divided by its OWN peak, which throws exactly
    # that away -- fine for drawing one device's shape, useless for saying whether
    # this device is brighter or dimmer than another. The planar overlay needs the
    # absolute one; see render_pp_summary_figure.
    radiance_signed_abs = rad_signed_sum / max(emitted_power_sum, G.channel_power_floor)
    mean_lee = (
        float(np.sum([rec[10] for rec in valid_records]) / emitted_power_sum)
        if emitted_power_sum > 0.0
        else float("nan")
    )
    min_lee = float(np.min(lee_values)) if lee_values.size else float("nan")
    max_lee = float(np.max(lee_values)) if lee_values.size else float("nan")
    print(
        f"[postprocess] full ensemble LEE (sum extracted top / sum normalization power): "
        f"{mean_lee:.6e}; individual min={min_lee:.6e}, max={max_lee:.6e} "
        f"over {lee_values.size} dipoles"
    )
    normalization_diffs = np.asarray([rec[12] for rec in records], dtype=float)
    normalization_diffs = normalization_diffs[np.isfinite(normalization_diffs)]
    normalization_tolerance = env_float("MSOPT_OLED_PP_POWER_NORM_TOL", 0.05)
    normalization_check = {
        "reference": "source_flux_box" if use_source_flux_box else "dipolepower",
        "dipolepower_box_samples": int(normalization_diffs.size),
        "dipolepower_box_median_rel_diff": (
            float(np.median(normalization_diffs)) if normalization_diffs.size else None
        ),
        "dipolepower_box_max_rel_diff": (
            float(np.max(normalization_diffs)) if normalization_diffs.size else None
        ),
        "comparison_tolerance": normalization_tolerance,
        "comparison_pass": bool(
            normalization_diffs.size == len(records)
            and np.max(normalization_diffs) <= normalization_tolerance
        ) if records else False,
    }
    manifest["power_normalization"] = normalization_check
    _write_manifest()
    if normalization_diffs.size:
        print(
            "[postprocess] dipolepower vs source-box: "
            f"median={np.median(normalization_diffs):.3%}, "
            f"max={np.max(normalization_diffs):.3%}, "
            f"check_pass={normalization_check['comparison_pass']} "
            "(source box remains the normalization reference)"
        )

    # step2b-style ensemble products: theta-phi map, signed plane cuts, cone shares.
    # step2b ensemble products (theta-phi map, cone shares, cumulative
    # extraction, channel ratios). Body in _postprocess_ensemble_products.
    cum_ang, cum_share, cum_abs = _postprocess_ensemble_products(
        G=G, angles=angles, angular_projection=angular_projection,
        complete=complete, design_hash=design_hash,
        emitted_power_sum=emitted_power_sum, manifest=manifest,
        mean_lee=mean_lee, performance_spec=performance_spec,
        planar_baseline=planar_baseline, pp_mode=pp_mode,
        spectrum_sum=spectrum_sum, theta=theta, ukx=ukx, uky=uky,
        use_source_flux_box=use_source_flux_box,
        _write_manifest=_write_manifest)

    # Per-polarization breakdown (only when the sweep ran more than one polarization):
    # each polarization's own incoherent angular profile and ensemble LEE, so x vs y (vs z)
    # can be inspected separately in addition to their combined sum.
    if len(pol_spectra) > 1:
        for pol, spec in pol_spectra.items():
            pol_ring = angle_profile(theta, spec, angles)
            pol_rad = radiance_from_spectrum(theta, spec, angles)
            np.savetxt(
                os.path.join(G.design_dir, f"OLED_postprocess_pol_{pol}_angle_profile.txt"),
                np.column_stack([angles, pol_ring, pol_rad]),
                header="theta_deg ring_flux radiance_norm",
            )
            pol_records = [
                rec for rec in records
                if rec[4] == pol and np.isfinite(rec[10]) and np.isfinite(rec[9]) and rec[9] > 0.0
            ]
            pol_lee = (
                float(np.sum([rec[10] for rec in pol_records]) / np.sum([rec[9] for rec in pol_records]))
                if pol_records
                else float("nan")
            )
            print(f"[postprocess] pol={pol}: ensemble LEE={pol_lee:.6e} over {len(pol_records)} dipoles")

    with open(os.path.join(G.design_dir, "OLED_postprocess_3x3_records.txt"), "w", encoding="utf-8") as fp:
        method = "n2f_single_cell_pml" if pp_mode == "single" else f"final_{tile_n}x{tile_n}_array_pml"
        fp.write(f"method {method}_central_cell_dipoles\n")
        fp.write(f"status {manifest['status']}\n")
        fp.write(f"requested_runs {requested_runs}\n")
        fp.write(f"successful_runs {len(records)}\n")
        fp.write(f"completion_fraction {len(records) / max(requested_runs, 1):.8e}\n")
        fp.write(f"ensemble_LEE {mean_lee:.8e}\n")
        fp.write(f"mean_LEE {mean_lee:.8e}\n")  # legacy reader compatibility
        fp.write(f"min_LEE {min_lee:.8e}\n")
        fp.write(f"max_LEE {max_lee:.8e}\n")
        fp.write(
            "dipole_idx x_um y_um z_um pol top_monitor_transmission "
            "source_power dipole_power source_box_power normalization_power "
            "extracted_top_power LEE dipole_box_rel_diff\n"
        )
        for rec in records:
            fp.write(
                "%d %.8e %.8e %.8e %s %.8e %.8e %.8e %.8e %.8e %.8e %.8e %.8e\n"
                % rec
            )

    # A': re-evaluate the OPTIMIZATION's OWN FoM on this incoherent far field.
    # This is the only true optimization-vs-postprocess comparison: it uses the
    # ring angles and target profile the design was actually optimized against
    # (loaded from the design's own OLED_angular_target.npz), not whatever
    # generic MSOPT_OLED_TARGET_ANGLES default the driver script carries.
    opt_target = load_optimization_angular_target(
        os.environ.get("MSOPT_OLED_POSTPROCESS_DESIGN", "").strip()
        or os.path.join(G.design_dir, "lastdesign.txt")
    )
    opt_match = optimization_target_match(theta, ukx, uky, spectrum_sum, opt_target) if opt_target else None
    if opt_match is not None:
        with open(os.path.join(G.design_dir, "OLED_postprocess_optimization_target_match.txt"), "w", encoding="utf-8") as fp:
            fp.write(f"# optimization target: {opt_target['path']}\n")
            fp.write("# FoM re-evaluated on the INCOHERENT postprocess far field\n")
            fp.write(f"# throughput {opt_match['throughput']:.8e}\n")
            fp.write(f"# match {opt_match['match']:.8e}\n")
            fp.write(f"# fom_throughput_x_match {opt_match['fom']:.8e}\n")
            fp.write(f"# total_variation_vs_target {opt_match['total_variation']:.8e}\n")
            fp.write("theta_deg achieved_share target_share in_range\n")
            for t, a, g, r in zip(opt_match["angle_thetas"], opt_match["achieved_profile"],
                                  opt_match["target_profile"], opt_match["in_range"]):
                fp.write(f"{t:.4f} {a:.6e} {g:.6e} {int(r)}\n")
        print(
            "[postprocess] optimization FoM re-evaluated on PP far field: "
            f"throughput={opt_match['throughput']:.4f}, match={opt_match['match']:.4f}, "
            f"FoM={opt_match['fom']:.4f}, TV vs target={opt_match['total_variation']:.4f}"
        )
        print("[postprocess]   per-ring achieved vs target: " + ", ".join(
            f"{t:.1f}deg {a * 100:.2f}%/{g * 100:.2f}%"
            for t, a, g in zip(opt_match["angle_thetas"], opt_match["achieved_profile"], opt_match["target_profile"])
        ))
        manifest["optimization_target_match"] = {
            "target_file": opt_target["path"],
            "angle_thetas_deg": [float(v) for v in opt_match["angle_thetas"]],
            "achieved_share": [float(v) for v in opt_match["achieved_profile"]],
            "target_share": [float(v) for v in opt_match["target_profile"]],
            "in_range": [bool(v) for v in opt_match["in_range"]],
            "throughput": float(opt_match["throughput"]),
            "match": float(opt_match["match"]),
            "fom": float(opt_match["fom"]),
            "total_variation": float(opt_match["total_variation"]),
        }
    else:
        print("[postprocess] NOTE: no optimization angular target found next to the design; "
              "the target columns below come from MSOPT_OLED_TARGET_* defaults and are NOT "
              "this design's optimization goal.")

    # B: decompose the incoherent emission into diffraction orders and compare
    # order power shares against the optimization's target order shares. This is
    # the apples-to-apples comparison (the optimization matches order shares, not
    # a continuous angular curve). Orders are grouped by their polar angle.
    order_info = build_target_orders(G, float(np.mean(G.visible_wavelengths)), G.window_x, G.window_y)
    propagating = np.sqrt(ukx ** 2 + uky ** 2) <= 1.0 + 1e-12
    order_labels = nearest_order_labels(ukx, uky, order_info["orders"], propagating)
    order_rows = {}
    for order_idx, o in enumerate(order_info["orders"]):
        p = float(np.sum(spectrum_sum[order_labels == order_idx]))
        key = round(float(o["theta_deg"]), 1)
        row = order_rows.setdefault(key, {"achieved": 0.0, "target": 0.0})
        row["achieved"] += p
        row["target"] += max(float(o["efficiency"]), 0.0)
    order_thetas = sorted(order_rows)
    achieved = np.asarray([order_rows[t]["achieved"] for t in order_thetas], dtype=float)
    target_eff = np.asarray([order_rows[t]["target"] for t in order_thetas], dtype=float)
    achieved_share = achieved / max(float(np.sum(achieved)), 1e-30)
    target_share = target_eff / max(float(np.sum(target_eff)), 1e-30)
    order_total_variation = 0.5 * float(np.sum(np.abs(achieved_share - target_share)))
    zero_order_idx = next((i for i, t in enumerate(order_thetas) if abs(float(t)) < 1e-9), None)
    zero_order_power = float(achieved[zero_order_idx]) if zero_order_idx is not None else 0.0
    zero_order_share = float(achieved_share[zero_order_idx]) if zero_order_idx is not None else 0.0
    print(
        f"[postprocess] integrated zero-order power={zero_order_power:.6e} W, "
        f"share={zero_order_share * 100:.3f}% "
        "(Voronoi region around the (0,0) diffraction order; not a single theta=0 pixel)"
    )

    with open(os.path.join(G.design_dir, "OLED_postprocess_order_shares.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"# zero_order_power {zero_order_power:.8e}\n")
        fp.write(f"# zero_order_share {zero_order_share:.8e}\n")
        fp.write(f"# total_variation {order_total_variation:.8e}\n")
        fp.write("theta_deg achieved_power achieved_share target_efficiency target_share\n")
        for t, a, ash, te, tsh in zip(order_thetas, achieved, achieved_share, target_eff, target_share):
            fp.write(f"{t:.3f} {a:.6e} {ash:.6e} {te:.6e} {tsh:.6e}\n")

    manifest["order_metrics"] = {
        "theta_deg": [float(v) for v in order_thetas],
        "achieved_share": [float(v) for v in achieved_share],
        "target_share": [float(v) for v in target_share],
        "total_variation": order_total_variation,
        "zero_order_power": zero_order_power,
        "zero_order_share": zero_order_share,
    }
    _write_manifest()

    # LEE read-out on the emission figure: this run's LEE and, when a planar
    # (no-design) reference run is given via MSOPT_OLED_PP_PLANAR_REFERENCE,
    # the bare-stack LEE and the enhancement factor. A planar run reports only
    # its own LEE (it IS the no-design case).
    planar_lee = None
    if not planar_baseline:
        planar_lee = load_planar_reference_lee(
            os.environ.get("MSOPT_OLED_PP_PLANAR_REFERENCE", "").strip())
        manifest["planar_reference_lee"] = planar_lee
        if planar_lee:
            manifest["lee_enhancement_vs_planar"] = float(mean_lee / planar_lee)
            print(f"[postprocess] LEE {mean_lee * 100:.2f}% vs planar {planar_lee * 100:.2f}% "
                  f"-> {mean_lee / planar_lee:.2f}x")

    path = os.path.join(G.design_dir, "OLED_postprocess_emission.png")
    cum_for_fig = ((cum_ang, cum_share, cum_abs)
                   if cum_ang is not None else None)
    render_emission_figure(angles, radiance_signed_norm, target_norm, order_thetas, achieved_share, target_share, path, "signed radiance", lee=mean_lee, planar_lee=planar_lee, cumulative=cum_for_fig)
    legacy_path = os.path.join(G.design_dir, "OLED_postprocess_3x3_emission.png")
    render_emission_figure(angles, radiance_signed_norm, target_norm, order_thetas, achieved_share, target_share, legacy_path, "signed radiance", lee=mean_lee, planar_lee=planar_lee, cumulative=cum_for_fig)
    print(f"[postprocess] saved emission (radiance + order-share) plot: {path}")

    # A planar run is the ONE that produces the reference every other run draws;
    # write it before the figure so this run's own plot already shows it.
    #
    # Refresh whenever the stored curve is not USABLE, not merely when the file is
    # absent. The old "and not os.path.exists(...)" meant the very first planar
    # measurement was permanent: a later planar run -- deliberately launched to
    # replace a reference from a different stack, grid or storage format -- would
    # silently decline to write, and the stale curve stayed. Someone running a
    # planar postprocess is asking for exactly this file.
    if planar_requested():
        _identity = planar_reference_identity(G, pp_grid_n)
        if load_planar_reference_curve(identity=_identity)[0] is None:
            save_planar_reference_curve(angles, radiance_signed_abs, lee=mean_lee,
                                        identity=_identity)
        else:
            print("[postprocess] planar reference already matches this configuration; kept")

    summary_path = os.path.join(G.design_dir, "PP_summary.png")
    render_pp_summary_figure(angles, radiance_signed_norm, target_norm,
                             summary_path, lee=mean_lee, planar_lee=planar_lee,
                             planar_identity=planar_reference_identity(G, pp_grid_n),
                             radiance_signed_abs=radiance_signed_abs)
    print(f"[postprocess] saved PP summary (radiance polar + vs theta): {summary_path}")

    per_path = save_per_dipole_emission_plot(G, angles, per_dipole, os.path.join(G.design_dir, "OLED_postprocess_per_dipole_emission.png"))
    if per_path:
        print(f"[postprocess] saved per-dipole angular emission plot ({len(per_dipole)} dipoles): {per_path}")

    # Coherence-case products: same angle-profile / order-share formats as the
    # incoherent reference, tagged per case, so the cases can be overlaid directly.
    for tag, th_c, sp_c, ux_c, uy_c in coh_cases:
        np.savetxt(
            os.path.join(G.design_dir, f"OLED_postprocess_{tag}_angle_profile.txt"),
            np.column_stack([angles, angle_profile(th_c, sp_c, angles), radiance_from_spectrum(th_c, sp_c, angles)]),
            header="theta_deg ring_flux radiance_norm",
        )
        c_prop = np.sqrt(ux_c ** 2 + uy_c ** 2) <= 1.0 + 1e-12
        c_labels = nearest_order_labels(ux_c, uy_c, order_info["orders"], c_prop)
        c_rows = {}
        for order_idx, o in enumerate(order_info["orders"]):
            key = round(float(o["theta_deg"]), 1)
            row = c_rows.setdefault(key, {"achieved": 0.0, "target": 0.0})
            row["achieved"] += float(np.sum(sp_c[c_labels == order_idx]))
            row["target"] += max(float(o["efficiency"]), 0.0)
        c_thetas = sorted(c_rows)
        c_ach = np.asarray([c_rows[t]["achieved"] for t in c_thetas], dtype=float)
        c_te = np.asarray([c_rows[t]["target"] for t in c_thetas], dtype=float)
        c_ash = c_ach / max(float(np.sum(c_ach)), 1e-30)
        c_tsh = c_te / max(float(np.sum(c_te)), 1e-30)
        with open(os.path.join(G.design_dir, f"OLED_postprocess_{tag}_order_shares.txt"), "w", encoding="utf-8") as fp:
            fp.write("theta_deg achieved_power achieved_share target_efficiency target_share\n")
            for t, a, ash, te, tsh in zip(c_thetas, c_ach, c_ash, c_te, c_tsh):
                fp.write(f"{t:.3f} {a:.6e} {ash:.6e} {te:.6e} {tsh:.6e}\n")
        print(f"[postprocess] {tag}: order shares " + ", ".join(f"{t:g}deg={s * 100:.2f}%" for t, s in zip(c_thetas, c_ash)))
    for t_no, th_c, sp_c in rand_trial_profiles:
        np.savetxt(
            os.path.join(G.design_dir, f"OLED_postprocess_randphase_trial{t_no:03d}_angle_profile.txt"),
            np.column_stack([angles, angle_profile(th_c, sp_c, angles), radiance_from_spectrum(th_c, sp_c, angles)]),
            header="theta_deg ring_flux radiance_norm",
        )

    total_seconds = time.time() - pp_t0
    manifest["timing"] = {
        "total_seconds": float(total_seconds),
        "dipole_sweep_seconds": float(sweep_seconds),
        "analysis_seconds": float(total_seconds - sweep_seconds),
        "simulations": int(run_idx),
        "seconds_per_simulation": float(sweep_seconds / max(run_idx, 1)),
    }
    _write_manifest()
    print(f"[postprocess] TOTAL wall time: {total_seconds / 60.0:.1f} min "
          f"(sweep {sweep_seconds / 60.0:.1f} min + analysis "
          f"{(total_seconds - sweep_seconds) / 60.0:.1f} min)")
    return manifest


# =============================================================================
# Optimizer harness helpers  (OLED_new.py main())
# =============================================================================


def session_test_banner(G, N_fom, extra=""):
    # MSOPT_OLED_SESSION_TEST early-exit print.
    msg = f"[session] N_fom={N_fom}, design_grid={G.design_grids}, boundary={G.bc_xy}, postprocess=supercell/PML"
    if extra:
        msg += f", {extra}"
    print(msg)


def save_result_plots(optimizer, design_dir):
    # result0..result6.png optimizer-history figures, ported from OLED_new.py main().
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
