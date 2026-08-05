"""Representative-single-dipole OLED outcoupling optimization (radial 3-D design).

Each of the N_fom channels is ONE x-polarized dipole at a representative radial
position in the EML plane; channel FoMs are combined by a NORMALIZED WEIGHTED
SUM (not mean).  Compact rewrite of legacy_20260731/OLED_lens.py on top of
oled_common; all MSOPT_* environment variables keep working.

The ONE intended behavior change vs the legacy script: the FoM angle basis is
now the shared 2-D k-ring ramp basis from oled_common (MSOPT_OLED_FOM_MODE,
default "kspace_2d") instead of the legacy 1-D phi=0 line basis.  The 1-D
basis only saw the (m,0) diffraction orders and was blind to the off-axis
(m,n) orders, which let ~50% of the power leak above 50 deg in the archived
runs.
"""

import os
import time

import numpy as np

import msopt as ms
import oled_common as oc

npa = oc.npa
ag_jacobian = oc.ag_jacobian

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
STACK = "microcavity"        # "microcavity" (optimized stack) or "legacy"
MC_COLOR = "green"           # red / green / blue -- also sets the wavelength
MC_STACK_KIND = "optimized"  # "optimized" (literature-derived) or "table"

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

# --- Representative-dipole channels ------------------------------------------
# Multiple dipole radii by default: a single on-axis dipole (r=0) barely excites
# the high-angle (off-normal) diffraction orders, so the FoM was blind to the
# leakage that off-axis dipoles dominate in the real (postprocess) device.
# Sampling several radii -- with areal weights (~r) so the incoherent average
# matches a uniform dipole sheet -- makes the FoM see and suppress that
# high-angle emission.
# How the EML is represented in the FoM.
#   "array"          one simulation, a uniformly spaced grid of same-polarization
#                    dipoles fired TOGETHER (OLED_new's formulation). N_fom = 1.
#   "representative" the previous scheme: one simulation per representative radius,
#                    each with a single dipole, combined as a weighted sum.
#
# WARNING, measured and documented in OLED_opt.py's header: a grid fired together
# is a COHERENT array, and a real EML is an INCOHERENT ensemble. On the 2026-07-30
# check against Done run 20260728_122008 the coherent 25-dipole FoM came out ~7x
# off at 0 deg for the same design, while the reciprocal engine matched the
# incoherent postprocess to ~0.003 in ring share. So this mode optimizes something
# the postprocess does not measure. Kept because it is what was asked for; compare
# against the postprocess before trusting the result.
DIPOLE_MODE = os.environ.get("MSOPT_OLED_DIPOLE_MODE", "array").strip().lower()
ARRAY_COUNT = oc.env_int("MSOPT_OLED_OPT_DIPOLE_COUNT", 25)

raw_r = oc.env_list_float("MSOPT_OLED_DIPOLE_RADII_FRAC", [0.1, 0.4, 0.8])
raw_phi = oc.env_list_float("MSOPT_OLED_DIPOLE_AZIMUTHS_DEG", [0.0])
raw_pol = oc.env_list_str("MSOPT_OLED_DIPOLE_POLARIZATIONS", ["x"])
raw_w = oc.env_list_float("MSOPT_OLED_DIPOLE_WEIGHTS", [1.0, 4.0, 8.0])
N_fom = max(len(raw_r), len(raw_phi), len(raw_pol), len(raw_w))
dipole_radii_frac = oc.broadcast(raw_r, N_fom, "MSOPT_OLED_DIPOLE_RADII_FRAC")
dipole_azimuths_deg = oc.broadcast(raw_phi, N_fom, "MSOPT_OLED_DIPOLE_AZIMUTHS_DEG")
dipole_polarizations = oc.broadcast(raw_pol, N_fom, "MSOPT_OLED_DIPOLE_POLARIZATIONS")
channel_weights = np.asarray(oc.broadcast(raw_w, N_fom, "MSOPT_OLED_DIPOLE_WEIGHTS"), dtype=float)

target_channels = []
if DIPOLE_MODE == "array":
    # One coherent grid per polarization -> one FoM channel per polarization.
    _pols = list(dict.fromkeys(dipole_polarizations))
    _gn = max(1, int(np.ceil(np.sqrt(max(1, ARRAY_COUNT)))))
    # Sub-cell CENTRES, not linspace endpoints. The cell is periodic and
    # active_radius equals the half-period here, so endpoints would drop dipoles
    # exactly on the boundary where the periodic image doubles them.
    _xs = (np.arange(_gn) + 0.5) / _gn * (2.0 * G.active_radius) - G.active_radius
    _ys = _xs.copy()
    for ci, pol in enumerate(_pols):
        pts = []
        for k in range(ARRAY_COUNT):
            row, col = divmod(k, _gn)
            pts.append((float(_xs[col]), float(_ys[row]), float(G.eml_c[2])))
        target_channels.append({
            "name": f"coherent_array_{pol}",
            "dipole_idx": ci,
            # centroid, so anything that still wants a single position keeps working
            "dipole_x": float(np.mean([p[0] for p in pts])),
            "dipole_y": float(np.mean([p[1] for p in pts])),
            "dipole_z": float(G.eml_c[2]),
            "dipoles": pts,
            "polarization": pol,
            "weight": 1.0,
        })
    N_fom = len(target_channels)
    channel_weights = np.ones(N_fom, dtype=float)
    print(f"[dipoles] COHERENT ARRAY: {ARRAY_COUNT} dipoles on a {_gn}x{_gn} grid over "
          f"+-{G.active_radius:g} um, fired together; {N_fom} channel(s) {_pols}")
else:
  for i, (rf, az, pol, weight) in enumerate(zip(dipole_radii_frac, dipole_azimuths_deg, dipole_polarizations, channel_weights)):
    a = np.deg2rad(az)
    target_channels.append(
        {
            "name": f"dipole_{i}_{pol}",
            "dipole_idx": i,
            "dipole_x": float(rf * G.active_radius * np.cos(a)),
            "dipole_y": float(rf * G.active_radius * np.sin(a)),
            "dipole_z": float(G.eml_c[2]),
            "polarization": pol,
            "weight": float(weight),
        }
    )


def combine_fom(vals):
    # Normalized weighted SUM over channels (not the oled_common mean); also the
    # function autograd differentiates in the adjoint_loop Case == 3 branch.
    vals = npa.maximum(npa.where(npa.isfinite(vals), vals, 0.0), G.fom_floor)
    weights = npa.asarray(channel_weights, dtype=float)
    weights = weights / npa.sum(weights) if float(np.sum(channel_weights)) > 0.0 else npa.ones_like(weights) / max(float(weights.size), 1.0)
    return npa.sum(vals * weights)


# --- Radial 3-D design mapping (63 x Nz parameters) --------------------------
DR_info = [G.design_s[0], G.design_s[1], G.design_s[2], 0, 1, 2]
DR_N_info = [G.Nx, G.Ny, G.Nz, G.resolution]
radial_design_radius = oc.env_float("MSOPT_OLED_RADIAL_RADIUS", 0.5 * min(G.design_s[0], G.design_s[1]))
radial_design_grids = oc.env_int("MSOPT_OLED_RADIAL_GRIDS", int(round(radial_design_radius * G.resolution)) + 1)
mapping = None
x0 = None
dJ_0 = np.zeros(G.design_cells)


def ensure_mapping():
    global mapping, x0
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
        x0 = G.grating_initial_density * np.ones(mapping.parameter_count)
    return mapping


def add_channel_dipoles(sim, channel):
    """Install a channel's emitter(s).

    A channel carrying "dipoles" is a grid fired TOGETHER: every dipole goes into
    one Lumerical group called "source" so the solver treats them as a single
    coherent excitation (OLED_new's arrangement). Otherwise it is the single
    representative dipole.
    """
    pts = channel.get("dipoles")
    if not pts:
        oc.add_dipole(G, sim, channel["dipole_x"], channel["dipole_y"],
                      channel["dipole_z"], channel["polarization"])
        return 1
    for k, (x, y, z) in enumerate(pts):
        oc.add_dipole(G, sim, x, y, z, channel["polarization"],
                      name=f"opt_dipole_{k}", enabled=True, group_name="source")
    return len(pts)


def build_optimization_problem():
    target_info = oc.build_target_orders(G)
    oc.save_target_orders(G, target_info)
    sims, opts, histories = [None] * N_fom, [None] * N_fom, [[] for _ in range(N_fom)]

    for idx, channel in enumerate(target_channels):
        sim = sims[idx] = oc.make_sim(G, [G.Sx, G.Sy, G.Sz])
        add_channel_dipoles(sim, channel)
        sim.add_monitor(G.target_monitor_name, G.target_monitor_c, G.target_monitor_s)

        if oc.env_flag("MSOPT_OLED_BULK_NORMALIZATION", "1"):
            oc.set_fdtd_background_index(sim.fdtd, G.bulk_reference_index)
            sim.run(name=f"bulk_reference_{idx}", save=True)
            oc.load_run_results(sim)
            bulk_power = oc.read_dipole_power(sim.fdtd, oc.source_freqs(G, sim)) or oc.read_source_power(sim.fdtd, oc.source_freqs(G, sim)) or G.channel_power_floor
            channel["bulk_reference_power"] = max(float(bulk_power), G.channel_power_floor)
            grid = sim.fdtd.getresult(G.target_monitor_name, "E")
            channel["angular_target"] = oc.angular_target_from_monitor_grid(G, grid["x"], grid["y"], target_info)
            if idx == 0:
                oc.save_angular_target_preview(G, channel["angular_target"])
            sim.fdtd.switchtolayout()
            oc.set_fdtd_background_index(sim.fdtd, G.background_index)
            print(f"[bulk] channel {idx}: n={G.bulk_reference_index:.4g}, power={channel['bulk_reference_power']:.6e}")
        else:
            channel["bulk_reference_power"] = np.nan
            channel["angular_target"] = None

        oc.add_stack(G, sim)
        sim.add_design_grid("design", G.design_c, G.design_s, G.design_high_index, G.design_low_index, G.design_grids, G.grating_initial_density * np.ones(G.design_grids), float(np.mean(G.visible_wavelengths)))
        sim.add_design_monitor()
        channel["design_incident_reference_power"] = oc.measure_design_incident_reference(G, channel)

        def J(Ex, Ey, Hx, Hy, channel_idx=idx, channel=channel):
            if channel["angular_target"] is None:
                channel["angular_target"] = oc.angular_target_from_monitor_grid(G, opts[channel_idx].xg, opts[channel_idx].yg, target_info)
            terms = oc.channel_fom_terms(G, Ex, Ey, Hx, Hy, channel)
            # Objective: FoM = throughput * match**ratio_emphasis (oc.angular_powers).
            fom = npa.clip(terms["fom"], 0.0, G.score_cap)
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
                if G.opt_emission_plot:
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
            FoM_size=G.target_monitor_s,
            FoM_center=G.target_monitor_c,
            adj_fwd=False,
            opt_idx=idx,
            broadband_adjoint=True,
        )

        def hook(problem, channel=channel):
            freqs = np.asarray(getattr(problem, "src_freqs", []), dtype=float).reshape(-1)
            freqs = freqs if freqs.size else oc.source_freqs(G, problem.sim)
            source_power = oc.read_source_power(problem.sim.fdtd, freqs)
            current_power = oc.read_dipole_power(problem.sim.fdtd, freqs)
            # Normalize by the fixed reference power incident on the design region
            # (measured once with the design + superstrate index-matched, so no
            # back-reflection). Falls back to bulk/source/dipole power if unavailable.
            norm_power, source = oc.choose_norm_power(G, channel.get("design_incident_reference_power"), channel.get("bulk_reference_power"), source_power, current_power)
            T = oc.read_transmission(problem.sim.fdtd, G.target_monitor_name)
            channel["last_normalization_power"] = norm_power
            channel["last_normalization_power_source"] = source
            channel["last_source_power"] = oc.valid_power(source_power) or np.nan
            channel["last_current_dipole_power"] = oc.valid_power(current_power) or np.nan
            channel["last_top_flux_sign"] = 1.0 if T >= 0.0 else -1.0
            channel["last_top_monitor_transmission"] = T
            # (The legacy diagnostic-only raw-flux calibration is gone: the shared
            # angular-target dicts no longer carry monitor_cell_area.)

        opt.forward_result_hook = hook
        print(f"[setup] channel {idx}: pos=({channel['dipole_x']:.3f},{channel['dipole_y']:.3f},{channel['dipole_z']:.3f}), pol={channel['polarization']}")

    return sims, opts, histories


def iter_plot(X, vals):
    # Per-iteration snapshots, exactly what the legacy adjoint_loop inlined.
    try:
        path = oc.save_current_design_sections(G, X, vals, mapping=mapping)
        print(f"[outcoupling] saved temporary design section: {path}")
    except Exception as exc:
        print(f"[outcoupling] skipped temporary design section: {exc}")
    if G.opt_emission_plot:
        try:
            epath = oc.save_optimization_emission_plot(G, target_channels=target_channels)
            if epath:
                print(f"[outcoupling] saved current emission plot: {epath}")
        except Exception as exc:
            print(f"[outcoupling] skipped emission plot: {exc}")


def main():
    if oc.env_flag("MSOPT_OLED_SESSION_TEST", "0"):
        oc.session_test_banner(G, N_fom)
        return

    start = time.time()
    post_only = oc.env_flag("MSOPT_OLED_POSTPROCESS_ONLY", "0")
    ensure_mapping()
    if not post_only:
        if ag_jacobian is None:
            raise RuntimeError("autograd is required for optimization. Install autograd or run with MSOPT_OLED_POSTPROCESS_ONLY=1.")
        sims, opts, histories = build_optimization_problem()
        print(
            f"[setup] period={G.Sx:g}x{G.Sy:g} um, design={G.design_grids}, N_fom={N_fom}, "
            f"bulk_n={G.bulk_reference_index:g}, opt_bc={G.bc_xy}"
        )
        optimizer = ms.Opt_MS2.OPT_Ms(
            x0, dJ_0,
            design_dir=G.design_dir,
            local_best_dir=G.local_dir,
            Born_k=99,
            Initial_LR=0.2,
            Raw=False,
        )
        optimizer.flag = True
        optimizer(mapping, N_fom, oc.adjoint_loop(G, opts, iter_plot_fn=iter_plot, combine=combine_fom))
        oc.save_result_plots(optimizer, G.design_dir)

    if oc.env_flag("MSOPT_OLED_POSTPROCESS", "1"):
        # MSOPT_OLED_POSTPROCESS_DESIGN lets the postprocess re-run on a design from a
        # previous run (e.g. a Done/ result) instead of this run's lastdesign.txt.
        design_path = os.environ.get("MSOPT_OLED_POSTPROCESS_DESIGN", "").strip()
        if not (design_path and os.path.exists(design_path)):
            design_path = os.path.join(G.design_dir, "lastdesign.txt")
        # A PLANAR run characterizes the bare stack, so it needs no design at
        # all: the design region is filled with the low index either way. Do not
        # let the "no design file" check abort it.
        planar_only = oc.planar_requested() and not os.path.exists(design_path)
        if planar_only:
            print("[postprocess] PLANAR stack characterization (no design required)")
            oc.run_postprocess(G, np.zeros(int(np.prod(G.design_grids)), dtype=float), mapping=None)
        elif os.path.exists(design_path):
            print(f"[postprocess] using design: {design_path}")
            oc.run_postprocess(G, np.loadtxt(design_path), mapping=mapping)
        else:
            print(f"[postprocess] skipped: {design_path} not found")
    print(f"Runtime setup time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
