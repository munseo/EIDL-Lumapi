"""
===============================================================
Step 2b — Meep coherence check (case1 / case2a / case2b / case3, original 36-dipole grid)
===============================================================

This is a focused follow-up to `step2_coherence_comparison.py`.

What it does
------------
1. Builds the same bare-LED Meep geometry as the older step-2 script.
2. Uses the exact same source ensemble as the original coherence test:
      6x6 endpoint grid, Ex-only, 36 dipoles total
3. Reuses an existing case1 incoherent reference JSON by default.
4. Re-runs:
      - Case 2a: simultaneous coherent emission with identical phase, phase = 0
      - Case 2b: simultaneous coherent emission with one random phase set
      - Case 3: simultaneous coherent emission with random phases, ensemble-averaged up to 20 trials
5. Saves:
      - dipole layout image
      - per-case radiation-pattern figures
      - per-trial radiation-pattern figures for case 3
      - JSON summary

Interpretation
--------------
Same-phase coherent and single random-phase coherent realizations include dipole-dipole cross terms.
Random-phase trial averaging should trend toward the incoherent source-wise sum, but with only
20 trials it should be interpreted as a trend check rather than a strict replacement for the
source-wise incoherent reference.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import meep as mp
import numpy as np
from matplotlib import pyplot as plt

mp.verbosity(1)


IS_MASTER = True
if hasattr(mp, "am_master"):
    try:
        IS_MASTER = mp.am_master()
    except Exception:
        IS_MASTER = True


def mprint(*a, **k):
    if IS_MASTER:
        print(*a, **k)


def save_json(path: Path, obj):
    if not IS_MASTER:
        return
    with open(path, "w") as f:
        json.dump(
            obj,
            f,
            indent=2,
            default=lambda x: (
                x.tolist()
                if isinstance(x, np.ndarray)
                else (float(x) if isinstance(x, np.floating) else None)
            ),
        )


# ================================================================
# 1. Parameters
# ================================================================
resolution = 40
target_wvl = 0.527
fcen = 1.0 / target_wvl
fwidth = 0.05 * fcen

n_GaN = 2.40
n_Sub = 2.49
n_Poly = 1.50

channel_w, channel_l = 1.05, 1.05
p_gan_h, mqw_h, n_gan_h = 0.20, 0.20, 0.70
sub_h = 0.20

pml_th = 0.50
PIXEL = 1.0 / resolution
mon_pml_gap_px = 2
mon_pml_gap = mon_pml_gap_px * PIXEL

CELL_XY = 7.0

gan_total_h = n_gan_h + mqw_h + p_gan_h
air_pad_top = max(4 * target_wvl, 2.0)
sz_raw = pml_th + sub_h + gan_total_h + air_pad_top + pml_th
sz = round(sz_raw * resolution) / resolution

z_bot = -sz / 2
z_sub_bot = z_bot + pml_th
z_sub_top = z_sub_bot + sub_h
z_gan_c = z_sub_top + gan_total_h / 2
z_mqw_c = z_sub_top + n_gan_h + mqw_h / 2
z_gan_top = z_sub_top + gan_total_h
src_z = z_mqw_c

z_top_mon = sz / 2 - pml_th - mon_pml_gap
ap_half = CELL_XY / 2 - pml_th - mon_pml_gap

SRC_BOX_INSET_PX = 2
src_box_inset = SRC_BOX_INSET_PX * PIXEL
src_box_half = channel_w / 2 - src_box_inset
src_box_z_lo = src_z - mqw_h / 2 + src_box_inset
src_box_z_hi = src_z + mqw_h / 2 - src_box_inset

AIR_FF_LAT_BOT_MARGIN = 0.5 * target_wvl
z_ff_lat_bot = z_gan_top + AIR_FF_LAT_BOT_MARGIN

SIM_DFT_TOL = 1e-4
SIM_MAX_TIME = 2000
EPS_RATIO = 1e-12
FF_R = 1e6
FF_N_THETA = 19
FF_N_PHI = 36

# Match the original step2_coherence_comparison.py source ensemble exactly.
EDGE_INSET = 2 * PIXEL
SOURCE_GRID_N = 6
CASE3_RANDOM_SEED = 1234
CASE3_TRIAL_COUNTS = (1, 2, 5, 10, 20)
MAX_TRIALS = max(CASE3_TRIAL_COUNTS)


# ================================================================
# 2. Geometry
# ================================================================
def make_geometry():
    return [
        mp.Block(
            mp.Vector3(mp.inf, mp.inf, pml_th + sub_h),
            center=mp.Vector3(0, 0, z_bot + (pml_th + sub_h) / 2),
            material=mp.Medium(index=n_Sub),
        ),
        mp.Block(
            mp.Vector3(mp.inf, mp.inf, gan_total_h),
            center=mp.Vector3(0, 0, z_gan_c),
            material=mp.Medium(index=n_Poly),
        ),
        mp.Block(
            mp.Vector3(channel_w, channel_l, gan_total_h),
            center=mp.Vector3(0, 0, z_gan_c),
            material=mp.Medium(index=n_GaN),
        ),
    ]


# ================================================================
# 3. Monitors
# ================================================================
def add_monitors(sim):
    sxy = 2 * src_box_half
    h = src_box_z_hi - src_box_z_lo
    zc = 0.5 * (src_box_z_lo + src_box_z_hi)
    src_box = sim.add_flux(
        fcen,
        0,
        1,
        mp.FluxRegion(
            mp.Vector3(0, 0, src_box_z_hi),
            mp.Vector3(sxy, sxy, 0),
            direction=mp.Z,
            weight=+1.0,
        ),
        mp.FluxRegion(
            mp.Vector3(0, 0, src_box_z_lo),
            mp.Vector3(sxy, sxy, 0),
            direction=mp.Z,
            weight=-1.0,
        ),
        mp.FluxRegion(
            mp.Vector3(+src_box_half, 0, zc),
            mp.Vector3(0, sxy, h),
            direction=mp.X,
            weight=+1.0,
        ),
        mp.FluxRegion(
            mp.Vector3(-src_box_half, 0, zc),
            mp.Vector3(0, sxy, h),
            direction=mp.X,
            weight=-1.0,
        ),
        mp.FluxRegion(
            mp.Vector3(0, +src_box_half, zc),
            mp.Vector3(sxy, 0, h),
            direction=mp.Y,
            weight=+1.0,
        ),
        mp.FluxRegion(
            mp.Vector3(0, -src_box_half, zc),
            mp.Vector3(sxy, 0, h),
            direction=mp.Y,
            weight=-1.0,
        ),
    )

    ap_sxy = 2 * ap_half
    ff_lat_h = z_top_mon - z_ff_lat_bot
    ff_lat_zc = 0.5 * (z_top_mon + z_ff_lat_bot)
    n2f = sim.add_near2far(
        fcen,
        0,
        1,
        mp.Near2FarRegion(
            mp.Vector3(0, 0, z_top_mon),
            mp.Vector3(ap_sxy, ap_sxy, 0),
            direction=mp.Z,
            weight=+1.0,
        ),
        mp.Near2FarRegion(
            mp.Vector3(+ap_half, 0, ff_lat_zc),
            mp.Vector3(0, ap_sxy, ff_lat_h),
            direction=mp.X,
            weight=+1.0,
        ),
        mp.Near2FarRegion(
            mp.Vector3(-ap_half, 0, ff_lat_zc),
            mp.Vector3(0, ap_sxy, ff_lat_h),
            direction=mp.X,
            weight=-1.0,
        ),
        mp.Near2FarRegion(
            mp.Vector3(0, +ap_half, ff_lat_zc),
            mp.Vector3(ap_sxy, 0, ff_lat_h),
            direction=mp.Y,
            weight=+1.0,
        ),
        mp.Near2FarRegion(
            mp.Vector3(0, -ap_half, ff_lat_zc),
            mp.Vector3(ap_sxy, 0, ff_lat_h),
            direction=mp.Y,
            weight=-1.0,
        ),
    )
    return src_box, n2f


# ================================================================
# 4. Far-field integral
# ================================================================
def integrate_upper_hemisphere(sim, n2f, R=FF_R, n_theta=FF_N_THETA, n_phi=FF_N_PHI):
    theta = np.linspace(0.0, np.pi / 2, n_theta)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    dth = theta[1] - theta[0]
    dph = phi[1] - phi[0]
    Sr = np.zeros((n_theta, n_phi))
    for it, th in enumerate(theta):
        sth, cth = np.sin(th), np.cos(th)
        for ip, ph in enumerate(phi):
            cph, sph = np.cos(ph), np.sin(ph)
            r_vec = mp.Vector3(R * sth * cph, R * sth * sph, R * cth)
            ff = sim.get_farfield(n2f, r_vec)
            Ex, Ey, Ez = ff[0], ff[1], ff[2]
            Hx, Hy, Hz = ff[3], ff[4], ff[5]
            Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
            Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
            Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
            Sr[it, ip] = Sx * sth * cph + Sy * sth * sph + Sz * cth
    w_th = np.full(n_theta, dth)
    w_th[0] *= 0.5
    w_th[-1] *= 0.5
    integrand = Sr * R**2 * np.sin(theta)[:, None]
    P_up = float(np.sum(integrand * w_th[:, None]) * dph)
    return P_up, Sr, theta, phi


# ================================================================
# 5. Source specifications
# ================================================================
def make_source_entry(x_centered, y_centered, component, pol_label, label):
    return {
        "x_centered": float(x_centered),
        "y_centered": float(y_centered),
        "component": component,
        "pol_label": pol_label,
        "label": label,
    }


def make_original_36src_specs():
    half_range_x = channel_w / 2 - EDGE_INSET
    half_range_y = channel_l / 2 - EDGE_INSET
    x_coords = np.linspace(-half_range_x, half_range_x, SOURCE_GRID_N)
    y_coords = np.linspace(-half_range_y, half_range_y, SOURCE_GRID_N)
    specs = []
    idx = 0
    for x in x_coords:
        for y in y_coords:
            idx += 1
            specs.append(make_source_entry(x, y, mp.Ex, "Ex", f"Ex@{idx:02d}"))
    return specs


SOURCE_SPECS = make_original_36src_specs()


CASE2_RANDOM_SEED = 0


def make_case2_random_phase(source_specs, seed=CASE2_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=len(source_specs))


def build_sources(source_specs, phases_rad=None):
    if phases_rad is None:
        phases_rad = np.zeros(len(source_specs))
    sources = []
    for src, phase in zip(source_specs, phases_rad):
        sources.append(
            mp.Source(
                src=mp.GaussianSource(fcen, fwidth=fwidth),
                component=src["component"],
                center=mp.Vector3(src["x_centered"], src["y_centered"], src_z),
                amplitude=np.exp(1j * phase),
            )
        )
    return sources


# ================================================================
# 6. Plot helpers
# ================================================================
def save_dipole_layout(outdir: Path):
    if not IS_MASTER:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    for src in SOURCE_SPECS:
        color = "tab:blue"
        marker = "o"
        ax.scatter(src["x_centered"], src["y_centered"], c=color, marker=marker, s=85)
        ax.text(src["x_centered"] + 0.010, src["y_centered"] + 0.010, src["label"], fontsize=6)
    ax.add_patch(
        plt.Rectangle(
            (-channel_w / 2, -channel_l / 2),
            channel_w,
            channel_l,
            fill=False,
            linewidth=1.5,
            linestyle="--",
            edgecolor="black",
        )
    )
    ax.set_title("Original step2 layout: 36 Ex dipoles on 6x6 endpoint grid")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "dipole_layout.png", dpi=160)
    plt.close(fig)


def save_radiation_pattern_figure(path: Path, Sr, theta, phi, title: str):
    if not IS_MASTER:
        return
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    I_theta = Sr.mean(axis=1)
    I_norm = I_theta / max(np.max(np.abs(I_theta)), 1e-30)

    fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1)
    im = ax0.imshow(
        Sr,
        origin="lower",
        aspect="auto",
        extent=[phi_deg[0], phi_deg[-1] + (phi_deg[1] - phi_deg[0]), theta_deg[0], theta_deg[-1]],
    )
    ax0.set_title("Upper-hemisphere radial Poynting")
    ax0.set_xlabel("phi [deg]")
    ax0.set_ylabel("theta [deg]")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = fig.add_subplot(1, 2, 2, projection="polar")
    theta_full = np.concatenate([-theta[::-1], theta[1:]])
    I_full = np.concatenate([I_norm[::-1], I_norm[1:]])
    ax1.plot(theta_full, I_full, lw=2.0)
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    ax1.set_thetamin(-90)
    ax1.set_thetamax(90)
    ax1.set_title("Azimuth-averaged radiation cut")

    fig.suptitle(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ================================================================
# 7. Single run helper
# ================================================================
def run_sim(sources):
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=mp.Vector3(CELL_XY, CELL_XY, sz),
        boundary_layers=[mp.PML(pml_th)],
        geometry=make_geometry(),
        sources=sources,
        force_complex_fields=True,
        eps_averaging=True,
    )
    src_box, n2f = add_monitors(sim)

    t0 = time.time()
    sim.run(until_after_sources=mp.stop_when_dft_decayed(SIM_DFT_TOL, 0, SIM_MAX_TIME))
    t_fdtd = time.time() - t0

    P_src = float(mp.get_fluxes(src_box)[0])

    t0 = time.time()
    P_up_ff, Sr, theta, phi = integrate_upper_hemisphere(sim, n2f)
    t_ff = time.time() - t0

    sim.reset_meep()
    del sim

    return {
        "P_src": P_src,
        "P_up_ff": P_up_ff,
        "eta_ff": P_up_ff / max(P_src, EPS_RATIO),
        "Sr": Sr,
        "theta": theta,
        "phi": phi,
        "t_fdtd": t_fdtd,
        "t_ff": t_ff,
    }


# ================================================================
# 8. Incoherent reference
# ================================================================
def run_incoherent_reference(outdir: Path):
    mprint(f"\n{'=' * 72}")
    mprint("Reference — source-wise incoherent average (36 individual runs)")
    mprint(f"{'=' * 72}")

    results = []
    for idx, src in enumerate(SOURCE_SPECS, start=1):
        mprint(f"  [{idx:02d}/{len(SOURCE_SPECS):02d}] {src['label']} ...", end="")
        r = run_sim(build_sources([src], phases_rad=np.zeros(1)))
        mprint(f"  {r['t_fdtd']:.0f}s  eta_ff={r['eta_ff'] * 100:.3f}%")
        results.append(r)

    P_src_avg = float(np.mean([r["P_src"] for r in results]))
    P_up_ff_avg = float(np.mean([r["P_up_ff"] for r in results]))
    eta_ff = P_up_ff_avg / max(P_src_avg, EPS_RATIO)
    Sr_avg = np.mean([r["Sr"] for r in results], axis=0)
    theta = results[0]["theta"]
    phi = results[0]["phi"]

    save_radiation_pattern_figure(
        outdir / "reference_incoherent_radiation.png",
        Sr_avg,
        theta,
        phi,
        "Incoherent reference radiation pattern",
    )

    return {
        "eta_ff": eta_ff,
        "P_src": P_src_avg,
        "P_up_ff": P_up_ff_avg,
        "Sr": Sr_avg,
        "theta": theta,
        "phi": phi,
        "n_runs": len(results),
        "per_source": [
            {
                "label": src["label"],
                "x_centered": src["x_centered"],
                "y_centered": src["y_centered"],
                "pol_label": src["pol_label"],
                "eta_ff": r["eta_ff"],
                "P_src": r["P_src"],
                "P_up_ff": r["P_up_ff"],
            }
            for src, r in zip(SOURCE_SPECS, results)
        ],
    }


# ================================================================
# 9. Case 2 — coherent simultaneous emission
# ================================================================
def summarize_against_reference(r, reference):
    return {
        "P_src": r["P_src"],
        "P_up_ff": r["P_up_ff"],
        "eta_ff": r["eta_ff"],
        "delta_eta_vs_incoherent": r["eta_ff"] - reference["eta_ff"],
        "rel_eta_error_vs_incoherent": abs(r["eta_ff"] - reference["eta_ff"]) / max(abs(reference["eta_ff"]), EPS_RATIO),
        "t_fdtd": r["t_fdtd"],
        "t_ff": r["t_ff"],
    }


def run_case2_same_phase(outdir: Path, reference):
    mprint(f"\n{'=' * 72}")
    mprint("Case 2a — same-phase coherent simultaneous emission")
    mprint(f"{'=' * 72}")

    phases = np.zeros(len(SOURCE_SPECS))
    mprint("  all phases = 0 ...", end="")
    r = run_sim(build_sources(SOURCE_SPECS, phases_rad=phases))
    mprint(f"  {r['t_fdtd']:.0f}s  eta_ff={r['eta_ff'] * 100:.3f}%")

    png_path = outdir / "case2a_same_phase_radiation.png"
    npz_path = outdir / "case2a_same_phase_radiation.npz"
    save_radiation_pattern_figure(
        png_path,
        r["Sr"],
        r["theta"],
        r["phi"],
        "Case 2a — same-phase coherent realization",
    )
    if IS_MASTER:
        np.savez(
            npz_path,
            Sr=r["Sr"],
            theta=r["theta"],
            phi=r["phi"],
            phases_rad=phases,
        )

    out = summarize_against_reference(r, reference)
    out.update({
        "description": "simultaneous coherent emission, all phases are zero",
        "radiation_png": str(png_path),
        "radiation_npz": str(npz_path),
        "phases_rad": phases.tolist(),
    })
    return out


def run_case2_random_single(outdir: Path, reference):
    mprint(f"\n{'=' * 72}")
    mprint("Case 2b — one random-phase coherent simultaneous emission")
    mprint(f"{'=' * 72}")

    phases = make_case2_random_phase(SOURCE_SPECS, seed=CASE2_RANDOM_SEED)
    mprint(f"  random_seed{CASE2_RANDOM_SEED} ...", end="")
    r = run_sim(build_sources(SOURCE_SPECS, phases_rad=phases))
    mprint(f"  {r['t_fdtd']:.0f}s  eta_ff={r['eta_ff'] * 100:.3f}%")

    png_path = outdir / "case2b_random_single_radiation.png"
    npz_path = outdir / "case2b_random_single_radiation.npz"
    save_radiation_pattern_figure(
        png_path,
        r["Sr"],
        r["theta"],
        r["phi"],
        f"Case 2b — single random-phase realization (seed {CASE2_RANDOM_SEED})",
    )
    if IS_MASTER:
        np.savez(
            npz_path,
            Sr=r["Sr"],
            theta=r["theta"],
            phi=r["phi"],
            phases_rad=phases,
        )

    out = summarize_against_reference(r, reference)
    out.update({
        "description": "simultaneous coherent emission, one random phase draw",
        "seed": CASE2_RANDOM_SEED,
        "radiation_png": str(png_path),
        "radiation_npz": str(npz_path),
        "phases_rad": phases.tolist(),
    })
    return out


# ================================================================
# 10. Case 3 — random-phase ensemble average
# ================================================================
def run_case3(outdir: Path, reference):
    mprint(f"\n{'=' * 72}")
    mprint(f"Case 3 — random-phase coherent trials (N up to {MAX_TRIALS})")
    mprint(f"{'=' * 72}")

    trial_dir = outdir / "case3_trial_fields"
    if IS_MASTER:
        trial_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(CASE3_RANDOM_SEED)
    all_phases = rng.uniform(0.0, 2.0 * np.pi, size=(MAX_TRIALS, len(SOURCE_SPECS)))

    trial_results = []
    P_src_cum = 0.0
    P_up_ff_cum = 0.0
    Sr_cum = None
    checkpoints = []

    for trial_idx in range(MAX_TRIALS):
        phases = all_phases[trial_idx]
        mprint(f"  trial {trial_idx + 1:02d}/{MAX_TRIALS:02d} ...", end="")
        r = run_sim(build_sources(SOURCE_SPECS, phases_rad=phases))
        mprint(f"  {r['t_fdtd']:.0f}s  eta_ff={r['eta_ff'] * 100:.3f}%")

        png_path = trial_dir / f"trial_{trial_idx + 1:03d}_radiation_field.png"
        npz_path = trial_dir / f"trial_{trial_idx + 1:03d}_radiation_field.npz"
        save_radiation_pattern_figure(
            png_path,
            r["Sr"],
            r["theta"],
            r["phi"],
            f"Case 3 trial {trial_idx + 1}",
        )
        if IS_MASTER:
            np.savez(
                npz_path,
                Sr=r["Sr"],
                theta=r["theta"],
                phi=r["phi"],
                phases_rad=phases,
            )

        trial_results.append(
            {
                "trial": trial_idx + 1,
                "eta_ff": r["eta_ff"],
                "P_src": r["P_src"],
                "P_up_ff": r["P_up_ff"],
                "t_fdtd": r["t_fdtd"],
                "t_ff": r["t_ff"],
                "phases_rad": phases.tolist(),
                "radiation_png": str(png_path),
                "radiation_npz": str(npz_path),
            }
        )

        P_src_cum += r["P_src"]
        P_up_ff_cum += r["P_up_ff"]
        Sr_cum = r["Sr"].copy() if Sr_cum is None else (Sr_cum + r["Sr"])

        n_done = trial_idx + 1
        if n_done in CASE3_TRIAL_COUNTS:
            P_src_avg = P_src_cum / n_done
            P_up_ff_avg = P_up_ff_cum / n_done
            eta_ff_avg = P_up_ff_avg / max(P_src_avg, EPS_RATIO)
            Sr_avg = Sr_cum / n_done
            cp_png = outdir / f"case3_checkpoint_{n_done:03d}_radiation.png"
            save_radiation_pattern_figure(
                cp_png,
                Sr_avg,
                r["theta"],
                r["phi"],
                f"Case 3 ensemble average after {n_done} trials",
            )
            checkpoints.append(
                {
                    "n_trials": n_done,
                    "P_src": P_src_avg,
                    "P_up_ff": P_up_ff_avg,
                    "eta_ff": eta_ff_avg,
                    "delta_eta_vs_incoherent": eta_ff_avg - reference["eta_ff"],
                    "rel_eta_error_vs_incoherent": abs(eta_ff_avg - reference["eta_ff"]) / max(abs(reference["eta_ff"]), EPS_RATIO),
                    "radiation_png": str(cp_png),
                }
            )

    return {
        "checkpoints": checkpoints,
        "per_trial": trial_results,
    }


# ================================================================
# 11. Summary plots
# ================================================================
def rel_diff(a, b):
    return abs(a - b) / max(abs(a), abs(b), EPS_RATIO)


def save_summary_plot(outdir: Path, reference, case2a, case2b, case3):
    if not IS_MASTER:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Eta comparison
    ax = axes[0, 0]
    labels = ["incoh_ref", "same_phase", "rand1"] + [f"avg_{cp['n_trials']}tr" for cp in case3["checkpoints"]]
    vals = [
        reference["eta_ff"] * 100.0,
        case2a["eta_ff"] * 100.0,
        case2b["eta_ff"] * 100.0,
    ] + [cp["eta_ff"] * 100.0 for cp in case3["checkpoints"]]
    ax.bar(np.arange(len(vals)), vals, color=["tab:blue", "tab:red", "tab:orange"] + ["tab:green"] * len(case3["checkpoints"]))
    ax.set_ylabel("eta_ff [%]")
    ax.set_title("Coherence comparison")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(alpha=0.3, axis="y")

    # Case3 convergence
    ax = axes[0, 1]
    trials = [cp["n_trials"] for cp in case3["checkpoints"]]
    rels = [cp["rel_eta_error_vs_incoherent"] * 100.0 for cp in case3["checkpoints"]]
    ax.plot(trials, rels, "o-", lw=2)
    ax.axhline(1.0, color="green", ls="--", lw=1, label="1%")
    ax.axhline(5.0, color="orange", ls="--", lw=1, label="5%")
    ax.set_xlabel("random-phase trial count")
    ax.set_ylabel("relative diff vs incoherent [%]")
    ax.set_title("Case 3 convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Case2 coherent mismatch
    ax = axes[1, 0]
    mismatch_vals = [
        case2a["rel_eta_error_vs_incoherent"] * 100.0,
        case2b["rel_eta_error_vs_incoherent"] * 100.0,
    ]
    ax.bar([0, 1], mismatch_vals, color=["tab:red", "tab:orange"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["same_phase", f"rand1_seed{case2b['seed']}"], rotation=20, ha="right")
    ax.set_ylabel("relative diff vs incoherent [%]")
    ax.set_title("Case 2 coherent mismatch")
    ax.grid(alpha=0.3, axis="y")

    # Case3 per-trial scatter
    ax = axes[1, 1]
    per_trial = case3["per_trial"]
    trial_ids = [r["trial"] for r in per_trial]
    trial_eta = [r["eta_ff"] * 100.0 for r in per_trial]
    running_avg = np.cumsum([r["eta_ff"] for r in per_trial]) / np.arange(1, len(per_trial) + 1)
    ax.scatter(trial_ids, trial_eta, s=25, alpha=0.7, label="single coherent realization")
    ax.plot(trial_ids, running_avg * 100.0, lw=2, label="running average")
    ax.axhline(reference["eta_ff"] * 100.0, color="tab:blue", ls="--", lw=1.5, label="incoherent ref")
    ax.set_xlabel("trial")
    ax.set_ylabel("eta_ff [%]")
    ax.set_title("Case 3 trial-by-trial behavior")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Meep coherence check — coherent cases and random-phase averaging vs incoherent reference")
    fig.savefig(outdir / "coherence_case23_summary.png", dpi=160)
    plt.close(fig)


# ================================================================
# 12. Reference loading
# ================================================================
def load_reference_from_existing_json(json_path: Path):
    with open(json_path, "r") as f:
        payload = json.load(f)

    case1 = payload["case1_incoherent"]
    return {
        "eta_ff": float(case1["eta_ff"]),
        "P_src": float(case1["P_src"]),
        "P_up_ff": float(case1["P_up_ff"]),
        "n_runs": int(case1.get("n_runs", 0)),
        "per_source": case1.get("per_position", []),
        "loaded_from_json": str(json_path),
        "is_external_reference": True,
    }


# ================================================================
# 13. Main
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Optional output directory. Default: timestamped folder next to this script.",
    )
    parser.add_argument(
        "--case1-json",
        type=str,
        default="/home/user/wonjung/3dpol/step2_coherence_20260417_233549/results.json",
        help=(
            "Optional existing step2 results.json to reuse case1_incoherent and skip "
            "new incoherent reference runs. Pass empty string to force recomputation."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_outdir = Path(__file__).resolve().parent / f"step2_coherence_case2abc_36src_{timestamp}"
    outdir = Path(args.outdir) if args.outdir else default_outdir
    if IS_MASTER:
        outdir.mkdir(parents=True, exist_ok=True)

    mprint(f"\n{'=' * 78}")
    mprint("Step 2b — Meep coherence check: case2a/case2b/case3 (original 36-dipole grid)")
    mprint(f"{'=' * 78}")
    mprint(f"  output dir   = {outdir}")
    mprint(f"  resolution   = {resolution}")
    mprint(f"  source set   = 6x6 endpoint grid, Ex-only = {len(SOURCE_SPECS)}")
    mprint(f"  case2a      = same-phase coherent simultaneous emission")
    mprint(f"  case2b      = one random-phase coherent simultaneous emission")
    mprint(f"  case3 trials = {CASE3_TRIAL_COUNTS}  # trend check only, max 20")
    mprint("  note         = this matches the original step2 coherence source layout")
    if args.case1_json:
        mprint(f"  case1 reuse  = {args.case1_json}")
    mprint(f"{'=' * 78}")

    save_dipole_layout(outdir)
    if args.case1_json:
        reference = load_reference_from_existing_json(Path(args.case1_json))
        mprint("\n[Reference]")
        mprint("  Reusing existing case1_incoherent from JSON.")
        mprint(f"  eta_ff = {reference['eta_ff'] * 100:.3f}%")
        mprint(f"  P_src  = {reference['P_src']:.4e}")
        mprint(f"  P_up   = {reference['P_up_ff']:.4e}")
        mprint("  Reference source layout is now matched to the original step2 setup.")
    else:
        reference = run_incoherent_reference(outdir)
    case2a = run_case2_same_phase(outdir, reference)
    case2b = run_case2_random_single(outdir, reference)
    case3 = run_case3(outdir, reference)
    save_summary_plot(outdir, reference, case2a, case2b, case3)

    payload = {
        "version": "step2b — coherence case2a/case2b/case3 original 36src",
        "setup": {
            "resolution": resolution,
            "cell_xy": CELL_XY,
            "channel_w": channel_w,
            "channel_l": channel_l,
            "source_layout": "6x6 endpoint grid, Ex-only (matches original step2)",
            "num_sources": len(SOURCE_SPECS),
            "case3_trial_counts": list(CASE3_TRIAL_COUNTS),
            "case3_random_seed": CASE3_RANDOM_SEED,
            "dipole_layout_png": str(outdir / "dipole_layout.png"),
            "case1_reference_json": args.case1_json if args.case1_json else None,
            "source_specs": SOURCE_SPECS,
        },
        "reference_incoherent": {
            "eta_ff": reference["eta_ff"],
            "P_src": reference["P_src"],
            "P_up_ff": reference["P_up_ff"],
            "n_runs": reference["n_runs"],
            "per_source": reference["per_source"],
            "radiation_png": (
                str(outdir / "reference_incoherent_radiation.png")
                if not reference.get("is_external_reference", False)
                else None
            ),
            "loaded_from_json": reference.get("loaded_from_json"),
            "is_external_reference": reference.get("is_external_reference", False),
        },
        "case2a_same_phase": case2a,
        "case2b_random_single": case2b,
        "case3_random_average": case3,
        "comparison": {
            "case2a_same_phase_rel_error_vs_incoherent": case2a["rel_eta_error_vs_incoherent"],
            "case2b_random_single_rel_error_vs_incoherent": case2b["rel_eta_error_vs_incoherent"],
            "case3_final_20trial_rel_error_vs_incoherent": case3["checkpoints"][-1]["rel_eta_error_vs_incoherent"],
        },
    }
    save_json(outdir / "results.json", payload)

    mprint(f"\nSaved -> {outdir}")
    mprint("  dipole_layout.png")
    if not reference.get("is_external_reference", False):
        mprint("  reference_incoherent_radiation.png")
    mprint("  case2a_same_phase_radiation.(png|npz)")
    mprint("  case2b_random_single_radiation.(png|npz)")
    mprint("  case3_trial_fields/trial_*_radiation_field.(png|npz)")
    mprint("  coherence_case23_summary.png")
    mprint("  results.json")


if __name__ == "__main__":
    main()
