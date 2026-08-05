"""Regenerate a finished postprocess run's TARGET-comparison products against the
design's OWN optimization target -- without re-running any simulation.

Why this exists
---------------
A postprocess driven by a generic script carries that script's
MSOPT_OLED_TARGET_* defaults, so its "target" columns/curves describe a goal the
design was never optimized for.  The real goal lives next to the design as
A/OLED_angular_target.npz (ring angles, normalized target power profile,
in-range mask).  This tool re-derives the target-dependent products from data
already stored in the run folder:

    OLED_postprocess_thetaphi_map.npz     -> achieved ring shares
    OLED_postprocess_3x3_angle_profile.txt -> radiance curve for the overlay

and writes
    OLED_postprocess_optimization_target_match.txt   (same format oled_common writes)
    OLED_postprocess_optimization_target_match.png

Ring power is reconstructed from the binned (theta, phi) map as
    P(theta) ~ <Sr>_phi * sin(theta) * cos(theta)
because the far-field spectrum is |E|^2 on the direction-cosine grid, where
dux duy = sin(theta) cos(theta) dtheta dphi.  Directions are assigned to their
nearest target ring (Voronoi partition in theta), which conserves total power.
Values therefore carry a small binning error versus a fresh run, which computes
the same quantities from the raw spectrum.

Usage
-----
    python OLED_pp_retarget_plots.py <run_dir> --target <angular_target.npz>
    python OLED_pp_retarget_plots.py <run_dir1> <run_dir2> --target <npz> --compare out.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_target(path):
    z = np.load(path)
    return (np.asarray(z["angle_thetas"], dtype=float).reshape(-1),
            np.asarray(z["target_profile"], dtype=float).reshape(-1),
            np.asarray(z["in_range"]).reshape(-1).astype(bool))


def ring_shares_from_map(map_npz, target_thetas):
    z = np.load(map_npz)
    th = np.asarray(z["theta_deg"], dtype=float)
    Sr = np.asarray(z["Sr"], dtype=float)
    weight = np.sin(np.radians(th)) * np.cos(np.radians(th))
    per_theta = Sr.mean(axis=1) * weight
    idx = np.argmin(np.abs(th[:, None] - target_thetas[None, :]), axis=1)
    power = np.zeros(target_thetas.size)
    np.add.at(power, idx, per_theta)
    total = float(np.sum(power))
    if total <= 0:
        raise RuntimeError(f"no propagating power in {map_npz}")
    return power / total, th, per_theta, idx


def ring_solid_angles(target_thetas, th_axis, idx):
    """Solid angle of each Voronoi ring, for converting target share -> radiance."""
    dth = float(np.mean(np.diff(th_axis))) if th_axis.size > 1 else 1.0
    omega = np.zeros(target_thetas.size)
    contrib = 2.0 * np.pi * np.sin(np.radians(th_axis)) * np.radians(dth)
    np.add.at(omega, idx, contrib)
    return omega


def evaluate(run_dir, target_path):
    thetas, target_profile, in_range = load_target(target_path)
    achieved, th_axis, _per_theta, idx = ring_shares_from_map(
        os.path.join(run_dir, "A", "OLED_postprocess_thetaphi_map.npz"), thetas)
    throughput = float(np.sum(achieved[in_range]))
    q = achieved / max(throughput, 1e-30)
    match = float(np.sum(np.minimum(q[in_range], target_profile[in_range])))
    tv = float(0.5 * np.sum(np.abs(achieved - target_profile)))
    omega = ring_solid_angles(thetas, th_axis, idx)
    return {
        "run_dir": run_dir, "target_path": target_path,
        "thetas": thetas, "target": target_profile, "in_range": in_range,
        "achieved": achieved, "throughput": throughput, "match": match,
        "fom": throughput * match, "total_variation": tv,
        "ring_solid_angle": omega,
    }


def write_txt(res):
    path = os.path.join(res["run_dir"], "A", "OLED_postprocess_optimization_target_match.txt")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(f"# optimization target: {res['target_path']}\n")
        fp.write("# FoM re-evaluated on the INCOHERENT postprocess far field\n")
        fp.write("# reconstructed from the stored (theta,phi) map (small binning error)\n")
        fp.write(f"# throughput {res['throughput']:.8e}\n")
        fp.write(f"# match {res['match']:.8e}\n")
        fp.write(f"# fom_throughput_x_match {res['fom']:.8e}\n")
        fp.write(f"# total_variation_vs_target {res['total_variation']:.8e}\n")
        fp.write("theta_deg achieved_share target_share in_range\n")
        for t, a, g, r in zip(res["thetas"], res["achieved"], res["target"], res["in_range"]):
            fp.write(f"{t:.4f} {a:.6e} {g:.6e} {int(r)}\n")
    return path


def plot_one(res):
    thetas, ach, tgt, ir = res["thetas"], res["achieved"], res["target"], res["in_range"]
    name = os.path.basename(os.path.normpath(res["run_dir"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    x = np.arange(thetas.size)
    ax.bar(x - 0.2, tgt * 100, 0.4, label="optimization target", color="0.35")
    ax.bar(x + 0.2, ach * 100, 0.4, label="postprocess (incoherent)", color="tab:blue")
    for i, r in enumerate(ir):
        if not r:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:red", alpha=0.10)
            ax.text(i, max(ach[i], tgt[i]) * 100 + 1.0, "suppress", ha="center", fontsize=7, color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in thetas])
    ax.set_xlabel("ring polar angle [deg]")
    ax.set_ylabel("power share [%]")
    ax.set_title("Achieved vs the design's OWN optimization target")
    ax.set_ylim(0.0, max(float(np.max(ach)), float(np.max(tgt))) * 100 * 1.42)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    ax.text(0.98, 0.98,
            f"throughput={res['throughput']:.3f}\nmatch={res['match']:.3f}\n"
            f"FoM=T*M={res['fom']:.3f}\nTV={res['total_variation']:.3f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7"))

    # radiance view: measured curve + target converted share -> radiance
    ax = axes[1]
    prof = np.loadtxt(os.path.join(res["run_dir"], "A", "OLED_postprocess_3x3_angle_profile.txt"))
    ax.plot(prof[:, 0], prof[:, 2], lw=2.0, color="tab:blue", label="postprocess radiance")
    # target_profile is already per-direction (see plot_emission): normalize to
    # max rather than dividing by the ring solid angle.
    scale = float(np.max(prof[:, 2]))
    ax.plot(thetas, tgt / max(float(np.max(tgt)), 1e-30) * scale, "o--",
            color="0.35", label="target (design's own ramp)")
    ax.plot(thetas, ach / max(float(np.max(ach)), 1e-30) * scale, "s-",
            color="tab:blue", alpha=0.6, label="achieved rings")
    ax.set_xlabel("theta [deg]")
    ax.set_ylabel("radiance [norm.]")
    ax.set_title("Radiance: measured curve vs ring targets")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Optimization-target comparison - {name}", fontsize=11)
    path = os.path.join(res["run_dir"], "A", "OLED_postprocess_optimization_target_match.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def mean_lee(run_dir):
    path = os.path.join(run_dir, "A", "OLED_postprocess_3x3_records.txt")
    for line in open(path, encoding="utf-8"):
        if line.startswith("mean_LEE"):
            return float(line.split()[1])
    return None


def planar_reference_lee(planar_dir):
    """mean LEE of a planar (no-design) run, rejecting a patterned run."""
    if not planar_dir:
        return None
    manifest = os.path.join(planar_dir, "A", "OLED_postprocess_manifest.json")
    if os.path.exists(manifest):
        import json
        with open(manifest, encoding="utf-8") as fp:
            m = json.load(fp)
        if not m.get("planar_baseline", False):
            raise SystemExit(f"{planar_dir} is not a planar-baseline run "
                             "(manifest planar_baseline is false)")
    return mean_lee(planar_dir)


def signed_radiance_from_map(map_npz):
    """Rebuild the emission figure's signed radiance from the stored map.

    Same convention as oled_common.directional_radiance: the +theta half is the
    kx>0 hemisphere, the -theta half is kx<0, so a left/right asymmetry stays
    visible instead of being mirrored away.
    """
    z = np.load(map_npz)
    th = np.asarray(z["theta_deg"], dtype=float)
    ph = np.asarray(z["phi_deg"], dtype=float)
    Sr = np.asarray(z["Sr"], dtype=float)
    cos_phi = np.cos(np.radians(ph))
    plus = Sr[:, cos_phi > 0].mean(axis=1)
    minus = Sr[:, cos_phi < 0].mean(axis=1)
    signed_theta = np.concatenate([-th[::-1], th[1:]])
    signed_value = np.concatenate([minus[::-1], plus[1:]])
    peak = float(np.max(signed_value))
    return signed_theta, signed_value / peak if peak > 0 else signed_value


def plot_emission(res):
    """Rewrite OLED_postprocess_emission.png against the design's OWN target.

    Same two-panel layout as the postprocess writes (polar radiance vs target,
    order-share bars), but every 'target' here is the optimization's actual ring
    target instead of the driver script's generic MSOPT_OLED_TARGET_* default.
    The original figure is preserved as *_genericTarget.png.
    """
    a_dir = os.path.join(res["run_dir"], "A")
    out = os.path.join(a_dir, "OLED_postprocess_emission.png")
    backup = os.path.join(a_dir, "OLED_postprocess_emission_genericTarget.png")
    if os.path.exists(out) and not os.path.exists(backup):
        os.replace(out, backup)

    thetas, ach, tgt, ir = res["thetas"], res["achieved"], res["target"], res["in_range"]
    # target_profile is build_ramp_target's per-direction ramp curve sampled at
    # the ring angles and normalized to sum 1, so it is ALREADY a per-direction
    # quantity: rescaling it by the ring solid angle would invent a spike at
    # theta=0 that the optimization never asked for. Normalize to max instead,
    # which is exactly the curve the FoM's ramp describes.
    tgt_rad = tgt / max(float(np.max(tgt)), 1e-30)
    ach_rad = ach / max(float(np.max(ach)), 1e-30)

    signed_theta, signed_val = signed_radiance_from_map(
        os.path.join(a_dir, "OLED_postprocess_thetaphi_map.npz"))

    fig = plt.figure(figsize=(13.0, 5.0), constrained_layout=True)
    ax0 = fig.add_subplot(1, 2, 1, projection="polar")
    ax0.plot(np.radians(signed_theta), signed_val, lw=1.8, color="tab:blue", label="signed radiance")
    # target: ring values interpolated across theta, mirrored to +/- theta
    dense = np.linspace(0.0, 90.0, 181)
    tgt_dense = np.interp(dense, thetas, tgt_rad, left=tgt_rad[0], right=0.0)
    ax0.plot(np.radians(np.concatenate([-dense[::-1], dense[1:]])),
             np.concatenate([tgt_dense[::-1], tgt_dense[1:]]),
             lw=1.8, color="tab:orange", label="target (design's own)")
    ax0.plot(np.radians(thetas), tgt_rad, "o", ms=5, color="tab:orange")
    ax0.plot(np.radians(-thetas), tgt_rad, "o", ms=5, color="tab:orange")
    ax0.set_theta_zero_location("N")
    ax0.set_theta_direction(-1)
    ax0.set_thetamin(-90)
    ax0.set_thetamax(90)
    ax0.set_title("Per-direction radiance vs target")
    ax0.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.30))

    ax1 = fig.add_subplot(1, 2, 2)
    x = np.arange(thetas.size)
    ax1.bar(x - 0.2, tgt, 0.4, color="tab:orange", label="target share")
    ax1.bar(x + 0.2, ach, 0.4, color="tab:blue", label="achieved share")
    for i, r in enumerate(ir):
        if not r:
            ax1.axvspan(i - 0.5, i + 0.5, color="tab:red", alpha=0.10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{t:.0f}" for t in thetas])
    ax1.set_xlabel("ring polar angle (deg)   shaded = target 0")
    ax1.set_ylabel("power share")
    ax1.set_title("Order power share: achieved vs target")
    ax1.set_ylim(0.0, max(float(np.max(ach)), float(np.max(tgt))) * 1.35)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3, axis="y")
    stats = [f"throughput={res['throughput']:.3f}",
             f"match={res['match']:.3f}",
             f"FoM=T*M={res['fom']:.3f}"]
    lee, planar_lee = res.get("lee"), res.get("planar_lee")
    if lee is not None:
        stats.append("")
        stats.append(f"LEE with design: {lee * 100:.2f}%")
        if planar_lee:
            stats.append(f"LEE no design:   {planar_lee * 100:.2f}%")
            stats.append(f"enhancement:     {lee / planar_lee:.2f}x")
        else:
            stats.append("LEE no design:   (not provided)")
    ax1.text(0.98, 0.98, "\n".join(stats), transform=ax1.transAxes, va="top", ha="right",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.92))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_compare(results, out_path):
    thetas = results[0]["thetas"]
    x = np.arange(thetas.size)
    width = 0.8 / (len(results) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    ax.bar(x - 0.4 + 0.5 * width, results[0]["target"] * 100, width,
           label="optimization target", color="0.35")
    for j, res in enumerate(results):
        ax.bar(x - 0.4 + (j + 1.5) * width, res["achieved"] * 100, width,
               label=f"{os.path.basename(os.path.normpath(res['run_dir']))[:24]} "
                     f"(FoM={res['fom']:.3f})")
    for i, r in enumerate(results[0]["in_range"]):
        if not r:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:red", alpha=0.10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in thetas])
    ax.set_xlabel("ring polar angle [deg]   (shaded = target 0, must be suppressed)")
    ax.set_ylabel("power share [%]")
    ax.set_title("Incoherent postprocess vs the designs' own optimization target")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--target", required=True, help="the design's OLED_angular_target.npz")
    ap.add_argument("--compare", default=None, help="also write a combined comparison PNG here")
    ap.add_argument("--planar", default=None,
                    help="planar (no-design) postprocess run dir, for the LEE read-out")
    args = ap.parse_args()
    planar_lee = planar_reference_lee(args.planar)
    results = []
    for run_dir in args.run_dirs:
        res = evaluate(run_dir, args.target)
        res["lee"] = mean_lee(run_dir)
        res["planar_lee"] = planar_lee
        print(f"[{os.path.basename(os.path.normpath(run_dir))}] "
              f"throughput={res['throughput']:.4f} match={res['match']:.4f} "
              f"FoM={res['fom']:.4f} TV={res['total_variation']:.4f}")
        print("   " + ", ".join(f"{t:.1f}deg {a*100:.2f}%/{g*100:.2f}%"
                                for t, a, g in zip(res["thetas"], res["achieved"], res["target"])))
        print("   ->", write_txt(res))
        print("   ->", plot_one(res))
        print("   ->", plot_emission(res))
        results.append(res)
    if args.compare and len(results) >= 1:
        print("   ->", plot_compare(results, args.compare))


if __name__ == "__main__":
    main()
