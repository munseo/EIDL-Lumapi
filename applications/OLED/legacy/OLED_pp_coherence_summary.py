"""
Slide-style coherence-comparison figures from re-postprocess runs.

Reads the A/ products written by OLED_lens/OLED_new/OLED_opt run_postprocess with
MSOPT_OLED_PP_COHERENT_CHECK=1 / MSOPT_OLED_PP_RANDOM_PHASE_TRIALS=N and builds,
per design, the Meep-style comparison figure:

  Case 1  incoherent per-dipole sum (6x6 grid)      -> OLED_postprocess_3x3_angle_profile.txt
  Case 2  same-phase simultaneous coherent          -> OLED_postprocess_coherent_same_phase_*
  Case 3  random-phase coherent, running average    -> OLED_postprocess_randphase_avgNNN_*
          (per-trial spread from randphase_trialNNN files)

plus the order-share table per case (0 deg transfer check).

Usage:
  python OLED_pp_coherence_summary.py <run_dir> [<run_dir> ...] [--outdir DIR]
  (<run_dir> = a Done/ re-postprocess run folder containing A/)
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt


AVG_TAGS = ["randphase_avg001", "randphase_avg002", "randphase_avg005", "randphase_avg010", "randphase_avg020"]


def load_txt(path):
    if not os.path.exists(path):
        return None
    try:
        return np.loadtxt(path)
    except Exception:
        return None


def load_profile(a_dir, tag):
    if tag == "incoherent":
        d = load_txt(os.path.join(a_dir, "OLED_postprocess_3x3_angle_profile.txt"))
    else:
        d = load_txt(os.path.join(a_dir, f"OLED_postprocess_{tag}_angle_profile.txt"))
    if d is None or d.ndim != 2:
        return None
    return d  # cols: theta_deg ring_flux radiance_norm [target_norm]


def load_shares(a_dir, tag):
    if tag == "incoherent":
        p = os.path.join(a_dir, "OLED_postprocess_order_shares.txt")
    else:
        p = os.path.join(a_dir, f"OLED_postprocess_{tag}_order_shares.txt")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p, "r", encoding="utf-8") as fp:
        header = fp.readline()
        for line in fp:
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(x) for x in parts[:5]])
    return np.asarray(rows) if rows else None  # theta, power, share, target_eff, target_share


def polar_cut(ax, theta_deg, radiance, color="tab:blue", lw=2.0, alpha=1.0, label=None):
    r = radiance / max(float(np.max(np.abs(radiance))), 1e-30)
    th = np.deg2rad(theta_deg)
    th_full = np.concatenate([-th[::-1], th[1:]])
    r_full = np.concatenate([r[::-1], r[1:]])
    ax.plot(th_full, r_full, color=color, lw=lw, alpha=alpha, label=label)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_yticklabels([])


def build_figure(run_dir, outdir):
    a_dir = os.path.join(run_dir, "A")
    name = os.path.basename(os.path.normpath(run_dir))
    incoh = load_profile(a_dir, "incoherent")
    if incoh is None:
        print(f"[skip] {name}: no incoherent angle profile")
        return None
    coh = load_profile(a_dir, "coherent_same_phase")
    avgs = [(t, load_profile(a_dir, t)) for t in AVG_TAGS]
    avgs = [(t, d) for t, d in avgs if d is not None]
    trials = []
    for p in sorted(glob.glob(os.path.join(a_dir, "OLED_postprocess_randphase_trial*_angle_profile.txt"))):
        d = load_txt(p)
        if d is not None and d.ndim == 2:
            m = re.search(r"trial(\d+)", p)
            trials.append((int(m.group(1)) if m else 0, d))

    n_polar = 2 + len(avgs) + (1 if coh is not None else 0)
    fig = plt.figure(figsize=(3.2 * max(n_polar, 4), 7.5))
    fig.suptitle(f"Multi-dipole coherence comparison - {name}", fontsize=13)

    col = 0
    ax = fig.add_subplot(2, max(n_polar, 4), col + 1, projection="polar")
    polar_cut(ax, incoh[:, 0], incoh[:, 2], color="tab:green")
    if incoh.shape[1] >= 4:
        polar_cut(ax, incoh[:, 0], incoh[:, 3], color="gray", lw=1.0, alpha=0.7, label="target")
    ax.set_title("Case 1\nincoherent sum (grid)", fontsize=9)
    col += 1

    if coh is not None:
        ax = fig.add_subplot(2, max(n_polar, 4), col + 1, projection="polar")
        polar_cut(ax, coh[:, 0], coh[:, 2], color="tab:red")
        ax.set_title("Case 2\nsame-phase coherent", fontsize=9)
        col += 1

    for tag, d in avgs:
        n = int(tag[-3:])
        ax = fig.add_subplot(2, max(n_polar, 4), col + 1, projection="polar")
        polar_cut(ax, d[:, 0], d[:, 2], color="tab:blue")
        polar_cut(ax, incoh[:, 0], incoh[:, 2], color="tab:green", lw=1.0, alpha=0.6)
        ax.set_title(f"Case 3\nrandom-phase avg x{n}", fontsize=9)
        col += 1

    # per-trial spread
    ax = fig.add_subplot(2, max(n_polar, 4), col + 1, projection="polar")
    for t_no, d in trials:
        polar_cut(ax, d[:, 0], d[:, 2], color="tab:blue", lw=0.8, alpha=0.35)
    polar_cut(ax, incoh[:, 0], incoh[:, 2], color="tab:green", lw=1.6)
    ax.set_title(f"Case 3 trials\n({len(trials)} single realizations)", fontsize=9)

    # ---- bottom row: order shares per case + convergence ----------------------
    tags = [("incoherent", "incoh")]
    if coh is not None:
        tags.append(("coherent_same_phase", "same-phase"))
    tags += [(t, f"avg{int(t[-3:])}") for t, _ in avgs]
    shares = [(lbl, load_shares(a_dir, t)) for t, lbl in tags]
    shares = [(lbl, s) for lbl, s in shares if s is not None]

    axb = fig.add_subplot(2, 2, 3)
    if shares:
        thetas = shares[0][1][:, 0]
        w = 0.8 / (len(shares) + 1)
        xs = np.arange(len(thetas))
        axb.bar(xs - 0.4 + 0.5 * w, shares[0][1][:, 4] * 100, w, color="k", alpha=0.55, label="target")
        for j, (lbl, s) in enumerate(shares):
            axb.bar(xs - 0.4 + (j + 1.5) * w, s[:, 2] * 100, w, label=lbl)
        axb.set_xticks(xs)
        axb.set_xticklabels([f"{t:g}" for t in thetas])
        axb.set_xlabel("order angle [deg]")
        axb.set_ylabel("power share [%]")
        axb.set_title("Order shares per case vs target")
        axb.legend(fontsize=7, ncol=2)
        axb.grid(alpha=0.3, axis="y")

    axc = fig.add_subplot(2, 2, 4)
    if shares and len(shares) > 1:
        ref = shares[0][1][:, 2]
        labels, rels = [], []
        for lbl, s in shares[1:]:
            rel = float(np.sum(np.abs(s[:, 2] - ref))) * 0.5 * 100  # total-variation distance, %
            labels.append(lbl)
            rels.append(rel)
        axc.plot(range(len(rels)), rels, "o-", lw=2)
        axc.set_xticks(range(len(rels)))
        axc.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axc.set_ylabel("share TV-distance vs incoherent [%]")
        axc.set_title("Convergence toward incoherent reference")
        axc.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(outdir, f"coherence_summary_{name}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    for rd in args.run_dirs:
        build_figure(rd, args.outdir)


if __name__ == "__main__":
    main()
