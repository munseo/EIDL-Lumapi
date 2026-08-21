"""Tune the microcavity for the brightest ON-AXIS stack, not the brightest one.

OLED_stack_design.py maximizes `outcoupled` and nothing else -- there is not one
reference to an angle, a viewing direction or on-axis intensity anywhere in it.
The stack it produced is the honest optimum of that objective, and for green it
peaks 26.5 degrees off axis with normal emission at 0.573 of the peak. For a
display that is the wrong trade: it costs front-of-screen luminance and shifts
colour with angle, because a microcavity resonance blue-shifts as it scans.

This script keeps everything else -- same five thicknesses, same material data,
same solver -- and changes only the question:

    maximize LEE   subject to   the far field peaking on axis

WHY THE CONSTRAINT IS CHEAP.  `components` returns every channel spectrum in one
call, and its outcoupled row is u * dP/du. Dividing by u recovers dP/du and
multiplying by cos(theta) carries that to dP/dOmega, the far-field radiance.
That costs microseconds, while LEE needs the adaptive quadrature and costs ~2 s.
So each sweep filters on the angular SHAPE first and prices only the survivors.

WHY MULTI-START.  The feasible set is not connected. Scanning one thickness at a
time around the current green stack puts on-axis peaks at HTL 90-140 nm AND again
at 240-260, at ETL 20-30 AND 110-120, at CPL 30-50 AND 150-160 -- different
resonance orders, with off-axis territory in between. A single coordinate ascent
cannot cross that gap, so several seeds are run and the best is kept.

WHAT IT FOUND FOR GREEN (2026-08-20)

    stack     LEE      peak     on-axis radiance
    current   0.5725   26.5deg  1.1956
    on-axis   0.5138    1.0deg  2.1791      HTL 155 ETL 30 cath 10 CPL 80, n 1.9

  +82% on axis for -10.3% LEE. Note HTL does not move; the gain comes from the
  ETL, the thinner cathode, and above all the CPL.

  --fix-ncpl, which keeps the present high-index capping layer, is much worse:
  LEE 0.4675 (-18.3%) and on-axis 1.0957, BELOW the current stack's own on-axis
  value. Constraining the cavity to peak on axis while holding n_CPL = 2.2 costs
  so much total extraction that even the normal direction loses. The whole gain
  depends on substituting a lower-index capping layer, which is a materials
  decision, not a thickness one -- so the flag exists to make that explicit
  rather than to offer a real alternative.

Usage
  python OLED_stack_onaxis.py                 # green, CPL index free
  python OLED_stack_onaxis.py --lam 0.62      # red
  python OLED_stack_onaxis.py --fix-ncpl 2.2  # keep today's capping material
  python OLED_stack_onaxis.py --json out.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import OLED_stack_design as sd                            # noqa: E402
import OLED_layered_dipole as ld                          # noqa: E402

# The stack in oled_common.microcavity_layers today, for the baseline row.
CURRENT = {0.53: dict(htl=155.0, etl=40.0, cath=13.0, cpl=70.0, n_cpl=2.2)}

# Seeds spread across the resonance orders the 1-D scans revealed.
SEEDS = [
    dict(htl=120.0, etl=25.0, cath=13.0, cpl=40.0, n_cpl=2.2),
    dict(htl=120.0, etl=115.0, cath=13.0, cpl=40.0, n_cpl=1.9),
    dict(htl=250.0, etl=25.0, cath=13.0, cpl=155.0, n_cpl=2.0),
    dict(htl=100.0, etl=25.0, cath=10.0, cpl=35.0, n_cpl=1.8),
    dict(htl=155.0, etl=30.0, cath=10.0, cpl=80.0, n_cpl=1.9),
]

U = np.sin(np.deg2rad(np.arange(0.25, 89.75, 0.25)))
TH = np.rad2deg(np.arcsin(U))
COS = np.cos(np.arcsin(U))


def radiance(st):
    """Far-field radiance dP/dOmega vs TH, in the solver's own units."""
    return (ld.components(st, U, "h")[ld.IDX_OUT] / U) * COS


def evaluate(lam, p, cathode="ag"):
    """LEE, peak angle, and radiance normalized by TOTAL emitted power.

    Dividing by the emitted total is what makes two different stacks comparable
    on axis: the Purcell factor differs between them, so radiance per unit
    SOURCE power is the only fair currency.
    """
    st = sd.build_stack(lam, p["htl"], p["etl"], p["cath"], p["cpl"], p["n_cpl"], cathode)
    u_max = sd._search_u_max(lam, cathode, "h")
    _pb, total, _n, _s = ld.integrate_all(st, "h", u_max, ld.make_bands(st, u_max), rtol=1e-4)
    rad = radiance(st) / float(total[ld.IDX_TOTAL])
    return {"lee": float(total[ld.IDX_OUT] / total[ld.IDX_TOTAL]),
            "peak_deg": float(TH[np.argmax(rad)]),
            "on_axis": float(rad[0]),
            "on_axis_over_peak": float(rad[0] / rad.max()),
            "radiance": rad}


def peaks_on_axis(lam, p, cathode="ag", tol=0.999):
    st = sd.build_stack(lam, p["htl"], p["etl"], p["cath"], p["cpl"], p["n_cpl"], cathode)
    r = radiance(st)
    return float(r[0] / r.max()) > tol


def lee_only(lam, p, cathode="ag"):
    return float(sd.outcoupled(lam, p["htl"], p["etl"], p["cath"], p["cpl"],
                               p["n_cpl"], cathode))


def ascend(lam, seed, keys, cathode="ag", rounds=6):
    """Coordinate ascent on LEE that never steps off the on-axis constraint."""
    cur = dict(seed)
    if not peaks_on_axis(lam, cur, cathode):
        return None
    best = lee_only(lam, cur, cathode)
    for _ in range(rounds):
        moved = False
        for k in keys:
            keep = cur[k]
            feasible = []
            for v in sd.GRID[k]:                 # the free check, first
                cur[k] = float(v)
                if peaks_on_axis(lam, cur, cathode):
                    feasible.append(float(v))
            cur[k] = keep
            for v in feasible:
                if v == keep:
                    continue
                cur[k] = v
                s = lee_only(lam, cur, cathode)
                if s > best + 1e-6:
                    best, keep, moved = s, v, True
                cur[k] = keep
            cur[k] = keep
        if not moved:
            break
    return cur, best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lam", type=float, default=0.53,
                    help="emission wavelength in um; must be one of the material table's")
    ap.add_argument("--cathode", default="ag", choices=("ag", "mgag"))
    ap.add_argument("--fix-ncpl", type=float, default=None,
                    help="hold the capping-layer index (e.g. 2.2 to keep today's material)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args()

    if args.lam not in sd.AG:
        ap.error(f"--lam must be one of {sorted(sd.AG)}")
    keys = ["htl", "etl", "cath", "cpl"] + ([] if args.fix_ncpl else ["n_cpl"])
    seeds = [dict(s) for s in SEEDS]
    if args.fix_ncpl:
        for s in seeds:
            s["n_cpl"] = float(args.fix_ncpl)

    base = CURRENT.get(args.lam)
    base_ev = evaluate(args.lam, base, args.cathode) if base else None
    if base_ev:
        print(f"[stack] current   LEE={base_ev['lee']:.4f}  peak={base_ev['peak_deg']:5.1f}deg  "
              f"on-axis={base_ev['on_axis']:.4e}  on-axis/peak={base_ev['on_axis_over_peak']:.3f}",
              flush=True)

    found = []
    for i, s in enumerate(seeds):
        out = ascend(args.lam, s, keys, args.cathode, args.rounds)
        if out is None:
            print(f"[stack] seed {i}: not on-axis, skipped", flush=True)
            continue
        p, _v = out
        ev = evaluate(args.lam, p, args.cathode)
        found.append((ev, p))
        print(f"[stack] seed {i}: LEE={ev['lee']:.4f}  peak={ev['peak_deg']:5.1f}deg  "
              f"on-axis={ev['on_axis']:.4e}   "
              + "  ".join(f"{k}={p[k]:g}" for k in ("htl", "etl", "cath", "cpl", "n_cpl")),
              flush=True)
    if not found:
        print("[stack] no on-axis stack found; widen the seeds or relax the constraint")
        return 1

    ev, p = max(found, key=lambda t: t[0]["lee"])
    print("\n  BEST on-axis stack:  "
          + "  ".join(f"{k}={p[k]:g}" for k in ("htl", "etl", "cath", "cpl", "n_cpl")))
    if base_ev:
        print(f"  LEE       {ev['lee']:.4f}   ({100 * (ev['lee'] / base_ev['lee'] - 1):+.1f}% vs current)")
        print(f"  on-axis   {ev['on_axis']:.4e}   ({ev['on_axis'] / base_ev['on_axis']:.3f}x current)")
        print(f"\n  {'theta':>7}{'current':>10}{'new':>10}{'new/cur':>10}   (both / current on-axis)")
        for x in (0, 10, 20, 30, 40, 45, 60, 75):
            a = np.interp(x, TH, base_ev["radiance"])
            b = np.interp(x, TH, ev["radiance"])
            print(f"  {x:>7}{a / base_ev['on_axis']:>10.3f}"
                  f"{b / base_ev['on_axis']:>10.3f}{b / a:>10.3f}")

    if args.json:
        def strip(d):
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
        json.dump({"lam_um": args.lam, "cathode": args.cathode,
                   "fix_ncpl": args.fix_ncpl,
                   "current": {"params": base, "metrics": strip(base_ev)} if base_ev else None,
                   "best": {"params": p, "metrics": strip(ev)},
                   "theta_deg": TH.tolist()},
                  open(args.json, "w"), indent=1)
        print(f"\n[stack] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
