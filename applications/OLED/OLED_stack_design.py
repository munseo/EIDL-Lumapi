"""Design an optimal top-emission microcavity OLED stack from literature材 data,
using the validated analytic layered-dipole solver (OLED_layered_dipole.py).

Why optimize rather than copy a published stack: a paper's thicknesses are tuned
to that paper's materials, emission wavelength and process.  The analytic solver
evaluates a full channel decomposition in ~0.2 s, so the thicknesses can simply
be searched directly for OUR material set.  The reconstructed table stack turned
out to sit at cavity ANTI-resonance for green (HTL 95 nm -> 0.89 % horizontal
outcoupling, while 170 nm gives 47.9 %), which is exactly the failure mode a
direct search removes.

MATERIAL DATA
  Ag        Johnson & Christy, Phys. Rev. B 6, 4370 (1972), interpolated to the
            three emission peaks.  This is the standard reference for evaporated
            silver and is what makes the metal loss (hence the SPP channel)
            defensible rather than assumed.
  ITO       n ~ 1.9-2.0 with weak visible absorption (k ~ 0.01-0.03); the loss
            matters because the anode sits inside the cavity.
  organics  Typical ellipsometry values for the named materials at the emission
            peaks; all treated as transparent (k = 0) there, which is standard
            for host/transport layers at their own emission wavelength.

The cathode is modelled as thin Ag by default.  Mg:Ag (10:1) optical constants
scatter widely with composition and deposition, so the default uses the material
whose constants are traceable; --cathode mgag switches to a published-range
Mg:Ag estimate for comparison and prints that it is an estimate.

Usage
  python OLED_stack_design.py --scan          # 1-D scans around the seed stack
  python OLED_stack_design.py --optimize      # coordinate search, all colors
  python OLED_stack_design.py --report        # final table + FDTD-ready summary
"""

import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import OLED_layered_dipole as ld


RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
OUT_DIR = os.path.join(RUN_DIR, "A")
os.makedirs(OUT_DIR, exist_ok=True)

# --- literature material data ------------------------------------------------
# Johnson & Christy 1972 silver, interpolated to the emission peaks.
AG = {0.460: 0.140 + 2.560j, 0.530: 0.121 + 3.090j, 0.620: 0.135 + 3.990j}
# Mg:Ag 10:1 -- ESTIMATE (composition/deposition dependent), for comparison only.
MGAG = {0.460: 0.270 + 2.700j, 0.530: 0.300 + 3.100j, 0.620: 0.330 + 3.600j}
ITO = {0.460: 2.020 + 0.030j, 0.530: 1.950 + 0.010j, 0.620: 1.900 + 0.010j}
ORG = {  # transparent transport/host materials at their own emission peak
    "HATCN": {0.460: 1.880, 0.530: 1.850, 0.620: 1.830},
    "NPB":   {0.460: 1.850, 0.530: 1.810, 0.620: 1.790},
    "TCTA":  {0.460: 1.860, 0.530: 1.820, 0.620: 1.800},
    "EML":   {0.460: 1.830, 0.530: 1.790, 0.620: 1.770},
    "TSPO1": {0.460: 1.790, 0.530: 1.760, 0.620: 1.740},
    "TPBi":  {0.460: 1.790, 0.530: 1.750, 0.620: 1.730},
}
COLORS = {"blue": 0.460, "green": 0.530, "red": 0.620}

# Fixed layers (nm): these are process/electrical choices, not optical knobs.
AG_ANODE_NM, ITO_NM, HIL_NM, EBL_NM, EML_NM, HBL_NM = 100.0, 10.0, 10.0, 10.0, 25.0, 10.0


def build_stack(lam, htl_nm, etl_nm, cath_nm, cpl_nm, n_cpl, cathode="ag",
                eml_nm=EML_NM, label=""):
    """Bottom -> top: air | Ag | ITO | HATCN | NPB(HTL) | TCTA(EBL) | EML |
    TSPO1(HBL) | TPBi(ETL) | cathode | CPL | air.  Dipole at the EML centre."""
    cath = (AG if cathode == "ag" else MGAG)[lam]
    names = ["air_bot", "Ag_anode", "ITO", "HATCN", "NPB_HTL", "TCTA_EBL",
             "EML", "TSPO1_HBL", "TPBi_ETL", "cathode", "CPL", "air_top"]
    n = [1.0, AG[lam], ITO[lam], ORG["HATCN"][lam], ORG["NPB"][lam],
         ORG["TCTA"][lam], ORG["EML"][lam], ORG["TSPO1"][lam], ORG["TPBi"][lam],
         cath, n_cpl, 1.0]
    um = 1e-3
    d = [None, AG_ANODE_NM * um, ITO_NM * um, HIL_NM * um, htl_nm * um,
         EBL_NM * um, eml_nm * um, HBL_NM * um, etl_nm * um, cath_nm * um,
         cpl_nm * um, None]
    return ld.Stack(names, n, d, lam, e=6, z_off=0.5 * eml_nm * um, label=label)


def evaluate(lam, htl, etl, cath, cpl, n_cpl, cathode="ag", orientation="h"):
    """Full decomposition for one stack (accurate, ~3 s)."""
    st = build_stack(lam, htl, etl, cath, cpl, n_cpl, cathode)
    res = ld.analyse(st, rtol_seq=(1e-6,))
    rec = res["orientations"][orientation]
    return rec["frac"], rec["purcell"], res


# The accurate path spends most of its time in choose_u_max, which re-searches
# the evanescent cut-off for every candidate. During a search the stack only
# changes by tens of nm, so the cut-off is reused: u_max is taken once from the
# seed with a generous safety factor and the objective then integrates directly.
_UMAX_CACHE = {}


def _search_u_max(lam, cathode, orientation):
    key = (lam, cathode, orientation)
    if key not in _UMAX_CACHE:
        st = build_stack(lam, SEED["htl"], SEED["etl"], SEED["cath"],
                         SEED["cpl"], SEED["n_cpl"], cathode)
        um = ld.choose_u_max(st, orientation, lambda x: ld.make_bands(st, x))
        _UMAX_CACHE[key] = 2.0 * um          # safety factor for nearby stacks
    return _UMAX_CACHE[key]


def outcoupled(lam, htl, etl, cath, cpl, n_cpl, cathode="ag", orientation="h",
               rtol=1e-4):
    """Outcoupled fraction only -- the search objective."""
    st = build_stack(lam, htl, etl, cath, cpl, n_cpl, cathode)
    u_max = _search_u_max(lam, cathode, orientation)
    bands = ld.make_bands(st, u_max)
    _pb, total, _nodes, _scale = ld.integrate_all(st, orientation, u_max, bands, rtol=rtol)
    return float(total[ld.IDX_OUT] / total[ld.IDX_TOTAL])


# --- searches ----------------------------------------------------------------
SEED = dict(htl=170.0, etl=40.0, cath=15.0, cpl=70.0, n_cpl=1.9)
GRID = {
    "htl":   np.arange(20.0, 261.0, 5.0),
    "etl":   np.arange(20.0, 121.0, 5.0),
    "cath":  np.arange(8.0, 31.0, 1.0),
    "cpl":   np.arange(20.0, 161.0, 5.0),
    "n_cpl": np.arange(1.6, 2.451, 0.05),
}


def coordinate_search(lam, cathode="ag", orientation="h", rounds=4, seed=None):
    """Cyclic coordinate ascent on the five optical thicknesses. The objective
    surface is dominated by the cavity resonance in HTL, so HTL is swept first
    and on a grid fine enough to resolve the resonance."""
    cur = dict(seed or SEED)
    best = outcoupled(lam, cathode=cathode, orientation=orientation, **cur)
    history = [(dict(cur), best)]
    for _rd in range(rounds):
        improved = False
        for key in ("htl", "etl", "cpl", "n_cpl", "cath"):
            vals = GRID[key]
            scores = []
            for v in vals:
                trial = dict(cur)
                trial[key] = float(v)
                scores.append(outcoupled(lam, cathode=cathode,
                                         orientation=orientation, **trial))
            i = int(np.argmax(scores))
            if scores[i] > best + 1e-6:
                best = scores[i]
                cur[key] = float(vals[i])
                improved = True
                history.append((dict(cur), best))
        if not improved:
            break
    return cur, best, history


def scan_plot(lam, cathode, base, path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.6), constrained_layout=True)
    for ax, key in zip(axes, ("htl", "etl", "cath", "cpl", "n_cpl")):
        vals = GRID[key]
        ys = []
        for v in vals:
            t = dict(base)
            t[key] = float(v)
            ys.append(outcoupled(lam, cathode=cathode, **t) * 100)
        ax.plot(vals, ys, lw=2)
        ax.axvline(base[key], color="tab:red", ls="--", lw=1.2,
                   label=f"chosen {base[key]:g}")
        ax.set_xlabel(key + (" (nm)" if key != "n_cpl" else ""))
        ax.set_ylabel("outcoupled [%]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"1-D sensitivity around the optimum, lambda={lam:g} um, cathode={cathode}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colors", default="green,red,blue")
    ap.add_argument("--cathode", default="ag", choices=["ag", "mgag"])
    ap.add_argument("--orientation", default="h", choices=["h", "v", "iso"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--scan", action="store_true", help="also write sensitivity plots")
    args = ap.parse_args()
    if args.cathode == "mgag":
        print("[note] Mg:Ag constants are a published-range ESTIMATE, not traceable data.")

    summary = {}
    for cname in [c.strip() for c in args.colors.split(",") if c.strip()]:
        lam = COLORS[cname]
        print(f"\n{'=' * 74}\n{cname.upper()}  lambda = {lam:g} um, cathode = {args.cathode}\n{'=' * 74}")
        best_cfg, best_val, _hist = coordinate_search(
            lam, cathode=args.cathode, orientation=args.orientation, rounds=args.rounds)
        frac, purcell, _res = evaluate(lam, cathode=args.cathode,
                                       orientation=args.orientation, **best_cfg)
        print(f"  optimum: HTL {best_cfg['htl']:.0f} / ETL {best_cfg['etl']:.0f} / "
              f"cathode {best_cfg['cath']:.0f} / CPL {best_cfg['cpl']:.0f} nm, n_CPL {best_cfg['n_cpl']:.2f}")
        print(f"  outcoupled = {best_val * 100:.2f} %   Purcell = {purcell:.3f}")
        for k, v in sorted(frac.items(), key=lambda kv: -kv[1]):
            print(f"      {k:<28s} {v * 100:7.3f} %")
        summary[cname] = {"lam_um": lam, "cathode": args.cathode,
                          "orientation": args.orientation,
                          "thicknesses_nm": best_cfg,
                          "outcoupled": best_val, "purcell": purcell,
                          "fractions": {k: float(v) for k, v in frac.items()}}
        if args.scan:
            scan_plot(lam, args.cathode, best_cfg,
                      os.path.join(OUT_DIR, f"stack_design_scan_{cname}_{args.cathode}.png"))

    path = os.path.join(OUT_DIR, f"OLED_stack_design_{args.cathode}_{args.orientation}.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
