"""
================================================================
Deterministic vs Trace Formulation 비교
================================================================
Bare LED 구조 (cell=7um, PML all sides) 에서:

  Method A — Deterministic decomposition
    6×6 midpoint grid, 9 unique Ex runs
    각 위치 개별 FDTD → power 가중 합산

  Method B — Trace formulation (cosine basis)
    MQW emission layer 위에 cosine basis source 배치
    K 개 basis function run → |field|² 합산
    K = 1, 2, 3, ... 으로 sweep 하여 수렴 확인

비교:
  - eta_ff 정확도 (method A 를 reference)
  - 총 wall time
  - basis 수 vs dipole 수 효율

Trace formulation 원리:
  N 개 incoherent dipole 에서 나오는 total power 는
    P_incoh = (1/N) Σ_i P(r_i)

  이를 orthonormal basis {ψ_k(x,y)} 로 분해하면
    P_incoh = Σ_k P(ψ_k)

  여기서 P(ψ_k) 는 basis function ψ_k 를 source amplitude 로
  넣고 돌린 FDTD 결과의 power.

  Basis: ψ_{mn}(x,y) = (2/L) cos(mπx/L) cos(nπy/L)
  (m,n = 0,1,2,... 에서 eigenvalue 큰 순서로 정렬)
================================================================
"""

import matplotlib
matplotlib.use("Agg")

import os
import json
import time
from datetime import datetime

import numpy as np
import meep as mp
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

def save_json(path, obj):
    if IS_MASTER:
        with open(path, "w") as f:
            json.dump(obj, f, indent=2,
                      default=lambda x: (x.tolist() if isinstance(x, np.ndarray)
                                         else (float(x) if isinstance(x, np.floating)
                                               else None)))


# ================================================================
# 1. Parameters
# ================================================================
resolution  = 40
target_wvl  = 0.527
fcen        = 1.0 / target_wvl
fwidth      = 0.05 * fcen

n_GaN  = 2.40
n_Sub  = 2.49
n_Poly = 1.50

channel_w, channel_l = 1.05, 1.05
p_gan_h, mqw_h, n_gan_h = 0.20, 0.20, 0.70
sub_h = 0.20

pml_th           = 0.50
PIXEL            = 1.0 / resolution
mon_pml_gap_px   = 2
mon_pml_gap      = mon_pml_gap_px * PIXEL

CELL_XY = 7.0

gan_total_h = n_gan_h + mqw_h + p_gan_h
air_pad_top = max(4 * target_wvl, 2.0)
sz_raw = pml_th + sub_h + gan_total_h + air_pad_top + pml_th
sz = round(sz_raw * resolution) / resolution

z_bot      = -sz / 2
z_sub_bot  = z_bot + pml_th
z_sub_top  = z_sub_bot + sub_h
z_gan_c    = z_sub_top + gan_total_h / 2
z_mqw_c    = z_sub_top + n_gan_h + mqw_h / 2
z_gan_top  = z_sub_top + gan_total_h
src_z      = z_mqw_c

z_top_mon  = sz/2 - pml_th - mon_pml_gap
ap_half    = CELL_XY / 2 - pml_th - mon_pml_gap

SRC_BOX_INSET_PX = 2
src_box_inset = SRC_BOX_INSET_PX * PIXEL
src_box_half  = channel_w / 2 - src_box_inset
src_box_z_lo  = src_z - mqw_h / 2 + src_box_inset
src_box_z_hi  = src_z + mqw_h / 2 - src_box_inset

AIR_FF_LAT_BOT_MARGIN = 0.5 * target_wvl
z_ff_lat_bot = z_gan_top + AIR_FF_LAT_BOT_MARGIN

EDGE_INSET   = 2 * PIXEL
SIM_DFT_TOL  = 1e-4
SIM_MAX_TIME = 2000
EPS_RATIO    = 1e-12
FF_R         = 1e6
FF_N_THETA   = 19
FF_N_PHI     = 36

# Emission region = MQW 면내 channel 영역
EMIT_LX = channel_w - 2 * EDGE_INSET
EMIT_LY = channel_l - 2 * EDGE_INSET

# Trace basis sweep: max order
MAX_BASIS_ORDER = 4   # (m,n) = 0..MAX_BASIS_ORDER → up to 25 basis functions


# ================================================================
# 2. Geometry
# ================================================================
def make_geometry():
    return [
        mp.Block(mp.Vector3(mp.inf, mp.inf, pml_th + sub_h),
                 center=mp.Vector3(0, 0, z_bot + (pml_th + sub_h)/2),
                 material=mp.Medium(index=n_Sub)),
        mp.Block(mp.Vector3(mp.inf, mp.inf, gan_total_h),
                 center=mp.Vector3(0, 0, z_gan_c),
                 material=mp.Medium(index=n_Poly)),
        mp.Block(mp.Vector3(channel_w, channel_l, gan_total_h),
                 center=mp.Vector3(0, 0, z_gan_c),
                 material=mp.Medium(index=n_GaN)),
    ]


# ================================================================
# 3. Monitors
# ================================================================
def add_monitors(sim):
    sxy = 2 * src_box_half
    h   = src_box_z_hi - src_box_z_lo
    zc  = 0.5 * (src_box_z_lo + src_box_z_hi)
    src_box = sim.add_flux(fcen, 0, 1,
        mp.FluxRegion(mp.Vector3(0, 0, src_box_z_hi),
                      mp.Vector3(sxy, sxy, 0), direction=mp.Z, weight=+1.0),
        mp.FluxRegion(mp.Vector3(0, 0, src_box_z_lo),
                      mp.Vector3(sxy, sxy, 0), direction=mp.Z, weight=-1.0),
        mp.FluxRegion(mp.Vector3(+src_box_half, 0, zc),
                      mp.Vector3(0, sxy, h), direction=mp.X, weight=+1.0),
        mp.FluxRegion(mp.Vector3(-src_box_half, 0, zc),
                      mp.Vector3(0, sxy, h), direction=mp.X, weight=-1.0),
        mp.FluxRegion(mp.Vector3(0, +src_box_half, zc),
                      mp.Vector3(sxy, 0, h), direction=mp.Y, weight=+1.0),
        mp.FluxRegion(mp.Vector3(0, -src_box_half, zc),
                      mp.Vector3(sxy, 0, h), direction=mp.Y, weight=-1.0),
    )

    ap_sxy = 2 * ap_half
    ff_lat_h  = z_top_mon - z_ff_lat_bot
    ff_lat_zc = 0.5 * (z_top_mon + z_ff_lat_bot)
    n2f = sim.add_near2far(fcen, 0, 1,
        mp.Near2FarRegion(mp.Vector3(0, 0, z_top_mon),
                          mp.Vector3(ap_sxy, ap_sxy, 0),
                          direction=mp.Z, weight=+1.0),
        mp.Near2FarRegion(mp.Vector3(+ap_half, 0, ff_lat_zc),
                          mp.Vector3(0, ap_sxy, ff_lat_h),
                          direction=mp.X, weight=+1.0),
        mp.Near2FarRegion(mp.Vector3(-ap_half, 0, ff_lat_zc),
                          mp.Vector3(0, ap_sxy, ff_lat_h),
                          direction=mp.X, weight=-1.0),
        mp.Near2FarRegion(mp.Vector3(0, +ap_half, ff_lat_zc),
                          mp.Vector3(ap_sxy, 0, ff_lat_h),
                          direction=mp.Y, weight=+1.0),
        mp.Near2FarRegion(mp.Vector3(0, -ap_half, ff_lat_zc),
                          mp.Vector3(ap_sxy, 0, ff_lat_h),
                          direction=mp.Y, weight=-1.0),
    )
    return src_box, n2f


# ================================================================
# 4. Far-field integral
# ================================================================
def integrate_upper_hemisphere(sim, n2f,
                                R=FF_R, n_theta=FF_N_THETA, n_phi=FF_N_PHI):
    theta = np.linspace(0.0, np.pi/2, n_theta)
    phi   = np.linspace(0.0, 2*np.pi, n_phi, endpoint=False)
    dth   = theta[1] - theta[0]
    dph   = phi[1]   - phi[0]
    Sr    = np.zeros((n_theta, n_phi))
    for it, th in enumerate(theta):
        sth, cth = np.sin(th), np.cos(th)
        for ip, ph in enumerate(phi):
            cph, sph = np.cos(ph), np.sin(ph)
            r_vec = mp.Vector3(R*sth*cph, R*sth*sph, R*cth)
            ff = sim.get_farfield(n2f, r_vec)
            Ex, Ey, Ez = ff[0], ff[1], ff[2]
            Hx, Hy, Hz = ff[3], ff[4], ff[5]
            Sx = 0.5*np.real(Ey*np.conj(Hz) - Ez*np.conj(Hy))
            Sy = 0.5*np.real(Ez*np.conj(Hx) - Ex*np.conj(Hz))
            Sz = 0.5*np.real(Ex*np.conj(Hy) - Ey*np.conj(Hx))
            Sr[it, ip] = Sx*sth*cph + Sy*sth*sph + Sz*cth
    w_th = np.full(n_theta, dth)
    w_th[0]  *= 0.5
    w_th[-1] *= 0.5
    integrand = Sr * R**2 * np.sin(theta)[:, None]
    P_up = float(np.sum(integrand * w_th[:, None]) * dph)
    return P_up, Sr, theta, phi


# ================================================================
# 5. Method A — Deterministic (6×6 midpoint)
# ================================================================
def run_point_dipole(x0, y0, label=""):
    sim = mp.Simulation(
        resolution=resolution,
        cell_size=mp.Vector3(CELL_XY, CELL_XY, sz),
        boundary_layers=[mp.PML(pml_th)],
        geometry=make_geometry(),
        sources=[mp.Source(src=mp.GaussianSource(fcen, fwidth=fwidth),
                           component=mp.Ex,
                           center=mp.Vector3(x0, y0, src_z))],
        force_complex_fields=True,
        eps_averaging=True,
    )
    src_box, n2f = add_monitors(sim)
    t0 = time.time()
    sim.run(until_after_sources=mp.stop_when_dft_decayed(
        SIM_DFT_TOL, 0, SIM_MAX_TIME))
    t_fdtd = time.time() - t0
    P_src  = float(mp.get_fluxes(src_box)[0])
    t0 = time.time()
    P_up_ff, Sr, theta, phi = integrate_upper_hemisphere(sim, n2f)
    t_ff = time.time() - t0
    sim.reset_meep()
    del sim
    return dict(x0=x0, y0=y0, P_src=P_src, P_up_ff=P_up_ff,
                eta_ff=P_up_ff / max(P_src, EPS_RATIO),
                Sr=Sr, theta=theta, phi=phi,
                t_fdtd=t_fdtd, t_ff=t_ff)


def run_deterministic():
    """Method A: 6×6 midpoint, 9 unique runs."""
    mprint(f"\n{'='*60}")
    mprint("Method A — Deterministic (6×6 midpoint, 9 unique runs)")
    mprint(f"{'='*60}")

    half_range = channel_w / 2 - EDGE_INSET
    coords = np.linspace(-half_range, half_range, 6)
    pos_coords = coords[coords > 1e-12]  # positive only, 3 values

    results = []
    t_total = 0
    for ix, x in enumerate(pos_coords):
        for iy, y in enumerate(pos_coords):
            idx = ix * len(pos_coords) + iy
            mprint(f"  [{idx}] ({x:.3f}, {y:.3f}) ...")
            r = run_point_dipole(x, y, f"det_{idx}")
            r["mult"] = 4
            t_total += r["t_fdtd"] + r["t_ff"]
            mprint(f"    {r['t_fdtd']:.0f}s  P_src={r['P_src']:.4e}  "
                   f"eta_ff={r['eta_ff']*100:.3f}%")
            results.append(r)

    w_total = sum(r["mult"] for r in results)
    P_src_avg   = sum(r["mult"] * r["P_src"]   for r in results) / w_total
    P_up_ff_avg = sum(r["mult"] * r["P_up_ff"] for r in results) / w_total
    eta_ff = P_up_ff_avg / max(P_src_avg, EPS_RATIO)

    mprint(f"\n  Deterministic result: eta_ff = {eta_ff*100:.3f}%")
    mprint(f"  Total wall time: {t_total:.0f}s ({len(results)} runs)")

    return dict(eta_ff=eta_ff, P_src_avg=P_src_avg, P_up_ff_avg=P_up_ff_avg,
                n_runs=len(results), t_total=t_total, results=results)


# ================================================================
# 6. Method B — Trace formulation (cosine basis)
# ================================================================
def make_cosine_basis_list(max_order):
    """
    Generate cosine basis functions sorted by spatial frequency.
    ψ_{mn}(x,y) = norm * cos(mπ(x-x0)/Lx) * cos(nπ(y-y0)/Ly)

    Returns list of (m, n, norm) tuples.
    Sorted by m²+n² (lowest frequency first).
    """
    Lx, Ly = EMIT_LX, EMIT_LY
    basis = []
    for m in range(max_order + 1):
        for n in range(max_order + 1):
            # normalization: ∫|ψ|² dA = 1 over emission region
            norm_x = 1.0 / np.sqrt(Lx) if m == 0 else np.sqrt(2.0 / Lx)
            norm_y = 1.0 / np.sqrt(Ly) if n == 0 else np.sqrt(2.0 / Ly)
            norm = norm_x * norm_y
            basis.append((m, n, norm, m*m + n*n))
    # sort by spatial frequency
    basis.sort(key=lambda x: (x[3], x[0], x[1]))
    return [(m, n, norm) for m, n, norm, _ in basis]


def run_basis_source(m, n, norm, label=""):
    """Run FDTD with cosine basis function as source amplitude."""
    Lx, Ly = EMIT_LX, EMIT_LY

    def amp_func(pt):
        # pt is relative to source center
        x, y = pt.x, pt.y
        val_x = norm if m == 0 else norm * np.cos(m * np.pi * x / (Lx / 2))
        val_y = 1.0  if n == 0 else np.cos(n * np.pi * y / (Ly / 2))
        if m == 0:
            val_x = 1.0 / np.sqrt(Lx)
        else:
            val_x = np.sqrt(2.0 / Lx) * np.cos(m * np.pi * (x + Lx/2) / Lx)
        if n == 0:
            val_y = 1.0 / np.sqrt(Ly)
        else:
            val_y = np.sqrt(2.0 / Ly) * np.cos(n * np.pi * (y + Ly/2) / Ly)
        return val_x * val_y

    sim = mp.Simulation(
        resolution=resolution,
        cell_size=mp.Vector3(CELL_XY, CELL_XY, sz),
        boundary_layers=[mp.PML(pml_th)],
        geometry=make_geometry(),
        sources=[mp.Source(
            src=mp.GaussianSource(fcen, fwidth=fwidth),
            component=mp.Ex,
            center=mp.Vector3(0, 0, src_z),
            size=mp.Vector3(EMIT_LX, EMIT_LY, 0),
            amp_func=amp_func,
        )],
        force_complex_fields=True,
        eps_averaging=True,
    )
    src_box, n2f = add_monitors(sim)
    t0 = time.time()
    sim.run(until_after_sources=mp.stop_when_dft_decayed(
        SIM_DFT_TOL, 0, SIM_MAX_TIME))
    t_fdtd = time.time() - t0
    P_src  = float(mp.get_fluxes(src_box)[0])
    t0 = time.time()
    P_up_ff, Sr, theta, phi = integrate_upper_hemisphere(sim, n2f)
    t_ff = time.time() - t0
    sim.reset_meep()
    del sim
    return dict(m=m, n=n, P_src=P_src, P_up_ff=P_up_ff,
                Sr=Sr, theta=theta, phi=phi,
                t_fdtd=t_fdtd, t_ff=t_ff)


def run_trace():
    """Method B: Trace formulation with cosine basis, cumulative convergence."""
    mprint(f"\n{'='*60}")
    mprint("Method B — Trace formulation (cosine basis)")
    mprint(f"{'='*60}")
    mprint(f"  Emission region: {EMIT_LX:.3f} × {EMIT_LY:.3f} um")
    mprint(f"  Max basis order: {MAX_BASIS_ORDER}")

    basis_list = make_cosine_basis_list(MAX_BASIS_ORDER)
    mprint(f"  Total basis functions: {len(basis_list)}")
    for i, (m, n, norm) in enumerate(basis_list):
        mprint(f"    [{i:>2d}] (m={m}, n={n})  norm={norm:.4f}")

    all_results = []
    cumulative = []
    t_total = 0
    P_src_cum = 0.0
    P_up_ff_cum = 0.0

    for i, (m, n, norm) in enumerate(basis_list):
        mprint(f"\n  Running basis [{i}] (m={m}, n={n}) ...")
        r = run_basis_source(m, n, norm, f"basis_{m}_{n}")
        t_run = r["t_fdtd"] + r["t_ff"]
        t_total += t_run
        mprint(f"    {r['t_fdtd']:.0f}s  P_src={r['P_src']:.4e}  "
               f"P_up_ff={r['P_up_ff']:.4e}")
        all_results.append(r)

        # Cumulative sum (trace = sum of all basis contributions)
        P_src_cum += r["P_src"]
        P_up_ff_cum += r["P_up_ff"]
        eta_ff_cum = P_up_ff_cum / max(P_src_cum, EPS_RATIO)

        cumulative.append(dict(
            K=i+1, m=m, n=n,
            P_src_cum=P_src_cum, P_up_ff_cum=P_up_ff_cum,
            eta_ff=eta_ff_cum,
            t_total=t_total,
        ))
        mprint(f"    Cumulative K={i+1}: eta_ff={eta_ff_cum*100:.3f}%  "
               f"(total {t_total:.0f}s)")

    mprint(f"\n  Final trace result (K={len(basis_list)}): "
           f"eta_ff = {cumulative[-1]['eta_ff']*100:.3f}%")
    mprint(f"  Total wall time: {t_total:.0f}s ({len(basis_list)} runs)")

    return dict(
        basis_list=[(m, n, float(norm)) for m, n, norm in basis_list],
        cumulative=cumulative,
        all_results=[{k: v for k, v in r.items() if k not in ("Sr", "theta", "phi")}
                     for r in all_results],
        t_total=t_total,
    )


# ================================================================
# 7. Main
# ================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"step1_trace_comparison_{timestamp}"
if IS_MASTER:
    os.makedirs(outdir, exist_ok=True)

mprint(f"\n{'='*74}")
mprint(f"Deterministic vs Trace Formulation Comparison")
mprint(f"{'='*74}")
mprint(f"  cell_xy  = {CELL_XY:.1f} um")
mprint(f"  output   = {outdir}/")
mprint(f"{'='*74}\n")

det_result = run_deterministic()
trace_result = run_trace()


# ================================================================
# 8. Comparison
# ================================================================
mprint(f"\n{'='*74}")
mprint("COMPARISON")
mprint(f"{'='*74}")

ref_eta = det_result["eta_ff"]
mprint(f"\n  Reference (deterministic 6×6): eta_ff = {ref_eta*100:.3f}%")
mprint(f"  {'K':>4s}  {'eta_ff':>10s}  {'vs ref':>10s}  {'time':>8s}  {'speedup':>8s}")

for c in trace_result["cumulative"]:
    diff = abs(c["eta_ff"] - ref_eta) / max(abs(ref_eta), EPS_RATIO) * 100
    speedup = det_result["t_total"] / max(c["t_total"], 1)
    mprint(f"  {c['K']:>4d}  {c['eta_ff']*100:>9.3f}%  {diff:>9.2f}%  "
           f"{c['t_total']:>7.0f}s  {speedup:>7.2f}x")

# Find minimum K for < 5% error
best_K = None
for c in trace_result["cumulative"]:
    diff = abs(c["eta_ff"] - ref_eta) / max(abs(ref_eta), EPS_RATIO) * 100
    if diff < 5.0 and best_K is None:
        best_K = c["K"]

if best_K:
    speedup = det_result["t_total"] / trace_result["cumulative"][best_K-1]["t_total"]
    mprint(f"\n  → K={best_K} basis functions achieve <5% error "
           f"with {speedup:.1f}x speedup vs deterministic")
else:
    mprint(f"\n  → No K achieved <5% error — need higher max_order")

mprint(f"\n{'='*74}\n")


# ================================================================
# 9. Plots
# ================================================================
if IS_MASTER:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) eta_ff convergence
    ax = axes[0]
    Ks = [c["K"] for c in trace_result["cumulative"]]
    etas = [c["eta_ff"] * 100 for c in trace_result["cumulative"]]
    ax.plot(Ks, etas, "o-", color="C1", ms=7, lw=1.5, label="Trace (cumulative)")
    ax.axhline(ref_eta * 100, color="C0", ls="--", lw=2, label="Deterministic ref")
    ax.axhspan(ref_eta*100*0.95, ref_eta*100*1.05,
               color="C0", alpha=0.1, label="±5%")
    ax.set_xlabel("K (# basis functions)")
    ax.set_ylabel("η_ff [%]")
    ax.set_title("Trace convergence vs deterministic")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) Wall time comparison
    ax = axes[1]
    t_trace = [c["t_total"] for c in trace_result["cumulative"]]
    ax.plot(Ks, t_trace, "o-", color="C1", ms=7, label="Trace")
    ax.axhline(det_result["t_total"], color="C0", ls="--", lw=2,
               label=f"Deterministic ({det_result['n_runs']} runs)")
    ax.set_xlabel("K (# basis functions)")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Computational cost")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) Error vs speedup
    ax = axes[2]
    errs = [abs(c["eta_ff"] - ref_eta) / max(abs(ref_eta), EPS_RATIO) * 100
            for c in trace_result["cumulative"]]
    speedups = [det_result["t_total"] / max(c["t_total"], 1)
                for c in trace_result["cumulative"]]
    ax.scatter(speedups, errs, c=Ks, cmap="viridis", s=80, edgecolors="k", zorder=3)
    for i, k in enumerate(Ks):
        ax.annotate(f"K={k}", (speedups[i], errs[i]),
                    fontsize=7, ha="left", va="bottom")
    ax.axhline(5.0, color="red", ls="--", lw=1, label="5% error")
    ax.axhline(1.0, color="orange", ls="--", lw=1, label="1% error")
    ax.set_xlabel("Speedup vs deterministic")
    ax.set_ylabel("Error vs deterministic [%]")
    ax.set_title("Efficiency frontier")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Deterministic vs Trace Formulation", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(outdir, "trace_comparison.png"),
                dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ================================================================
# 10. Save JSON
# ================================================================
payload = {
    "version": "deterministic vs trace comparison",
    "setup": {
        "resolution": resolution, "cell_xy": CELL_XY,
        "emission_region": [EMIT_LX, EMIT_LY],
        "max_basis_order": MAX_BASIS_ORDER,
    },
    "deterministic": {
        "eta_ff": det_result["eta_ff"],
        "n_runs": det_result["n_runs"],
        "t_total": det_result["t_total"],
    },
    "trace": {
        "cumulative": trace_result["cumulative"],
        "basis_list": trace_result["basis_list"],
        "t_total": trace_result["t_total"],
    },
    "best_K_5pct": best_K,
}
save_json(os.path.join(outdir, "results.json"), payload)

mprint(f"Saved → {outdir}/")
mprint(f"  results.json, trace_comparison.png\n")
