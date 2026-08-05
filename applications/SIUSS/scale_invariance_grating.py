import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import msopt as ms


# ---------------------------------------------------------------------------
# Maxwell scale-invariance test on a 3D-PBC 1D grating.
#
# Base case (scale a = 1):
#   - center wavelength  = 500 nm
#   - period             = 500 nm  (1D grating along x, uniform along y)
#   - n_bg = 1.0, n_grating = 2.0
#   - grating thickness  = 300 nm, grating bar width = 250 nm (50% duty)
#   - simulation height  = 1.5 um, PML on top/bottom, periodic in x and y
#       bottom -> center : substrate (n_grating) + patterned grating layer
#       center -> top    : air
#   - normal-incidence plane wave from the top (air), E along x
#   - E-field recorded on the y = 0, x-z plane
#
# Maxwell's equations are invariant if every length AND the wavelength are
# multiplied by the same factor a. We rerun the whole problem for
# a in {0.5, 0.8, 1.0, 1.2, 1.5, 2.0}. Using a fixed points-per-wavelength mesh,
# the discretized grid is identical for every scale, so the center-wavelength
# field on the (normalized) x-z plane should be identical. We quantify that with
# the NMSE against the a = 1 reference.
# ---------------------------------------------------------------------------


def env_float(name, default):
    return float(os.environ.get(name, str(default)))


def env_int(name, default):
    return int(os.environ.get(name, str(default)))


def env_list_float(name, default):
    raw = os.environ.get(name, "").replace(";", ",").replace(" ", ",")
    vals = [v for v in raw.split(",") if v.strip()]
    return [float(v) for v in vals] if vals else list(default)


RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
out_dir = os.path.join(RUN_DIR, "SIUSS_scale_invariance") + os.sep
os.makedirs(out_dir, exist_ok=True)

# Base geometry / physics (units of um; unit = 1e-6).
unit = 1e-6
center_wl0_um = env_float("SIUSS_CENTER_WL_UM", 0.5)     # 500 nm
period0_um = env_float("SIUSS_PERIOD_UM", 0.5)           # 500 nm
grating_thick0_um = env_float("SIUSS_GRATING_THICK_UM", 0.3)   # 300 nm
grating_width0_um = env_float("SIUSS_GRATING_WIDTH_UM", 0.25)  # 250 nm bar
sim_height0_um = env_float("SIUSS_SIM_HEIGHT_UM", 1.5)   # 1.5 um

n_bg = env_float("SIUSS_N_BG", 1.0)
n_grating = env_float("SIUSS_N_GRATING", 2.0)
n_substrate = env_float("SIUSS_N_SUBSTRATE", n_grating)  # substrate under the grating

# y-extent of the periodic cell (structure is uniform along y). A modest span is
# enough since the grating is 1D; keep it periodic for a genuine 3D-PBC run.
period_y0_um = env_float("SIUSS_PERIOD_Y_UM", period0_um)

points_per_wavelength = env_int("SIUSS_PPW", 40)  # fixed ppw -> scale-invariant mesh
polarization_angle = env_float("SIUSS_POL_DEG", 0.0)  # 0 -> E along x (TM vs 1D grating)
source_bandwidth = env_float("SIUSS_SRC_BANDWIDTH", 0.2)

scales = env_list_float("SIUSS_SCALES", [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
reference_scale = env_float("SIUSS_REFERENCE_SCALE", 1.0)


def grating_index_spec(scale):
    # Constant, non-dispersive index -> passed as a single-element list.
    return [n_grating]


def run_and_get_E(sim, monitor, run_name, retries=4):
    """Run the FDTD sim and read the monitor E field, re-running if the GPU
    session returns no monitor results (intermittent session-run failures)."""
    last_exc = None
    for attempt in range(1, retries + 1):
        sim.run(name=run_name, save=True)
        try:
            result = sim.fdtd.getresult(monitor, "E")
            if result is not None and "E" in result:
                return result
            raise RuntimeError("empty result payload")
        except Exception as exc:
            last_exc = exc
            print(
                f"[scale-test] monitor '{monitor}' has no 'E' after run "
                f"(attempt {attempt}/{retries}); re-running."
            )
            try:
                sim.fdtd.switchtolayout()
            except Exception:
                pass
    try:
        available = sim.fdtd.getresult(monitor)
    except Exception:
        available = "<could not list results>"
    raise RuntimeError(
        f"monitor '{monitor}' produced no 'E' result after {retries} runs. "
        f"Available results:\n{available}"
    ) from last_exc


def build_and_run(scale):
    """Build and run the scaled grating, return E and normalized x-z coords."""
    a = float(scale)
    wl_um = center_wl0_um * a
    px = period0_um * a
    py = period_y0_um * a
    gt = grating_thick0_um * a
    gw = grating_width0_um * a
    H = sim_height0_um * a
    z_top = 0.5 * H

    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[px, py, H],
        points_per_wavelength=points_per_wavelength,  # fixed ppw -> identical mesh
        unit=unit,
        background_index=n_bg,
        center_wl=wl_um,
        N_f=1,
        bc_x="Periodic",
        bc_y="Periodic",
        bc_z="PML",
    )

    # Substrate: fills the lower region below the grating layer, z in [-H/2, -gt].
    sub_bottom = -z_top
    sub_top = -gt
    sub_h = sub_top - sub_bottom
    if sub_h > 0:
        sim.add_geo(
            center=[0.0, 0.0, 0.5 * (sub_bottom + sub_top)],
            size=[px, py, sub_h],
            index=[n_substrate],
            name="substrate",
            wavelength=wl_um,
        )

    # Grating layer: z in [-gt, 0], one high-index bar (width gw) centered at x=0;
    # the remaining gap stays background (air). Grating top sits at the center z=0.
    sim.add_geo(
        center=[0.0, 0.0, -0.5 * gt],
        size=[gw, py, gt],
        index=grating_index_spec(a),
        name="grating_bar",
        wavelength=wl_um,
    )

    # Normal-incidence plane wave launched from inside the substrate, propagating
    # +z (upward) into the grating, then diffracting into the air above.
    z_src = -0.5 * z_top  # inside the substrate, below the grating layer
    sim.add_source(
        mode="plane",
        name="plane_src",
        center=[0.0, 0.0, z_src],
        size=[px, py, 0.0],   # zero-thickness along z -> injection axis z
        direction="Forward",
        src_wl=[wl_um],
        bandwidth=source_bandwidth,
        pol=polarization_angle,
        theta=0.0,
        phi=0.0,
    )

    # E-field monitor on the y = 0, x-z plane (zero y-span -> 2D Y-normal).
    sim.add_monitor(name="Exz", center=[0.0, 0.0, 0.0], size=[px, 0.0, H])

    run_name = os.path.join(out_dir, f"grating_scale_{a:.3f}")
    result = run_and_get_E(sim, "Exz", run_name)
    E = np.asarray(result["E"], dtype=np.complex128)
    x = np.ravel(np.asarray(result["x"], dtype=float))
    z = np.ravel(np.asarray(result["z"], dtype=float))
    E = np.squeeze(E)  # drop singleton y and frequency axes -> (Nx, Nz, 3)
    if E.ndim != 3 or E.shape[-1] != 3:
        raise ValueError(f"unexpected monitor E shape {E.shape} for scale {a}")
    if E.shape[0] == z.size and E.shape[1] == x.size:
        E = np.transpose(E, (1, 0, 2))  # normalize to (Nx, Nz, 3)
    if E.shape[0] != x.size or E.shape[1] != z.size:
        raise ValueError(f"monitor axes mismatch: E={E.shape}, x={x.size}, z={z.size}")

    try:
        sim.fdtd.close()
    except Exception:
        pass

    # Normalized coordinates (scale-independent): x/period, z/height in [-0.5, 0.5].
    xn = x / (px)
    zn = z / (H)
    return {"E": E, "xn": xn, "zn": zn, "scale": a}


def interp2_complex(F, xs, zs, xt, zt):
    """Separable bilinear interpolation of a complex 2D field onto (xt, zt)."""
    Fr, Fi = F.real, F.imag
    tmp_r = np.empty((xt.size, F.shape[1]), dtype=float)
    tmp_i = np.empty((xt.size, F.shape[1]), dtype=float)
    for j in range(F.shape[1]):
        tmp_r[:, j] = np.interp(xt, xs, Fr[:, j])
        tmp_i[:, j] = np.interp(xt, xs, Fi[:, j])
    out_r = np.empty((xt.size, zt.size), dtype=float)
    out_i = np.empty((xt.size, zt.size), dtype=float)
    for i in range(xt.size):
        out_r[i, :] = np.interp(zt, zs, tmp_r[i, :])
        out_i[i, :] = np.interp(zt, zs, tmp_i[i, :])
    return out_r + 1j * out_i


def resample_field(rec, xn_ref, zn_ref):
    """Resample all 3 field components onto the reference normalized grid."""
    E = rec["E"]
    comps = [interp2_complex(E[:, :, c], rec["xn"], rec["zn"], xn_ref, zn_ref) for c in range(3)]
    return np.stack(comps, axis=-1)


def nmse(ref, test):
    num = float(np.sum(np.abs(test - ref) ** 2))
    den = float(np.sum(np.abs(ref) ** 2))
    return num / den if den > 0 else np.nan


def nmse_phase_aligned(ref, test):
    # Optimal global complex scalar alpha minimizing ||ref - alpha*test||^2.
    denom = complex(np.vdot(test, test))
    if abs(denom) == 0:
        return np.nan
    alpha = complex(np.vdot(test, ref)) / denom
    num = float(np.sum(np.abs(ref - alpha * test) ** 2))
    den = float(np.sum(np.abs(ref) ** 2))
    return num / den if den > 0 else np.nan


def field_magnitude(E):
    return np.sqrt(np.sum(np.abs(E) ** 2, axis=-1))


def save_field_figure(records, xn_ref, zn_ref):
    n = len(records)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, rec in enumerate(records):
        ax = axes[k // ncols][k % ncols]
        ax.axis("on")
        Eg = resample_field(rec, xn_ref, zn_ref)
        mag = field_magnitude(Eg)
        im = ax.imshow(
            mag.T,
            origin="lower",
            extent=(xn_ref[0], xn_ref[-1], zn_ref[0], zn_ref[-1]),
            aspect="auto",
            cmap="inferno",
        )
        ax.set_title(f"|E|, scale a={rec['scale']:.2f}")
        ax.set_xlabel("x / period")
        ax.set_ylabel("z / height")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("y=0 x-z plane |E| (normalized coordinates)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    path = os.path.join(out_dir, "scale_fields_xz.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[scale-test] saved field montage: {path}")


def save_nmse_figure(scales_sorted, nmse_raw, nmse_aligned):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(scales_sorted, np.maximum(nmse_raw, 1e-18), "o-", label="NMSE (raw)")
    ax.semilogy(scales_sorted, np.maximum(nmse_aligned, 1e-18), "s--", label="NMSE (phase-aligned)")
    ax.set_xlabel("scale factor a  (wavelength & geometry)")
    ax.set_ylabel("NMSE vs a=%.2f field" % reference_scale)
    ax.set_title("Maxwell scale-invariance: center-wavelength field vs reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "scale_nmse.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[scale-test] saved NMSE curve: {path}")


def main():
    if os.environ.get("SIUSS_SESSION_TEST", "0").strip().lower() in ("1", "true", "yes", "on"):
        print("[scale-test] session test (no Lumerical run):")
        print(f"  scales={scales}, ppw={points_per_wavelength}, pol={polarization_angle} deg")
        for a in scales:
            print(
                f"  a={a:.2f}: wl={center_wl0_um*a*1e3:.0f} nm, period={period0_um*a*1e3:.0f} nm, "
                f"grating {grating_width0_um*a*1e3:.0f}/{period0_um*a*1e3:.0f} nm x {grating_thick0_um*a*1e3:.0f} nm, "
                f"H={sim_height0_um*a*1e3:.0f} nm"
            )
        return

    start = time.time()
    records = []
    for a in scales:
        print(f"[scale-test] running scale a={a:.3f} ...")
        rec = build_and_run(a)
        records.append(rec)
        np.savez(
            os.path.join(out_dir, f"field_scale_{a:.3f}.npz"),
            E=rec["E"], xn=rec["xn"], zn=rec["zn"], scale=rec["scale"],
        )

    # Reference field (a = reference_scale), used as the common normalized grid.
    ref_idx = int(np.argmin([abs(r["scale"] - reference_scale) for r in records]))
    ref = records[ref_idx]
    xn_ref, zn_ref = ref["xn"], ref["zn"]
    E_ref = resample_field(ref, xn_ref, zn_ref)

    order = np.argsort([r["scale"] for r in records])
    scales_sorted, nmse_raw, nmse_aligned = [], [], []
    lines = ["scale  NMSE_raw  NMSE_phase_aligned"]
    for i in order:
        rec = records[i]
        E_test = resample_field(rec, xn_ref, zn_ref)
        r = nmse(E_ref, E_test)
        ra = nmse_phase_aligned(E_ref, E_test)
        scales_sorted.append(rec["scale"])
        nmse_raw.append(r)
        nmse_aligned.append(ra)
        lines.append(f"{rec['scale']:.3f}  {r:.6e}  {ra:.6e}")
        print(f"[scale-test] a={rec['scale']:.3f}: NMSE_raw={r:.6e}, NMSE_aligned={ra:.6e}")

    scales_sorted = np.asarray(scales_sorted)
    nmse_raw = np.asarray(nmse_raw)
    nmse_aligned = np.asarray(nmse_aligned)

    with open(os.path.join(out_dir, "scale_nmse.txt"), "w", encoding="utf-8") as fp:
        fp.write(f"reference_scale {reference_scale:.3f}\n")
        fp.write(f"points_per_wavelength {points_per_wavelength}\n")
        fp.write("\n".join(lines) + "\n")

    save_field_figure(records, xn_ref, zn_ref)
    save_nmse_figure(scales_sorted, nmse_raw, nmse_aligned)

    worst = float(np.max(nmse_raw[scales_sorted != reference_scale])) if np.any(scales_sorted != reference_scale) else 0.0
    print(f"[scale-test] worst-case NMSE (excluding reference) = {worst:.3e}")
    print(f"[scale-test] done in {time.time() - start:.1f} s. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
