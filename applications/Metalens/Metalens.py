import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from autograd import numpy as npa
from mpi4py import MPI

import lumapi
import msopt as ms


class _NoOpProfiler:
    class _Step:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def step(self, *args, **kwargs):
        return self._Step()


prof = _NoOpProfiler()

seed = 240
np.random.seed(seed)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get("EIDL_RUN_DIR", os.getcwd())).resolve()
design_dir = RUN_DIR / "A"
local_best_dir = RUN_DIR / "Local_bests"
design_dir.mkdir(parents=True, exist_ok=True)
local_best_dir.mkdir(parents=True, exist_ok=True)

start = time.time()


# =============================================================================
# Metalens setup from the reference sketch
# =============================================================================
wavelength = 0.63
resolution = 80
bandwidth = 0.0

aperture_x = 5.0
aperture_y = 5.0
focal_length = 2.0

simulation_height = 2.5
substrate_h = 0.10
spacer_h = 0.05
design_h = 0.2
bottom_air_h = 0.10
top_air_h = simulation_height - bottom_air_h - substrate_h - spacer_h - design_h

Sx = aperture_x
Sy = aperture_y
Sz = simulation_height

X_min = -0.5 * Sx
Y_min = -0.5 * Sy
Z_min = -0.5 * Sz
Z_max = 0.5 * Sz

z_cursor = Z_min + bottom_air_h

substrate_s = [Sx, Sy, substrate_h]
substrate_c = [0, 0, z_cursor + 0.5 * substrate_h]
z_cursor += substrate_h

spacer_s = [Sx, Sy, spacer_h]
spacer_c = [0, 0, z_cursor + 0.5 * spacer_h]
z_cursor += spacer_h

design_s = [aperture_x, aperture_y, design_h]
design_c = [0, 0, z_cursor + 0.5 * design_h]
design_top_z = z_cursor + design_h
z_cursor += design_h

source_s = [Sx, Sy, 0]
source_c = [0, 0, Z_min + 0.02]

input_monitor_s = [Sx, Sy, 0]
input_monitor_c = [0, 0, Z_min + 0.08]

focus_window_x = 1.0
focus_window_y = 1.0

fom_s = [focus_window_x, focus_window_y, 0]
fom_c = [0, 0, design_top_z + focal_length]

vertical_monitor_s = [Sx, 0, Sz]
vertical_monitor_c = [0, 0, 0]

Nx = int(round(design_s[0] * resolution)) + 1
Ny = int(round(design_s[1] * resolution)) + 1
Nz = int(round(design_s[2] * resolution)) + 1
design_grids = [Nx, Ny, Nz]
design_cells = Nx * Ny * Nz

N_fom = 1

air_n = [1.0]
substrate_n = [1.45]
spacer_n = [1.45]
high_n = [3.0]
low_n = [1.0]

target_NA = 0.80

min_feature_size = 0.1
min_gap_size = 0.1


def _field_components(result):
    arr = np.array(result["E"], dtype=np.complex128)
    return [arr[..., 0][:, :, :, 0], arr[..., 1][:, :, :, 0], arr[..., 2][:, :, :, 0]]


def _h_components(result):
    arr = np.array(result["H"], dtype=np.complex128)
    return [arr[..., 0][:, :, :, 0], arr[..., 1][:, :, :, 0], arr[..., 2][:, :, :, 0]]


def normalized_focus_intensity(E_x, E_y, E_z):
    Ex = E_x[:, :, :, 0] if E_x.ndim == 4 else E_x
    Ey = E_y[:, :, :, 0] if E_y.ndim == 4 else E_y
    Ez = E_z[:, :, :, 0] if E_z.ndim == 4 else E_z
    intensity = npa.abs(Ex) ** 2 + npa.abs(Ey) ** 2 + npa.abs(Ez) ** 2
    return npa.sum(intensity) / (Input_intensity + 1e-30)


def field_intensity(fields):
    return float(
        np.real(
            np.sum(
                np.abs(fields[0]) ** 2
                + np.abs(fields[1]) ** 2
                + np.abs(fields[2]) ** 2
            )
        )
    )


def save_field_preview(fields, path, title):
    plt.figure(figsize=(8, 3))
    for idx, name in enumerate(["Ex", "Ey", "Ez"]):
        data = np.squeeze(np.real(fields[idx]))
        plt.subplot(1, 3, idx + 1)
        plt.imshow(data, cmap="RdBu")
        plt.title(name)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_density_images(density, suffix):
    rho = np.asarray(density, dtype=float)
    if rho.size != design_cells:
        print(f"[postprocess] skip density image: expected {design_cells}, got {rho.size}")
        return
    rho = rho.reshape(design_grids)
    ix = Nx // 2
    iy = Ny // 2
    iz = Nz // 2
    x = np.linspace(-0.5 * design_s[0], 0.5 * design_s[0], Nx)
    y = np.linspace(-0.5 * design_s[1], 0.5 * design_s[1], Ny)
    z = np.linspace(design_c[2] - 0.5 * design_s[2], design_c[2] + 0.5 * design_s[2], Nz)

    plots = [
        (rho[:, iy, :].T, (x[0], x[-1], z[0], z[-1]), "x (um)", "z (um)", "xz"),
        (rho[ix, :, :].T, (y[0], y[-1], z[0], z[-1]), "y (um)", "z (um)", "yz"),
        (rho[:, :, iz].T, (x[0], x[-1], y[0], y[-1]), "x (um)", "y (um)", "xy_center"),
        (np.mean(rho, axis=2).T, (x[0], x[-1], y[0], y[-1]), "x (um)", "y (um)", "xy_zavg"),
    ]
    for data, extent, xlabel, ylabel, name in plots:
        plt.figure(figsize=(5.5, 5))
        image = plt.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        plt.colorbar(image, label="density")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(design_dir / f"metalens_density_{suffix}_{name}.png", dpi=200)
        plt.close()


def add_static_stack(sim):
    sim.add_geo(
        center=substrate_c,
        size=substrate_s,
        index=substrate_n,
        name="substrate",
    )
    sim.add_geo(
        center=spacer_c,
        size=spacer_s,
        index=spacer_n,
        name="spacer",
    )


DR_info = [design_s[0], design_s[1], design_s[2], 0, 1, 2]
DR_N_info = [Nx, Ny, Nz, resolution]
def build_mapping():
    return ms.Opt_MS2.Mapping(
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
        Is_slanted_grating=False,
    )



# =============================================================================
# Normalization simulation: incident plane-wave power/intensity
# =============================================================================
Input_power = 1.0
Input_intensity = 1.0

with prof.step("Step1_norm_FDTD_init", locals(), globals()):
    sim_norm = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[Sx, Sy, Sz],
        resolution=resolution,
        unit=1e-6,
        background_index=air_n[0],
        center_wl=wavelength,
        N_f=1,
    )
    # add_static_stack(sim_norm)
    sim_norm.add_source(
        mode="plane",
        name="source",
        center=source_c,
        size=source_s,
        direction="forward",
        src_wl=[wavelength],
        bandwidth=bandwidth,
        pol=0,
        single=True,
    )
    sim_norm.add_monitor(name="input_monitor", center=input_monitor_c, size=input_monitor_s, N_f=1)
    sim_norm.add_monitor(name="vertical_monitor", center=vertical_monitor_c, size=vertical_monitor_s, N_f=1)

with prof.step("Step2_norm_FDTD", locals(), globals()):
    sim_norm.run(name="norm_plane_wave", save=True)

with prof.step("Step3_norm_postprocess", locals(), globals()):
    Ein = _field_components(sim_norm.fdtd.getresult("input_monitor", "E"))
    Hin = _h_components(sim_norm.fdtd.getresult("input_monitor", "H"))
    Eprop = _field_components(sim_norm.fdtd.getresult("vertical_monitor", "E"))
    Input_power = ms.Opt_MS2.Cross_product(Ein, Hin)
    Input_intensity = field_intensity(Ein)
    print(f"Input_power={Input_power}")
    print(f"Input_intensity={Input_intensity}")
    save_field_preview(Eprop, design_dir / "norm_vertical_profile.png", "Normalization propagation field")
    sim_norm.fdtd.close()


# =============================================================================
# Optimization simulation
# =============================================================================
with prof.step("Step4_Optimization_Init", locals(), globals()):
    sim = [None] * N_fom
    opt = [None] * N_fom
    Foms_history = [[0], [0]]

    x0 = np.ones(Nx*Ny)*0.5
    mapping_preview = build_mapping()
    initial_design = mapping_preview(x0, 1.0)
    save_density_images(initial_design, "initial")
    np.savetxt(design_dir / "initial_density_2d.txt", x0)

    sim[0] = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[Sx, Sy, Sz],
        resolution=resolution,
        unit=1e-6,
        background_index=air_n[0],
        center_wl=wavelength,
        N_f=1,
    )
    # add_static_stack(sim[0])
    sim[0].add_source(
        mode="plane",
        name="source",
        center=source_c,
        size=source_s,
        direction="forward",
        src_wl=[wavelength],
        bandwidth=bandwidth,
        pol=0,
        single=True,
    )
    sim[0].add_design_grid(
        name="design",
        center=design_c,
        size=design_s,
        index1=high_n,
        index2=low_n,
        design_grids=design_grids,
        density=initial_design,
    )
    sim[0].add_design_monitor()
    sim[0].add_monitor(name="FoM_monitor", center=fom_c, size=fom_s, N_f=1)

    def J_focus(E_x, E_y, E_z, H_x, H_y):
        E_out = [E_x[:, :, :, 0], E_y[:, :, :, 0], E_z[:, :, :, 0]]
        H_out = [H_x[:, :, :, 0], H_y[:, :, :, 0], np.zeros_like(H_x[:, :, :, 0])]
        focus_eff = normalized_focus_intensity(E_x, E_y, E_z)
        power = ms.Opt_MS2.Cross_product(E_out, H_out) / (Input_power + 1e-30)

        if isinstance(focus_eff, float):
            Foms_history[0].append(np.real(focus_eff))
        if isinstance(power, float):
            Foms_history[1].append(np.real(power))

        print(f"focus_window_intensity={focus_eff}, transmitted_power={power}")
        return focus_eff

    opt[0] = ms.Lumerical_utill.LumericalOptimizationProblem(
        sim[0],
        objective_functions=[J_focus],
        objective_arguments=[0, 1, 2, 3, 4],
        FoM_size=fom_s,
        FoM_center=fom_c,
        adj_fwd=False,
        opt_idx=0,
    )

    mapping = build_mapping()
    evaluation_historyM = [[0]]
    forward_cnt = []
    adjoint_cnt = []
    Foms_history0 = [[0], [0]]

    def Adjoint_loop(X, N_fom, Case=True):
        with prof.step("Metalens.Adjoint_loop", locals(), globals()):
            if Case == 3:
                dJ_du = X[0][0]
                evaluation_historyM[0].append(N_fom[0])
                print(f"grad max:{np.max(np.abs(dJ_du))}")
                print(f"grad mean:{np.mean(np.abs(dJ_du))}")
                return dJ_du

            f0s, dJ_dus = [0] * N_fom, [0] * N_fom
            for idx in range(N_fom):
                if isinstance(X, str):
                    f0s[idx], dJ_dus[idx] = opt[idx](need_gradient=Case)
                    adjoint_cnt.append(0)
                else:
                    f0s[idx], dJ_dus[idx] = opt[idx](
                        rho_vector=[npa.clip(X, 0.0, 1.0)],
                        need_value=True,
                        need_gradient=Case,
                    )
                    forward_cnt.append(0)
                    if Case:
                        adjoint_cnt.append(0)

            f0 = np.mean(f0s)
            if Case:
                for hist_idx in range(2):
                    Foms_history0[hist_idx].append(Foms_history[hist_idx][-1])
                if isinstance(X, str):
                    return dJ_dus
                return f0, f0s, dJ_dus
            return f0, f0s

    dJ_0 = np.zeros(design_cells)
    optimizer = ms.Opt_MS2.OPT_Ms(
        x0,
        dJ_0,
        design_dir=str(design_dir) + os.sep,
        local_best_dir=str(local_best_dir) + os.sep,
        Born_k=99,
        Initial_LR=0.2,
    )
    optimizer.flag = True


with prof.step("Step5_Optimization", locals(), globals()):
    optimizer(mapping, N_fom, Adjoint_loop)


with prof.step("Step6_Postprocess", locals(), globals()):
    print(f"forward_cnt={len(forward_cnt)}")
    print(f"adjoint_cnt={len(adjoint_cnt)}")
    for idx, name in enumerate(["focus_window_intensity", "power"]):
        np.savetxt(design_dir / f"FoM_{name}.txt", Foms_history[idx])
        np.savetxt(design_dir / f"FoM0_{name}.txt", Foms_history0[idx])

    plt.figure()
    for idx in range(optimizer.bt_tol):
        plt.plot(optimizer.wrong_evaluation_history[idx], "r-", alpha=0.5)
        plt.plot(optimizer.wrong_evaluation_history2[idx], "b-", alpha=0.5)
    plt.plot(optimizer.evaluation_history, "k-")
    plt.grid(True)
    plt.xlabel("Iteration")
    plt.ylabel("FoM")
    plt.tight_layout()
    plt.savefig(design_dir / "result_fom.png", dpi=180)
    plt.close()

    history_plots = [
        ("binarization", optimizer.binarization_history, "Binarized"),
        ("learning_rate", optimizer.learning_rate_history, "LR"),
        ("grad_mean", optimizer.grad_mean_history, "Mean grad"),
        ("grad_max", optimizer.grad_max_history, "Max grad"),
        ("beta", optimizer.beta_history, "Beta"),
    ]
    for name, values, ylabel in history_plots:
        plt.figure()
        plt.plot(values, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        if name == "learning_rate":
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(design_dir / f"result_{name}.png", dpi=180)
        plt.close()

    last_design = design_dir / "lastdesign.txt"
    if last_design.exists():
        final_design = np.loadtxt(last_design)
        save_density_images(final_design, "final")

    runtime = time.time() - start
    summary_path = design_dir / "benchmark_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("single_wavelength_metalens_benchmark\n")
        handle.write(f"wavelength_um {wavelength}\n")
        handle.write(f"resolution_grids_per_um {resolution}\n")
        handle.write(f"simulation_size_um {Sx} {Sy} {Sz}\n")
        handle.write(f"design_size_um {design_s[0]} {design_s[1]} {design_s[2]}\n")
        handle.write(f"design_grids {Nx} {Ny} {Nz}\n")
        handle.write(f"index_range {low_n[0]} {high_n[0]}\n")
        handle.write(f"focal_length_um {focal_length}\n")
        handle.write(f"focus_window_um {focus_window_x} {focus_window_y}\n")
        handle.write("fom focal_plane_mean_absE2_over_incident_absE2\n")
        handle.write(f"runtime_sec {runtime:.6f}\n")
        handle.write(f"iterations {len(optimizer.evaluation_history)}\n")
        handle.write(f"forward_calls {len(forward_cnt)}\n")
        handle.write(f"adjoint_calls {len(adjoint_cnt)}\n")
        if optimizer.evaluation_history:
            handle.write(f"final_fom {optimizer.evaluation_history[-1]}\n")

    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Benchmark summary: {summary_path}")
