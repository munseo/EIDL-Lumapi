import os
import time
import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as npa
from mpi4py import MPI


import lumapi
import msopt as ms

# =============================================================================
# Patched from original Meep/MPA Optimization.py
# Core policy:
# - Keep Mapping / OPT_Ms / Adjoint_loop / FoM logic as much as possible
# - Replace solver backend with Lumerical backend used in HD example
# - Fixed LN structures use anisotropic tensor materials
# - Design region uses anisotropic imported nk grid
# =============================================================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

seed = 240
np.random.seed(seed)
mode=[0, 1]  # mode[0] = forward, mode[1] = backward

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, 'A') + os.sep
local_best_dir = os.path.join(RUN_DIR, 'Local_bests') + os.sep
os.makedirs(design_dir, exist_ok=True)
os.makedirs(local_best_dir, exist_ok=True)


# =============================================================================
# Original high-level flags
# =============================================================================
Constrain = True
Modules = True
Optimization = True
Output = True

for seq in [0]:
    Overlap = True
    full_scale = False

    Min_s_top = 0.1  # um
    Capsulation = True

    if Capsulation:
        Geometry_profile = True
        Wavelengths = 1.55  # um
        Target_band = 1

        Main_Parameters = True
        Parameter_activation = True
        Monitor_Profile = True
        Source_profile = True
        Material_profile = True

        if Main_Parameters:
            resolution = 25

            design_region_x = round(0.3, 2)
            design_region_y = round(7.0, 2)
            design_region_z = round(7.0, 2)
            design_region_resolution = int(resolution)
            design_s = [design_region_x, design_region_y, design_region_z]
            design_c = [0, 0, 0]
            

            TOX_h = round(2.0, 2)
            Core_h = design_region_x
            BOX_h = round(2.0, 2)

            w_top = round(1.0, 2)


        if Parameter_activation:
            waveguide_length_I = round(1.0, 2)
            waveguide_length_O = round(1.0, 2)
            pml_2_src = round(2.0 / resolution, 2)
            mon_2_pml = round(2.0 / resolution, 2)

            Sy = round(waveguide_length_I + design_region_y + waveguide_length_O, 2)
            Sz = round(waveguide_length_I + design_region_z + waveguide_length_O, 2)
            Sx = round(BOX_h + Core_h + TOX_h, 2)

            X_min = round(-0.5 * Sx, 2)
            Y_min = round(-0.5 * Sy, 2)
            Z_min = round(-0.5 * Sz, 2)

            X_max = round(0.5 * Sx, 2)
            Y_max = round(0.5 * Sy, 2)
            Z_max = round(0.5 * Sz, 2)

            Nx = int(design_region_resolution * design_region_x) + 1
            Ny = int(design_region_resolution * design_region_y) + 1
            Nz = int(design_region_resolution * design_region_z) + 1
            design_cells = Nx * Ny * Nz
            design_grids = [Nx, Ny, Nz]

        if Monitor_Profile:
            source_center1 = [0, 0, Z_min + pml_2_src]
            source_center2 = [0, Y_min + pml_2_src, 0]
            src_c=[source_center1, source_center2]

            source_size1 = [Sx, Sy, 0]
            source_size2 = [Sx, 0, Sz]
            src_s=[source_size1, source_size2]

            Input_monitor_center = [0, 0, Z_min + waveguide_length_I]
            Output_monitor_center = [0, 0, Z_max - mon_2_pml]
            i_mon_c=[Input_monitor_center, [0, Y_min + waveguide_length_I, 0]]
            o_mon_c=[Output_monitor_center, [0, Y_max - mon_2_pml, 0]]
            
            dft_monitor_size = [Sx,Sy, 0]
            o_mon_s=[dft_monitor_size, source_size2]
            
            Top_cen = [X_min + BOX_h + Core_h / 2, 0, 0]
            Top_size = [0, Sy, Sz]
            Side_cen = [0, 0, 0]
            Side_size = [Sx, 0, Sz]

        if Material_profile:
            SiN_n = [2.0, 2.0, 2.0]
            LC_1_n = [1.5, 1.685, 1.5]
            LC_0_n = [1.5, 1.5, 1.685]
            LC_n = [LC_0_n, LC_1_n]
            SiO2_n = [1.44, 1.44, 1.44]
            Air_n = [1.0, 1.0, 1.0]

    # =========================================================================
    # Lumerical normalization / target-field preparation
    # =========================================================================
    # We create two source-normalization sims corresponding to mode[0], mode[1]
    # and use their output fields as target/noise references.
    # =========================================================================
    N_fom = 2
    Input_ints = [0] * 2
    Input_powers = [0] * 2
    Target_E_Fields = [None] * 2
    Input_E_Fields = [None] * 2
    sim_norm = [None] * 2

    for src_idx in [0, 1]:
        sim_norm[src_idx] = ms.Lumerical_utill.LumericalFDTDSimulator(
            sim_size=[Sx, Sy, Sz],
            resolution=resolution,
            unit=1e-6,
            background_index=1.44,
            center_wl=Wavelengths,
            N_f=1,
        )

        fdtd = sim_norm[src_idx].fdtd

        # Full straight input guide for normalization
        if src_idx == 0:
            sim_norm[src_idx].add_geo(
                center=[X_min + BOX_h + 0.5 *Core_h, 0, 0],
                size=[Core_h, w_top, 2*Sz],
                index=SiN_n,
                name='input_wg',
            )
            sim_norm[src_idx].add_source(
                mode='eigen',
                name='source_norm',
                center=src_c[src_idx],
                size=src_s[src_idx],
                direction='forward',
                src_wl=[Wavelengths],
                bandwidth=0.0,
                mode_num=int(1),
            )
        else:
            sim_norm[src_idx].add_geo(
                center=[X_min + BOX_h + 0.5 *Core_h, 0, 0],
                size=[Core_h, 2*Sy, w_top],
                index=SiN_n,
                name='target_wg',
            )

            sim_norm[src_idx].add_source(
                mode='eigen',
                name='source_norm',
                center=src_c[src_idx],
                size=src_s[src_idx],
                direction='forward',
                src_wl=[Wavelengths],
                bandwidth=0.0,
                mode_num=int(1),
            )

        sim_norm[src_idx].add_monitor(
            name='input_monitor',
            center=i_mon_c[src_idx],
            size=src_s[src_idx],
            N_f=1,
        )
        sim_norm[src_idx].add_monitor(
            name='output_monitor',
            center=o_mon_c[src_idx],
            size=o_mon_s[src_idx],
            N_f=1,
        )

        sim_norm[src_idx].run(name=f'norm_mode_{src_idx}', save=True)

        E_in = sim_norm[src_idx].fdtd.getresult('input_monitor', 'E')
        H_out = sim_norm[src_idx].fdtd.getresult('output_monitor', 'H')
        E_out = sim_norm[src_idx].fdtd.getresult('output_monitor', 'E')

        Ein_all = np.array(E_in['E'], dtype=np.complex128)
        Hout_all = np.array(H_out['H'], dtype=np.complex128)
        Eout_all = np.array(E_out['E'], dtype=np.complex128)

        Ein = [Ein_all[..., 0][:, :, :, 0], Ein_all[..., 1][:, :, :, 0], Ein_all[..., 2][:, :, :, 0]]
        Hout = [Hout_all[..., 0][:, :, :, 0], Hout_all[..., 1][:, :, :, 0], Hout_all[..., 2][:, :, :, 0]]
        Eout = [Eout_all[..., 0][:, :, :, 0], Eout_all[..., 1][:, :, :, 0], Eout_all[..., 2][:, :, :, 0]]

        print(np.sum(np.abs(Eout[0])))
        print(np.sum(np.abs(Eout[1])))
        print(np.sum(np.abs(Eout[2])))
        # === Save mode profile as PNG ===
        try:
            # The monitor is a plane normal to the propagation axis, so one spatial
            # axis is singleton (y-normal source -> shape (Nx,1,Nz); z-normal -> (Nx,Ny,1)).
            # squeeze() drops it to a 2D transverse profile so imshow works for both.
            Ex_plot = np.squeeze(np.real(Eout[0]))
            Ey_plot = np.squeeze(np.real(Eout[1]))
            Ez_plot = np.squeeze(np.real(Eout[2]))

            plt.figure(figsize=(8,3))
            plt.subplot(1,3,1)
            plt.imshow(Ex_plot, cmap='RdBu', origin='lower', aspect='auto')
            plt.title('Ex')
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(Ey_plot, cmap='RdBu', origin='lower', aspect='auto')
            plt.title('Ey')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(Ez_plot, cmap='RdBu', origin='lower', aspect='auto')
            plt.title('Ez')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(design_dir, f'mode_profile_{src_idx}.png'))
            plt.close()
        except Exception as e:
            print(f'Failed to save mode profile PNG: {e}')

        Input_E_Fields[src_idx] = Ein
        Target_E_Fields[src_idx] = Eout
        Input_powers[src_idx] = ms.Opt_MS2.Cross_product(Eout, Hout)
        Input_ints[src_idx] =npa.sum((npa.abs(Eout[0])**2) + (npa.abs(Eout[1])**2) + (npa.abs(Eout[2])**2))
        print(Input_powers[src_idx])

        sim_norm[src_idx].fdtd.switchtolayout()

        # Save reference fields for post-processing
        print(f"  Saving reference fields for mode {src_idx}...")
        try:
            ref_dir = os.path.join(design_dir, 'reference_fields')
            os.makedirs(ref_dir, exist_ok=True)

            np.savez(
                os.path.join(ref_dir, f'mode_{src_idx}_fields.npz'),
                mode_num=np.array([int(mode[src_idx])], dtype=np.int32),
                wavelength=np.array([Wavelengths], dtype=np.float64),
                input_Ex=Input_E_Fields[src_idx][0],
                input_Ey=Input_E_Fields[src_idx][1],
                input_Ez=Input_E_Fields[src_idx][2],
                target_Ex=Target_E_Fields[src_idx][0],
                target_Ey=Target_E_Fields[src_idx][1],
                target_Ez=Target_E_Fields[src_idx][2],
                power=np.array([float(np.real(Input_powers[src_idx]))], dtype=np.float64),
                intensity=np.array([float(np.real(Input_ints[src_idx]))], dtype=np.float64),
            )
            print(f"    ✓ Saved reference fields to reference_fields/mode_{src_idx}_fields.npz")
        except Exception as e:
            print(f"    Warning: Could not save reference fields: {e}")

    # =========================================================================
    # LC diagonal barrier (a separate object, since LC is a 3rd material that the
    # SiN/SiO2 design grid cannot represent). A rect thin in x, long along the y=z
    # diagonal, wall_thickness thick, rotated 45 deg about x. It is given a lower
    # mesh order than the design import (default 2) so it OVERRIDES the design in
    # the barrier band. The material is the switchable LC (LC_n[idx]).
    # =========================================================================
    wall_thickness = 1.0  # um (perpendicular to the y=z line)

    def add_lc_wall(sim_obj, idx, name="LC_wall"):
        fdtd = sim_obj.fdtd
        xc = X_min + BOX_h + 0.5 * Core_h                         # design/core x-center
        diag_len = np.sqrt(1.0) * max(design_region_y, design_region_z) + 2.0
        fdtd.addrect()
        fdtd.set("name", name)
        fdtd.set("x", X_max * 1e-6)
        fdtd.set("x span", 2 * TOX_h * 1e-6)
        fdtd.set("y", 0.0)
        fdtd.set("z", 0.0)
        fdtd.set("y span", diag_len * 1e-6)                      # length along the diagonal (pre-rotation)
        fdtd.set("z span", wall_thickness * 1e-6)               # perpendicular thickness
        fdtd.set("first axis", "x")
        fdtd.set("rotation 1", 45.0)                             # 45 deg about x -> centered on y=z
        # switchable anisotropic LC material
        sim_obj._set_object_index(LC_n[idx], object_name=name,
                                  material_name=f"LC_wall_mat_{idx}", wavelength=Wavelengths)
        # override the design import (mesh order 2) inside the barrier band
        try:
            fdtd.set("override mesh order from material database", True)
            fdtd.set("mesh order", 1)
        except Exception as exc:
            print(f"[LC_wall] could not set mesh order: {exc}")

    # =========================================================================
    # Optimization simulation objects
    # =========================================================================
    sim = [0] * N_fom
    opt = [0] * N_fom
    ob_list = [0] * N_fom
    Foms_history = [[0], [0]]

    for idx in range(N_fom):
        sim[idx] = ms.Lumerical_utill.LumericalFDTDSimulator(
            sim_size=[Sx, Sy, Sz],
            resolution=resolution,
            unit=1e-6,
            background_index=1.44,
            center_wl=Wavelengths,
            N_f=1,
        )

        # # LN substrate/slab
        # sim[idx].add_geo(
        #     center=[X_min + BOX_h + 0.5 * (Core_h+TOX_h), 0, 0],
        #     size=[(Core_h+TOX_h), Sy, design_region_z],
        #     index=LC_n[idx],
        #     name='LC_tranche',
        # )

        # Full straight input guide for normalization
        # Full straight input guide for normalization
        sim[idx].add_geo(
            center=[X_min + BOX_h + 0.5 *Core_h, 0, Z_min],
            size=[Core_h, w_top, 2*waveguide_length_I],
            index=SiN_n,
            name='input_wg1',
        )

        sim[idx].add_geo(
            center=[X_min + BOX_h + 0.5 *Core_h, 0,  Z_max],
            size=[Core_h, w_top, 2*waveguide_length_O],
            index=SiN_n,
            name='output_wg1',
        )


        sim[idx].add_geo(
            center=[X_min + BOX_h + 0.5 *Core_h, Y_min, 0],
            size=[Core_h, 2*waveguide_length_I, w_top],
            index=SiN_n,
            name='input_wg2',
        )

        sim[idx].add_geo(
            center=[X_min + BOX_h + 0.5 *Core_h, Y_max, 0],
            size=[Core_h, 2*waveguide_length_O, w_top],
            index=SiN_n,
            name='output_wg2',
        )


        Initial_geo = np.ones((Ny * Nz,)) * 0.5
        mapping_preview = ms.Opt_MS2.Mapping(
            Symmetry_sim=False,
            Sym_geo_width=False,
            Sym_geo_length=False,
            Sym_offdiag=True,
            Sym_geo_C2=False,
            Is_waveguide=[True, False, False, 2],
            DR_info=[design_region_x, design_region_y, design_region_z, 1, 2, 0],
            DR_N_info=[Nx, Ny, Nz, design_region_resolution],
            Mask_info=[w_top, w_top],
            Mask_pixels=0,
            MFS=Min_s_top,
            MGS=Min_s_top,
        )
        design_3d = mapping_preview(Initial_geo, 1.0)

        sim[idx].add_design_grid(
            name='design',
            center=[X_min + BOX_h + 0.5 * (Core_h), 0, Z_min + waveguide_length_I + 0.5 * design_region_z],
            size=[design_region_x, design_region_y, design_region_z],
            index1=SiN_n,
            index2=SiO2_n,
            design_grids=design_grids,
            density=design_3d,
        )
        # LC diagonal barrier overriding the design in the y=z band (switchable LC)
        add_lc_wall(sim[idx], idx)
        sim[idx].add_design_monitor()

        sim[idx].add_source(
            mode='eigen',
            name='source',
            center=src_c[0],
            size=src_s[0],
            direction='forward',
            src_wl=[Wavelengths],
            bandwidth=0.0,
            mode_num=int(1),
        )

        sim[idx].add_monitor(
            name='FoM_monitor',
            center=o_mon_c[idx],
            size=o_mon_s[idx],
            N_f=1,
        )

    # FoM function kept close to original script
    def J0(E_x, E_y, E_z, H_x, H_y, H_z):
        E_out = [E_x[:, :, :, 0], E_y[:, :, :, 0], E_z[:, :, :, 0]]
        H_out = [H_x[:, :, :, 0], H_y[:, :, :, 0], H_z[:, :, :, 0]]

        Purity = ms.Opt_MS2.Overlap_intg(Target_E_Fields[0], E_out, normalization=True)
        Purity0 = ms.Opt_MS2.Overlap_intg(Target_E_Fields[0], E_out, normalization=True, self_norm= Input_ints[0])

        Power = ms.Opt_MS2.Cross_product(E_out, H_out)
        print(Power / Input_powers[0])
        print(Power)

        if isinstance(Purity, float):
            Foms_history[0].append(np.real(Purity))
            print(f'Purity: {Purity} & Tran: {Power / Input_powers[0]}, eff: {Purity0}')
        if isinstance(Power / Input_powers[0], float):
            Foms_history[1].append(np.real(Power / Input_powers[0]))

        FoM = Purity0

        return FoM
    
    def J1(E_x, E_y, E_z, H_x, H_y, H_z):
        E_out = [E_x[:, :, :, 0], E_y[:, :, :, 0], E_z[:, :, :, 0]]
        H_out = [H_x[:, :, :, 0], H_y[:, :, :, 0], H_z[:, :, :, 0]]

        Purity = ms.Opt_MS2.Overlap_intg(Target_E_Fields[1], E_out, normalization=True)
        Purity0 = ms.Opt_MS2.Overlap_intg(Target_E_Fields[1], E_out, normalization=True, self_norm= Input_ints[0])

        Power = ms.Opt_MS2.Cross_product(E_out, H_out,axis=1)
        print(Power / Input_powers[0])
        print(Power)

        if isinstance(Purity, float):
            Foms_history[0].append(np.real(Purity))
            print(f'Purity: {Purity} & Refl: {Power / Input_powers[0]}, eff: {Purity0}')
        if isinstance(Power / Input_powers[0], float):
            Foms_history[1].append(np.real(Power / Input_powers[0]))

        FoM = Power

        return FoM

    Js=[J0, J1]
    adj_dir = [False, False]

    for idx in range(N_fom):
        opt[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim[idx],
            objective_functions=[Js[idx]],
            objective_arguments=[0, 1, 2, 3, 4, 5],
            FoM_size=o_mon_s[idx],
            FoM_center=o_mon_c[idx],
            adj_fwd=adj_dir[idx],
            opt_idx=idx,
        )

    # =========================================================================
    # Mapping and optimizer (kept from original script)
    # =========================================================================
    if Constrain:
        Is_waveguide = [True, False, False, 2]
        DR_info = [design_region_x, design_region_y, design_region_z, 1, 2, 0]
        DR_N_info = [Nx, Ny, Nz, design_region_resolution]
        Mask_info = [w_top, w_top]
        MFS = Min_s_top
        MGS = Min_s_top

        evaluation_historyM = []
        for i in range(N_fom):
            evaluation_historyM.append([0])

        mapping = ms.Opt_MS2.Mapping(
            Symmetry_sim=False,
            Sym_geo_width=False,
            Sym_geo_length=False,
            Sym_offdiag=True,
            Sym_geo_C2=False,
            Is_waveguide=Is_waveguide,
            DR_info=DR_info,
            DR_N_info=DR_N_info,
            Mask_info=Mask_info,
            Mask_pixels=0,
            MFS=MFS,
            MGS=MGS,
        )


    def design_to_grid(design, beta=1.0):
        rho = np.asarray(design, dtype=float)
        if rho.size == design_cells:
            return rho.reshape(design_grids)
        raise ValueError(f"expected {design_cells} or {Nx * Ny} design values, got {rho.size}")


    def format_design_plot_status(f0_vals=None):
        lines = []

        lines.append(f"{"Transmission"}={np.mean(f0_vals[0]):.3e}")
        lines.append(f"{"Reflection"}={np.mean(f0_vals[1]):.3e}")
        return "\n".join(lines)


    def save_current_design_sections(design, f0_vals=None):
        rho = design_to_grid(npa.clip(design, 0.0, 1.0))
        x_axis = np.linspace(-0.5 * design_s[0], 0.5 * design_s[0], Nx)
        y_axis = np.linspace(-0.5 * design_s[1], 0.5 * design_s[1], Ny)
        z_axis = np.linspace(design_c[2] - 0.5 * design_s[2], design_c[2] + 0.5 * design_s[2], Nz)

        plt.subplots(1, 1, figsize=(10, 4.8))
        plt.imshow(rho[-1], extent=(y_axis[0], y_axis[-1], z_axis[0], z_axis[-1]), cmap="binary", vmin=0.0, vmax=1.0, aspect= "equal")
        plt.xlabel("z (um)")
        plt.ylabel("y (um)")
        plt.title("y-z section at z=top")

        status_text = format_design_plot_status(f0_vals)
        plt.suptitle("Current design sections")
        if status_text:
            plt.text(0.5, 0.02, status_text, ha="center", va="bottom", fontsize=8.5)
            plt.tight_layout(rect=(0.0, 0.16, 1.0, 0.92))
        else:
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

        path = os.path.join(design_dir, "design_iter_temp.png")
        plt.savefig(path, dpi=200)
        plt.close()
        return path


    if Optimization:
        forward_cnt = []
        adjoint_cnt = []
        Foms_history0 = [[0], [0]]

        # Length (z) weighting of the adjoint gradient dJ/du: weight decreases from
        # `grad_w_near` (near the input port, low z) to `grad_w_far` (far end). The
        # gradient is (Nx, Ny, Nz) C-order flat (z = axis 2), and the design is
        # x-extruded, so the weight depends on z only. z-index 0 is the input side
        # (design z starts at Z_min + waveguide_length_I). Flip the linspace if your
        # input ends up on the far side.
        grad_w_near = 1.0
        grad_w_far = 1.0
        _z_weight = np.linspace(grad_w_near, grad_w_far, Nz)           # (Nz,)
        grad_length_weight = np.broadcast_to(_z_weight[None, None, :], (Nx, Ny, Nz)).reshape(-1)

        def apply_length_weight(g):
            g = np.asarray(g, dtype=float)
            if g.size == design_cells:
                return (g.reshape(-1) * grad_length_weight).reshape(g.shape)
            return g

        # ---- Freeze the design in the LC barrier band ----------------------------
        # The physical LC wall (add_lc_wall) OVERRIDES the design in the y=z band, so
        # the SiN/SiO2 design there is irrelevant. We still pin it to SiO2 (density 0)
        # and zero its gradient so the optimizer wastes no effort under the wall and
        # the design plot reads cleanly. Same (Nx,Ny,Nz)=(x,y,z) frame / thickness as
        # add_lc_wall. density 1 = SiN (index1), density 0 = SiO2 (index2).
        _xg = np.linspace(-0.5 * design_region_x, 0.5 * design_region_x, Nx)   # um, rel. design center
        _yg = np.linspace(-0.5 * design_region_y, 0.5 * design_region_y, Ny)
        _zg = np.linspace(-0.5 * design_region_z, 0.5 * design_region_z, Nz)
        _Xg, _Yg, _Zg = np.meshgrid(_xg, _yg, _zg, indexing="ij")              # each (Nx, Ny, Nz)

        overlay_mask = np.zeros((Nx, Ny, Nz), dtype=bool)      # True = frozen cell
        overlay_density = np.zeros((Nx, Ny, Nz), dtype=float)  # value under the wall (0 = SiO2)

        # y=z diagonal band, same perpendicular thickness as the physical LC wall:
        # perp distance to line (y - z = 0) is |y - z|/sqrt(2) <= wall_thickness/2.
        _wall = np.abs(_Zg-_Yg) <= np.sqrt(1.0) * (0.5 * wall_thickness)
        overlay_mask[_wall] = True
        overlay_density[_wall] = 0.0   # SiO2 filler (physically overridden by LC)

        overlay_mask_flat = overlay_mask.reshape(-1)
        overlay_density_flat = overlay_density.reshape(-1)
        print(f"[overlay] forcing {int(overlay_mask.sum())}/{design_cells} design cells")

        def apply_overlay(X):
            X = np.asarray(X, dtype=float).copy()
            if X.size == design_cells:
                X.reshape(-1)[overlay_mask_flat] = overlay_density_flat[overlay_mask_flat]
            return X

        def freeze_overlay_grad(g):
            g = np.asarray(g, dtype=float).copy()
            if g.size == design_cells:
                g.reshape(-1)[overlay_mask_flat] = 0.0
            return g

        def Adjoint_loop(X, N_fom, Case=True):
            if Case == 3:
                if len(N_fom) == 1:
                    dJ_du = X[0][0]
                else:
                    dJ_du = ms.Opt_MS2.Goal_Attainment(X[1], N_fom, X[0])
                    
                    dJ_du = dJ_du.reshape(Nx,Ny,Nz)
                for i in range(len(N_fom)):
                    evaluation_historyM[i].append(N_fom[i])
                return dJ_du.flatten()
            else:
                f0s, dJ_dus = [0] * N_fom, [0] * N_fom
                #if not isinstance(X, str):
                #    X = apply_overlay(X)   # impose the fixed structure this iteration
                for i in range(N_fom):
                    if isinstance(X, str):
                        f0s[i], dJ_dus[i] = opt[i](need_gradient=Case)
                        adjoint_cnt.append(0)
                        for j in range(2):
                            Foms_history0[j].append(Foms_history[j][-1])
                    else:
                        f0s[i], dJ_dus[i] = opt[i](rho_vector=[npa.clip(X, 0.0, 1.0)], need_value=True, need_gradient=Case)
                        forward_cnt.append(0)
                        if Case:
                            adjoint_cnt.append(0)
                            for j in range(2):
                                Foms_history0[j].append(Foms_history[j][-1])
                    if Case:
                        dJ_dus[i] = apply_length_weight(dJ_dus[i])
                        #dJ_dus[i] = freeze_overlay_grad(dJ_dus[i])   # freeze forced cells
                f0 = np.mean(f0s)
                if Case:
                    if isinstance(X, str):
                        return dJ_dus
                    else:
                        return f0, f0s, dJ_dus
                else:
                    path = save_current_design_sections(X, f0s)
                    print(f"[outcoupling] saved temporary design section: {path}")
                    return f0, f0s

        n = Ny * Nz
        x0 = np.ones((n,)) * 0.5
        dJ_0 = np.zeros((n * Nx,))

        My_opt = ms.Opt_MS2.OPT_Ms(x0, dJ_0, Born_k=99)
        My_opt.flag = True
        My_opt(mapping, N_fom, Adjoint_loop)

        print(len(forward_cnt))
        print(len(adjoint_cnt))

        for idx in range(2):
            np.savetxt(os.path.join(design_dir,  f'FoMS{idx}.txt'), Foms_history[idx])
            np.savetxt(os.path.join(design_dir,  f'FoMS0{idx}.txt'), Foms_history0[idx])

        plt.figure()
        for i in range(My_opt.bt_tol):
            plt.plot(My_opt.wrong_evaluation_history[i], 'r-')
        for i in range(My_opt.bt_tol):
            plt.plot(My_opt.wrong_evaluation_history2[i], 'b-')
        plt.plot(My_opt.evaluation_history, 'k-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('FoM')
        plt.savefig(os.path.join(design_dir,  'result1.png'))
        plt.close()

        plt.figure()
        for i in range(2):
            plt.plot(Foms_history0[i])
        plt.plot(My_opt.evaluation_history, 'k-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('FoM')
        plt.savefig(os.path.join(design_dir,  'result0.png'))
        plt.close()

        plt.figure()
        plt.plot(My_opt.binarization_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Binarized')
        plt.savefig(os.path.join(design_dir,  'result2.png'))
        plt.close()

        plt.figure()
        plt.plot(My_opt.learning_rate_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.savefig(os.path.join(design_dir, 'result4.png'))
        plt.close()

        plt.figure()
        plt.plot(My_opt.grad_mean_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Mean grad')
        plt.savefig(os.path.join(design_dir, 'result5.png'))
        plt.close()

        plt.figure()
        plt.plot(My_opt.grad_max_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Max grad')
        plt.savefig(os.path.join(design_dir,  'result6.png'))
        plt.close()

        plt.figure()
        plt.plot(My_opt.beta_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Beta')
        plt.savefig(os.path.join(design_dir,  'result3.png'))
        plt.close()

        if os.path.exists(os.path.join(design_dir, 'lastdesign.txt')):
            Opt_design = np.loadtxt(os.path.join(design_dir, 'lastdesign.txt'))
            z_slice = npa.reshape(Opt_design, (Nx, Ny, Nz))
            plt.figure()
            plt.imshow(z_slice[-1], origin="lower", cmap='binary')  # y up, not flipped
            plt.axis('off')
            plt.savefig(os.path.join(design_dir, 'eps_last.png'))
            plt.close()
