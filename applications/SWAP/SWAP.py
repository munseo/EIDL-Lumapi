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
        mode = [1, 3]       # target Lumerical mode numbers
        Target_band = 1

        Main_Parameters = True
        Parameter_activation = True
        Monitor_Profile = True
        Source_profile = True
        Material_profile = True

        if Main_Parameters:
            resolution = 50

            design_region_x = round(0.3, 2)
            design_region_y = round(6.0, 2)
            design_region_z = round(10.0, 2)
            design_region_resolution = int(resolution)

            SiO2_h = round(0.65, 2)
            LNsub_h = round(0.2, 2)
            LNwg_h = design_region_x

            min_g = round(2 * design_region_x / 2.0, 2)
            input_w_top = round(2.0, 2)
            input_w_bot = round(input_w_top + min_g, 2)
            w_top = round(2.0, 2)
            w_bot = round(w_top + min_g, 2)

        if Parameter_activation:
            Lpml = round(10.0 / resolution, 2)
            waveguide_length_I = round(0.5, 2)
            waveguide_length_O = round(0.5, 2)
            pml_2_src = round(2.0 / resolution, 2)
            mon_2_pml = round(2.0 / resolution, 2)

            Sy = design_region_y + 1.0
            Sz = round(waveguide_length_I + design_region_z + waveguide_length_O, 2)
            Sx = round(SiO2_h + LNsub_h + LNwg_h + SiO2_h, 2)

            X_min = round(-0.5 * Sx, 2)
            Y_min = round(-0.5 * Sy, 2)
            Z_min = round(-0.5 * Sz, 2)

            X_max = round(0.5 * Sx, 2)
            Y_max = round(0.5 * Sy, 2)
            Z_max = round(0.5 * Sz, 2)

            Nx = int(design_region_resolution * design_region_x) + 1
            Ny = int(design_region_resolution * design_region_y) + 1
            Nz = int(design_region_resolution * design_region_z) + 1
            design_grids = [Nx, Ny, Nz]

        if Monitor_Profile:
            source_center = [0, 0, Z_min + pml_2_src]
            source_size = [Sx, Sy, 0]
            Input_monitor_center = [0, 0, Z_min + waveguide_length_I]
            Output_monitor_center = [0, 0, Z_max - mon_2_pml]
            dft_monitor_size = [Sx,Sy, 0]
            Top_cen = [X_min + SiO2_h + LNsub_h + LNwg_h / 2, 0, 0]
            Top_size = [0, Sy, Sz]
            Side_cen = [0, 0, 0]
            Side_size = [Sx, 0, Sz]

        if Material_profile:
            LN_eps = [4.8855, 4.5836, 4.8855]
            LN_n = [np.sqrt(v) for v in LN_eps]
            SiO2_n = [1.44, 1.44, 1.44]
            Air_n = [1.0, 1.0, 1.0]

    # =========================================================================
    # Lumerical normalization / target-field preparation
    # =========================================================================
    # We create two source-normalization sims corresponding to mode[0], mode[1]
    # and use their output fields as target/noise references.
    # =========================================================================
    N_fom = 1
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

        # Lower SiO2 region
        sim_norm[src_idx].add_geo(
            center=[X_min + 0.5 * SiO2_h, 0, 0],
            size=[SiO2_h, Sy, Sz],
            index=[1.44],
            name='sio2_lower',
        )

        # LN substrate/slab
        sim_norm[src_idx].add_geo(
            center=[X_min + SiO2_h + 0.5 * LNsub_h, 0, 0],
            size=[LNsub_h, Sy, Sz],
            index=LN_n,
            name='ln_sub',
        )

        # Full straight input guide for normalization
        sim_norm[src_idx].add_waveguide(
            center=[X_min + SiO2_h + LNsub_h + 0.5 * LNwg_h, 0, 0],
            length=Sz,
            height=LNwg_h,
            top_width=input_w_top,
            bottom_width=input_w_bot,
            index=LN_n,
            name='input_wg',
            prop_axis='z',
        )

        sim_norm[src_idx].add_source(
            mode='eigen',
            name='source',
            center=source_center,
            size=source_size,
            direction='forward',
            src_wl=[Wavelengths],
            bandwidth=0.0,
            mode_num=int(mode[src_idx]),
        )

        sim_norm[src_idx].add_monitor(
            name='input_monitor',
            center=Input_monitor_center,
            size=dft_monitor_size,
            N_f=1,
        )
        sim_norm[src_idx].add_monitor(
            name='output_monitor',
            center=Output_monitor_center,
            size=dft_monitor_size,
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
            Ex_plot = np.real(Eout[0])
            Ey_plot = np.real(Eout[1])
            Ez_plot = np.real(Eout[2])
	
            plt.figure(figsize=(8,3))
            plt.subplot(1,3,1)
            plt.imshow(Ex_plot, cmap='RdBu')
            plt.title('Ex')
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(Ey_plot, cmap='RdBu')
            plt.title('Ey')
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(Ez_plot, cmap='RdBu')
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

        # Lower SiO2 region
        sim[idx].add_geo(
            center=[X_min + 0.5 * SiO2_h, 0, 0],
            size=[SiO2_h, Sy, Sz],
            index=[1.44],
            name='sio2_lower',
        )

        # LN substrate/slab
        sim[idx].add_geo(
            center=[X_min + SiO2_h + 0.5 * LNsub_h, 0, 0],
            size=[LNsub_h, Sy, Sz],
            index=LN_n,
            name='ln_sub',
        )

        # Full straight input guide for normalization
        sim[idx].add_waveguide(
            center=[X_min + SiO2_h + LNsub_h + 0.5 * LNwg_h, 0, Z_min + 0.5 * waveguide_length_I],
            length=waveguide_length_I,
            height=LNwg_h,
            top_width=input_w_top,
            bottom_width=input_w_bot,
            index=LN_n,
            name='input_wg',
            prop_axis='z',
        )

        sim[idx].add_waveguide(
            center=[X_min + SiO2_h + LNsub_h + 0.5 * LNwg_h, 0, Z_max - 0.5 * waveguide_length_O],
            length=waveguide_length_O,
            height=LNwg_h,
            top_width=w_top,
            bottom_width=w_bot,
            index=LN_n,
            name='output_wg',
            prop_axis='z',
        )

        Initial_geo = np.ones((Ny * Nz,)) * 0.5
        mapping_preview = ms.Opt_MS2.Mapping(
            Symmetry_sim=False,
            Sym_geo_width=False,
            Sym_geo_length=True,
            Sym_geo_C2=False,
            Is_waveguide=[True, False, True, 2],
            DR_info=[design_region_x, design_region_y, design_region_z, 1, 2, 0],
            DR_N_info=[Nx, Ny, Nz, design_region_resolution],
            Mask_info=[input_w_top, w_top],
            Mask_pixels=0,
            MFS=Min_s_top,
            MGS=min_g,
        )
        design_3d = mapping_preview(Initial_geo, 1.0)

        sim[idx].add_design_grid(
            name='design',
            center=[X_min + SiO2_h + LNsub_h + 0.5 * LNwg_h, 0, Z_min + waveguide_length_I + 0.5 * design_region_z],
            size=[design_region_x, design_region_y, design_region_z],
            index1=LN_n,
            index2=SiO2_n,
            design_grids=design_grids,
            density=design_3d,
        )
        sim[idx].add_design_monitor()

        sim[idx].add_source(
            mode='eigen',
            name='source',
            center=source_center,
            size=source_size,
            direction='forward',
            src_wl=[Wavelengths],
            bandwidth=0.0,
            mode_num=int(mode[1]),
        )

        sim[idx].add_monitor(
            name='FoM_monitor',
            center=Output_monitor_center,
            size=dft_monitor_size,
            N_f=1,
        )

        # FoM function kept close to original script
        def J0(E_x, E_y, E_z, H_x, H_y):
            E_out = [E_x[:, :, :, 0], E_y[:, :, :, 0], E_z[:, :, :, 0]]
            H_out = [H_x[:, :, :, 0], H_y[:, :, :, 0], np.zeros_like(H_x[:, :, :, 0])]

            Purity = ms.Opt_MS2.Overlap_intg(Target_E_Fields[0], E_out, normalization=True)
            Purity0 = ms.Opt_MS2.Overlap_intg(Target_E_Fields[0], E_out, normalization=True, self_norm= Input_ints[1])
            
            Noise = ms.Opt_MS2.Overlap_intg(Target_E_Fields[1], E_out, normalization=True)
            Noise0 = ms.Opt_MS2.Overlap_intg(Target_E_Fields[1], E_out, normalization=True, self_norm= Input_ints[1])

            Power = ms.Opt_MS2.Cross_product(E_out, H_out)
            print(Power)
            print(f'Purity: {Purity} & Power: {Power / Input_powers[1]}, Noise: {Noise}, eff: {Purity0}')

            if isinstance(Purity, float):
                Foms_history[0].append(np.real(Purity))
            if isinstance(Power / Input_powers[1], float):
                Foms_history[1].append(np.real(Power / Input_powers[1]))

            FoM = Purity0

            return FoM

        opt[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim[idx],
            objective_functions=[J0],
            objective_arguments=[0, 1, 2, 3, 4],
            FoM_size=dft_monitor_size,
            FoM_center=Output_monitor_center,
            adj_fwd=False,
            opt_idx=idx,
        )

    # =========================================================================
    # Mapping and optimizer (kept from original script)
    # =========================================================================
    if Constrain:
        Is_waveguide = [True, False, True, 2]
        DR_info = [design_region_x, design_region_y, design_region_z, 1, 2, 0]
        DR_N_info = [Nx, Ny, Nz, design_region_resolution]
        Mask_info = [input_w_top, w_top]
        MFS = Min_s_top
        MGS = min_g

        evaluation_historyM = []
        for i in range(N_fom):
            evaluation_historyM.append([0])

        mapping = ms.Opt_MS2.Mapping(
            Symmetry_sim=False,
            Sym_geo_width=False,
            Sym_geo_length=True,
            Sym_geo_C2=False,
            Is_waveguide=Is_waveguide,
            DR_info=DR_info,
            DR_N_info=DR_N_info,
            Mask_info=Mask_info,
            Mask_pixels=0,
            MFS=MFS,
            MGS=MGS,
        )

    if Optimization:
        forward_cnt = []
        adjoint_cnt = []
        Foms_history0 = [[0], [0]]

        def Adjoint_loop(X, N_fom, Case=True):
            if Case == 3:
                if len(N_fom) == 1:
                    dJ_du = X[0][0]
                else:
                    dJ_du = ms.Opt_MS2.Goal_Attainment(X[1], N_fom, X[0])
                for i in range(len(N_fom)):
                    evaluation_historyM[i].append(N_fom[i])
                return dJ_du
            else:
                f0s, dJ_dus = [0] * N_fom, [0] * N_fom
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
                f0 = np.mean(f0s)
                if Case:
                    if isinstance(X, str):
                        return dJ_dus
                    else:
                        return f0, f0s, dJ_dus
                else:
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
            plt.imshow(z_slice[-1], cmap='binary')
            plt.axis('off')
            plt.savefig(os.path.join(design_dir, 'eps_last.png'))
            plt.close()
