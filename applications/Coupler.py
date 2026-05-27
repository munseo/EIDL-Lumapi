import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
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

RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
local_best_dir = os.path.join(RUN_DIR, "Local_bests") + os.sep
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
        Wavelengths = 0.78  # um
        Main_Parameters = True
        Parameter_activation = True
        Monitor_Profile = True
        Source_profile = True
        Material_profile = True

        if Main_Parameters:
            resolution = 50

            design_region_x = round(18, 2)
            design_region_y = round(18, 2)
            design_region_z = round(0.12, 2)
            design_region_resolution = int(resolution)

            Si_h = round(2.0, 2)
            SiO2_h = round(2.0, 2)
            LNsub_h = round(0.12, 2)
            LNwg_h = design_region_z
            Air_h= round(2.0, 2)

            min_g = round(2 * design_region_z / 2.356, 2)
            input_w_top = round(1.8, 2)
            input_w_bot = round(input_w_top + min_g, 2)

        if Parameter_activation:
            waveguide_length_I = round(1.0, 2)
            Side_gap = round(1.0, 2)
            pml_2_src = round(2.0 / resolution, 2)
            mon_2_pml = round(1.0, 2)

            Sx = Side_gap + design_region_y + Side_gap
            Sy = waveguide_length_I + design_region_x + Side_gap
            Sz = round(Si_h + SiO2_h + LNsub_h + LNwg_h + SiO2_h + Air_h, 2)

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
            source_center = [0, Y_min + pml_2_src, 0]
            Input_monitor_center = [0, source_center[1] + waveguide_length_I, 0]
            source_size = [Sx/2, 0, Sz/2]
            
            Output_monitor_center = [0, 0, Z_max-mon_2_pml]
            dft_monitor_size = [Sx, Sy, 0]


        if Material_profile:
            LN_eps = [4.749, 5.1059, 5.1059]
            LN_n = [np.sqrt(v) for v in LN_eps]
            Si_n = [3.69, 3.69, 3.69]
            SiO2_n = [1.45, 1.45, 1.45]
            Air_n = [1.0, 1.0, 1.0]

    # =========================================================================
    # Lumerical normalization / target-field preparation
    # =========================================================================
    # We create two source-normalization sims corresponding to mode[0], mode[1]
    # and use their output fields as target/noise references.
    # =========================================================================
    N_fom = 1
    Input_powers = [0]
    sim_norm = [None]

    with prof.step("Step0_load_beam_fields", locals(), globals()):
        filename='Target'
        if not os.path.exists(filename+'.npz'):
            ms.Lumerical_module.Beam_save(
                filename,
                dft_monitor_size,
                Output_monitor_center,
                resolution,
                0,
                Wavelengths,
                0.062,
                -24,
                Pol='X',
            )
        Target_field=ms.Lumerical_module.vectoral_beam_load(filename)
        # === Save mode profile as PNG ===
        try:
            Ex_plot = np.real(Target_field[0])
            Ey_plot = np.real(Target_field[1])
            Ez_plot = np.real(Target_field[2])

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
            plt.savefig(os.path.join(design_dir, f'target_profile.png'))
            plt.close()
        except Exception as e:
            print(f'Failed to save mode profile PNG: {e}')

    for src_idx in [0]:
        with prof.step(f"Step1_FDTD_init[{src_idx}]", locals(), globals()):
            sim_norm[src_idx] = ms.Lumerical_utill.LumericalFDTDSimulator(
                sim_size=[Sx, Sy, Sz],
                resolution=resolution,
                unit=1e-6,
                background_index=SiO2_n[0],
                center_wl=Wavelengths,
                N_f=1,
            )

            fdtd = sim_norm[src_idx].fdtd

            # Lower SiO2 region
            sim_norm[src_idx].add_geo(
                center=[0, 0, Z_min],
                size=[Sx*2, Sy*2, Si_h*2],
                index=[Si_n[0]],
                name='si_substrate',
            )

            # sim_norm[src_idx].add_geo(
            #     center=[0, 0, Z_min + Si_h + 0.5 * SiO2_h],
            #     size=[Sx, Sy, SiO2_h],
            #     index=[SiO2_n[0]],
            #     name='BOX',
            # )

            # LN substrate/slab
            sim_norm[src_idx].add_geo(
                center=[0, 0, Z_min + Si_h + SiO2_h + 0.5 * LNsub_h],
                size=[Sx*2, Sy*2, LNsub_h],
                index=LN_n,
                name='ln_sub',
            )

            # sim_norm[src_idx].add_geo(
            #     center=[0, 0, Z_max - Air_h - 0.5 * SiO2_h],
            #     size=[Sx, Sy, SiO2_h],
            #     index=[SiO2_n[0]],
            #     name='TOX',
            # )

            sim_norm[src_idx].add_geo(
                center=[0, 0, Z_max],
                size=[Sx*2, Sy*2, Air_h*2],
                index=[Air_n[0]],
                name='TOX',
            )

            # Full straight input guide for normalization
            sim_norm[src_idx].add_waveguide(
                center=[0, 0, Z_min + Si_h + SiO2_h + LNsub_h + 0.5 * LNwg_h],
                length=Sy*2,
                height=LNwg_h,
                top_width=input_w_top,
                bottom_width=input_w_bot,
                index=LN_n,
                name='long_wg',
                prop_axis='y',
            )

            sim_norm[src_idx].add_source(
                mode='eigen',
                name='source',
                center=source_center,
                size=source_size,
                direction='forward',
                src_wl=[Wavelengths],
                bandwidth=0.0,
                mode_num=int(1),
            )

            sim_norm[src_idx].add_monitor(
                name='input_monitor',
                center=Input_monitor_center,
                size=source_size,
                N_f=1,
            )

        with prof.step(f"Step2_norm_FDTD[{src_idx}]", locals(), globals()):
            sim_norm[src_idx].run(name=f'norm_mode_{src_idx}', save=True)

        with prof.step(f"Step3_Preprocess[{src_idx}]", locals(), globals()):
            E_in = sim_norm[src_idx].fdtd.getresult('input_monitor', 'E')
            H_in = sim_norm[src_idx].fdtd.getresult('input_monitor', 'H')

            Ein_all = np.array(E_in['E'], dtype=np.complex128)
            Hin_all = np.array(H_in['H'], dtype=np.complex128)


            Ein = [Ein_all[..., 0][:, :, :, 0], Ein_all[..., 1][:, :, :, 0], Ein_all[..., 2][:, :, :, 0]]
            Hin = [Hin_all[..., 0][:, :, :, 0], Hin_all[..., 1][:, :, :, 0], Hin_all[..., 2][:, :, :, 0]]

            # === Save mode profile as PNG ===
            try:
                Ex_plot = np.real(Ein[0])
                Ey_plot = np.real(Ein[1])
                Ez_plot = np.real(Ein[2])

                plt.figure(figsize=(8,3))
                plt.subplot(1,3,1)
                plt.imshow(np.squeeze(Ex_plot), cmap='RdBu')
                plt.title('Ex')
                plt.axis('off')

                plt.subplot(1,3,2)
                plt.imshow(np.squeeze(Ey_plot), cmap='RdBu')
                plt.title('Ey')
                plt.axis('off')

                plt.subplot(1,3,3)
                plt.imshow(np.squeeze(Ez_plot), cmap='RdBu')
                plt.title('Ez')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(design_dir, f'mode_profile.png'))
                plt.close()
            except Exception as e:
                print(f'Failed to save mode profile PNG: {e}')

            Input_powers[src_idx] = ms.Opt_MS2.Cross_product(Ein, Hin, axis=1)
            print(Input_powers[src_idx])

            sim_norm[src_idx].fdtd.switchtolayout()


    # =========================================================================
    # Optimization simulation objects
    # =========================================================================
    with prof.step("Step4_Optimization_Init", locals(), globals()):
        sim = [0] * N_fom
        opt = [0] * N_fom
        ob_list = [0] * N_fom
        Foms_history = [[0], [0]]

        for idx in range(N_fom):
            sim[idx] = sim_norm[idx]
            sim[idx].fdtd.setnamed(f"long_wg", 'enabled', False)

            # Full straight input guide for normalization
            sim[idx].add_waveguide(
                center=[0, Y_min, Z_min + Si_h + SiO2_h + LNsub_h + 0.5*LNwg_h],
                length=waveguide_length_I*2,
                height=LNwg_h,
                top_width=input_w_top,
                bottom_width=input_w_bot,
                index=LN_n,
                name='input_wg',
                prop_axis='y',
            )

            Initial_geo = np.ones((Nx * Ny,)) * 0.5
            mapping_preview = ms.Opt_MS2.Mapping(
                Symmetry_sim=False,
                Sym_geo_width=False,
                Sym_geo_length=False,
                Sym_geo_C2=False,
                Is_waveguide=[True, False, True, 2],
                DR_info=[design_region_x, design_region_y, design_region_z, 0, 1, 2],
                DR_N_info=[Nx, Ny, Nz, design_region_resolution],
                Mask_info=[input_w_top, 0],
                Mask_pixels=0,
                MFS=Min_s_top,
                MGS=min_g,
            )
            design_3d = mapping_preview(Initial_geo, 1.0)

            sim[idx].add_design_grid(
                name='design',
                center=[0, 0, Z_min + Si_h + SiO2_h + LNsub_h + 0.5 * LNwg_h],
                size=[design_region_x, design_region_y, design_region_z],
                index1=LN_n,
                index2=SiO2_n,
                design_grids=design_grids,
                density=design_3d,
            )
            sim[idx].add_design_monitor()

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

                Purity = ms.Opt_MS2.Overlap_intg(Target_field, E_out, normalization=True)
                Power = ms.Opt_MS2.Cross_product(E_out, H_out)
                print(Power)
                print(f'Purity: {Purity} & Power: {Power / Input_powers[0]}')

                if isinstance(Purity, float):
                    Foms_history[0].append(np.real(Purity))
                if isinstance(Power / Input_powers[0], float):
                    Foms_history[1].append(np.real(Power / Input_powers[0]))

                if Purity < 0.9:
                    FoM = Purity
                else:
                    FoM = Purity * (Power / Input_powers[0])
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
            DR_info = [design_region_x, design_region_y, design_region_z, 0, 1, 2]
            DR_N_info = [Nx, Ny, Nz, design_region_resolution]
            Mask_info = [input_w_top, 0]
            MFS = Min_s_top
            MGS = min_g

            evaluation_historyM = []
            for i in range(N_fom):
                evaluation_historyM.append([0])

            mapping = ms.Opt_MS2.Mapping(
                Symmetry_sim=False,
                Sym_geo_width=False,
                Sym_geo_length=False,
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
                with prof.step("Coupler.Adjoint_loop", locals(), globals()):
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

            n = Nx * Ny
            x0 = np.ones((n,)) * 0.5
            dJ_0 = np.zeros((n * Nz,))

            My_opt = ms.Opt_MS2.OPT_Ms(x0, dJ_0, Born_k=99)
            My_opt.flag = True

    with prof.step("Step5_Optimization", locals(), globals()):
        My_opt(mapping, N_fom, Adjoint_loop)

    with prof.step("Step6_Postprocess", locals(), globals()):
        print(len(forward_cnt))
        print(len(adjoint_cnt))

        for idx in range(2):
            np.savetxt(design_dir + f'FoMS{idx}.txt', Foms_history[idx])
            np.savetxt(design_dir + f'FoMS0{idx}.txt', Foms_history0[idx])

        plt.figure()
        for i in range(My_opt.bt_tol):
            plt.plot(My_opt.wrong_evaluation_history[i], 'r-')
        for i in range(My_opt.bt_tol):
            plt.plot(My_opt.wrong_evaluation_history2[i], 'b-')
        plt.plot(My_opt.evaluation_history, 'k-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('FoM')
        plt.savefig(design_dir + 'result1.png')
        plt.close()

        plt.figure()
        for i in range(2):
            plt.plot(Foms_history0[i])
        plt.plot(My_opt.evaluation_history, 'k-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('FoM')
        plt.savefig(design_dir + 'result0.png')
        plt.close()

        plt.figure()
        plt.plot(My_opt.binarization_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Binarized')
        plt.savefig(design_dir + 'result2.png')
        plt.close()

        plt.figure()
        plt.plot(My_opt.learning_rate_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.savefig(design_dir + 'result4.png')
        plt.close()

        plt.figure()
        plt.plot(My_opt.grad_mean_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Mean grad')
        plt.savefig(design_dir + 'result5.png')
        plt.close()

        plt.figure()
        plt.plot(My_opt.grad_max_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Max grad')
        plt.savefig(design_dir + 'result6.png')
        plt.close()

        plt.figure()
        plt.plot(My_opt.beta_history, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration')
        plt.ylabel('Beta')
        plt.savefig(design_dir + 'result3.png')
        plt.close()

        if os.path.exists(os.path.join(design_dir, 'lastdesign.txt')):
            Opt_design = np.loadtxt(os.path.join(design_dir, 'lastdesign.txt'))
            z_slice = npa.reshape(Opt_design, (Nx, Ny, Nz))
            plt.figure()
            plt.imshow(z_slice[-1], cmap='binary')
            plt.axis('off')
            plt.savefig(os.path.join(design_dir, 'eps_last.png'))
            plt.close()
