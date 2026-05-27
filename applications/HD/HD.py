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

# =============================================================================
# Patched from legacy HD optimization script
# Core policy:
# - Keep FoM / Mapping / Adjoint_loop logic as much as possible
# - Update path handling and plotting for the current repo layout
# - Use the current Lumerical session-backed simulator API style
# =============================================================================

seed = 240
np.random.seed(seed)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
local_best_dir = os.path.join(RUN_DIR, "Local_bests") + os.sep
os.makedirs(design_dir, exist_ok=True)
os.makedirs(local_best_dir, exist_ok=True)
start = time.time()
# === Parameters ===
wavelength = 0.8
resolution = 50
fcen = 1 / wavelength
bandwidth = 0.2
fwidth = fcen * bandwidth

# -----------------------------------------------------------------------------
# Geometry building blocks
# Define per-axis components first, then derive the total simulation size from
# their sums so local geometry changes propagate safely.
# -----------------------------------------------------------------------------
sub_h = 0.5
design_h = 0.8
src_2_mon = 0.2
src_2_geo = 0.8
mon_2_pml = 0.1

design_span_x = 8.0
design_span_y = 8.0
x_side_gap = 0.5
y_side_gap = 0.5

Sx = design_span_x + 2 * x_side_gap
Sy = design_span_y + 2 * y_side_gap
Sz = sub_h + design_h + src_2_geo + src_2_mon + mon_2_pml

""" Custom coordinate """
X_min= round(0.5*-Sx, 2)       
Y_min= round(0.5*-Sy, 2)           
Z_min= round(0.5*-Sz, 2)           

X_max= round(0.5*Sx, 2)              
Y_max= round(0.5*Sy, 2)             
Z_max= round(0.5*Sz, 2)  



sub_s = [2 * Sx, 2 * Sy, 2 * wavelength]
sub_c = [0, 0, Z_min + sub_h - 0.5 * sub_s[2]]

design_s = [design_span_x, design_span_y, design_h]
design_c = [0, 0, Z_min + sub_h + 0.5 * design_h]

src_s = [Sx, Sy, 0]
src_c = [0, 0, design_c[2] + 0.5 * design_h + src_2_geo]

out_s = [Sx, Sy, 0]
out_c = [0, 0, src_c[2] + src_2_mon]

norm_s= [Sx, Sy, 0]
norm_c= [0, 0, src_c[2] - src_2_geo]

Nx = int(round(design_s[0] * resolution))+1
Ny = int(round(design_s[1] * resolution))+1
Nz = int(round(design_s[2] * resolution))+1
design_grids = [Nx, Ny, Nz]
design_cells = Nx * Ny * Nz              # number of parameters

N_fom=2


filename='FWD_ln3x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, -3, wavelength, 0.225, -src_2_geo, Pol='X')
FWD_Fields_n=ms.Lumerical_module.vectoral_beam_load(filename)

filename='FWD_lp3x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, 3, wavelength, 0.225, -src_2_geo, Pol='X')
FWD_Fields_p=ms.Lumerical_module.vectoral_beam_load(filename)





filename='Tar_ln3x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, -3, wavelength, 0.225, 1.0, Pol='X')
Tm3=ms.Lumerical_module.vectoral_beam_load(filename)

filename='Tar_lp3x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, 3, wavelength, 0.225, 1.0, Pol='X')
T3=ms.Lumerical_module.vectoral_beam_load(filename)

filename='Tar_ln2x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, -2, wavelength, 0.225, 1.0, Pol='X')
Tm2=ms.Lumerical_module.vectoral_beam_load(filename)

filename='Tar_lp2x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, 2, wavelength, 0.225, 1.0, Pol='X')
T2=ms.Lumerical_module.vectoral_beam_load(filename)

filename='Tar_ln1x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, -1, wavelength, 0.225, 1.0, Pol='X')
Tm1=ms.Lumerical_module.vectoral_beam_load(filename)

filename='Tar_lp1x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, 1, wavelength, 0.225, 1.0, Pol='X')
T1=ms.Lumerical_module.vectoral_beam_load(filename)

filename='Tar_l0x'
if not os.path.exists(filename+'.npz'):
    ms.Lumerical_module.Beam_save(filename, src_s, src_c, resolution, 0, wavelength, 0.225, 1.0, Pol='X')
T0=ms.Lumerical_module.vectoral_beam_load(filename)



FWD_fields=[(FWD_Fields_p), (FWD_Fields_n), Tm3, Tm2, Tm1, T0, T1, T2, T3]

for idx in range(9):
    FWD_amp=np.real(np.sqrt(np.conj(FWD_fields[idx])*FWD_fields[idx]))
    norm=np.max(FWD_amp)

    plt.subplot(2,3,1)
    plt.imshow(FWD_amp[0]/norm, cmap='RdBu', alpha=0.9, clim=(0,1.0))
    plt.axis('off')
    plt.title('$|E_x|$'.format(idx))
    plt.subplot(2,3,2)
    plt.imshow(FWD_amp[1]/norm, cmap='RdBu', alpha=0.9, clim=(0,1.0))
    plt.axis('off')
    plt.title('$|E_y|$'.format(idx)) 
    plt.subplot(2,3,3)
    plt.imshow(FWD_amp[2]/norm, cmap='RdBu', alpha=0.9, clim=(0,1.0))
    plt.axis('off')
    plt.title('$|E_z|$'.format(idx))  

    plt.subplot(2,3,4)
    plt.imshow(np.arctan(np.imag(FWD_fields[idx][0])/np.real(FWD_fields[idx][0])), cmap='binary', clim=(-3.14,3.14))
    plt.axis('off')
    plt.title('$arg(E_x)$'.format(idx))
    plt.subplot(2,3,5)
    plt.imshow(np.arctan(np.imag(FWD_fields[idx][1])/np.real(FWD_fields[idx][1])), cmap='binary', clim=(-3.14,3.14))
    plt.axis('off')
    plt.title('$arg(E_y)$'.format(idx)) 
    plt.subplot(2,3,6)
    plt.imshow(np.arctan(np.imag(FWD_fields[idx][2])/np.real(FWD_fields[idx][2])), cmap='binary', clim=(-3.14,3.14))
    plt.axis('off')
    plt.title('$arg(E_z)$'.format(idx))  


    plt.savefig(os.path.join(design_dir, f"mode{idx}.png"))
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure








""" FDTD Simulator """
FDTD_norm=True
if FDTD_norm:
    """ FDTD module """
    sim=[0]*N_fom
    Input_E=[0]*N_fom
    Input_H=[0]*N_fom

    Input_E0s=[0]*N_fom
    Input_H0s=[0]*N_fom

    Input_int=[0]*N_fom
    Input_tot=[0]*N_fom
    Input_Power=[0]*N_fom

    for idx in range(N_fom):
        sim[idx] = ms.Lumerical_utill.LumericalFDTDSimulator(
            sim_size=[Sx,Sy,Sz],
            resolution= resolution,
            unit= 1e-6,   
            background_index=1.0,
            center_wl=wavelength,
            N_f=1,
        )
        """ Source """
        sim[idx].add_source(
            #mode="eigen",
            mode="custom",
            name="source",
            center=src_c,
            size=src_s,
            direction="backward",
            src_wl=[wavelength],
            bandwidth=0,
            Fields=(FWD_fields[idx]),
        )
        """ Dft monitors """
        sim[idx].add_monitor(
            name='input_monitor',
            center=[0, 0, src_c[2]-src_2_geo],
            size=out_s,
        )

        """ Dft monitors """
        sim[idx].add_monitor(
            name='vertical_monitor',
            center=[0,0,0],
            size=[Sx, 0, Sz],
        )




        """ Normalization run """
        sim[idx].run(name=f"initialize_{idx}", save=True)


        E_in = sim[idx].fdtd.getresult('input_monitor', 'E')
        H_in = sim[idx].fdtd.getresult('input_monitor', 'H')
        E_prop = sim[idx].fdtd.getresult('vertical_monitor', 'E')

        Ein_all = np.array(E_in['E'], dtype=np.complex128)
        Hin_all = np.array(H_in['H'], dtype=np.complex128)
        Eprop_all = np.array(E_prop['E'], dtype=np.complex128)

        Ein = [Ein_all[..., 0][:, :, :, 0], Ein_all[..., 1][:, :, :, 0], Ein_all[..., 2][:, :, :, 0]]
        Hin = [Hin_all[..., 0][:, :, :, 0], Hin_all[..., 1][:, :, :, 0], Hin_all[..., 2][:, :, :, 0]]
        Eprop = [Eprop_all[..., 0][:, :, :, 0], Eprop_all[..., 1][:, :, :, 0], Eprop_all[..., 2][:, :, :, 0]]


        # === Save mode profile as PNG ===
        try:
            Ex_plot = np.real(Eprop[0])
            Ey_plot = np.real(Eprop[1])
            Ez_plot = np.real(Eprop[2])

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
            plt.savefig(os.path.join(design_dir, f'mode_profile_{idx}.png'))
            plt.close()
        except Exception as e:
            print(f'Failed to save mode profile PNG: {e}')


        Input_E[idx] = Ein
        Input_Power[idx] = ms.Opt_MS2.Cross_product(Ein, Hin)
        Input_int[idx] =npa.sum((npa.abs(Ein[0])**2) + (npa.abs(Ein[1])**2) + (npa.abs(Ein[2])**2))
        print(Input_Power[idx])

        sim[idx].fdtd.switchtolayout()
        sim[idx].fdtd.setnamed('input_monitor', 'enabled', False)
        sim[idx].fdtd.setnamed('vertical_monitor', 'enabled', False)


        """ Material and Geometry """
        sim[idx].add_geo(
            center=sub_c,
            size=sub_s,
            index=[1.45],
            name='substrate',
        )

        Initial_geo=np.zeros(design_cells)
        sim[idx].add_design_grid( 
                name="design", 
                center=design_c, 
                size=design_s, 
                index1=[2.024], 
                index2=[1.0], 
                design_grids=design_grids,
                density=Initial_geo
            )
        sim[idx].add_design_monitor()
        sim[idx].fdtd.switchtolayout()
    initialize = time.time()
    # x0=np.random.uniform(0.45, 0.55, n)
    x0=0.5*np.ones(Nx * Ny)
print(Input_Power)
""" Adjoint Optimizer """   
Optimization=True

Foms_history_n = []
Foms_history_p = []
for idx in range(8):
    Foms_history_n.append([0])
    Foms_history_p.append([0])

if Optimization:
    """ Figure of Merit Function """
    def J0(E_x, E_y, E_z, H_x, H_y):
        E_out = [E_x[:, :, :, 0], E_y[:, :, :, 0], E_z[:, :, :, 0]]
        H_out = [H_x[:, :, :, 0], H_y[:, :, :, 0], np.zeros_like(H_x[:, :, :, 0])]

        Purity= ms.Opt_MS2.Overlap_intg(T3, E_out, normalization=True, self_norm=Input_int[0])

        Pm3= ms.Opt_MS2.Overlap_intg(Tm3, E_out, normalization=True)
        Pm2= ms.Opt_MS2.Overlap_intg(Tm2, E_out, normalization=True)
        Pm1= ms.Opt_MS2.Overlap_intg(Tm1, E_out, normalization=True)
        P0= ms.Opt_MS2.Overlap_intg(T0, E_out, normalization=True)
        P1= ms.Opt_MS2.Overlap_intg(T1, E_out, normalization=True)
        P2= ms.Opt_MS2.Overlap_intg(T2, E_out, normalization=True)
        P3= ms.Opt_MS2.Overlap_intg(T3, E_out, normalization=True)

        print(Purity)
        Power=ms.Opt_MS2.Cross_product(E_out, H_out)/Input_Power[0]
        Foms=[Pm3, Pm2, Pm1, P0, P1, P2, P3, Power]
        for idx in range(8):
            if isinstance(Foms[idx], float):
                Foms_history_n[idx].append(np.real(Foms[idx]))

        FoM=(Purity)
        return (FoM)

    def J1(E_x, E_y, E_z, H_x, H_y):
        E_out = [E_x[:, :, :, 0], E_y[:, :, :, 0], E_z[:, :, :, 0]]
        H_out = [H_x[:, :, :, 0], H_y[:, :, :, 0], np.zeros_like(H_x[:, :, :, 0])]

        Pm3= ms.Opt_MS2.Overlap_intg(Tm3, E_out, normalization=True)
        Pm2= ms.Opt_MS2.Overlap_intg(Tm2, E_out, normalization=True)
        Pm1= ms.Opt_MS2.Overlap_intg(Tm1, E_out, normalization=True)
        P0= ms.Opt_MS2.Overlap_intg(T0, E_out, normalization=True)
        P1= ms.Opt_MS2.Overlap_intg(T1, E_out, normalization=True)
        P2= ms.Opt_MS2.Overlap_intg(T2, E_out, normalization=True)
        P3= ms.Opt_MS2.Overlap_intg(T3, E_out, normalization=True)

        Power=ms.Opt_MS2.Cross_product(E_out, H_out)/Input_Power[1]
        Intensity= npa.sum((npa.abs(E_out[0])**2) + (npa.abs(E_out[1])**2))/Input_int[1]
        Purity= ms.Opt_MS2.Overlap_intg(Tm3, E_out, normalization=True, self_norm=Input_int[1])
        print(Intensity)
        Foms=[Pm3, Pm2, Pm1, P0, P1, P2, P3, Power]
        for idx in range(8):
            if isinstance(Foms[idx], float):
                Foms_history_p[idx].append(np.real(Foms[idx]))

        FoM=Purity
        return (FoM)

    
    
    Js=[J0,J1]

    """ Optimization module """
    opt=[0]*N_fom
    for idx in range(N_fom):
        opt[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim[idx],
            objective_functions=[Js[idx]],
            objective_arguments=[0,1,2,3,4],
            FoM_size=out_s,
            FoM_center=out_c,
            adj_fwd=False,
            opt_idx=idx,
        )

    # A,F=opt[0].fd_ad(rho_vector=x0,N_cells=design_cells,num_gradients=10,step=1e-5)
    if True:
        DR_info= [design_s[0], design_s[1], design_s[2], 0, 1, 2]
        DR_N_info= [Nx, Ny, Nz, resolution]
        MFS= round(0.2, 2) 
        MGS= round(0.2, 2)

        mapping = ms.Opt_MS2.Mapping(
            Symmetry_sim= False,
            Sym_geo_width= False,
            Sym_geo_length= False,
            Sym_geo_C2= False,
            DR_info= DR_info,
            DR_N_info= DR_N_info,
            Mask_pixels= 5,
            MFS= MFS,
            MGS= MGS,
        )

        evaluation_historyM = [] 
        for i in range(N_fom):
            evaluation_historyM.append([0])
        forward_cnt=[]
        adjoint_cnt=[]

        Foms_history_nph= [[0],[0]]

        def Adjoint_loop(X, N_fom, Case=True):
            if Case==3: # Case 3: gradient analysis
                # Input: X=[dJ_dus, 1st order momentum gradient], N_fom=f0s
                if len(N_fom) == 1:
                    dJ_du=((X[0][0]))  
                else:
                    dJ_du=(X[0][0])-(X[0][1])
                    print(f" grad max:{np.max(np.abs(dJ_du))}")
                    print(f" grad mean:{np.mean(np.abs(dJ_du))}\n")
                for i in range(0, len(N_fom)):
                    evaluation_historyM[i].append(N_fom[i])
                return dJ_du
            else: # Case: True / False -> Last run
                #Input: X= geometry, N_fom= len(Foms)
                f0s, dJ_dus = [0]*N_fom, [0]*N_fom
                for i in range(0, N_fom): # Forward & Adjoint
                    if isinstance(X, str):
                        f0s[i], dJ_dus[i] = opt[i](need_gradient=Case)
                        adjoint_cnt.append(0)
                    else:
                        f0s[i], dJ_dus[i] = opt[i](rho_vector=[npa.clip(X, 0.0, 1.0)], need_gradient=Case)
                        forward_cnt.append(0)
                        if Case:
                            adjoint_cnt.append(0)

                f0=(1.0+((Foms_history_n[7][-1]-Foms_history_p[7][-1])/(Foms_history_n[7][-1]+Foms_history_p[7][-1])))/2
                # f0=f0s[0][0]
                for idx in range(7):
                    print(f"ell:+3 to {idx-3}:{np.round(Foms_history_p[idx][-1],3)} ell:-3 to {idx-3}:{np.round(Foms_history_n[idx][-1],3)}")
                    # print(f"ell:-3 to {idx-3}:{np.round(Foms_history_n[idx][-1],3)}")
                print(f"\n Power P:{np.round(Foms_history_p[7][-1],5)}, Power N:{np.round(Foms_history_n[7][-1],5)}")
                # print(f"\n Power N:{np.round(Foms_history_n[7][-1],5)}")
                # f0=f0s[0][0]
                if Case:      
                    for i in range(0, N_fom):
                        Foms_history_nph[i].append(f0s[i][0])                 
                    if isinstance(X, str):
                        return dJ_dus
                    else:
                        return f0, f0s, dJ_dus
                else:
                    return f0, f0s
            ##########################
        dJ_0 = np.zeros(design_cells) # initial gradient
        ##########################
        My_opt=ms.Opt_MS2.OPT_Ms(
            x0,
            dJ_0, 
            Born_k=99, 
            Initial_LR=0.2, 
        )
        My_opt(mapping, N_fom, Adjoint_loop)
        print(len(forward_cnt))
        print(len(adjoint_cnt))
        end = time.time()

        plt.figure()
        for i in range(My_opt.bt_tol):
            plt.plot(My_opt.wrong_evaluation_history[i], "r-")
        for i in range(My_opt.bt_tol):
            plt.plot(My_opt.wrong_evaluation_history2[i], "b-")
        plt.plot(My_opt.evaluation_history, "k-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("FoM")
        plt.savefig(design_dir+"result1.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        plt.figure()
        plt.plot(My_opt.evaluation_history, "k-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("FoM")
        plt.savefig(design_dir+"result0.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        plt.figure()
        plt.plot(My_opt.binarization_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Binarized")
        plt.savefig(design_dir+"result2.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        plt.figure()
        plt.plot(My_opt.learning_rate_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("LR")
        plt.yscale('log')
        plt.savefig(design_dir+"result4.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        plt.figure()
        plt.plot(My_opt.grad_mean_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Mean grad")
        plt.savefig(design_dir+"result5.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure

        plt.figure()
        plt.plot(My_opt.grad_max_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Max grad")
        plt.savefig(design_dir+"result6.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        
        plt.figure()
        plt.plot(My_opt.beta_history, "o-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("Beta")
        plt.savefig(design_dir+"result3.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure


        np.savetxt(os.path.join(design_dir, "Purity.txt"), Foms_history_nph)
        for i in range(8):
            np.savetxt(os.path.join(design_dir, f"FoMs_p{i}.txt"), Foms_history_p[i][:])
            np.savetxt(os.path.join(design_dir, f"FoMs_n{i}.txt"), Foms_history_n[i][:])

        plt.figure()
        plt.plot(Foms_history_nph[0], "r-")
        plt.plot(Foms_history_nph[1], "g-")
        plt.plot(My_opt.evaluation_history, "k-")
        plt.grid(True)
        plt.xlabel("Iteration")
        plt.ylabel("FoM")
        plt.savefig(design_dir+"result00.png")
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure


        last_design_path = os.path.join(design_dir, 'lastdesign.txt')
        if os.path.exists(last_design_path):
            Opt_design = np.loadtxt(last_design_path)

            z_slice = npa.reshape(Opt_design, (Nx, Ny, Nz))
            plt.figure()
            plt.imshow(z_slice[:, :, 0], cmap='binary')
            plt.axis('off')
            plt.savefig(os.path.join(design_dir, 'eps_last.png'))
            plt.cla()
            plt.clf()
            plt.close()

        """Post process"""
        Output_Int=[0,0]
        Output_Tot=[0,0]
        for idx in range(N_fom):
            sim[idx].fdtd.switchtolayout()
            sim[idx].add_monitor(
                name='eff_monitor',
                center=out_c,
                size=[Sx, Sy, 0],
            )
            sim[idx].fdtd.setnamed('source', 'enabled', True)
            sim[idx].run(name=f"Postprocess{idx}", save=True)
            Output_Ex=sim[idx].fdtd.getresult('eff_monitor',"Ex")
            # Input_Ex=np.array(Input_Ex[:,:,:,0])
            
            Output_Ey=sim[idx].fdtd.getresult('eff_monitor',"Ey")
            # Input_Ey=np.array(Input_Ey[:,:,:,0])

            Output_Ez=sim[idx].fdtd.getresult('eff_monitor',"Ez")
            # Input_Ez=np.array(Input_Ez[:,:,:,0])
            Output_E=np.array([Output_Ex[:,:,:,0], Output_Ey[:,:,:,0], Output_Ez[:,:,:,0]])
            sim[idx].fdtd.switchtolayout()
            Output_Int[idx]=Output_E*np.conj(Output_E)
            Output_Tot[idx]=Output_E[0]*np.conj(Output_E[0])+Output_E[1]*np.conj(Output_E[1])+Output_E[2]*np.conj(Output_E[2])


        Norm=np.max(np.real(Output_Tot))

        plt.subplot(4,2,2)
        plt.imshow(np.squeeze(np.real(Output_Int[0][0]))/Norm, cmap='Blues', alpha=0.9, clim=(0,1.0))
        plt.axis('off')
        plt.title('$N:|E_x|^2$')
        plt.subplot(4,2,3)
        plt.imshow(np.squeeze(np.real(Output_Int[0][1]))/Norm, cmap='Blues', alpha=0.9, clim=(0,1.0))
        plt.axis('off')
        plt.title('$N:|E_y|^2$') 
        plt.subplot(4,2,4)
        plt.imshow(np.squeeze(np.real(Output_Int[0][2]))/Norm, cmap='Blues', alpha=0.9, clim=(0,1.0))
        plt.axis('off')
        plt.title('$N:|E_z|^2$')  


        plt.subplot(4,2,6)
        plt.imshow(np.squeeze(np.real(Output_Int[1][0]))/Norm, cmap='Reds', alpha=0.9, clim=(0,1.0))
        plt.axis('off')
        plt.title('$P:|E_x|^2$')
        plt.subplot(4,2,7)
        plt.imshow(np.squeeze(np.real(Output_Int[1][1]))/Norm, cmap='Reds', alpha=0.9, clim=(0,1.0))
        plt.axis('off')
        plt.title('$P:|E_y|^2$') 
        plt.subplot(4,2,8)
        plt.imshow(np.squeeze(np.real(Output_Int[1][2]))/Norm, cmap='Reds', alpha=0.9, clim=(0,1.0))
        plt.axis('off')
        plt.title('$P:|E_z|^2$')  

        plt.savefig(os.path.join(design_dir, "Output_field.png"))
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure





        postpp = time.time()

        print(f"Total run time: {postpp - start:.2f} seconds")
        print(f"Initialization time: {initialize - start:.2f} seconds")
        print(f"Optimization time: {end - initialize:.2f} seconds")
        print(f"Postprocessing time: {postpp - end:.2f} seconds")
