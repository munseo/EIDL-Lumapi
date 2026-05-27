"""
Optimizer for LNOI
History:
24/11/10 - Created
24/11/11 - Bi-direction backtracker added
24/11/12 - Capsulization on applicapable general parts

Last update: 2024/11/12 by munseong
"""
import scipy
import math
import numpy as np
import meep as mp
import meep.adjoint as mpa
import autograd.numpy as npa
from autograd import tensor_jacobian_product, grad
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from mpi4py import MPI

import os
import msopt as ms

comm= MPI.COMM_WORLD
rank= comm.Get_rank()

RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, "A") + os.sep
local_best_dir = os.path.join(RUN_DIR, "Local_bests") + os.sep
mp.verbosity(1)
# mp.verbosity.mpb =0
if not os.path.exists(design_dir):
    os.makedirs(design_dir)
if not os.path.exists(local_best_dir):
    os.makedirs(local_best_dir)

Constrain= True
Modules= True
Optimization= True
Output= True

Min_s_top= 0.1  # 100 nm

Capsulation= True
if Capsulation:
    #############################
    Geometry_profile= True      #
    Wavelengths = [0.775, 1.55] #
    Forward_modes = [2, 1]      #= 2(775nm TE00), 3(1550nm TE00)
    Backward_mode = 3           #= 8(775nm qTE20), 1(1550nm TE10)
    is_3D = True               # 2D approx?
    is_Multi = False            #"""" V1 """"
    Target_band = 1             #"""" 1: 1550, 0: 775 """"  
    #############################
    #############################
    Main_Parameters= True       #
    Parameter_activation= True  #
    Monitor_Profile= True       #
    Source_profile= True        #
    Material_profile= True      #########################################################################
    #                        ####  Set main parameters  ####                                            #
    if Main_Parameters:                                                                                 #-Parameters for Simulation--%
        resolution= int(50)                                                                             #-Resolution-----  |
        Mask_thick= 20                                                                                  #-Tickness of Mask |
        # design region                                                                                 #                     
        design_region_x = round(0.24, 2)                                                                 #-Design size-|
        design_region_y = round(3.2 + 2*round(Mask_thick/resolution,2), 2)                              #             |
        design_region_z = round(22.0 + 2*round(Mask_thick/resolution,2), 2)                             #             |
        design_region_resolution = int(resolution)                                                      #-------------|
        # Hights (x: thickness)                                                                         #-Hights (X)-----|
        SiO2_h= round(0.8, 2) # SiO2 padding                                                            #                |
        LNsub_h= round(0.06, 2) # LN substrate                                                          #                |
        LNwg_h= design_region_x # LN waveguide                                                          #----------------|
        # Width (y: width)                                                                              #-Waveguide (Y)--|
        min_g= round(0.2, 2) # minimum gap                                                             #                |
        input_w_top= round(1.85, 2)  # top width (input waveguide)                                      #                |
        input_w_bot= round(input_w_top + min_g, 2)                                                      #                |
                                                                                                        #                |
        w_top= round(1.85, 2) # top width (output waveguide)<---------------------------                #                |
        w_bot= round(w_top+ min_g, 2) #bottom width (1222 nm)                                           #----------------|
    #                     ####  Set main components and monitors  ####                                  #
    if Parameter_activation:                                                                            #--Variables for simulation--%
        Lpml= round(10.0/ resolution, 2)                                                                #-Tickness of PML|  
        # Waveguide length (z: length)                                                                  #
        waveguide_length= round(0.5, 2) # 입출력단 길이                                                  #-Propagation (Z)|
        pml_2_src= round(0.3, 2)                                                                        #                |
        mon_2_pml= round(0.3, 2)                                                                        #                |
        Prop_length= design_region_z                                                                    #----------------|
                                                                                                        #
        # Overall volume                                                                                #
        Sy = design_region_y# Width: w/o pml, 3um                                                       #             |
        Sz = round(waveguide_length+ Prop_length+ waveguide_length, 2) # 길이 (진행방향)                 #-------------|
        if is_3D:                                                                                       # 3D case     |
            Sx = round(SiO2_h+ LNsub_h+ LNwg_h+ SiO2_h, 2)                                              #-XYZ length--|
            X_tot= Sx + 2*Lpml                                                                          #             |
            cell= mp.Vector3(Sx+ 2*Lpml, Sy+ 2*Lpml, Sz+ 2*Lpml)                                        #-Total Volume|
        else:                                                                                           # 2D case     |
            Cross_section= mp.Vector3(round(SiO2_h+ LNsub_h+ LNwg_h+ SiO2_h, 2), Sy, 0)                 # Mode monitor|
            Sx = 0                                                                                      #             |
            cell= mp.Vector3(Sy+ 2*Lpml, Sz+ 2*Lpml)                                                    #-Total Volume|
        pml_layers = [mp.PML(Lpml)]#, direction=mp.Y)]                                                  #-PML---------|
                                                                                                        #
        # 공간 비례 변수 (pml 제외 유효 공간)                                                             #     
        X_min= round(0.5*-round(SiO2_h+ LNsub_h+ LNwg_h+ SiO2_h, 2), 2)                                 #-Min points--|
        Y_min= round(0.5*-Sy, 2)                                                                        #             |
        Z_min= round(0.5*-Sz, 2)                                                                        #-------------|
                                                                                                        #
        X_max= round(0.5*round(SiO2_h+ LNsub_h+ LNwg_h+ SiO2_h, 2), 2)                                  #-Max points--|
        Y_max= round(0.5*Sy, 2)                                                                         #             |
        Z_max= round(0.5*Sz, 2)                                                                         #-------------|
        if is_3D:                                                                                       # 3D case     |
            Nx = int(design_region_resolution * design_region_x)+ 1                                     #-Grid points-|
        else:                                                                                           # 2D case     |
            Nx = 1                                                                                      #-Grid points-|
        Ny = int(design_region_resolution * design_region_y)+ 1                                         #             |
        Nz = int(design_region_resolution * design_region_z)+ 1                                         #-------------|
    if Monitor_Profile:                                                                                 #---Center & Size of Monitor-%
        # Adjoint source profile<-------------------------------시뮬레이션 조건 바뀌면 체크               #
        if is_3D:                                                                                       # 3D case     |
            Adjoint_center= mp.Vector3(0, 0, Z_min+ 0.2)                                                #-Adjoint-----|
                                                                                                        #
            # Fundamental source (TE00)                                                                 #
            source_center= mp.Vector3(0, 0, Z_min+ pml_2_src)                                           #-Source------|
            source_size= mp.Vector3(X_tot, Sy+ 2*Lpml, 0)                                               #-------------|
                                                                                                        #
            dft_monitor_center= mp.Vector3(0, 0, Z_min+ 0.2)                                            #-Output------|
            dft_monitor_size= mp.Vector3(Sx, Sy, 0)                                                     #-------------|
                                                                                                        #
            Incidence_center= mp.Vector3(0, 0, Z_min+ waveguide_length)                                 #-Input-------|
            Incidence_size= mp.Vector3(Sx, Sy, 0)                                                       #-------------|
            Eff_center= mp.Vector3(0, 0, Z_max- mon_2_pml)                                              #-Transmission|
            Eff_size= mp.Vector3(Sx, Sy, 0)                                                             #-------------|
            Ref_center= mp.Vector3(0, 0, Z_min+ 0.2)                                                    #-Transmission|
            Ref_size= mp.Vector3(Sx, Sy, 0)                                                             #-------------|
            kpoint = mp.Vector3(0,0,1)                                                                  #-K vector----|
        else:                                                                                           # 2D case     |
            Adjoint_center= mp.Vector3(0, Z_min+ 0.2)                                                   #-Adjoint-----|
            # Fundamental source (TE00)                                                                 #
            source_center= mp.Vector3(0, Z_min+ pml_2_src)                                              #-Source------|
            source_size= mp.Vector3(Sy+ 2*Lpml, 0)                                                      #-------------|
                                                                                                        #
            dft_monitor_center= mp.Vector3(0, Z_min+ 0.2)                                               #-Output------|
            dft_monitor_size= mp.Vector3(Sy, 0)                                                         #-------------|
                                                                                                        #
            Incidence_center= mp.Vector3(0, Z_min+ waveguide_length)                                    #-Input-------|
            Incidence_size= mp.Vector3(Sy, 0)                                                           #-------------|
            Eff_center= mp.Vector3(0, Z_max- mon_2_pml)                                                 #-Transmission|
            Eff_size= mp.Vector3(Sy, 0)                                                                 #-------------|
            Ref_center= mp.Vector3(0, Z_min+ 0.2)                                                       #-Transmission|
            Ref_size= mp.Vector3(Sy, 0)                                                                 #-------------|     
            kpoint = mp.Vector3(0,1)                                                                    #-K vector----|       
    if Source_profile:                                                                                  #----Sources for simulation--%
        Sources=[0]*2                                                                                   #             |
        fcen=[0]*2                                                                                      #             |
        fwidth=[0]*2                                                                                    #             |
        for i in [0, 1]:                                                                                #             |
            fcen[i]= 1.0/ Wavelengths[i] # 1550 nm                                                      #             |
            bandwidth= 0.2*(Wavelengths[i]/Wavelengths[1]) # 200 nm for 1550 nm, 100 nm for 775 nm      #-Frequancy------|
            fmax=1/(Wavelengths[i]-bandwidth*0.5)                                                       #             |
            fmin=1/(Wavelengths[i]+bandwidth*0.5)                                                       #             |
            fwidth[i] = fmax-fmin                                                                       #----------------|   
            src = mp.GaussianSource(frequency= fcen[i], fwidth= fwidth[i])#, is_integrated=True)        #-Unit source-|
            Sources[i] = [                                                                              #-Input Src --|
                mp.EigenModeSource(                                                                     #             |
                    src,                                                                                #             |
                    size= source_size,                                                                  #             |
                    center= source_center,                                                              #             |
                    direction= mp.NO_DIRECTION,                                                         #             |
                    eig_kpoint= kpoint,                                                                 #             |
                    eig_band= Forward_modes[i],                                                         #             |
                    eig_match_freq= True,                                                               #             |
                ),                                                                                      #             |
            ]                                                                                           #-------------|
    if Material_profile:                                                                                #             |
        LiNbO3s= [                                                                                      #             |
            mp.Medium(epsilon_diag=mp.Vector3(5.1092, 4.7515, 5.1092)),                                 #             |
            mp.Medium(epsilon_diag=mp.Vector3(4.8855, 4.5836, 4.8855)),                                 #             |
        ]                                                                                               #             |
        SiO2s= [                                                                                        #             |
            mp.Medium(index=1.45),                                                                      #             |
            mp.Medium(index=1.44),                                                                      #             |
        ]                                                                                               #             |
        Core=[                                                                                          #             |
            mp.Medium(index=1.9938),                                                                    #             |
            mp.Medium(index=1.7805),                                                                    #             |
        ]                                                                                               #             |
        Clad=[                                                                                          #             |
            mp.Medium(index=1.652),                                                                     #             |
            mp.Medium(index=1.4766),                                                                    #             |
        ]                                                                                               #             |
        Air=mp.Medium(index=1.0)                                                                        #             |
    ########################    Should be equal to that of Optimization    ############################################

design= np.loadtxt("lastdesign.txt")
ms.Lumerical_module.GDS_converter(design, Nx, Ny, Nz, is_3D)
