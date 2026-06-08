"""
Module for fabrication constraints
History:
24/11/07 - Created
24/11/11 - C2 symmetry and Air mask option added for metasurface geometry
24/11/12 - Local erosion added (for middle layer extension; LNOI)
24/11/15 - Relexed

24/12/18
Last update: 2024/11/15 by munseong
"""
import numpy as np
import autograd.numpy as npa

from . import Filters as Ft

def get_conic_radius(b, eta_e):
    if (eta_e >= 0.5) and (eta_e < 0.75):
        return b / (2 * np.sqrt(eta_e - 0.5))
    elif (eta_e >= 0.75) and (eta_e <= 1):
        return b / (2 - 2 * np.sqrt(1 - eta_e))
    else:
        raise ValueError("(eta_e) must be between 0.5 and 1.")
#-----------------------------------------------------------------------------|
def tanh_projection_m(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    if beta == npa.inf:
        return npa.where(x > eta, 1.0, 0.0) # save less -> more erosion
    else:
        if eta== -0.5:
            eta=0.5
            x=npa.where(x == eta, 0.51, x)
        LR=npa.tanh(beta * eta)
        HR=npa.tanh(beta * (1 - eta))
        if beta < 30:
            return (LR + npa.tanh(beta*(x-eta)))/(LR+HR)
        else:
            return npa.where(x < eta, (LR + npa.tanh(beta*(x-eta)))/(2*LR), (HR + npa.tanh(beta*(x-eta)))/(2*HR))
"""
2D to 3D Sub mapping moudule for x-cut LN waveguide's slanted side wall (67 deg)

The code was optimized for X: heights, Y: width, Z: length (propagation axis)
"""
def apply_grbg_cell_symmetry(x, N_width, N_length, cell_gap=0):
    """
    Apply GRBG symmetry logic to the entire 2D design array.
    - x_ref: 2D array (N_width, N_length)
    - N_cells_width: number of cells along vertical (height) axis
    - N_cells_length: number of cells along horizontal (width) axis
    - cell_gap: number of pixels between cells to mask as 0 (optional)
    """
    x_ref = npa.reshape(x,(N_width,N_length))
    N_width, N_length = x_ref.shape
    cell_h = N_width // 2
    cell_w = N_length // 2

    for i in range(2):
        for j in range(2):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w

            cell = x_ref[y0:y1, x0:x1]
            h, w = cell.shape
            half_h, half_w = h // 2, w // 2

            # GRBG subregions
            G1 = cell[:half_h, :half_w]
            R  = cell[:half_h, half_w:]
            B  = cell[half_h:, :half_w]
            G2 = cell[half_h:, half_w:]

            # Symmetries
            G1_sym = 0.5 * (G1 + npa.rot90(G1, 2))
            G2_sym = npa.flipud(npa.fliplr(G1_sym))
            R_sym  = 0.5 * (R + R[:, ::-1])  # mirror_y
            B_sym  = 0.5 * (B + B[::-1, :])  # mirror_x

            # Update cell
            new_cell = cell.copy()
            new_cell = new_cell.at[:half_h, :half_w].set(G1_sym)
            new_cell = new_cell.at[half_h:, half_w:].set(G2_sym)
            new_cell = new_cell.at[:half_h, half_w:].set(R_sym)
            new_cell = new_cell.at[half_h:, :half_w].set(B_sym)

            x_ref = x_ref.at[y0:y1, x0:x1].set(new_cell)

            # Mask out border pixels if gap > 0
            if cell_gap > 0:
                if y0 + cell_gap < y1:
                    x_ref = x_ref.at[y0:y0+cell_gap, x0:x1].set(0)
                    x_ref = x_ref.at[y1-cell_gap:y1, x0:x1].set(0)
                if x0 + cell_gap < x1:
                    x_ref = x_ref.at[y0:y1, x0:x0+cell_gap].set(0)
                    x_ref = x_ref.at[y0:y1, x1-cell_gap:x1].set(0)
    return x_ref



def get_reference_layer(#-- Input parameters ---------------------|
    Symmetry_in_Sim, # Check the symmetry in simulation           |
    Width_symmetry,  # Width symmetry in geometry                 |
    Length_symmetry, # Length symmetry in geometry                |
    C2_symmetry,     # C2 symmetry in geometry                    |
    QR_symmetry,     # Quater Rotation symmetry                   |
    Pseudo_Cyl,      # Peudo cylindrical symmetry                 |
    Is_waveguide,    #                                            |
    Sub_pixels,      #                                            |
    #                                                             |
    DR_width, # Disign region width                               |
    N_width,  # Number of DR width grids                          |
    #                                                             |
    DR_length, # Disign region length                             |
    N_length,  # Number of DR length grids                        |
    #                                                             |
    DR_res, # Design region resolution                            |
    #                                                             |
    Min_size_top, # Minimum linwidth of top layer                 |
    Min_gap, # Minimum gap width of ref layer                     |
    #                                                             |
    input_w_top, # Top width of input waveguide                   |
    w_top, # Top width of output waveguide                        |
    #                                                             |
    x, # Input material density array                             |
    #                                                             |
    beta,     # Binrization parameter                             |
    Mask_pixels, # Number of pixels for mask region               |
):                                                                #-- Impose MFS and MGS on x ------------------------|
    eta_Ref = 0.05                                                      # Thresholding point of tanh projection       |
    Single_pixel= round(1/DR_res,3)                                                          #                        |
    Total_R = get_conic_radius(Min_size_top+ Min_gap, 1-eta_Ref)                             # Radius for MFS + MGS   |
    MGS_R = get_conic_radius((Min_gap), 1-eta_Ref)                                           # Radius for MGS         |
    N_total = round(Total_R/(4*Single_pixel))                     # Total number of Erosion & Dilation with R= 4pixel |
    N_Half = round(Total_R/(8*Single_pixel))                      # Half number of Erosion & Dilation with R= 4pixel  |
    # Set filter R, number of dilation & erosion & smoothing ---------------------------------------------------------|
    if N_total > 1:                               # Total_R > 5 pixel ------------------------------------------------|
        N_total = 2                                                                          #                        |
        if N_Half < 1:                                      # Half_R < 5 pixel ---------------------------------------|
            N_Half = 1                                                                       #                        |
        N_dilation = N_total*N_Half                               # Number of dilation = 2*N_Half                     |
        Dilation_R = get_conic_radius((Min_gap + Min_size_top)/(N_dilation), 1-eta_Ref)      # R for unit dilation    |
        Half_R = (Min_gap + Min_size_top)/2                                                  # R for Half smoothing   |
        if Min_gap > Min_size_top:                         # MGS > MFS  ----------------------------------------------|
            MGS_R = get_conic_radius((Min_gap - Min_size_top)/2, 1-eta_Ref)                  # R for half erosion     |
            N_MGS = round(MGS_R/(4*Single_pixel))              # Number of unit erosion with R= 4pixel                |
            if N_MGS < 1:                                      # MGS_R < 2 pixel -------------------------------------|
                N_MGS = 1                                                                    #                        |
            Erosion_R = get_conic_radius((Min_gap - Min_size_top)/(2*N_MGS), 1-eta_Ref)      # R for unit erosion     |
            Smoothing_E = True                                                   # Half smoothing during erosion      |
            Smoothing_D = False                                                  #                                    |
            N_erosion = N_Half + N_MGS                                           # Number of erosion = N_Half + N_MGS |
        else:                                               # MGS =< MFS ---------------------------------------------|
            N_MGS = round(MGS_R/(4*Single_pixel))              # Number of unit erosion with R= 4pixel                |
            if N_MGS < 1:                                      # MGS_R < 2 pixel -------------------------------------|
                N_MGS = 1                                                                    #                        |
            Erosion_R = get_conic_radius((Min_gap)/(N_MGS), 1-eta_Ref)                       # R for unit erosion     |
            Smoothing_E = False                                                  #                                    |
            Smoothing_D = True                                                   # Half smoothing during dilation     |
            N_erosion = N_MGS                                                    # Number of erosion = N_MGS          |
    else:                                         # Total_R <= 5 pixel -----------------------------------------------|
        N_dilation = 1                                                           # Single dilation                    |
        N_erosion = 1                                                            # Single erosion                     |
        Dilation_R = Total_R                                                     # R for dilation = MGS + MFS         |
        Erosion_R = MGS_R                                                        # R for erosion = MGS                |
        Smoothing_D = False                                                      # Deactivate half smoothing          |
        Smoothing_E = False                                                      # Deactivate half smoothing          |
    #-----------------------------------------------------------------------------------------------------------------|
    if Mask_pixels == 0:                                                                     #                        |
        if Is_waveguide:                                                                     #                        |
            Air_pad= int((Dilation_R + Single_pixel)/Single_pixel)+1                         # Size of Air pad        |
        else:                                                                                #                        |
            Air_pad= int(((Min_gap)/2+ Dilation_R + Single_pixel)/Single_pixel)+1            # Size of Air pad        |
            if QR_symmetry:
                if N_width != N_length:
                    print("Width and Length should be equal for the rotation!")
                    return 0
        if Sub_pixels[0]:
            x= apply_grbg_cell_symmetry(x, N_width, N_length, Sub_pixels[1])
        x= npa.pad(
            npa.reshape(x,(N_width,N_length)),
            ((Air_pad, Air_pad), (Air_pad, Air_pad)),
            mode="constant", constant_values=0)
        x= x.flatten()                                                                       #                        |
        N_length += 2*Air_pad                                                                #                        |
        N_width  += 2*Air_pad                                                                #                        |
        # Keep the padded physical span exactly consistent with the padded grid count.
        # Rounding Air_pad / DR_res can make Filters.mesh_grid expect one fewer grid.
        DR_width = (N_width - 1) / DR_res                                                    #                        |
        DR_length = (N_length - 1) / DR_res                                                  #                        |
        print("Use Air pad")                                                                 #                        |
    else:                                                                                    #                        |
        Air_pad= 0                                                                           #                        |
    Minimum_mask= (Min_gap + Min_size_top)/2+ Dilation_R + Single_pixel                      # Minimum size of Mask   |
    if Mask_pixels <= Minimum_mask/Single_pixel:                                             #                        |
        Mask_pixels = int(Minimum_mask/Single_pixel)+1                                       #                        |
        # print(f"Mask size changed to {Mask_pixels} Pixels ")                               #                        |
    width_grid = np.linspace(-DR_width /2, DR_width /2, N_width)                             #                        |
    length_grid = np.linspace(-DR_length /2, DR_length /2, N_length)                         #                        |
    W_g, L_g = np.meshgrid(width_grid, length_grid, sparse=True, indexing="ij")              #                        |
    border_mask = (                                                                          # Cladding mask ---------|
        (W_g >= W_g[N_width-1-Mask_pixels])                                                  #                        |
        | (W_g <= W_g[Mask_pixels])                                                          #                        |
        | (L_g <= L_g[:,Mask_pixels])                                                        #                        |
        | (L_g >= L_g[:,N_length-1-Mask_pixels])                                             #                        |
    )                                                                                        #------------------------|
    Air_mask = border_mask.copy()                                                            # Mask ------------------|
    if Is_waveguide: #-----------------------------------------------------------------------# Define the mask region |
        Mask_sz_ii = round(0.5*(input_w_top)/Single_pixel)                                   # Core mask -------------|
        Mask_sz_io = round(0.5*(w_top)/Single_pixel)                                         #                        |
        down_wg_mask_init= (                                                                 #                        |
            (L_g <=L_g[:,Mask_pixels])                                                       #                        |
            & (np.abs(W_g)<=W_g[int((N_width-1)/2+ Mask_sz_ii)])                             #                        |
            )                                                                                #                        |
        up_wg_mask_init= (                                                                   #                        |
            (L_g >=L_g[:,N_length-1-Mask_pixels])                                            #                        |
            & (np.abs(W_g)<=W_g[round((N_width-1)/2+Mask_sz_io)])                            #                        |
            )                                                                                #                        |
        Li_mask_init = down_wg_mask_init | up_wg_mask_init                                   #                        |
        Air_mask[Li_mask_init] = False                                                       #                        |
    #-----------------------------------------------------------------------------------------------------------------|
    x_ref = npa.reshape(x,(N_width,N_length))  # Reshape for flip ----------------------------------------------------|
    if Width_symmetry:                                                                       # Width Mirror ----------|
        if Symmetry_in_Sim:                                                                  #   Y Mirror Sim Case    |
            print('Gradient has Width symmetry')                                             #                        |
            x_ref = ((npa.flipud(x_ref)) + x_ref)/2      
            print('ODD pixels & EVEN grids') 
        else:                                                                  
            # if N_width/2 == int(N_width/2):                
            x_ref = ((npa.flipud(x_ref)) + x_ref)/2      
            print('ODD pixels & EVEN grids')             
            # else:                                          
            #     x_center=x_ref[int(N_width/2),:] <----------------Currently Unsupports Autograd, Need to fix
            #     x_ref = ((npa.flipud(x_ref)) + x_ref)/2   
            #     x_ref[int(N_width/2),:] = x_center
            #     print('EVEN pixels & ODD grids')          
    if Length_symmetry:                                                                      # Length Mirror ---------|
        x_ref = (npa.fliplr(x_ref) + x_ref)/2                                                #                        |
    if C2_symmetry:                                                                          # C2 --------------------|
        x_ref = (npa.rot90(x_ref,2)+ x_ref)/2                                                #                        |
    if QR_symmetry:                                                                          # 45 deg rot ------------|
        #x_ref = (npa.rot90(x_ref,2).transpose()+ x_ref)/2                                   #   off diag             |
        x_ref = (x_ref.transpose()+ x_ref)/2                                                 #   diag                 |
    if Pseudo_Cyl:
        x_ref = (npa.fliplr(x_ref) + x_ref)/2
        x_ref = (npa.flipud(x_ref) + x_ref)/2
        x_ref = (x_ref.transpose() + x_ref)/2
    x_ref = x_ref.flatten()                                                                  #                        |
    if Is_waveguide:                                                                         # Masking            ----|
        x_ref = npa.where(Li_mask_init.flatten(), 1, npa.where(Air_mask.flatten(), 0, x_ref))#                        |
    else:                                                                                    #                        |
        x_ref = npa.where(Air_mask.flatten(), 0, x_ref)                                      #                        |
    #-----------------------------------------------------------------------------------------------------------------|
    x_copy = tanh_projection_m(x_ref, beta, 0.5)                                            # Binarization ----------|
    # x_copy = tanh_projection_m(x_copy, beta, -0.5)                                           # Binarization ----------|
    Dilation_index= 0                                                                        #                        |
    while Dilation_index < N_dilation:    #------- Activate multiple dilation N_dilation times -----------------------|
        x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Dilation_R, DR_width, DR_length, DR_res)    #             |
        x_copy = tanh_projection_m(x_copy, beta, eta_Ref)                                 # Unit dilation             |
        if Smoothing_D and Dilation_index == round(N_dilation/2)-1: # MGS < MFS, Smoothing when MFS = MGS ------------|
            x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)              #   |
            x_copy = tanh_projection_m(x_copy, beta, 0.5)                                 # Half smoothing        #   |
            x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)              #   |
            x_copy = tanh_projection_m(x_copy, beta, 0.5)                                 # Compensation smoothing    |
        Dilation_index +=1                                                                                        #   |
    Erosion_index= 0                                                                                              #   |
    while Erosion_index < N_erosion:      #--------- Activate multiple erosion N_erosion times -----------------------|
        if Smoothing_E:                             # MGS >= MFS -----------------------------------------------------|
            while Erosion_index < N_Half:            # Erosion with Dilation radius ----------------------------------|
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Dilation_R, DR_width, DR_length, DR_res)      #   |
                x_copy = tanh_projection_m(x_copy, beta, 1-eta_Ref)                       # Unit erosion          #   |
                Erosion_index +=1                                                                                 #   |
            if Erosion_index == N_Half:              # Smoothing when MFS = MGS --------------------------------------|
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)          #   |
                x_copy = tanh_projection_m(x_copy, beta, 0.5)                             # Half smoothing        #   |
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)          #   |
                x_copy = tanh_projection_m(x_copy, beta, 0.5)                             # Compensation smoothing    |
        x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Erosion_R, DR_width, DR_length, DR_res)               #   |
        x_copy = tanh_projection_m(x_copy, beta, 1-eta_Ref)                               # Unit erosion              |
        Erosion_index +=1                                                                                         #   |
    if Air_pad > 0:                                                                                               #   |
        x_copy= x_copy[Air_pad:(N_width-Air_pad),Air_pad:(N_length-Air_pad)]                                      #   |
    x_copy = x_copy.flatten()                                                                                     #   |
    del x_ref                                     # Clear Memory -----------------------------------------------------|
    return x_copy                                                                                # Reference layer ---|
#---------------------------------------------------------------------------------------------------------------------|


def get_reference_layer_1D(#-- Input parameters ---------------------------------|
    #                                                             |
    DR_length, # Disign region length                             |
    N_length,  # Number of DR length grids                        |
    peudo_width,
    peudo_N_width,
    DR_res, # Design region resolution                            |
    #                                                             |
    Min_size_top, # Minimum linwidth of top layer                 |
    Min_gap, # Minimum gap width of ref layer                     |
    #                                                             |
    x, # Input material density array                             |
    #                                                             |
    beta,     # Binrization parameter                             |
    is_slanted,
):                                                                #-- Impose MFS and MGS on x ------------------------|
    eta_Ref = 0.05                                                      # Thresholding point of tanh projection       |
    Single_pixel= round(1/DR_res,3)                                                          #                        |
    Total_R = get_conic_radius(Min_size_top+ Min_gap, 1-eta_Ref)                             # Radius for MFS + MGS   |
    MGS_R = get_conic_radius((Min_gap), 1-eta_Ref)                                           # Radius for MGS         |
    N_total = round(Total_R/(4*Single_pixel))                     # Total number of Erosion & Dilation with R= 4pixel |
    N_Half = round(Total_R/(8*Single_pixel))                      # Half number of Erosion & Dilation with R= 4pixel  |
    # Set filter R, number of dilation & erosion & smoothing ---------------------------------------------------------|
    if N_total > 1:                               # Total_R > 5 pixel ------------------------------------------------|
        N_total = 2                                                                          #                        |
        if N_Half < 1:                                      # Half_R < 5 pixel ---------------------------------------|
            N_Half = 1                                                                       #                        |
        N_dilation = N_total*N_Half                               # Number of dilation = 2*N_Half                     |
        Dilation_R = get_conic_radius((Min_gap + Min_size_top)/(N_dilation), 1-eta_Ref)      # R for unit dilation    |
        Half_R = (Min_gap + Min_size_top)/2                                                  # R for Half smoothing   |
        if Min_gap > Min_size_top:                         # MGS > MFS  ----------------------------------------------|
            MGS_R = get_conic_radius((Min_gap - Min_size_top)/2, 1-eta_Ref)                  # R for half erosion     |
            N_MGS = round(MGS_R/(4*Single_pixel))              # Number of unit erosion with R= 4pixel                |
            if N_MGS < 1:                                      # MGS_R < 2 pixel -------------------------------------|
                N_MGS = 1                                                                    #                        |
            Erosion_R = get_conic_radius((Min_gap - Min_size_top)/(2*N_MGS), 1-eta_Ref)      # R for unit erosion     |
            Smoothing_E = True                                                   # Half smoothing during erosion      |
            Smoothing_D = False                                                  #                                    |
            N_erosion = N_Half + N_MGS                                           # Number of erosion = N_Half + N_MGS |
        else:                                               # MGS =< MFS ---------------------------------------------|
            N_MGS = round(MGS_R/(4*Single_pixel))              # Number of unit erosion with R= 4pixel                |
            if N_MGS < 1:                                      # MGS_R < 2 pixel -------------------------------------|
                N_MGS = 1                                                                    #                        |
            Erosion_R = get_conic_radius((Min_gap)/(N_MGS), 1-eta_Ref)                       # R for unit erosion     |
            Smoothing_E = False                                                  #                                    |
            Smoothing_D = True                                                   # Half smoothing during dilation     |
            N_erosion = N_MGS                                                    # Number of erosion = N_MGS          |
    else:                                         # Total_R <= 5 pixel -----------------------------------------------|
        N_dilation = 1                                                           # Single dilation                    |
        N_erosion = 1                                                            # Single erosion                     |
        Dilation_R = Total_R                                                     # R for dilation = MGS + MFS         |
        Erosion_R = MGS_R                                                        # R for erosion = MGS                |
        Smoothing_D = False                                                      # Deactivate half smoothing          |
        Smoothing_E = False                                                      # Deactivate half smoothing          |
    #-----------------------------------------------------------------------------------------------------------------|
    L_g = np.linspace(-DR_length /2, DR_length /2, N_length)
    Air_pad= int(((Min_gap)/2+ Dilation_R + Single_pixel)/Single_pixel)+1                # Size of Air pad        |
    x_2D = npa.where(L_g <= L_g[Air_pad], 1, npa.where(L_g >= L_g[N_length-1-Air_pad], 1, x))
    x_2D = npa.tile(x_2D,(peudo_N_width,1))
    x_2D= npa.pad(                                  
        x_2D,                                       
        ((Air_pad, Air_pad), (Air_pad, Air_pad)),   
        mode="constant", constant_values=0)         
    DR_length = DR_length+ 2*Air_pad*Single_pixel
    N_length  += 2*Air_pad 
    DR_width = peudo_width + 2*Air_pad*Single_pixel
    N_width = peudo_N_width + 2*Air_pad
    # raise ValueError(f"{x_2D.size}, {Air_pad}, {x.size}, {peudo_N_width}")
    #-----------------------------------------------------------------------------------------------------------------|
    x_copy = tanh_projection_m(x_2D.flatten(), beta, 0.5)                                   # Binarization ----------|
    # x_copy = tanh_projection_m(x_copy.flatten(), beta, -0.5)                                 # Binarization ----------|
    Dilation_index= 0                                                                        #                        |
    while Dilation_index < N_dilation:    #------- Activate multiple dilation N_dilation times -----------------------|
        x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Dilation_R, DR_width, DR_length, DR_res)    #             |
        x_copy = tanh_projection_m(x_copy, beta, eta_Ref)                                 # Unit dilation             |
        if Smoothing_D and Dilation_index == round(N_dilation/2)-1: # MGS < MFS, Smoothing when MFS = MGS ------------|
            x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)              #   |
            x_copy = tanh_projection_m(x_copy, beta, 0.5)                                 # Half smoothing        #   |
            x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)              #   |
            x_copy = tanh_projection_m(x_copy, beta, 0.5)                                 # Compensation smoothing    |
        Dilation_index +=1                                                                                        #   |
    Erosion_index= 0                                                                                              #   |
    while Erosion_index < N_erosion:      #--------- Activate multiple erosion N_erosion times -----------------------|
        if Smoothing_E:                             # MGS >= MFS -----------------------------------------------------|
            while Erosion_index < N_Half:            # Erosion with Dilation radius ----------------------------------|
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Dilation_R, DR_width, DR_length, DR_res)      #   |
                x_copy = tanh_projection_m(x_copy, beta, 1-eta_Ref)                       # Unit erosion          #   |
                Erosion_index +=1                                                                                 #   |
            if Erosion_index == N_Half:              # Smoothing when MFS = MGS --------------------------------------|
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)          #   |
                x_copy = tanh_projection_m(x_copy, beta, 0.5)                             # Half smoothing        #   |
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)          #   |
                x_copy = tanh_projection_m(x_copy, beta, 0.5)                             # Compensation smoothing    |
        x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Erosion_R, DR_width, DR_length, DR_res)               #   |
        x_copy = tanh_projection_m(x_copy, beta, 1-eta_Ref)                               # Unit erosion              |
        Erosion_index +=1                                                                                         #   |
    if Air_pad > 0:                                                                                               #   |
        x_copy= x_copy[Air_pad:(N_width-Air_pad),Air_pad:(N_length-Air_pad)]                                      #   |
        N_length -= 2*Air_pad
        N_width -= 2*Air_pad
    if not is_slanted:
        x_copy= x_copy[int((N_width)/2),:]                                                                       #   |
    x_copy = x_copy.flatten()                                                                                     #   | 
    del x_2D                                      # Clear Memory -----------------------------------------------------|
    return x_copy                                                                                # Reference layer ---|
#---------------------------------------------------------------------------------------------------------------------|

def eta_z(z, eta_top, bottom, top,
): #- Height dependent threshold for linear filter -------| eta_bot=0.5
    norm_h=(z-bottom)/(top-bottom)                        #
    Size= 2*(norm_h)*(1-np.sqrt(1-1*eta_top))             # Size_rad*2
    if Size < 1.0:                                        #
        eta=(Size**2)/4 + 0.5                             #
    else:                                                 #
        eta=1 - ((Size-2)**2)/4                           #
    return eta # 0.5 ~ eta_top                            #
#---------------------------------------------------------|

def Local_dilation(
    Reference_layer, 
    DR_width, 
    DR_length, 
    DR_res, 
    beta, 
    eta_gap, 
    N_local, 
    Local_Size, 
    const,
    Number_of_local_region,
    LR_temp,
    is_Dilation: bool = True,
): #----------------------------- Dilation by heights for local region-----------------------------------------------------|
    if is_Dilation:                                                           # Activate dilation                          |
        D_const=1                                                             #--------------------------------------------|
        local_h0=0                                                            #                                            |
    else:                                                                     # Activate erosion                           |
        D_const=0                                                             #--------------------------------------------|
        local_h0=2                                                            #                                            |
    Local_x = []                                                              #                                            |
    local_h = N_local                                                         # initialize the local height index as a top |
    eta_bot = 0.5-eta_gap                                                         # Maximum dilation size for bottom layer |
    Local_R = get_conic_radius(Local_Size, 1-eta_bot)                                                            #         |
    filtered_slice = Ft.conic_filter(npa.clip(Reference_layer, 0.0, 1.0), Local_R, DR_width, DR_length, DR_res)  #         |
    #----------------------------------------------------------------------------------------------------------------------|
    if LR_temp+1 == Number_of_local_region or beta < 30:       # Total Number of projection -> Number_of_local_region      |
        while local_h > local_h0:                              #- h: Top to Bottom, linear dilation by heights ------------|
            eta_dh= eta_z(local_h, 1-eta_bot, 1, N_local+const) - eta_gap*D_const                  # dilation threshold (h)|
            z_slice = tanh_projection_m(filtered_slice, beta, eta_dh)                                        # Dilation    |
            z_slice = z_slice.flatten()                                                                      #             |
            Local_x = npa.concatenate((z_slice, Local_x), axis=0)                                            #             |
            local_h -=1                                                                                      #             |
        del filtered_slice                             # Clear Memory -----------------------------------------------------|
        del Reference_layer                            # Clear Memory -----------------------------------------------------|
        return Local_x, z_slice                                # 3D geo                                      #             |
    else:                                                      # Total Number of projection (Npj)-> Number_of_local_region |
        eta_dh= eta_z(1+(N_local-1)*(1-D_const), 1-eta_bot, 1, N_local+const) - eta_gap*D_const    # dilation threshold (h)|
        Next_top = tanh_projection_m(filtered_slice, beta, eta_dh)                                 #                       |
        Next_top = Next_top.flatten()                          # Reference layer for the next section, Npj = LR_temp + 1   |
        LR_idx = LR_temp                                       #-----------------------------------------------------------|
        Local_top= filtered_slice                              #-----------------------------------------------------------|
        while LR_idx < Number_of_local_region-1:               # Npj ->LR_temp + (Number_of_local_region - LR_temp -1) ----|
            eta_dh= eta_z(1+(N_local-1)*D_const, 1-eta_bot, 1, N_local) - eta_gap*D_const          # dilation threshold (h)|
            Local_top = tanh_projection_m(Local_top, beta, eta_dh)                                             # Dilation  |
            Local_top = Ft.conic_filter(npa.clip(Local_top, 0.0, 1.0), Local_R, DR_width, DR_length, DR_res)   #           |
            LR_idx +=1                                         #-----------------------------------------------------------|
        while local_h > local_h0:                              # Npj -> Number_of_local_region ----------------------------|
            eta_dh= eta_z(local_h, 1-eta_bot, 1, N_local+const) - eta_gap*D_const                  # dilation threshold (h)|
            z_slice = tanh_projection_m(Local_top, beta, eta_dh)                                             # Dilation    |
            z_slice = z_slice.flatten()                                                                      #             |
            Local_x = npa.concatenate((z_slice, Local_x), axis=0)                                            #             |
            local_h -=1                                                                                      #             |
        del filtered_slice                             # Clear Memory -----------------------------------------------------|
        del Local_top                                  # Clear Memory -----------------------------------------------------|
        del z_slice                                    # Clear Memory -----------------------------------------------------|
        del Reference_layer                            # Clear Memory -----------------------------------------------------|
        return Local_x, Next_top                         # 3D geo, 2D Reference plane for next subsection    #             |
#--------------------------------------------------------------------------------------------------------------------------|

def Vertical_sidewall(
    Reference_layer,
    N_height,
):
    x2 = []
    h_temp=0
    while h_temp < N_height:
        x2 = npa.concatenate((Reference_layer,x2), axis=0)
        h_temp+=1
    x= x2.flatten()
    return x

def Slant_sidewall(
    Reference_layer,
    DR_width, 
    DR_length, 
    DR_res,
    beta,
    Min_gap,
    N_height,
    Number_of_local_region,
    is_Middle: bool = False,
):                           #-- Impose Slanted angle. Than, structs 3D density array map --|
    x2 = []                                                        #                        |
    eta_gap = 0.45                                                 #                        |
    if beta < 20:                                                  #                        |
        Number_of_local_region = 2                                 #                        |
    Nl=int(N_height/Number_of_local_region)                        #                        |
    Nl_r=int(N_height-(Nl*Number_of_local_region))                 #                        |
    Local_Size= Min_gap*((Nl)/(N_height))                          #                        |
    LR_temp = 0                                                    #----------------------------------|
    if is_Middle:                                                  # Case1: Reference layer is middle |
        Number_of_local_region=int(Number_of_local_region/2)       #                                  |
        Local_Size_e= Min_gap*((Nl+2)/(N_height))                  #                                  |
        if Number_of_local_region ==1:                             #                                  |
            Main_NDL= Nl_r + Nl                                    #                                  |
            Main_DLS= Min_gap*(Nl_r + Nl)/(N_height)               #                                  |
        else:                                                      #                                  |
            Main_NDL= Nl                                           #                                  |
            Main_DLS= Local_Size                                   #                                  |
        Layer0, z_slice= Local_dilation(                           # Mid to Top             |---------|
            Reference_layer= Reference_layer,                      #                        |
            DR_width= DR_width,                                    #                        |
            DR_length= DR_length,                                  #                        |
            DR_res= DR_res,                                        #                        |
            beta= beta,                                            #                        |
            eta_gap= eta_gap,                                      #                        |
            N_local= Nl+2,                                         #                        |
            Local_Size= Local_Size_e,                              #                        |
            const= 0,                                              #                        |
            Number_of_local_region= Number_of_local_region,        #                        |
            LR_temp= LR_temp,                                      #                        |
            is_Dilation= False,                                    #                        |
        )                                                          #                        |
        x2= npa.concatenate((x2, Layer0), axis=0)                  #                        |
        LR_temp +=1                                                #                        |
        del Layer0          # Clear Memory -------------------------------------------------|
        if Number_of_local_region>1:      # Multiple sub sections---------------------------|
            while LR_temp < Number_of_local_region:                # Sub sections x (NHR-1)-|
                Layer, z_slice= Local_dilation(                    # Eroded layer           |
                    Reference_layer=z_slice,                       #                        |
                    DR_width= DR_width,                            #                        |
                    DR_length= DR_length,                          #                        |
                    DR_res= DR_res,                                #                        |
                    beta= beta,                                    #                        |
                    eta_gap= eta_gap,                              #                        |
                    N_local= Nl+2,                                 #                        |
                    Local_Size= Local_Size_e,                      #                        |
                    const= 0,                                      #                        |
                    Number_of_local_region= Number_of_local_region,#                        |
                    LR_temp= LR_temp,                              #                        |
                    is_Dilation= False,                            #                        |
                )                                                  #                        |
                x2= npa.concatenate((x2, Layer), axis=0)           #                        |
                del Layer   # Clear Memory -------------------------------------------------|
                LR_temp +=1              # -------------------------------------------------|
        LR_temp = 0                                                #----------------------------------|
    else:                                                          # Case2: Reference layer is top    |
        Main_NDL= Nl                                               #                                  |
        Main_DLS= Local_Size                                       #                                  |
    Layer0, z_slice= Local_dilation(                               # Top to Bot (Main)      |---------|
        Reference_layer= Reference_layer,                          #                        |
        DR_width= DR_width,                                        #                        |
        DR_length= DR_length,                                      #                        |
        DR_res= DR_res,                                            #                        |
        beta= beta,                                                #                        |
        eta_gap= eta_gap,                                          #                        |
        N_local= Main_NDL,                                         #                        |
        Local_Size= Main_DLS,                                      #                        |
        const= 0.00,                                               #                        |
        Number_of_local_region= Number_of_local_region,            #                        |
        LR_temp= LR_temp,                                          #                        |
    )                                                              #                        |
    x2= npa.concatenate((Layer0, x2), axis=0)                      #                        |
    LR_temp +=1                                                    #                        |
    del Layer0          # Clear Memory -----------------------------------------------------|
    del Reference_layer # Clear Memory -----------------------------------------------------|
    if Number_of_local_region>1:     # Multiple sub sections--------------------------------|
        while LR_temp < Number_of_local_region-1:                  # Sub sections x (NR-2) -|
            Layer, z_slice= Local_dilation(                        # Dilated layer          |
                Reference_layer=z_slice,                           #                        |
                DR_width= DR_width,                                #                        |
                DR_length= DR_length,                              #                        |
                DR_res= DR_res,                                    #                        |
                beta= beta,                                        #                        |
                eta_gap= eta_gap,                                  #                        |
                N_local= Nl,                                       #                        |
                Local_Size= Local_Size,                            #                        |
                const= 1,                                          #                        |
                Number_of_local_region= Number_of_local_region,    #                        |
                LR_temp= LR_temp,                                  #                        |
            )                                                      #                        |
            x2= npa.concatenate((Layer, x2), axis=0)               #                        |
            del Layer   # Clear Memory -----------------------------------------------------|
            LR_temp +=1              # -----------------------------------------------------|
        Layer, z_slice= Local_dilation(                            # Last sub section (Bot)-|
            Reference_layer=z_slice,                               #                        |
            DR_width= DR_width,                                    #                        |
            DR_length= DR_length,                                  #                        |
            DR_res= DR_res,                                        #                        |
            beta= beta,                                            #                        |
            eta_gap= eta_gap,                                      #                        |
            N_local= Nl_r + Nl,                                    #                        |
            Local_Size= Min_gap*(Nl_r + Nl)/(N_height),            #                        |
            const= 1,                                              #                        |
            Number_of_local_region= Number_of_local_region,        #                        |
            LR_temp= LR_temp,                                      #                        |
        )                                                          #                        |
        x2= npa.concatenate((Layer, x2), axis=0)                   #                        |
        del Layer       # Clear Memory -----------------------------------------------------|
    x= x2.flatten()                                                #                        |
    del z_slice         # Clear Memory -----------------------------------------------------|
    del x2              # Clear Memory -----------------------------------------------------|
    return x                                                       # 3D slanted geometry----|
#-------------------------------------------------------------------------------------------|

def Grating(#-- Input parameters ---------------------------------|
    #                                                             |
    DR_width, # Disign region width                               |
    N_width,  # Number of DR width grids                          |
    DR_res, # Design region resolution                            |
    #                                                             |
    Min_size_top, # Minimum linwidth of top layer                 |
    Min_gap, # Minimum gap width of ref layer                     |
    #                                                             |
    x, # Input material density array                             |
    #                                                             |
    beta,     # Binrization parameter                             |
    Mask_pixels, # Number of pixels for mask region               |
    Disk_R,
    Is_Cyl,
):                                                                #-- Impose MFS and MGS on x ------------------------|
    eta_Ref = 0.05                                                      # Thresholding point of tanh projection       |
    Single_pixel= round(1/DR_res,3)                                                          #                        |
    grid_size = int(2* N_width-1)
    grid = npa.linspace(-DR_width, DR_width, grid_size)
    X, Y = npa.meshgrid(grid, grid, indexing='ij')
    R = npa.sqrt(X**2 + Y**2)
    dr = DR_width / N_width
    r_idx = npa.clip((R/dr).astype(int), 0, N_width-1)
    x_2D = x[r_idx]
    if Disk_R > 0.0:
        x_2D = npa.where(R < Disk_R, 1.0, x_2D)
    Total_R = get_conic_radius(Min_size_top+ Min_gap, 1-eta_Ref)                             # Radius for MFS + MGS   |
    MGS_R = get_conic_radius((Min_gap), 1-eta_Ref)                                           # Radius for MGS         |
    N_total = round(Total_R/(4*Single_pixel))                     # Total number of Erosion & Dilation with R= 4pixel |
    N_Half = round(Total_R/(8*Single_pixel))                      # Half number of Erosion & Dilation with R= 4pixel  |
    # Set filter R, number of dilation & erosion & smoothing ---------------------------------------------------------|
    if N_total > 1:                               # Total_R > 5 pixel ------------------------------------------------|
        N_total = 2                                                                          #                        |
        if N_Half < 1:                                      # Half_R < 5 pixel ---------------------------------------|
            N_Half = 1                                                                       #                        |
        N_dilation = N_total*N_Half                               # Number of dilation = 2*N_Half                     |
        Dilation_R = get_conic_radius((Min_gap + Min_size_top)/(N_dilation), 1-eta_Ref)      # R for unit dilation    |
        Half_R = (Min_gap + Min_size_top)/2                                                  # R for Half smoothing   |
        if Min_gap > Min_size_top:                         # MGS > MFS  ----------------------------------------------|
            MGS_R = get_conic_radius((Min_gap - Min_size_top)/2, 1-eta_Ref)                  # R for half erosion     |
            N_MGS = round(MGS_R/(4*Single_pixel))              # Number of unit erosion with R= 4pixel                |
            if N_MGS < 1:                                      # MGS_R < 2 pixel -------------------------------------|
                N_MGS = 1                                                                    #                        |
            Erosion_R = get_conic_radius((Min_gap - Min_size_top)/(2*N_MGS), 1-eta_Ref)      # R for unit erosion     |
            Smoothing_E = True                                                   # Half smoothing during erosion      |
            Smoothing_D = False                                                  #                                    |
            N_erosion = N_Half + N_MGS                                           # Number of erosion = N_Half + N_MGS |
        else:                                               # MGS =< MFS ---------------------------------------------|
            N_MGS = round(MGS_R/(4*Single_pixel))              # Number of unit erosion with R= 4pixel                |
            if N_MGS < 1:                                      # MGS_R < 2 pixel -------------------------------------|
                N_MGS = 1                                                                    #                        |
            Erosion_R = get_conic_radius((Min_gap)/(N_MGS), 1-eta_Ref)                       # R for unit erosion     |
            Smoothing_E = False                                                  #                                    |
            Smoothing_D = True                                                   # Half smoothing during dilation     |
            N_erosion = N_MGS                                                    # Number of erosion = N_MGS          |
    else:                                         # Total_R <= 5 pixel -----------------------------------------------|
        N_dilation = 1                                                           # Single dilation                    |
        N_erosion = 1                                                            # Single erosion                     |
        Dilation_R = Total_R                                                     # R for dilation = MGS + MFS         |
        Erosion_R = MGS_R                                                        # R for erosion = MGS                |
        Smoothing_D = False                                                      # Deactivate half smoothing          |
        Smoothing_E = False                                                      # Deactivate half smoothing          |
    #-----------------------------------------------------------------------------------------------------------------|
    if Mask_pixels == 0:                                                                     #                        |
        Air_pad= int(((Min_gap)/2+ Dilation_R + Single_pixel)/Single_pixel)+1                # Size of Air pad        |
        x_2D= npa.pad(                                  
            x_2D,                                       
            ((Air_pad, Air_pad), (Air_pad, Air_pad)),   
            mode="constant", constant_values=0)         
        DR_width = 2*(DR_width+ round(Air_pad/DR_res,2))
        N_width  =grid_size+ 2*Air_pad 
        DR_length = DR_width 
        N_length = N_width
        x_2D = x_2D.copy()
        x_2D[:Air_pad, :] = 0
        x_2D[-Air_pad:, :] = 0
        x_2D[:, :Air_pad] = 0
        x_2D[:, -Air_pad:] = 0
    #-----------------------------------------------------------------------------------------------------------------|
    x_copy = tanh_projection_m(x_2D.flatten(), beta, 0.5)                                    # Binarization ----------|
    Dilation_index= 0                                                                        #                        |
    while Dilation_index < N_dilation:    #------- Activate multiple dilation N_dilation times -----------------------|
        x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Dilation_R, DR_width, DR_length, DR_res)    #             |
        x_copy = tanh_projection_m(x_copy, beta, eta_Ref)                                 # Unit dilation             |
        if Smoothing_D and Dilation_index == round(N_dilation/2)-1: # MGS < MFS, Smoothing when MFS = MGS ------------|
            x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)              #   |
            x_copy = tanh_projection_m(x_copy, beta, 0.5)                                 # Half smoothing        #   |
            x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)              #   |
            x_copy = tanh_projection_m(x_copy, beta, 0.5)                                 # Compensation smoothing    |
        Dilation_index +=1                                                                                        #   |
    Erosion_index= 0                                                                                              #   |
    while Erosion_index < N_erosion:      #--------- Activate multiple erosion N_erosion times -----------------------|
        if Smoothing_E:                             # MGS >= MFS -----------------------------------------------------|
            while Erosion_index < N_Half:            # Erosion with Dilation radius ----------------------------------|
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Dilation_R, DR_width, DR_length, DR_res)      #   |
                x_copy = tanh_projection_m(x_copy, beta, 1-eta_Ref)                       # Unit erosion          #   |
                Erosion_index +=1                                                                                 #   |
            if Erosion_index == N_Half:              # Smoothing when MFS = MGS --------------------------------------|
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)          #   |
                x_copy = tanh_projection_m(x_copy, beta, 0.5)                             # Half smoothing        #   |
                x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Half_R, DR_width, DR_length, DR_res)          #   |
                x_copy = tanh_projection_m(x_copy, beta, 0.5)                             # Compensation smoothing    |
        x_copy= Ft.conic_filter(npa.clip(x_copy, 0.0, 1.0), Erosion_R, DR_width, DR_length, DR_res)               #   |
        x_copy = tanh_projection_m(x_copy, beta, 1-eta_Ref)                               # Unit erosion              |
        Erosion_index +=1                                                                                         #   |
    if Air_pad > 0:                                                                                               #   |
        x_copy= x_copy[Air_pad:(N_width-Air_pad),Air_pad:(N_length-Air_pad)]                                      #   |
        N_length -= 2*Air_pad                                                                                     #   |
    if Is_Cyl:                                                                                                    #   |
        x_copy= x_copy[int((N_length)/2):,int((N_length)/2)]                                                      #   |
    x_copy = x_copy.flatten()                                                                                     #   | 
    del x_2D                                      # Clear Memory -----------------------------------------------------|
    return x_copy                                                                                # Reference layer ---|
#---------------------------------------------------------------------------------------------------------------------|
