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

Use_wall=True

# =============================================================================
# Original high-level flags
# =============================================================================
Constrain = True
Modules = True
Optimization = True
Output = True
# Waveguide angle from the LC wall axis (z). Crossing angle = 2*theta.
# 3D-confirmed operating point (Switch_3D_sweep, 48 points):
#   t=0.40 um, w=6.0 um, 2*theta=54.50 deg, LC wall 0.575 um
#   -> T=0.716 (transmit state), R=0.824 (reflect state), vertically SINGLE mode.
# Alternatives on the same margin, same w and d (see A/LC_3D_confirmed.txt):
#   t=0.49 -> 2*theta=59.16, T 0.707 / R 0.831  (shortest device, T margin only +0.007)
#   t=0.30 -> 2*theta=46.34, T 0.738 / R 0.815  (best T, crossing ~33% longer)
# theta sets the TIR angular margin = incidence angle - critical angle, where the
# critical angle comes from the WALL's own vertical mode index (1.483 here), not
# bulk n_o. This margin is the one knob that decides R; 4-6 deg is the usable
# window and this point sits at 4.5 deg. +-1 deg of crossing angle is the whole
# tolerance -- 1 deg moves T by ~0.035 and R by ~0.030.
# Put the WAVEGUIDE inside the design region instead of freezing it.
#
# With the waveguide frozen to SiN, the only free cells were the SiO2 background,
# where the guided field is evanescent -- the adjoint gradient there is negligible
# and the 42-iteration run drove all of it to SiO2 and added nothing but a 2-cell
# boundary artifact. The transmit side stayed stuck at purity 0.60 (power 0.88
# crosses, but it does not re-form the output mode).
#
# When True the waveguide stops being a constraint and becomes the STARTING POINT:
#   - only the LC wall stays frozen (to LC),
#   - the initial density IS the CAD waveguide footprint, so iteration 1 is the
#     structure we already validated in 3D (T 0.716 / R 0.824),
#   - the CAD waveguide rectangles are clipped to OUTSIDE the design region so
#     they cannot double up with the imported design.
# The optimizer can then reshape the guide where it meets the wall -- taper,
# widen, offset -- which is the one lever that can act on mode overlap.
DESIGN_INCLUDES_WAVEGUIDE = False

theta=[27.25, -27.25]

for seq in [0]:
    Overlap = True
    full_scale = False

    Min_s_top = 0.1  # um
    Capsulation = True

    if Capsulation:
        Geometry_profile = True
        Wavelengths = 1.55  # um
        mode = [3, 1]       # target Lumerical mode numbers
        Target_band = 1

        Main_Parameters = True
        Parameter_activation = True
        Monitor_Profile = True
        Source_profile = True
        Material_profile = True

        if Main_Parameters:
            resolution = 25

            design_region_x = round(0.40, 2)     # SiN core thickness (= Core_h)
            design_region_y = round(30.0, 2)
            design_region_z = round(30.0, 2)
            design_region_resolution = int(resolution)
            design_s = [design_region_x, design_region_y, design_region_z]
            design_c = [0, 0, 0]
            

            TOX_h = round(2.0, 2)      # oxide ABOVE the core
            Core_h = design_region_x
            # Air on top of the upper oxide. The domain used to stop at the TOX
            # surface on an oxide background, i.e. the PML sat directly on oxide and
            # there was no air anywhere. The LC wall is the etched trench, so it
            # reaches that surface and its vertical modes -- which set the critical
            # angle -- depend on what caps it.
            Air_h = round(1.5, 2)
            # The FDTD region is centred on x=0, and a great deal of this script
            # (index monitor, mode-profile plots, source window) reads naturally only
            # when x=0 IS the middle of the core. Adding air on top alone would push
            # the core 0.75 um below x=0 and every one of those would silently point
            # at the wrong plane. So the MODELLED buried oxide is grown to keep the
            # core centred. This is not a change to the device: the mode decays into
            # the BOX within ~0.25 um, so 2.0 um and 3.5 um of modelled substrate are
            # optically the same simulation -- the extra oxide only replaces PML.
            BOX_h = round(TOX_h + Air_h, 2)

            # Guide width. It buys T and R at the same time (a wider guide has a
            # narrower angular spectrum, so less of the beam falls below the
            # critical angle, and a longer diffraction length across the wall).
            # 3D: w=4 um fails at every thickness tried; 5 um is the bare minimum;
            # 6 um is the practical one.
            w_top = round(6.0, 2)


        if Parameter_activation:
            waveguide_length_I = round(2.0, 2)
            waveguide_length_O = round(2.0, 2)
            pml_2_src = round(2.0 / resolution, 2)
            # Must clear the PML, not just the boundary: Lumerical's default is 8
            # PML layers = 8 cells, so 2 cells put the FOM monitor inside the
            # absorber and under-read the transmitted power.
            mon_2_pml = round(12.0 / resolution, 2)

            Sy = round(waveguide_length_I + design_region_y + waveguide_length_O, 2)
            Sz = round(waveguide_length_I + design_region_z + waveguide_length_O, 2)
            Sx = round(BOX_h + Core_h + TOX_h + Air_h, 2)

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
            # x is no longer the domain centre once the air cap is added, so the mode
            # source is centred on the CORE instead of on x=0.
            x_core_c = X_min + BOX_h + 0.5 * Core_h
            source_center2 = [x_core_c, Y_max*0.5, Z_min + 0.5]
            source_center1 = [x_core_c, Y_min*0.5, Z_min + 0.5]
            src_c=[source_center1, source_center2]

            source_size = [Sx*0.5, Sy*0.5, 0]

            Output_monitor_center = [0, 0, Z_max - mon_2_pml]
            dft_monitor_size = [Sx,Sy, 0]
            
            Top_cen = [X_min + BOX_h + Core_h / 2, 0, 0]
            Top_size = [0, Sy, Sz]
            Side_cen = [0, 0, 0]
            Side_size = [Sx, 0, Sz]

        if Material_profile:
            SiN_n = [2.0, 2.0, 2.0]
            LC_0_n = [1.5, 1.685, 1.5]
            LC_1_n = [1.5, 1.5, 1.685]
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

    def port_flux(E, H, alpha):
        S=[0]*3
        S[0]= npa.real(E[1]*npa.conjugate(H[2]) - E[2]*npa.conjugate(H[1]))
        S[1]= npa.real(E[2]*npa.conjugate(H[0]) - E[0]*npa.conjugate(H[2]))
        S[2]= npa.real(E[0]*npa.conjugate(H[1]) - E[1]*npa.conjugate(H[0]))
        if alpha != 0.0:
            a_r = np.deg2rad(alpha)
            ca, sa = float(np.cos(a_r)), float(np.sin(a_r))
            d_hat = np.array([0, sa, ca])
        return (npa.abs(npa.sum(S[1])) * d_hat[1] + npa.abs(npa.sum(S[2])) * d_hat[2])


    def add_tilted_mode_source(sim, name, center, size, src_wl, alpha):
        fdtd=sim.fdtd
        um = 1e-6
        wl_min = float(np.min(src_wl))
        wl_max = float(np.max(src_wl))
        fdtd.addmode()
        fdtd.set("name", name)
        fdtd.set("injection axis", "z-axis")
        fdtd.set("direction", "forward")
        fdtd.set("x", center[0]* um)
        fdtd.set("y", center[1]* um)
        fdtd.set("z", center[2]* um)
        if size[0] !=0:
            fdtd.set("x span", size[0]* um)
        if size[1] !=0:
            fdtd.set("y span", size[1]* um)
        if size[2] !=0:
            fdtd.set("z span", size[2]* um)
        fdtd.set("theta", -alpha)
        fdtd.set("phi", -90)
        
        fdtd.set("wavelength start", wl_min * um)
        fdtd.set("wavelength stop", wl_max * um)
        fdtd.set('mode selection', 'user select')
        fdtd.set('selected mode number', 0)

    def add_air_cap(sim_obj):
        """Air above the upper oxide. The simulator background is SiO2, so without
        this block the PML would terminate an infinite oxide and the LC wall would
        be capped by oxide instead of air -- which shifts the wall's vertical mode
        indices and therefore the critical angle the whole design is built on."""
        if Air_h <= 0.0:
            return
        x_lo = X_min + BOX_h + Core_h + TOX_h
        sim_obj.add_geo(
            center=[0.5 * (x_lo + X_max), 0.0, 0.0],
            size=[X_max - x_lo, 2.0 * Sy, 2.0 * Sz],
            index=Air_n,
            name='air_cap',
            wavelength=Wavelengths,
        )

    def add_tilted_waveguide(sim, name, center, size, index, alpha):
        fdtd=sim.fdtd
        fdtd.addrect()
        fdtd.set("name", name)
        fdtd.set("x", center[0]*1e-6)
        fdtd.set("y", center[1]*1e-6)
        fdtd.set("z", center[2]*1e-6)
        if size[0] !=0:
            fdtd.set("x span", size[0]*1e-6)
        if size[1] !=0:
            fdtd.set("y span", size[1]*1e-6)
        if size[2] !=0:
            fdtd.set("z span", size[2]*1e-6)
        fdtd.set("first axis", "x")
        fdtd.set("rotation 1", -alpha)
        sim._set_object_index(index, object_name=name,
                                material_name=f"name", wavelength=Wavelengths)
        # override the design import (mesh order 2) inside the barrier band

    for src_idx in [0, 1]:
        sim_norm[src_idx] = ms.Lumerical_utill.LumericalFDTDSimulator(
            sim_size=[Sx, Sy, Sz],
            resolution=resolution,
            unit=1e-6,
            background_index=1.44,
            center_wl=Wavelengths,
            N_f=1,
        )
        add_air_cap(sim_norm[src_idx])
        # Full straight input guide for normalization
        add_tilted_waveguide(
            sim=sim_norm[src_idx],
            name='input_wg',
            center=[X_min + BOX_h + 0.5 *Core_h, 0, 0],
            size=[Core_h, w_top, 2*Sz],
            index=SiN_n,
            alpha=theta[src_idx],
        )
        add_tilted_mode_source(
            sim=sim_norm[src_idx],
            name='source_norm',
            center=src_c[src_idx],
            size=source_size,
            src_wl=[Wavelengths],
            alpha=theta[src_idx],
        )
        sim_norm[src_idx].add_monitor(
            name='output_monitor',
            center=Output_monitor_center,
            size=dft_monitor_size,
            N_f=1,
        )
        sim_norm[src_idx].add_monitor(
            name='Cross_monitor',
            center=[X_min + BOX_h + 0.5 * (Core_h), 0, Z_min + waveguide_length_I + 0.5 * design_region_z],
            size=[0, Sy, Sz],
            N_f=1,
        )

        sim_norm[src_idx].fdtd.addindex()
        sim_norm[src_idx].fdtd.set("monitor type", "2D x-normal")
        sim_norm[src_idx].fdtd.set("name", "mon_index")
        # Must sit in the CORE, not at x=0. With the air cap the stack is no longer
        # symmetric about x=0, so x=0 falls in the upper oxide and the index plot
        # would show no waveguides at all. Top_cen[0] is the core mid-plane.
        sim_norm[src_idx].fdtd.set("x", Top_cen[0] * 1e-6)
        sim_norm[src_idx].fdtd.set("y", 0.0)
        sim_norm[src_idx].fdtd.set("z", 0.0)
        sim_norm[src_idx].fdtd.set("y span", Sy * 1e-6)
        sim_norm[src_idx].fdtd.set("z span", Sz * 1e-6)

        sim_norm[src_idx].run(name=f'norm_mode_{src_idx}', save=True)

        H_c = sim_norm[src_idx].fdtd.getresult('Cross_monitor', 'H')
        # Physical axes for the plots below. Without them imshow falls back to pixel
        # indices, which hides both the aspect ratio and where the mode actually sits.
        yc_ax = np.ravel(np.asarray(H_c['y'], dtype=float)) * 1e6
        zc_ax = np.ravel(np.asarray(H_c['z'], dtype=float)) * 1e6

        Hc_all = np.array(H_c['H'], dtype=np.complex128)
        Hxc = Hc_all[..., 0][:, :, :, 0]
        Hxc_plot = np.squeeze(np.real(Hxc))#/np.max(np.abs(np.flatten(np.real(Hx))))

        H_out = sim_norm[src_idx].fdtd.getresult('output_monitor', 'H')
        E_out = sim_norm[src_idx].fdtd.getresult('output_monitor', 'E')
        xo_ax = np.ravel(np.asarray(E_out['x'], dtype=float)) * 1e6
        yo_ax = np.ravel(np.asarray(E_out['y'], dtype=float)) * 1e6

        Hout_all = np.array(H_out['H'], dtype=np.complex128)
        Eout_all = np.array(E_out['E'], dtype=np.complex128)

        Hout = [Hout_all[..., 0][:, :, :, 0], Hout_all[..., 1][:, :, :, 0], Hout_all[..., 2][:, :, :, 0]]
        Eout = [Eout_all[..., 0][:, :, :, 0], Eout_all[..., 1][:, :, :, 0], Eout_all[..., 2][:, :, :, 0]]


        try:
            ires = sim_norm[src_idx].fdtd.getresult("mon_index", "index")
            ny = np.real(np.squeeze(np.asarray(ires["index_y"])))
            nz = np.real(np.squeeze(np.asarray(ires["index_z"])))
            yi = np.ravel(np.asarray(ires["y"])) * 1e6
            zi = np.ravel(np.asarray(ires["z"])) * 1e6
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            for ax, dat, lab in ((axes[0], ny, "n for E||x"), (axes[1], nz, "n for E||y")):
                im = ax.imshow(dat, extent=(zi[0], zi[-1], yi[0], yi[-1]),
                               origin="lower", cmap="viridis", aspect="equal",
                               vmin=1.4, vmax=2.0)
                ax.set_title(lab)
                ax.set_xlabel("z (um)")
                ax.set_ylabel("y (um)")
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(os.path.join(design_dir, f"LC2D_index_{src_idx}.png"), dpi=160)
            plt.close(fig)
        except Exception as exc:
            print(f"[index] plot skipped: {exc}")


        # === Save mode profile as PNG ===
        try:
            # The monitor is a plane normal to the propagation axis, so one spatial
            # axis is singleton (y-normal source -> shape (Nx,1,Nz); z-normal -> (Nx,Ny,1)).
            # squeeze() drops it to a 2D transverse profile so imshow works for both.
            Ex_plot = np.squeeze(np.real(Eout[0]))
            Ey_plot = np.squeeze(np.real(Eout[1]))
            Ez_plot = np.squeeze(np.real(Eout[2]))

            # output_monitor is z-normal spanning [Sx, Sy], so the arrays are
            # (Nx, Ny): rows are the layer normal x, columns the in-plane y. imshow
            # draws rows vertically, so the extent has to be (y..., x...) and the
            # origin has to be "lower" or the stack comes out upside down. The old
            # aspect=shape[0]/shape[1] squashed the profile ~3x and printed pixel
            # indices instead of um.
            ext_o = (yo_ax[0], yo_ax[-1], xo_ax[0], xo_ax[-1])
            fig = plt.figure(figsize=(13, 7))
            for k, (dat, lab) in enumerate(((Ex_plot, "Ex  (layer normal)"),
                                            (Ey_plot, "Ey  (in-plane, sees the LC)"),
                                            (Ez_plot, "Ez  (along propagation)"))):
                ax = fig.add_subplot(2, 3, k + 1)
                v = float(np.max(np.abs(dat))) or 1.0
                im = ax.imshow(dat, cmap="RdBu", extent=ext_o, origin="lower",
                               aspect="equal", vmin=-v, vmax=v)
                ax.set_title(lab, fontsize=10)
                ax.set_xlabel("y (um)")
                ax.set_ylabel("x (um)")
                fig.colorbar(im, ax=ax, fraction=0.046)

            # Cross_monitor is x-normal in the core: (Ny, Nz), rows y and columns z.
            ax = fig.add_subplot(2, 1, 2)
            im = ax.imshow(Hxc_plot ** 2, cmap="inferno",
                           extent=(zc_ax[0], zc_ax[-1], yc_ax[0], yc_ax[-1]),
                           origin="lower", aspect="equal")
            ax.set_title("|Hx|^2 on the core plane (normalization run: input guide only)",
                         fontsize=10)
            ax.set_xlabel("z (um)   <- input   output ->")
            ax.set_ylabel("y (um)")
            fig.colorbar(im, ax=ax, fraction=0.046)

            fig.suptitle(f"NORM mode {src_idx}: guide at {theta[src_idx]:+.2f} deg, "
                         f"w={w_top:g} um, core {Core_h:g} um at x=0")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(os.path.join(design_dir, f'mode_profile_{src_idx}.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f'Failed to save mode profile PNG: {e}')

        Target_E_Fields[src_idx] = Eout
        # src 0 propagates along z (straight output) -> S_z (axis 2); src 1 propagates
        # along y (reflection output) -> S_y (axis 1). Using the default axis=2 for the
        # y-mode gave a wrong (near-zero) reference power.
        Input_powers[src_idx] = port_flux(Eout, Hout, alpha=theta[src_idx])
        Input_ints[src_idx] = npa.sum((npa.abs(Eout[0])**2) + (npa.abs(Eout[1])**2) + (npa.abs(Eout[2])**2))
        print(f"[norm {src_idx}] Input_power={Input_powers[src_idx]:.6e}, Input_int={Input_ints[src_idx]:.6e}")

        sim_norm[src_idx].fdtd.switchtolayout()

    # =========================================================================
    # Optimization simulation objects
    # =========================================================================
    sim = [0] * N_fom
    opt = [0] * N_fom
    ob_list = [0] * N_fom
    Foms_history = [[0], [0]]

    # =========================================================================
    # Exact fixed-material masks inside the design import
    # =========================================================================
    # The two physical waveguides are Lumerical rectangles rotated by -theta
    # about x.  The LC wall is a horizontal rectangle (rotation 0) that overrides
    # the SiN/SiO2 design import with mesh order 1.  Reproduce those exact CAD
    # footprints in the mapped density, then remove the LC intersection from the
    # effective waveguide mask so the fixed-one and fixed-zero contracts are
    # unambiguous and disjoint:
    #
    #   waveguide union -> density 1 (SiN)
    #   LC wall         -> density 0 (SiO2 filler, physically overridden by LC)
    #
    # Since these npa.where operations are the final mapping operation, autograd
    # also gives exactly zero design-variable gradient in every frozen cell.
    #
    # The band BETWEEN the LC wall and the waveguides is deliberately left out of
    # both masks: that coupling gap is where the switching behaviour is decided,
    # so freezing it to the CAD waveguide would remove exactly the degrees of
    # freedom the optimizer needs there.  lc_clearance sets the half-width of the
    # free band measured from each LC wall face.
    # LC wall (etched trench) thickness. Thicker is a NET LOSS: 3D at w=8, margin
    # 4 deg gave d 0.575 -> 0.8 -> 1.1 um moving R 0.818 -> 0.917 -> 0.959 while T
    # fell 0.776 -> 0.668 -> 0.561. Thinner loses R faster than it gains T.
    wall_thickness = 0.575
    lc_clearance = 0.3
    lc_wall_rotation_deg = 0.0
    lc_wall_length = np.sqrt(2.0) * max(design_region_y, design_region_z)

    _mask_y = np.linspace(-0.5 * design_region_y, 0.5 * design_region_y, Ny)
    _mask_z = np.linspace(-0.5 * design_region_z, 0.5 * design_region_z, Nz)
    waveguide_cad_footprint_2d = np.zeros((Ny, Nz), dtype=bool)
    for alpha in theta:
        waveguide_cad_footprint_2d |= ms.Sub_Mapping.rotated_rectangle_mask_2d(
            _mask_y,
            _mask_z,
            span_1=w_top,
            span_2=2.0 * Sz,
            rotation_deg=-float(alpha),
        )
    if Use_wall:
        lc_mask_2d = ms.Sub_Mapping.rotated_rectangle_mask_2d(
            _mask_y,
            _mask_z,
            span_1=wall_thickness,
            span_2=lc_wall_length,
            rotation_deg=lc_wall_rotation_deg,
        )
        # Same wall widened by lc_clearance on both faces: everything inside it
        # is kept out of the waveguide mask, so the LC-to-waveguide gap stays a
        # free design variable instead of being frozen to SiN.
        lc_exclusion_2d = ms.Sub_Mapping.rotated_rectangle_mask_2d(
            _mask_y,
            _mask_z,
            span_1=wall_thickness + 2.0 * lc_clearance,
            span_2=lc_wall_length,
            rotation_deg=lc_wall_rotation_deg,
        )
    else:
        lc_mask_2d = np.zeros((Ny, Nz), dtype=bool)
        lc_exclusion_2d = np.zeros((Ny, Nz), dtype=bool)

    removed_overlap_mask_2d = waveguide_cad_footprint_2d & lc_exclusion_2d
    waveguide_mask_2d = waveguide_cad_footprint_2d & ~lc_exclusion_2d
    free_gap_mask_2d = lc_exclusion_2d & ~lc_mask_2d      # neither fixed 1 nor fixed 0
    if np.any(waveguide_mask_2d & lc_mask_2d):
        raise RuntimeError("effective waveguide and LC masks must be disjoint")
    if np.any(waveguide_mask_2d & free_gap_mask_2d):
        raise RuntimeError("the LC clearance band must not be frozen as waveguide")

    if DESIGN_INCLUDES_WAVEGUIDE:
        # nothing is pinned to SiN; the waveguide enters as the initial density
        fixed_one_mask = np.zeros(Ny * Nz, dtype=bool)
    else:
        fixed_one_mask = waveguide_mask_2d.reshape(-1)
    fixed_zero_mask = lc_mask_2d.reshape(-1)
    if fixed_one_mask.size != Ny * Nz or fixed_zero_mask.size != Ny * Nz:
        raise RuntimeError("fixed material masks do not match the reference-layer grid")
    print(
        "[fixed masks] "
        f"waveguides=1: {int(fixed_one_mask.sum())}/{Ny * Nz} per layer, "
        f"LC=0: {int(fixed_zero_mask.sum())}/{Ny * Nz} per layer, "
        "overlap: 0, "
        f"LC intersections removed from waveguide mask: {int(removed_overlap_mask_2d.sum())}, "
        f"free LC-waveguide gap (clearance {lc_clearance:g}um): {int(free_gap_mask_2d.sum())} cells per layer"
    )

    np.savez(
        os.path.join(design_dir, "switch_fixed_masks.npz"),
        y_um=_mask_y,
        z_um=_mask_z,
        waveguide_cad_footprint_mask=waveguide_cad_footprint_2d,
        waveguide_one_mask=waveguide_mask_2d,
        lc_zero_mask=lc_mask_2d,
        lc_exclusion_mask=lc_exclusion_2d,
        free_gap_mask=free_gap_mask_2d,
        lc_clearance_um=float(lc_clearance),
        removed_waveguide_lc_overlap=removed_overlap_mask_2d,
        combined_fixed_density=np.where(
            waveguide_mask_2d, 1.0, np.where(lc_mask_2d, 0.0, np.nan)
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, data, title in (
        (axes[0], waveguide_mask_2d, "two waveguides: fixed 1"),
        (axes[1], lc_mask_2d, "LC wall: fixed 0"),
        (
            axes[2],
            np.where(
                waveguide_mask_2d, 1.0,
                np.where(lc_mask_2d, 0.0, np.where(free_gap_mask_2d, 0.5, np.nan)),
            ),
            f"combined (grey = free LC gap, {lc_clearance:g}um)",
        ),
    ):
        image = ax.imshow(
            data.astype(float),
            origin="lower",
            extent=(_mask_z[0], _mask_z[-1], _mask_y[0], _mask_y[-1]),
            aspect="equal",
            vmin=0.0,
            vmax=1.0,
            cmap="gray_r",
        )
        ax.set_title(title)
        ax.set_xlabel("z (um)")
        ax.set_ylabel("y (um)")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(design_dir, "switch_fixed_masks.png"), dpi=180)
    plt.close(fig)

    def build_switch_mapping():
        # Is_waveguide selects Sub_Mapping's custom-mask branch.  Supplying
        # Fixed_waveguide_masks bypasses its legacy straight edge-port mask and
        # applies these exact CAD-aligned masks after filtering/projection.
        return ms.Opt_MS2.Mapping(
            Symmetry_sim=False,
            Sym_geo_width=False,
            Sym_geo_length=True,
            Sym_offdiag=False,
            Sym_geo_C2=False,
            Is_waveguide=[True, False, False, 2],
            DR_info=[design_region_x, design_region_y, design_region_z, 1, 2, 0],
            DR_N_info=[Nx, Ny, Nz, design_region_resolution],
            Mask_info=[0.0, 0.0],
            Mask_pixels=0,
            MFS=Min_s_top,
            MGS=Min_s_top,
            Fixed_waveguide_masks={
                "one": fixed_one_mask,
                "zero": fixed_zero_mask,
            },
        )

    def add_waveguide_outside_design(sim_obj, name, alpha, index):
        """Feed/collect arms drawn ONLY outside the design region.

        Inside the design box the structure has to come from the imported design
        grid alone; a CAD rectangle there would sit on top of it and pin the shape
        the optimizer is supposed to be free to change. The guide leaves the box
        through its +-z faces at arm distance s0 = (design_region_z/2)/cos(alpha),
        so each arm becomes two segments beyond that point.
        """
        a = np.deg2rad(alpha)
        ca_, sa_ = np.cos(a), np.sin(a)
        s0 = (0.5 * design_region_z) / max(ca_, 1e-9)
        seg = 2.0 * Sz - s0                      # long enough to reach the PML
        for sgn in (+1.0, -1.0):
            sc = sgn * (s0 + 0.5 * seg)
            add_tilted_waveguide(
                sim=sim_obj,
                name=f"{name}_{'p' if sgn > 0 else 'm'}",
                center=[X_min + BOX_h + 0.5 * Core_h, sc * sa_, sc * ca_],
                size=[Core_h, w_top, seg],
                index=index,
                alpha=alpha,
            )

    def add_lc_wall(sim_obj, idx, name="LC_wall"):
        fdtd = sim_obj.fdtd
        xc = X_min + BOX_h + 0.5 * (Core_h + TOX_h)
        fdtd.addrect()
        fdtd.set("name", name)
        fdtd.set("x", xc * 1e-6)
        fdtd.set("x span", (TOX_h + design_region_x) * 1e-6)
        fdtd.set("y", 0.0)
        fdtd.set("z", 0.0)
        fdtd.set("z span", lc_wall_length * 1e-6)
        fdtd.set("y span", wall_thickness * 1e-6)
        fdtd.set("first axis", "x")
        fdtd.set("rotation 1", lc_wall_rotation_deg)
        sim_obj._set_object_index(
            LC_n[idx],
            object_name=name,
            material_name=f"LC_wall_mat_{idx}",
            wavelength=Wavelengths,
        )
        try:
            fdtd.set("override mesh order from material database", True)
            fdtd.set("mesh order", 1)
        except Exception as exc:
            print(f"[LC_wall] could not set mesh order: {exc}")

    for idx in range(N_fom):
        sim[idx] = ms.Lumerical_utill.LumericalFDTDSimulator(
            sim_size=[Sx, Sy, Sz],
            resolution=resolution,
            unit=1e-6,
            background_index=1.44,
            center_wl=Wavelengths,
            N_f=1,
        )
        add_air_cap(sim[idx])
        if not Use_wall:
            sim[idx].add_geo(
                center=[X_min + BOX_h +Core_h+ (TOX_h), 0, 0],
                size=[(2*TOX_h), design_region_y, design_region_z],
                index=LC_n[idx],
                name='LC_tranche',
            )

        if DESIGN_INCLUDES_WAVEGUIDE:
            add_waveguide_outside_design(sim[idx], 'input_wg1', theta[0], SiN_n)
            add_waveguide_outside_design(sim[idx], 'input_wg2', theta[1], SiN_n)
        else:
            add_tilted_waveguide(
                sim=sim[idx],
                name='input_wg1',
                center=[X_min + BOX_h + 0.5 *Core_h, 0, 0],
                size=[Core_h, w_top, 2*Sz],
                index=SiN_n,
                alpha=theta[0],
            )

            add_tilted_waveguide(
                sim=sim[idx],
                name='input_wg2',
                center=[X_min + BOX_h + 0.5 *Core_h, 0, 0],
                size=[Core_h, w_top, 2*Sz],
                index=SiN_n,
                alpha=theta[1],
            )


        if DESIGN_INCLUDES_WAVEGUIDE:
            # start ON the validated CAD structure rather than on grey
            Initial_geo = waveguide_mask_2d.reshape(-1).astype(float)
        else:
            Initial_geo = np.ones((Ny * Nz,)) * 0.5
        mapping_preview = build_switch_mapping()
        design_3d = mapping_preview(Initial_geo, 1.0)

        if Use_wall:
            add_lc_wall(sim[idx], idx, name=f"LC_wall_{idx}")
            idx2=SiO2_n
        else:
            idx2=LC_n[idx]

        sim[idx].add_design_grid(
            name='design',
            center=[X_min + BOX_h + 0.5 * (Core_h), 0, Z_min + waveguide_length_I + 0.5 * design_region_z],
            size=[design_region_x, design_region_y, design_region_z],
            index1=SiN_n,
            index2=idx2,
            design_grids=design_grids,
            density=design_3d,
        )

        sim[idx].add_design_monitor()

        add_tilted_mode_source(
            sim=sim[idx],
            name='source',
            center=src_c[0],
            size=source_size,
            src_wl=[Wavelengths],
            alpha=theta[0],
        )


        sim[idx].add_monitor(
            name='FoM_monitor',
            center=Output_monitor_center,
            size=dft_monitor_size,
            N_f=1,
        )

        sim[idx].add_monitor(
            name='Cross_monitor',
            center=[X_min + BOX_h + 0.5 * (Core_h), 0, Z_min + waveguide_length_I + 0.5 * design_region_z],
            size=[0, Sy, Sz],
            N_f=1,
        )

    # FoM function kept close to original script
    def J0(E_x, E_y, E_z, H_x, H_y, H_z):
        E_out = [E_x[:, :, :, 0], E_y[:, :, :, 0], E_z[:, :, :, 0]]
        H_out = [H_x[:, :, :, 0], H_y[:, :, :, 0], H_z[:, :, :, 0]]

        Purity = ms.Opt_MS2.Overlap_intg(Target_E_Fields[0], E_out, normalization=True)
        Purity0 = ms.Opt_MS2.Overlap_intg(Target_E_Fields[0], E_out, normalization=True, self_norm=Input_ints[0])

        Power = port_flux(E_out, H_out, alpha=theta[0])
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

        # State-1 target = the y-output (reflection) mode; normalize the overlap by the
        # state-1 reference intensity (Input_ints[1]), NOT state 0.
        Purity = ms.Opt_MS2.Overlap_intg(Target_E_Fields[1], E_out, normalization=True)
        Purity0 = ms.Opt_MS2.Overlap_intg(Target_E_Fields[1], E_out, normalization=True, self_norm=Input_ints[0])

        Power = port_flux(E_out, H_out, alpha=theta[1])      # S_y for the y-output
        if isinstance(Purity, float):
            Foms_history[0].append(np.real(Purity))
            print(f'Purity: {Purity} & Refl: {Power / Input_powers[0]}, eff: {Purity0}')
        if isinstance(Power / Input_powers[0], float):
            Foms_history[1].append(np.real(Power / Input_powers[0]))

        # Same scale ([0,1] efficiency) as J0 so Goal_Attainment can balance the two.
        FoM = Purity0

        return FoM

    Js=[J0, J1]
    adj_dir = [False, False]
    # N_fom=1
    for idx in range(N_fom):
        # idx=1
        opt[idx] = ms.Lumerical_utill.LumericalOptimizationProblem(
            sim[idx],
            objective_functions=[Js[idx]],
            objective_arguments=[0, 1, 2, 3, 4, 5],
            FoM_size=dft_monitor_size,
            FoM_center=Output_monitor_center,
            adj_fwd=adj_dir[idx],
            opt_idx=idx,
        )

    # =========================================================================
    # Mapping and optimizer (kept from original script)
    # =========================================================================
    if Constrain:
        # Exact tilted guide/LC masks follow the Is_waveguide custom-mask branch.
        Is_waveguide = [True, False, False, 2]
        DR_info = [design_region_x, design_region_y, design_region_z, 1, 2, 0]
        DR_N_info = [Nx, Ny, Nz, design_region_resolution]
        Mask_info = [0.0, 0.0]
        MFS = Min_s_top
        MGS = Min_s_top

        evaluation_historyM = []
        for i in range(N_fom):
            evaluation_historyM.append([0])

        mapping = build_switch_mapping()


    def save_current_field_profile(design, f0_vals=None):
        rho=design.reshape(design_grids)
        plt.figure(figsize=(10, 4.8))
        plt.subplot(1, 3, 1)
        plt.imshow(rho[0], cmap="binary", vmin=0.0, vmax=1.0, aspect= (rho[0].shape[0]/rho[0].shape[1]))
        for idx in range(N_fom):
            # This field snapshot is diagnostic only, but it still starts a new
            # solver process.  A transient FlexNet shortage can therefore make
            # run() return without monitor data.  Do not let that abort the
            # optimization: rebuild the layout and rerun until H is readable.
            max_attempts = max(
                1, int(os.environ.get("LUMERICAL_PROFILE_RESULT_RETRIES", "20"))
            )
            retry_delay = max(
                0.0,
                float(os.environ.get("LUMERICAL_PROFILE_RESULT_RETRY_DELAY", "10")),
            )
            H_out = None
            last_error = None
            for attempt in range(1, max_attempts + 1):
                opt[idx].sim.fdtd.switchtolayout()
                opt[idx].sim.fdtd.setnamed('source', 'enabled', True)
                opt[idx].sim.run(name=f'temp_case_{idx}', save=True)
                try:
                    H_out = opt[idx].sim.fdtd.getresult('Cross_monitor', 'H')
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == max_attempts:
                        raise RuntimeError(
                            f"temp_case_{idx} returned without readable "
                            f"Cross_monitor/H after {max_attempts} attempts"
                        ) from exc
                    print(
                        f"[field profile] temp_case_{idx} returned without "
                        f"readable Cross_monitor/H; retrying in "
                        f"{retry_delay:g}s ({attempt}/{max_attempts - 1}). "
                        f"Last error: {exc}"
                    )
                    time.sleep(retry_delay)
            opt[idx].sim.fdtd.switchtolayout()
            Hout_all = np.array(H_out['H'], dtype=np.complex128)
            Hx = Hout_all[..., 0][:, :, :, 0]
            Hx_plot = np.squeeze(np.real(Hx))#/np.max(np.abs(np.flatten(np.real(Hx))))
            plt.subplot(1, 3, idx+2)
            plt.imshow(Hx_plot*Hx_plot, cmap="RdBu", aspect= (Hx_plot.shape[0]/Hx_plot.shape[1]))

        lines = []

        lines.append(f"state0 transmit eff={np.mean(f0_vals[0]):.3e}")
        if len(f0_vals) > 1:
            lines.append(f"state1 reflect eff={np.mean(f0_vals[1]):.3e}")
        plt.suptitle("Current design sections")
        if lines:
            plt.text(0.5, 0.02, "\n".join(lines), ha="center", va="bottom", fontsize=8.5)
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
        # Fixed guide/LC cells are already part of `mapping`.  Their derivatives
        # are zeroed by the final npa.where operations during Jacobian backprop;
        # no separate forward overlay or manual gradient edit is required.

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

        def Adjoint_loop(X, N_fom, Case=True):
            if Case == 3:
                if len(N_fom) == 1:
                    dJ_du = X[0][0]
                else:
                    dJ_du = ms.Opt_MS2.Goal_Attainment(X[1], N_fom, X[0])
                    # dJ_du = X[0][1] #+ X[0][1]
                    dJ_du = dJ_du.reshape(Nx,Ny,Nz)
                for i in range(len(N_fom)):
                    evaluation_historyM[i].append(N_fom[i])
                return dJ_du.flatten()
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
                    if Case:
                        dJ_dus[i] = apply_length_weight(dJ_dus[i])
                f0 = np.mean(f0s)
                if Case:
                    if isinstance(X, str):
                        return dJ_dus
                    else:
                        return f0, f0s, dJ_dus
                else:
                    path = save_current_field_profile(X, f0s)
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
            plt.imshow(z_slice[-1], cmap='binary')
            plt.axis('off')
            plt.savefig(os.path.join(design_dir, 'eps_last.png'))
            plt.close()
