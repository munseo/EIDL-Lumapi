"""Top-emission RGB microcavity OLED stack, reconstructed from the layer table,
characterized with the shared incoherent-dipole postprocess.

Stack (bottom -> top; emission exits UPWARD through the semi-transparent cathode):

    Ag anode mirror   100 nm    reflector
    ITO                10 nm    anode contact
    HAT-CN (HIL)       10 nm
    NPB (HTL)      140/95/60    * cavity-length tuning per color
    TCBTA (EBL)        10 nm
    EML             30/25/20    * dipole emission plane (dipoles at its center)
    TSPO1 (HBL)        10 nm
    TPBi (ETL)         30 nm
    Mg:Ag cathode      12 nm    * semi-transparent top mirror -> microcavity
    NPB (CPL)       70/60/50    * outcoupling capping layer
    air

ASSUMPTIONS -- the table does not contain these, so they are stated explicitly
and are all overridable by environment variable:

 1. PEAK WAVELENGTHS.  The table's indices are "@ Peak" but the peaks are not
    given.  Defaults R=0.620, G=0.530, B=0.460 um
    (MSOPT_MC_WAVELENGTH_UM overrides the active color's value).
 2. METAL EXTINCTION k.  The table lists n only.  Metals are dominated by k, so
    values are taken from Johnson & Christy (Ag) and an Ag-dominated alloy
    estimate (Mg:Ag 10:1):
        Ag     k = 4.05 / 3.32 / 2.66   (R/G/B)
        Mg:Ag  k = 3.60 / 3.10 / 2.70   (R/G/B)
    Override with MSOPT_MC_AG_K / MSOPT_MC_MGAG_K.
 3. RANGE-VALUED n.  Ag "0.06~0.12" and EML "1.78~1.81" are ranges; they are
    mapped across the colors as Ag 0.12/0.09/0.06 and EML 1.81/1.79/1.78
    (R/G/B).  Override with MSOPT_MC_AG_N / MSOPT_MC_EML_N.

The run is PLANAR by construction: this stack has no patterning, and the design
region the shared postprocess requires is a thin AIR slab above the CPL, so the
simulated structure is exactly the table's stack under air.  MSOPT_OLED_PP_PLANAR
is forced on, which also tags the manifest as a no-design reference.

MESH.  The thinnest layers are 10 nm, so the default resolution is 250 /um
(4 nm cells, ~2.5 cells per thin layer, 3 cells across the 12 nm cathode).  That
resolves the stack only marginally: treat absolute LEE from a single run as
provisional until MSOPT_OLED_RESOLUTION is swept (a convergence check).

Usage:
    run OLED_microcavity.py -th 8 -GPU <id>          # env MSOPT_MC_COLOR=green
    MSOPT_MC_COLOR=red   run OLED_microcavity.py ...
    MSOPT_MC_DRY_RUN=1 python OLED_microcavity.py    # print the stack, no session
"""

import os
import types

import numpy as np

import oled_common as oc


# =============================================================================
# Color-dependent table data
# =============================================================================
# MSOPT_MC_STACK selects which stack is built:
#   "optimized" (default) : thicknesses found by OLED_stack_design.py with the
#                           validated analytic solver, on Johnson & Christy Ag
#                           and literature organic/ITO indices
#   "table"               : the originally reconstructed layer table
# The table stack sits at cavity ANTI-resonance (green horizontal outcoupling
# 0.89 % vs 57 % optimized), which is why it is no longer the default.
TABLE_COLORS = {
    "red":   {"wavelength_um": 0.620, "htl_nm": 140.0, "eml_nm": 30.0, "cpl_nm": 70.0,
              "etl_nm": 30.0, "cath_nm": 12.0, "n_cpl": 1.800,
              "ag_n": 0.12, "ag_k": 4.05, "mgag_k": 3.60, "eml_n": 1.81},
    "green": {"wavelength_um": 0.530, "htl_nm":  95.0, "eml_nm": 25.0, "cpl_nm": 60.0,
              "etl_nm": 30.0, "cath_nm": 12.0, "n_cpl": 1.800,
              "ag_n": 0.09, "ag_k": 3.32, "mgag_k": 3.10, "eml_n": 1.79},
    "blue":  {"wavelength_um": 0.460, "htl_nm":  60.0, "eml_nm": 20.0, "cpl_nm": 50.0,
              "etl_nm": 30.0, "cath_nm": 12.0, "n_cpl": 1.800,
              "ag_n": 0.06, "ag_k": 2.66, "mgag_k": 2.70, "eml_n": 1.78},
}

# Optimized stack. Ag from Johnson & Christy (1972); organics/ITO from typical
# ellipsometry at each emission peak; ITO carries its (small) visible loss.
# Analytic horizontal-dipole outcoupling: R 61.8 %, G 57.3 %, B 50.3 %.
OPT_COLORS = {
    "red":   {"wavelength_um": 0.620, "htl_nm": 205.0, "eml_nm": 25.0, "cpl_nm": 75.0,
              "etl_nm": 50.0, "cath_nm": 10.0, "n_cpl": 2.45,
              "ag_n": 0.135, "ag_k": 3.990, "mgag_k": 3.60, "eml_n": 1.770,
              "n_ito": 1.900, "k_ito": 0.010, "n_hil": 1.830, "n_htl": 1.790,
              "n_ebl": 1.800, "n_hbl": 1.740, "n_etl": 1.730},
    "green": {"wavelength_um": 0.530, "htl_nm": 155.0, "eml_nm": 25.0, "cpl_nm": 70.0,
              "etl_nm": 40.0, "cath_nm": 13.0, "n_cpl": 2.20,
              "ag_n": 0.121, "ag_k": 3.090, "mgag_k": 3.10, "eml_n": 1.790,
              "n_ito": 1.950, "k_ito": 0.010, "n_hil": 1.850, "n_htl": 1.810,
              "n_ebl": 1.820, "n_hbl": 1.760, "n_etl": 1.750},
    "blue":  {"wavelength_um": 0.460, "htl_nm": 125.0, "eml_nm": 25.0, "cpl_nm": 70.0,
              "etl_nm": 20.0, "cath_nm": 11.0, "n_cpl": 2.00,
              "ag_n": 0.140, "ag_k": 2.560, "mgag_k": 2.70, "eml_n": 1.830,
              "n_ito": 2.020, "k_ito": 0.030, "n_hil": 1.880, "n_htl": 1.850,
              "n_ebl": 1.860, "n_hbl": 1.790, "n_etl": 1.790},
}

# Fixed by process/electrical requirements, not optical knobs.
ITO_NM, HIL_NM, EBL_NM, HBL_NM = 10.0, 10.0, 10.0, 10.0
AG_NM = 100.0
# Table-stack fallbacks for the indices the optimized set overrides per color.
N_ITO, N_HIL, N_HTL, N_EBL, N_HBL, N_ETL = 1.820, 1.850, 1.800, 1.760, 1.750, 1.740
N_MGAG = 0.250


def build_microcavity_config():
    which = os.environ.get("MSOPT_MC_STACK", "optimized").strip().lower()
    if which not in ("optimized", "table"):
        raise ValueError("MSOPT_MC_STACK must be 'optimized' or 'table'")
    table = OPT_COLORS if which == "optimized" else TABLE_COLORS
    color = os.environ.get("MSOPT_MC_COLOR", "green").strip().lower()
    if color not in table:
        raise ValueError(f"MSOPT_MC_COLOR must be one of {sorted(table)}")
    spec = dict(table[color])
    spec["stack_kind"] = which
    spec["wavelength_um"] = oc.env_float("MSOPT_MC_WAVELENGTH_UM", spec["wavelength_um"])
    spec["ag_n"] = oc.env_float("MSOPT_MC_AG_N", spec["ag_n"])
    spec["ag_k"] = oc.env_float("MSOPT_MC_AG_K", spec["ag_k"])
    spec["mgag_k"] = oc.env_float("MSOPT_MC_MGAG_K", spec["mgag_k"])
    spec["eml_n"] = oc.env_float("MSOPT_MC_EML_N", spec["eml_n"])
    wl = spec["wavelength_um"]

    # Lateral cell: planar, so it only has to be wide enough that the PML does
    # not clip the dipole near field. Small keeps the fine mesh affordable.
    period = oc.env_float("MSOPT_OLED_PERIOD_X_UM", 1.0)
    os.environ.setdefault("MSOPT_OLED_PERIOD_X_UM", str(period))
    os.environ.setdefault("MSOPT_OLED_PERIOD_Y_UM", str(period))
    os.environ.setdefault("MSOPT_OLED_RESOLUTION", "250")
    os.environ.setdefault("MSOPT_OLED_WAVELENGTHS", str(wl))
    os.environ.setdefault("MSOPT_OLED_PP_MODE", "single")     # no periodicity to tile
    # Monitor height above the stack: far enough that evanescent tails are gone
    # (>~0.5 lambda), CLOSE enough that the escape cone has not spread wider than
    # the domain before reaching it. 0.4 um of added air puts it ~0.9 lambda up.
    os.environ.setdefault("MSOPT_OLED_PP_N2F_FAR_Z_UM", "0.4")
    # The structure is planar, so emission is position-independent by symmetry:
    # one centred dipole per polarization is the whole answer, and keeping the
    # dipole at r=0 removes the off-centre clipping that a spread-out grid
    # suffers (an edge dipole's escape cone leaves the domain much sooner).
    os.environ.setdefault("MSOPT_OLED_PP_DIPOLE_GRID", "1")
    os.environ.setdefault("MSOPT_OLED_PP_KEEP_FSP", "0")      # drop .fsp + sibling dir + log
    os.environ["MSOPT_OLED_PP_PLANAR"] = "low"                # design slab = air

    # The stack now lives in oled_common, and build_config derives Sz, every z
    # coordinate, the EML plane, the design slab and all monitors from it. That
    # replaces the earlier approach of patching nine fields on the returned
    # config by hand, which is what mis-placed the far-field monitor twice.
    layer_specs, _spec = oc.microcavity_layers(color, spec["stack_kind"])
    os.environ.setdefault("MSOPT_OLED_AIR_TOP_UM", "0.60")
    os.environ.setdefault("MSOPT_OLED_DESIGN_H_UM", "0.05")   # thin AIR slab (planar)
    G = oc.build_config(period_x_default=period, layer_specs=layer_specs,
                        eml_layer_name=oc.MICROCAVITY_EML_LAYER)
    G.visible_wavelengths = np.asarray([wl], dtype=float)

    # --- lateral size from the capture angle -------------------------------
    # The top monitor sits h above the stack, so light leaving the dipole at
    # angle theta lands at r = h*tan(theta). If the domain is narrower than
    # that, the lateral PML eats the escape cone BEFORE it is measured: LEE
    # comes out far too low and (with a spread dipole grid) becomes spuriously
    # position dependent. Size the pad from the angle we intend to capture
    # instead of guessing it.
    pp_res = G.pp_resolution
    inset = oc.env_float("MSOPT_OLED_PP_MONITOR_BOUNDARY_INSET_UM",
                         0.50 + 2.0 / max(float(pp_res), 1.0))
    pp_far = oc.env_float("MSOPT_OLED_PP_N2F_FAR_Z_UM", 0.4)
    struct_top = G.design_c[2] + 0.5 * G.design_s[2]
    monitor_h = (G.Sz + pp_far) / 2.0 - inset - (struct_top - pp_far / 2.0)
    capture_deg = oc.env_float("MSOPT_MC_CAPTURE_ANGLE_DEG", 60.0)
    half_needed = monitor_h * float(np.tan(np.deg2rad(capture_deg))) + inset
    pad = max(half_needed - 0.5 * G.Sx, 0.1)
    os.environ.setdefault("MSOPT_OLED_PP_N2F_PAD_UM", f"{pad:.4f}")
    G.mc_monitor_h, G.mc_capture_deg = monitor_h, capture_deg
    G.mc_pad = float(os.environ["MSOPT_OLED_PP_N2F_PAD_UM"])

    G.mc_color, G.mc_spec = color, spec
    G.mc_stack_h = sum(l["size"][2] for l in G.stack_layers)
    return G


def print_stack(G):
    spec = G.mc_spec
    print("=" * 78)
    print(f"RGB microcavity OLED stack [{G.mc_spec['stack_kind']}] -- color = {G.mc_color.upper()}, "
          f"lambda = {spec['wavelength_um']:.3f} um")
    print("=" * 78)
    print(f"  cell           = {G.Sx:g} x {G.Sy:g} x {G.Sz:.4f} um, "
          f"resolution {G.resolution}/um ({1000.0 / G.resolution:.1f} nm mesh)")
    print(f"  stack height   = {G.mc_stack_h * 1000:.0f} nm")
    print(f"  cells          = {int(G.Sx * G.resolution)} x {int(G.Sy * G.resolution)} x "
          f"{int(G.Sz * G.resolution)} = {G.Sx * G.Sy * G.Sz * G.resolution ** 3 / 1e6:.0f} M")
    print(f"  {'layer':<18s}{'t (nm)':>8s}{'n':>8s}{'k':>8s}   {'z range (um)':>22s}")
    for layer in G.stack_layers:
        idx = layer["index"]
        h = layer["size"][2]
        z0 = layer["center"][2] - 0.5 * h
        print(f"  {layer['name']:<18s}{h * 1000:8.1f}{idx['n'][0]:8.3f}{idx['k'][0]:8.3f}   "
              f"{z0:+10.4f} .. {z0 + h:+10.4f}")
    print(f"  {'design slab (AIR)':<18s}{G.design_s[2] * 1000:8.1f}{1.0:8.3f}{0.0:8.3f}   "
          f"{G.design_c[2] - 0.5 * G.design_s[2]:+10.4f} .. "
          f"{G.design_c[2] + 0.5 * G.design_s[2]:+10.4f}")
    print(f"  dipole plane (EML center) z = {G.eml_c[2]:+.4f} um")
    print(f"  top monitor               z = {G.target_monitor_c[2]:+.4f} um")
    print(f"  design grid    = {G.design_grids} ({G.design_cells} cells)")
    lat = G.Sx + 2.0 * G.mc_pad
    print(f"  postprocess    = monitor {G.mc_monitor_h:.3f} um "
          f"({G.mc_monitor_h / G.mc_spec['wavelength_um']:.2f} lambda) above the stack, "
          f"capture {G.mc_capture_deg:g} deg -> pad {G.mc_pad:.3f} um, "
          f"lateral {lat:.2f} um ({int(lat * G.pp_resolution)} cells)")
    print("=" * 78)


def main():
    G = build_microcavity_config()
    print_stack(G)
    if oc.env_flag("MSOPT_MC_DRY_RUN", "0"):
        print("[dry-run] geometry only; no Lumerical session started.")
        return
    rho = np.zeros(G.design_cells, dtype=float)     # planar; run_postprocess re-zeros it too
    oc.run_postprocess(G, rho)


if __name__ == "__main__":
    main()
