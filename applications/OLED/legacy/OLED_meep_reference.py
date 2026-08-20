"""Lumerical port of the two Meep bare-LED reference scripts in this directory.

PURPOSE
-------
`step1_trace_comparison.py` (Method A) and `step2b_coherence_case2abc_36src.py`
are the lab's Meep reference numbers for a BARE LED (no OLED stack).  This
script reproduces the SAME structure, the SAME source ensembles and the SAME
reported quantities in Lumerical, so the two can be diffed field-by-field and
the OLED postprocess in `oled_common.py` can be trusted (or corrected).

It therefore builds its OWN geometry -- the bare-LED stack of the Meep scripts --
and deliberately does NOT call `oled_common.build_config`, which hardcodes the
OLED (Ag/TPBi/EML/TCTA/ITO/SiO2) stack.  Everything that is geometry
independent IS reused from `oled_common`: `make_sim`-style session creation
through `ms.Lumerical_utill.LumericalFDTDSimulator`, `add_dipole`,
`add_flux_box_monitors`, `read_transmission`, `read_source_power`,
`read_dipole_power`, `valid_power`, `source_freqs`, `load_run_results`
(GPU layout-mode reload guard), `n2f_spectrum`, `theta_phi_map`,
`save_radiation_map_figure`, `minimum_source_box_clearance`.

CASES (--case, default "all")
-----------------------------
  trace   step1 Method A: 6x6 grid -> the 3 positive x and 3 positive y
          coordinates -> 9 unique Ex dipole runs, each with multiplicity 4,
          power-weighted average (P_src_avg, P_up_ff_avg, eta_ff).
  case1   step2b incoherent reference: 36 individual Ex dipole runs on the 6x6
          endpoint grid (EDGE_INSET = 2 pixels), per-source (incoherent) average.
  case2a  all 36 dipoles simultaneously, identical phase (one run).
  case2b  all 36 simultaneously, ONE random phase draw (seed 0).
  case3   random-phase ensemble, MAX_TRIALS = 20 (seed 1234), running average
          with checkpoints at 1, 2, 5, 10, 20.

MEEP -> LUMERICAL QUANTITY MAP
------------------------------
  Meep                                        | Lumerical (this script)
  --------------------------------------------+-----------------------------------------
  mp.Simulation(cell_size, resolution, PML)   | ms.Lumerical_utill.LumericalFDTDSimulator
                                              |   (sim_size, resolution, bc_*="PML")
  mp.PML(0.50)                                | FDTD "pml layers" = 0.50 * resolution = 20
                                              |   (Lumerical's default 8 layers = 0.20 um)
  mp.Block(mp.inf, mp.inf, h)                 | sim.add_geo(...) rect of span CELL_XY+2 um
  mp.Medium(index=n)                          | add_geo(index=[n])   (non-dispersive)
  eps_averaging=True                          | Lumerical default conformal mesh refinement
  force_complex_fields=True                   | n/a: DFT monitors are complex by construction
  mp.Source(GaussianSource(fcen,0.05*fcen),   | oc.add_dipole(...) electric dipole,
            component=mp.Ex)                  |   theta=90, phi=0, wl start = wl stop = 0.527
  amplitude=exp(1j*phase)                     | dipole property "phase" [deg] = degrees(phase)
  stop_when_dft_decayed(1e-4, 0, 2000)        | FDTD "auto shutoff min"=1e-4 (set by the
                                              |   framework) + "simulation time" cap
                                              |   (2000 Meep units == 6671 fs)
  add_flux(...) 6-face box at src_box_*       | oc.add_flux_box_monitors("meepref_srcbox")
  mp.get_fluxes(src_box)[0]  -> P_src         | sum(sign * transmission(face)) * sourcepower()
  (no equivalent)                             | dipolepower()  -> P_src_dipolepower (primary)
  add_near2far(top + 4 lateral) + hemisphere  | sum of the OUTWARD flux through the SAME 5
    integral of Sr R^2 sin(th) dth dph        |   surfaces (identical by energy conservation
    -> P_up_ff                                |   for a lossless enclosing surface)
  sim.get_farfield(n2f, r)  -> Sr(theta,phi)  | oc.n2f_spectrum -> farfield3d/farfieldux/
                                              |   farfielduy on the TOP plane, re-binned by
                                              |   oc.theta_phi_map into Sr(theta,phi)
  eta_ff = P_up_ff / P_src                    | eta_ff = P_up_ff / P_src  (denominator chosen
                                              |   by MSOPT_MEEPREF_ETA_DENOM, see below)
  np.random.default_rng(seed).uniform(0,2pi)  | same RNG/seeds, converted to degrees

WHERE AN EXACT PORT WAS IMPOSSIBLE (and what was done instead)
--------------------------------------------------------------
 1. step1 Method B (cosine-basis "trace" formulation) is NOT ported.  It needs
    Meep's `amp_func`, an arbitrary spatially varying amplitude on an extended
    source; Lumerical has no equivalent for a dipole sheet.  Only Method A
    (the deterministic 9-run decomposition) is reproduced.
 2. step1's docstring calls its grid a "6x6 midpoint grid", but its CODE is
    `np.linspace(-half_range, half_range, 6)` -- an ENDPOINT grid identical to
    step2b's.  The CODE was ported (positive coords = 0.095, 0.285, 0.475).
 3. Near-to-far: Meep feeds 5 surfaces (top + 4 lateral) into ONE n2f transform
    and integrates the far field over the upper hemisphere.  Lumerical's
    farfield3d projects from a SINGLE planar monitor.  So:
      * P_up_ff  = direct outward Poynting flux through those same 5 surfaces
        (mathematically the same number for a lossless enclosing surface);
      * the ANGULAR PATTERN comes from the TOP plane only and therefore omits
        the power that leaves sideways.  Compare pattern SHAPE, not absolute
        level, with the Meep Sr map.
 4. Sr(theta, phi): Meep samples exact nodes on a 19 x 36 (theta, phi) grid at
    R = 1e6.  farfield3d returns a uniform direction-cosine grid, which
    `oc.theta_phi_map` re-bins into (theta, phi) BIN MEANS on the same 19 x 36
    shape.  Shapes are comparable; absolute Sr scales are not (Meep's Sr is a
    far-field intensity at R, this is a per-direction mean of the calibrated
    top-plane power spectrum).
 5. Source spectrum: Meep uses a Gaussian pulse (fwidth = 0.05 fcen) and DFTs at
    fcen; the Lumerical dipole is set with wavelength start = stop = 0.527 um,
    the established single-wavelength pattern of the sibling OLED scripts.  The
    single-frequency steady-state results are equivalent; the transients are not.
 6. Meep tolerates a point source lying exactly ON a flux-region face; Lumerical
    does not (an on-face source loses ~1/2, a corner one ~3/4 of its box power).
    BOTH Meep grids put the outermost dipoles at x,y = +-0.475 um, which is
    EXACTLY src_box_half.  The box is kept at the Meep coordinates by default
    (faithful port) and every run records `src_box_min_clearance_um`; a warning
    is printed when it is <= 0.  MSOPT_MEEPREF_SRC_BOX_INSET_PX=1 shrinks the
    inset so the box strictly encloses the grid, for a corrected cross-check.
 7. `oled_common.read_dipole_power` sums only POSITIVE dipolepower entries
    (`finite_sum`).  In a coherent array an individual dipole can ABSORB
    (negative dipolepower), so that convention over-estimates P_src for
    case2a/2b/3.  This script uses a SIGNED sum as the primary
    `P_src_dipolepower` and also records `P_src_dipolepower_positive_only`
    (the oled_common value) so the bias is measurable -- this is one of the
    oled_common behaviours the comparison is meant to check.
 8. Phase sign convention: Meep's `amplitude = exp(+i phi)` vs Lumerical's
    dipole "phase" property.  A global sign flip does not change case2a or the
    case3 ensemble statistics, but it does change the single realization
    case2b.  MSOPT_MEEPREF_PHASE_SIGN=-1 flips it if the realization disagrees.
 9. RNG: identical `np.random.default_rng(seed)` streams.  Meep draws
    uniform(0, 2pi) and this script converts to degrees; `oled_common` draws
    uniform(0, 360) directly -- the same numbers to floating-point rounding.
10. Meep builds a FRESH mp.Simulation per run; here ONE Lumerical session is
    reused (switchtolayout -> delete/re-add dipoles -> run), the sibling
    scripts' pattern.  Geometry and monitors are identical across runs.
11. mp.inf blocks become finite rects of span CELL_XY + 2 um, and the substrate
    block is extended 1 um BELOW the domain floor (physically irrelevant, it
    only avoids a material edge exactly on the boundary cell).

ENV KNOBS (prefix MSOPT_MEEPREF_)
---------------------------------
  RESOLUTION            40      cells per um (Meep `resolution`)
  WAVELENGTH_UM         0.527   single wavelength
  CELL_XY_UM            7.0     lateral cell size (PML included, as in Meep)
  N_DIPOLES_GRID        6       source grid N (6 -> 36 dipoles / 9 trace runs)
  CASE3_TRIALS          20      case3 ensemble size
  KEEP_FSP              0       0 -> delete each run's .fsp, its sibling
                                directory AND its _p0.log after extraction
  PML_LAYERS            20      = round(0.50 * RESOLUTION)
  SIM_TIME_FS           1000    Lumerical simulation-time cap (Meep's 2000
                                time units would be 6671 fs)
  SRC_BOX_INSET_PX      2       Meep SRC_BOX_INSET_PX (see deviation 6)
  FF_N_THETA / FF_N_PHI 19 / 36 Sr(theta,phi) map shape (Meep FF_N_THETA/PHI)
  N2F_POINTS            181     farfield3d grid points per axis
  ETA_DENOM             dipolepower | box  -- denominator of the headline eta_ff
  PHASE_SIGN            +1      see deviation 8
  SESSION_TEST          0       1 == --dry-run
  MSOPT_MEEPREF_OUTDIR  <RUN_DIR>/A  output folder override

OUTPUTS (in $EIDL_RUN_DIR/A/)
-----------------------------
  dipole_layout.png, results.json, coherence_case23_summary.png,
  <case>_radiation.(png|npz), case3_trial_fields/trial_*_radiation_field.(png|npz),
  case3_checkpoint_*_radiation.png

DO NOT run this on a busy GPU without checking free disk: with KEEP_FSP=0 each
run's project is removed right after extraction, which is why the default is 0.
"""

import argparse
import json
import os
import shutil
import sys
import time
import types
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import msopt as ms  # noqa: E402  (framework entry point, as in the sibling scripts)
import oled_common as oc  # noqa: E402


# =============================================================================
# 0. Output location (same convention as the sibling scripts)
# =============================================================================

RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
DESIGN_DIR = os.path.abspath(
    os.environ.get("MSOPT_MEEPREF_OUTDIR", os.path.join(RUN_DIR, "A"))
) + os.sep


def env_int(name, default):
    return oc.env_int(f"MSOPT_MEEPREF_{name}", default)


def env_float(name, default):
    return oc.env_float(f"MSOPT_MEEPREF_{name}", default)


def env_flag(name, default="0"):
    return oc.env_flag(f"MSOPT_MEEPREF_{name}", default)


def env_str(name, default):
    return os.environ.get(f"MSOPT_MEEPREF_{name}", default).strip()


# =============================================================================
# 1. Parameters  -- byte-for-byte the same expressions as the Meep scripts
# =============================================================================

resolution = env_int("RESOLUTION", 40)
target_wvl = env_float("WAVELENGTH_UM", 0.527)
fcen = 1.0 / target_wvl
fwidth = 0.05 * fcen                      # Meep pulse width; no Lumerical equivalent

n_GaN = 2.40
n_Sub = 2.49
n_Poly = 1.50

channel_w, channel_l = 1.05, 1.05
p_gan_h, mqw_h, n_gan_h = 0.20, 0.20, 0.70
sub_h = 0.20

pml_th = 0.50
PIXEL = 1.0 / resolution
mon_pml_gap_px = 2
mon_pml_gap = mon_pml_gap_px * PIXEL

CELL_XY = env_float("CELL_XY_UM", 7.0)

gan_total_h = n_gan_h + mqw_h + p_gan_h
air_pad_top = max(4 * target_wvl, 2.0)
sz_raw = pml_th + sub_h + gan_total_h + air_pad_top + pml_th
sz = round(sz_raw * resolution) / resolution

z_bot = -sz / 2
z_sub_bot = z_bot + pml_th
z_sub_top = z_sub_bot + sub_h
z_gan_c = z_sub_top + gan_total_h / 2
z_mqw_c = z_sub_top + n_gan_h + mqw_h / 2
z_gan_top = z_sub_top + gan_total_h
src_z = z_mqw_c

z_top_mon = sz / 2 - pml_th - mon_pml_gap
ap_half = CELL_XY / 2 - pml_th - mon_pml_gap

SRC_BOX_INSET_PX = env_float("SRC_BOX_INSET_PX", 2)
src_box_inset = SRC_BOX_INSET_PX * PIXEL
src_box_half = channel_w / 2 - src_box_inset
src_box_z_lo = src_z - mqw_h / 2 + src_box_inset
src_box_z_hi = src_z + mqw_h / 2 - src_box_inset

AIR_FF_LAT_BOT_MARGIN = 0.5 * target_wvl
z_ff_lat_bot = z_gan_top + AIR_FF_LAT_BOT_MARGIN

EDGE_INSET = 2 * PIXEL
# The Meep scripts guard their ratios with max(P, EPS_RATIO), EPS_RATIO = 1e-12.
# That is safe in Meep, whose powers are dimensionless and O(1), but it is NOT
# portable: Lumerical reports absolute watts and this bare-LED emits ~1e-15 W,
# so max(P_src, 1e-12) returned the GUARD for every run and every eta became
# P_up*1e12 instead of P_up/P_src.  Ratios here are therefore taken with
# safe_ratio(), which only guards against a non-positive/non-finite denominator.
EPS_RATIO = 1e-12          # kept only for reference to the Meep scripts


def safe_ratio(numerator, denominator):
    """numerator/denominator, or None when the denominator is unusable.

    Never substitutes a constant for the denominator -- a scale-dependent floor
    silently produces wrong efficiencies (see the EPS_RATIO note above).
    """
    if numerator is None or denominator is None:
        return None
    num, den = float(numerator), float(denominator)
    if not np.isfinite(den) or den <= 0.0 or not np.isfinite(num):
        return None
    return num / den
FF_N_THETA = env_int("FF_N_THETA", 19)
FF_N_PHI = env_int("FF_N_PHI", 36)

SOURCE_GRID_N = env_int("N_DIPOLES_GRID", 6)
CASE2_RANDOM_SEED = 0
CASE3_RANDOM_SEED = 1234
CASE3_TRIAL_COUNTS = (1, 2, 5, 10, 20)
MAX_TRIALS = env_int("CASE3_TRIALS", max(CASE3_TRIAL_COUNTS))

# --- Lumerical-only knobs -----------------------------------------------------
PML_LAYERS = env_int("PML_LAYERS", int(round(pml_th * resolution)))
SIM_TIME_FS = env_float("SIM_TIME_FS", 1000.0)
# Meep SIM_MAX_TIME = 2000 Meep time units == 2000 * (1 um / c) == 6671.3 fs
MEEP_MAX_TIME_FS = 2000.0 * (1e-6 / 299792458.0) * 1e15
N2F_POINTS = env_int("N2F_POINTS", 181)
KEEP_FSP = env_flag("KEEP_FSP", "0")
PHASE_SIGN = 1.0 if env_float("PHASE_SIGN", 1.0) >= 0 else -1.0
ETA_DENOM = env_str("ETA_DENOM", "dipolepower").lower()
if ETA_DENOM not in ("dipolepower", "box"):
    raise ValueError("MSOPT_MEEPREF_ETA_DENOM must be 'dipolepower' or 'box'.")

INF_SPAN = CELL_XY + 2.0           # mp.inf substitute for the layer blocks
SUB_OVERHANG = 1.0                 # substrate extended below the domain floor

TOP_MONITOR = "meepref_ff_zp"      # the +z face of the far-field box
FF_PREFIX = "meepref_ff"
SRC_BOX_PREFIX = "meepref_srcbox"
DIPOLE_FMT = "meepref_dipole_{:03d}"

ALL_CASES = ("trace", "case1", "case2a", "case2b", "case3")


# =============================================================================
# 2. Minimal config namespace for the geometry-independent oled_common helpers
# =============================================================================

def make_G():
    """The few attributes `oc.add_dipole` / `oc.source_freqs` read from `G`.

    This intentionally REPLACES oc.build_config (which hardcodes the OLED
    stack) with the bare-LED numbers of the Meep scripts.
    """
    return types.SimpleNamespace(
        visible_wavelengths=np.asarray([target_wvl], dtype=float),
        resolution=resolution,
        background_index=1.0,
        bc_xy="PML",
        design_dir=DESIGN_DIR,
        # NOT EPS_RATIO: this floor is compared against absolute watts, and this
        # bare LED emits ~1e-15 W, so a 1e-12 floor would replace every real
        # power (same trap as the eta guard above).
        channel_power_floor=1e-30,
    )


G = make_G()


# =============================================================================
# 3. Source ensembles  (identical expressions to the Meep scripts)
# =============================================================================

def make_source_entry(x_centered, y_centered, label, mult=1):
    return {
        "x_centered": float(x_centered),
        "y_centered": float(y_centered),
        "component": "Ex",
        "pol_label": "Ex",
        "label": label,
        "mult": int(mult),
    }


def make_original_36src_specs():
    """step2b: 6x6 ENDPOINT grid, Ex-only, 36 dipoles."""
    half_range_x = channel_w / 2 - EDGE_INSET
    half_range_y = channel_l / 2 - EDGE_INSET
    x_coords = np.linspace(-half_range_x, half_range_x, SOURCE_GRID_N)
    y_coords = np.linspace(-half_range_y, half_range_y, SOURCE_GRID_N)
    specs = []
    idx = 0
    for x in x_coords:
        for y in y_coords:
            idx += 1
            specs.append(make_source_entry(x, y, f"Ex@{idx:02d}"))
    return specs


def make_trace_specs():
    """step1 Method A: the POSITIVE coordinates of the same grid, mult=4 each.

    NOTE: step1 calls this a "midpoint" grid in its docstring, but its code is
    `np.linspace(-half_range, half_range, 6)` -- the endpoint grid above.
    """
    half_range = channel_w / 2 - EDGE_INSET
    coords = np.linspace(-half_range, half_range, SOURCE_GRID_N)
    pos_coords = coords[coords > 1e-12]
    specs = []
    for ix, x in enumerate(pos_coords):
        for iy, y in enumerate(pos_coords):
            idx = ix * len(pos_coords) + iy
            specs.append(make_source_entry(x, y, f"det_{idx}", mult=4))
    return specs


SOURCE_SPECS = make_original_36src_specs()
TRACE_SPECS = make_trace_specs()


def make_case2_random_phase(n_sources, seed=CASE2_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=n_sources)


def make_case3_random_phases(n_trials, n_sources, seed=CASE3_RANDOM_SEED):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=(n_trials, n_sources))


def phases_to_degrees(phases_rad):
    return PHASE_SIGN * np.degrees(np.asarray(phases_rad, dtype=float))


# =============================================================================
# 4. Resolved-geometry report (the --dry-run review table)
# =============================================================================

def geometry_lines():
    nx = int(round(CELL_XY * resolution))
    ny = int(round(CELL_XY * resolution))
    nz = int(round(sz * resolution))
    ff_lat_h = z_top_mon - z_ff_lat_bot
    ff_lat_zc = 0.5 * (z_top_mon + z_ff_lat_bot)
    src_box_h = src_box_z_hi - src_box_z_lo
    src_box_zc = 0.5 * (src_box_z_lo + src_box_z_hi)
    L = []
    A = L.append
    A("=" * 78)
    A("RESOLVED GEOMETRY  (Lumerical port of the Meep bare-LED reference)")
    A("=" * 78)
    A("[materials / layers]")
    A(f"  n_GaN                 = {n_GaN:.4f}   (channel {channel_w:g} x {channel_l:g} um)")
    A(f"  n_Sub                 = {n_Sub:.4f}")
    A(f"  n_Poly                = {n_Poly:.4f}   (surrounding matrix)")
    A(f"  background index      = 1.0000   (air above z_gan_top)")
    A(f"  p_GaN / MQW / n_GaN   = {p_gan_h:g} / {mqw_h:g} / {n_gan_h:g} um  "
      f"(gan_total_h = {gan_total_h:g} um)")
    A(f"  sub_h                 = {sub_h:g} um")
    A("")
    A("[cell]")
    A(f"  resolution            = {resolution} cells/um   (PIXEL = {PIXEL:.6g} um)")
    A(f"  wavelength            = {target_wvl:g} um   (fcen = {fcen:.6g}, "
      f"meep fwidth = {fwidth:.6g})")
    A(f"  CELL_XY               = {CELL_XY:g} um  (x and y, PML included)")
    A(f"  pml_th                = {pml_th:g} um  -> Lumerical 'pml layers' = {PML_LAYERS}")
    A(f"  air_pad_top           = max(4*wvl, 2.0) = {air_pad_top:.6g} um")
    A(f"  sz_raw                = pml+sub+gan+air+pml = {sz_raw:.6g} um")
    A(f"  sz                    = round(sz_raw*res)/res = {sz:.6g} um")
    A(f"  cell counts           = {nx} x {ny} x {nz} = {nx*ny*nz/1e6:.3f} M cells")
    A("")
    A("[z coordinates]  (identical expressions to the Meep scripts)")
    A(f"  z_bot                 = -sz/2                       = {z_bot:+.6f}")
    A(f"  z_sub_bot             = z_bot + pml_th              = {z_sub_bot:+.6f}")
    A(f"  z_sub_top             = z_sub_bot + sub_h           = {z_sub_top:+.6f}")
    A(f"  z_gan_c               = z_sub_top + gan_total_h/2   = {z_gan_c:+.6f}")
    A(f"  z_mqw_c (= src_z)     = z_sub_top + n_gan_h + mqw_h/2 = {z_mqw_c:+.6f}")
    A(f"  z_gan_top             = z_sub_top + gan_total_h     = {z_gan_top:+.6f}")
    A(f"  z_top_mon             = sz/2 - pml_th - mon_pml_gap = {z_top_mon:+.6f}")
    A(f"  ap_half               = CELL_XY/2 - pml_th - gap    = {ap_half:+.6f}")
    A(f"  z_ff_lat_bot          = z_gan_top + 0.5*wvl         = {z_ff_lat_bot:+.6f}")
    A(f"  mon_pml_gap           = {mon_pml_gap_px} px          = {mon_pml_gap:.6f}")
    A("")
    A("[blocks as built in Lumerical]  (mp.inf -> finite span "
      f"{INF_SPAN:g} um)")
    A(f"  substrate  center=(0, 0, {0.5*((z_bot - SUB_OVERHANG) + z_sub_top):+.6f})  "
      f"size=({INF_SPAN:g}, {INF_SPAN:g}, {z_sub_top - z_bot + SUB_OVERHANG:.6f})  n={n_Sub}")
    A(f"             [Meep: center z {z_bot + (pml_th + sub_h)/2:+.6f}, "
      f"height {pml_th + sub_h:.6f}; extended {SUB_OVERHANG:g} um below the floor]")
    A(f"  poly       center=(0, 0, {z_gan_c:+.6f})  "
      f"size=({INF_SPAN:g}, {INF_SPAN:g}, {gan_total_h:.6f})  n={n_Poly}")
    A(f"  GaN chan   center=(0, 0, {z_gan_c:+.6f})  "
      f"size=({channel_w:g}, {channel_l:g}, {gan_total_h:.6f})  n={n_GaN}")
    A("")
    A("[source flux box]  (Meep src_box, 6 faces)")
    A(f"  SRC_BOX_INSET_PX      = {SRC_BOX_INSET_PX:g}  -> inset {src_box_inset:.6f} um")
    A(f"  src_box_half          = {src_box_half:+.6f}")
    A(f"  src_box_z_lo / _hi    = {src_box_z_lo:+.6f} / {src_box_z_hi:+.6f}")
    A(f"  box center/size       = (0, 0, {src_box_zc:+.6f}) / "
      f"({2*src_box_half:.6f}, {2*src_box_half:.6f}, {src_box_h:.6f})")
    A(f"  faces                 = {SRC_BOX_PREFIX}_(xp,xm,yp,ym,zp,zm)")
    A("")
    A("[far-field surface]  (Meep near2far: top + 4 lateral; +bottom here as a check)")
    A(f"  top      {FF_PREFIX}_zp  center=(0, 0, {z_top_mon:+.6f})  "
      f"size=({2*ap_half:.6f}, {2*ap_half:.6f}, 0)      sign=+1")
    A(f"  lateral  {FF_PREFIX}_xp  center=({+ap_half:+.6f}, 0, {ff_lat_zc:+.6f})  "
      f"size=(0, {2*ap_half:.6f}, {ff_lat_h:.6f})  sign=+1")
    A(f"  lateral  {FF_PREFIX}_xm  center=({-ap_half:+.6f}, 0, {ff_lat_zc:+.6f})  "
      f"size=(0, {2*ap_half:.6f}, {ff_lat_h:.6f})  sign=-1")
    A(f"  lateral  {FF_PREFIX}_yp  center=(0, {+ap_half:+.6f}, {ff_lat_zc:+.6f})  "
      f"size=({2*ap_half:.6f}, 0, {ff_lat_h:.6f})  sign=+1")
    A(f"  lateral  {FF_PREFIX}_ym  center=(0, {-ap_half:+.6f}, {ff_lat_zc:+.6f})  "
      f"size=({2*ap_half:.6f}, 0, {ff_lat_h:.6f})  sign=-1")
    A(f"  bottom   {FF_PREFIX}_zm  center=(0, 0, {z_ff_lat_bot:+.6f})  "
      f"size=({2*ap_half:.6f}, {2*ap_half:.6f}, 0)      NOT in P_up (diagnostic influx)")
    A(f"  ff_lat_h / ff_lat_zc  = {ff_lat_h:.6f} / {ff_lat_zc:+.6f}")
    A(f"  angular readout       = farfield3d on {TOP_MONITOR}, {N2F_POINTS}x{N2F_POINTS} "
      f"-> Sr map {FF_N_THETA} x {FF_N_PHI}")
    A("")
    A("[dipoles]  Ex-oriented (theta=90, phi=0) at z = z_mqw_c = "
      f"{src_z:+.6f}")
    A(f"  EDGE_INSET            = 2 px = {EDGE_INSET:.6f} um")
    half_range_x = channel_w / 2 - EDGE_INSET
    coords = np.linspace(-half_range_x, half_range_x, SOURCE_GRID_N)
    A(f"  grid coords ({SOURCE_GRID_N})       = ["
      + ", ".join(f"{c:+.6f}" for c in coords) + "]")
    A(f"  case1/2a/2b/3 sources = {len(SOURCE_SPECS)} (full {SOURCE_GRID_N}x{SOURCE_GRID_N} grid)")
    for s in SOURCE_SPECS:
        A(f"      {s['label']:>8s}  ({s['x_centered']:+.6f}, {s['y_centered']:+.6f})")
    A(f"  trace sources         = {len(TRACE_SPECS)} unique runs, multiplicity 4 each")
    for s in TRACE_SPECS:
        A(f"      {s['label']:>8s}  ({s['x_centered']:+.6f}, {s['y_centered']:+.6f})  "
          f"mult={s['mult']}")
    clear_36 = oc.minimum_source_box_clearance(
        [(s["x_centered"], s["y_centered"], src_z) for s in SOURCE_SPECS],
        [0.0, 0.0, src_box_zc],
        [2 * src_box_half, 2 * src_box_half, src_box_h],
    )
    clear_tr = oc.minimum_source_box_clearance(
        [(s["x_centered"], s["y_centered"], src_z) for s in TRACE_SPECS],
        [0.0, 0.0, src_box_zc],
        [2 * src_box_half, 2 * src_box_half, src_box_h],
    )
    A("")
    A("[source-box clearance]  (Meep tolerates 0, Lumerical does not -- deviation 6)")
    A(f"  min clearance, 36-grid = {clear_36:+.6f} um"
      + ("   <-- WARNING: sources ON the box face" if clear_36 is not None and clear_36 <= 0 else ""))
    A(f"  min clearance, trace   = {clear_tr:+.6f} um"
      + ("   <-- WARNING: sources ON the box face" if clear_tr is not None and clear_tr <= 0 else ""))
    A("")
    A("[run control]")
    A(f"  auto shutoff min      = 1e-4  (Meep SIM_DFT_TOL)")
    A(f"  simulation time       = {SIM_TIME_FS:g} fs "
      f"(Meep SIM_MAX_TIME=2000 == {MEEP_MAX_TIME_FS:.1f} fs)")
    A(f"  eta_ff denominator    = {ETA_DENOM}")
    A(f"  phase sign            = {PHASE_SIGN:+.0f}")
    A(f"  case2 seed / case3 seed / trials = {CASE2_RANDOM_SEED} / "
      f"{CASE3_RANDOM_SEED} / {MAX_TRIALS}")
    A(f"  keep .fsp             = {int(KEEP_FSP)}")
    A(f"  output dir            = {DESIGN_DIR}")
    A("=" * 78)
    return L


# =============================================================================
# 5. Lumerical session: geometry + monitors
# =============================================================================

def build_sim():
    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[CELL_XY, CELL_XY, sz],
        resolution=resolution,
        unit=1e-6,
        background_index=1.0,
        center_wl=target_wvl,
        N_f=1,
        bc_x="PML",
        bc_y="PML",
        bc_z="PML",
    )
    fdtd = sim.fdtd

    # Meep's PML thickness is a LENGTH; Lumerical's is a layer count.
    try:
        fdtd.setnamed("FDTD", "pml layers", int(PML_LAYERS))
        print(f"[setup] pml layers = {PML_LAYERS} ({PML_LAYERS * PIXEL:g} um)")
    except Exception as exc:
        print(f"[setup] warning: could not set 'pml layers' ({exc}); "
              f"Lumerical default (8 layers = {8 * PIXEL:g} um) is in use")
    try:
        fdtd.setnamed("FDTD", "simulation time", float(SIM_TIME_FS) * 1e-15)
    except Exception as exc:
        print(f"[setup] warning: could not set 'simulation time': {exc}")
    try:
        fdtd.setnamed("FDTD", "auto shutoff min", 1e-4)   # Meep SIM_DFT_TOL
    except Exception:
        pass

    # --- geometry (Meep make_geometry) ---------------------------------------
    sub_top = z_sub_top
    sub_bot = z_bot - SUB_OVERHANG
    sim.add_geo(
        [0.0, 0.0, 0.5 * (sub_bot + sub_top)],
        [INF_SPAN, INF_SPAN, sub_top - sub_bot],
        [n_Sub], "substrate", target_wvl,
    )
    sim.add_geo(
        [0.0, 0.0, z_gan_c], [INF_SPAN, INF_SPAN, gan_total_h],
        [n_Poly], "poly_matrix", target_wvl,
    )
    sim.add_geo(
        [0.0, 0.0, z_gan_c], [channel_w, channel_l, gan_total_h],
        [n_GaN], "gan_channel", target_wvl,
    )

    # --- monitors ------------------------------------------------------------
    src_box_h = src_box_z_hi - src_box_z_lo
    src_box_zc = 0.5 * (src_box_z_lo + src_box_z_hi)
    src_box_faces = oc.add_flux_box_monitors(
        sim, SRC_BOX_PREFIX,
        [0.0, 0.0, src_box_zc],
        [2 * src_box_half, 2 * src_box_half, src_box_h],
    )

    ff_lat_h = z_top_mon - z_ff_lat_bot
    ff_lat_zc = 0.5 * (z_top_mon + z_ff_lat_bot)
    ff_faces = oc.add_flux_box_monitors(
        sim, FF_PREFIX,
        [0.0, 0.0, ff_lat_zc],
        [2 * ap_half, 2 * ap_half, ff_lat_h],
    )
    # Meep's near2far surface is top + 4 lateral; the bottom face is kept only
    # as a closed-box energy-balance diagnostic and is EXCLUDED from P_up.
    ff_up_faces = [(name, sign) for name, sign in ff_faces
                   if not name.endswith("_zm")]
    ff_bottom_faces = [(name, sign) for name, sign in ff_faces
                       if name.endswith("_zm")]

    ctx = {
        "sim": sim,
        "src_box_faces": src_box_faces,
        "ff_up_faces": ff_up_faces,
        "ff_bottom_faces": ff_bottom_faces,
        "src_box_center": [0.0, 0.0, src_box_zc],
        "src_box_size": [2 * src_box_half, 2 * src_box_half, src_box_h],
        "n_dipoles_present": 0,
    }
    return ctx


# =============================================================================
# 6. Result readers
# =============================================================================

def read_dipole_power_signed(fdtd, freqs_hz):
    """SIGNED sum of dipolepower over sources/frequencies.

    oled_common.read_dipole_power keeps only POSITIVE entries; in a coherent
    array a dipole can absorb, so the positive-only sum over-estimates the
    emitted power (deviation 7).
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float).reshape(-1)
    try:
        vals = fdtd.dipolepower(freqs_hz)
    except Exception:
        fdtd.putv("meepref_dp_freqs", freqs_hz)
        fdtd.eval("meepref_dp_values = dipolepower(meepref_dp_freqs);")
        vals = fdtd.getv("meepref_dp_values")
    vals = np.real(np.asarray(vals, dtype=np.complex128)).reshape(-1)
    vals = vals[np.isfinite(vals)]
    return float(np.sum(vals)) if vals.size else None


def signed_flux_power(fdtd, faces, source_power):
    """Outward power through a signed set of faces (sign convention of
    oled_common.read_flux_box_power, but WITHOUT dropping negative results so a
    sign/geometry problem stays visible)."""
    if source_power is None:
        return None
    total = 0.0
    for name, sign in faces:
        total += float(sign) * oc.read_transmission(fdtd, name)
    return float(total) * float(source_power)


def drop_last_fsp(sim):
    """MSOPT_MEEPREF_KEEP_FSP=0: delete the just-analyzed run's .fsp, its
    SIBLING DIRECTORY and its _p0.log (copied from oled_common._drop_last_fsp:
    removing only the .fsp once filled the disk)."""
    if KEEP_FSP:
        return
    fsp = getattr(sim, "_last_run_fsp_path", None)
    if not fsp:
        return
    stem = fsp[:-4] if fsp.endswith(".fsp") else fsp
    for path in (fsp, f"{stem}_p0.log"):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
    try:
        if os.path.isdir(stem):
            shutil.rmtree(stem, ignore_errors=True)
    except Exception:
        pass


# =============================================================================
# 7. One simulation run
# =============================================================================

def set_dipoles(ctx, specs, phases_rad=None):
    sim = ctx["sim"]
    fdtd = sim.fdtd
    fdtd.switchtolayout()
    for k in range(max(ctx["n_dipoles_present"], len(specs))):
        oc.delete_object(fdtd, DIPOLE_FMT.format(k))
    phases_deg = (np.zeros(len(specs)) if phases_rad is None
                  else phases_to_degrees(phases_rad))
    for k, spec in enumerate(specs):
        name = DIPOLE_FMT.format(k)
        oc.add_dipole(G, sim, spec["x_centered"], spec["y_centered"], src_z,
                      "x", name)
        try:
            fdtd.setnamed(name, "phase", float(phases_deg[k]))
        except Exception as exc:
            if abs(float(phases_deg[k])) > 1e-12:
                raise RuntimeError(f"could not set phase on {name}: {exc}")
    ctx["n_dipoles_present"] = len(specs)
    return phases_deg


def run_once(ctx, specs, tag, phases_rad=None):
    """Run one Lumerical simulation and extract every Meep-equivalent quantity."""
    sim = ctx["sim"]
    fdtd = sim.fdtd
    phases_deg = set_dipoles(ctx, specs, phases_rad)

    t0 = time.time()
    sim.run(name=f"meepref_{tag}", save=True)
    t_fdtd = time.time() - t0

    t0 = time.time()
    oc.load_run_results(sim)
    freqs = oc.source_freqs(G, sim)
    src_power = oc.valid_power(oc.read_source_power(fdtd, freqs))
    dip_signed = read_dipole_power_signed(fdtd, freqs)
    dip_positive = oc.read_dipole_power(fdtd, freqs)      # oled_common convention
    P_src_box = signed_flux_power(fdtd, ctx["src_box_faces"], src_power)
    P_up = signed_flux_power(fdtd, ctx["ff_up_faces"], src_power)
    P_ff_bottom_in = signed_flux_power(fdtd, ctx["ff_bottom_faces"], src_power)
    T_top = oc.read_transmission(fdtd, TOP_MONITOR)
    P_top = max(float(T_top), 0.0) * (src_power if src_power else 0.0)

    theta = spectrum = ukx = uky = None
    try:
        theta, raw_spec, ukx, uky = oc.n2f_spectrum(sim, TOP_MONITOR, na=N2F_POINTS)
        raw_sum = float(np.sum(np.maximum(raw_spec, 0.0)))
        if np.isfinite(raw_sum) and raw_sum > 0.0 and P_top > 0.0:
            spectrum = np.maximum(raw_spec, 0.0) * P_top / raw_sum
        else:
            spectrum = np.maximum(raw_spec, 0.0)
    except Exception as exc:
        print(f"[run] warning: far-field projection failed for {tag}: {exc}")
    t_ff = time.time() - t0

    clearance = oc.minimum_source_box_clearance(
        [(s["x_centered"], s["y_centered"], src_z) for s in specs],
        ctx["src_box_center"], ctx["src_box_size"],
    )

    P_src = dip_signed if ETA_DENOM == "dipolepower" else P_src_box
    eta_ff = safe_ratio(P_up, P_src)

    r = {
        "tag": tag,
        "n_sources": len(specs),
        "P_src": P_src,
        "P_src_dipolepower": dip_signed,
        "P_src_dipolepower_positive_only": dip_positive,
        "P_src_box": P_src_box,
        "P_up_ff": P_up,
        "eta_ff": eta_ff,
        "eta_ff_dipolepower": (safe_ratio(P_up, dip_signed)
                               if (P_up is not None and dip_signed) else None),
        "eta_ff_box": (safe_ratio(P_up, P_src_box)
                       if (P_up is not None and P_src_box) else None),
        "source_power": src_power,
        "top_monitor_transmission": float(T_top),
        "P_top_plane": P_top,
        "P_ff_box_bottom_influx": (-P_ff_bottom_in
                                   if P_ff_bottom_in is not None else None),
        "src_box_min_clearance_um": clearance,
        "t_fdtd": t_fdtd,
        "t_ff": t_ff,
        "theta": theta,
        "spectrum": spectrum,
        "ukx": ukx,
        "uky": uky,
        "phases_deg": np.asarray(phases_deg, dtype=float),
    }
    if clearance is not None and clearance <= 0.0:
        print(f"[run] warning: {tag}: dipole(s) lie ON a source-box face "
              f"(clearance {clearance:+.4g} um); P_src_box is unreliable "
              f"(Meep tolerates this, Lumerical does not)")
    drop_last_fsp(sim)
    return r


def log_run(r):
    """One line per run, appended to the '...' prefix printed before the run
    (same shape as the Meep scripts' progress log)."""
    def fmt(v, spec="{:.4e}"):
        return spec.format(v) if isinstance(v, float) and np.isfinite(v) else "n/a"

    eta = r["eta_ff"]
    print(f"  {r['t_fdtd']:.0f}s  P_src={fmt(r['P_src'])}  "
          f"P_up={fmt(r['P_up_ff'])}  eta_ff="
          + (f"{eta * 100:.3f}%" if eta is not None else "n/a")
          + f"  [box {fmt(r['P_src_box'])}, "
          f"dip+ {fmt(r['P_src_dipolepower_positive_only'])}]", flush=True)


# =============================================================================
# 8. Figures
# =============================================================================

def save_dipole_layout(path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for s in SOURCE_SPECS:
        ax.scatter(s["x_centered"], s["y_centered"], c="tab:blue", marker="o", s=85)
        ax.text(s["x_centered"] + 0.010, s["y_centered"] + 0.010, s["label"], fontsize=6)
    for s in TRACE_SPECS:
        ax.scatter(s["x_centered"], s["y_centered"], facecolors="none",
                   edgecolors="tab:red", marker="o", s=220, linewidths=1.6)
    ax.add_patch(plt.Rectangle((-channel_w / 2, -channel_l / 2), channel_w, channel_l,
                               fill=False, linewidth=1.5, linestyle="--",
                               edgecolor="black", label="GaN channel"))
    ax.add_patch(plt.Rectangle((-src_box_half, -src_box_half),
                               2 * src_box_half, 2 * src_box_half,
                               fill=False, linewidth=1.2, linestyle=":",
                               edgecolor="tab:green", label="source flux box"))
    ax.set_title(f"{SOURCE_GRID_N}x{SOURCE_GRID_N} endpoint grid: {len(SOURCE_SPECS)} Ex "
                 f"dipoles (blue)\nred rings = {len(TRACE_SPECS)} unique step1 Method-A runs "
                 f"(mult 4)")
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_radiation_figure(path, theta, spectrum, ukx, uky, title, npz_path=None,
                          phases_rad=None):
    """Meep step2b-style figure: Sr(theta,phi) heatmap + azimuth-averaged polar cut."""
    if theta is None or spectrum is None:
        return None
    th_axis, ph_axis, Sr = oc.theta_phi_map(
        theta, ukx, uky, spectrum, n_theta=FF_N_THETA, n_phi=FF_N_PHI)
    oc.save_radiation_map_figure(path, th_axis, ph_axis, Sr, title)
    if npz_path is not None:
        np.savez(npz_path, Sr=Sr, theta=np.deg2rad(th_axis), phi=np.deg2rad(ph_axis),
                 phases_rad=(np.asarray(phases_rad, dtype=float)
                             if phases_rad is not None else np.zeros(0)))
    return Sr


def mean_radiation_figure(path, runs, title, npz_path=None):
    """Average the calibrated spectra of several runs, then render one figure."""
    usable = [r for r in runs if r.get("spectrum") is not None]
    if not usable:
        return None
    spec = np.mean([np.asarray(r["spectrum"]) for r in usable], axis=0)
    r0 = usable[0]
    return save_radiation_figure(path, r0["theta"], spec, r0["ukx"], r0["uky"],
                                 title, npz_path=npz_path)


def save_summary_plot(path, payload):
    ref = payload.get("reference_incoherent")
    c2a = payload.get("case2a_same_phase")
    c2b = payload.get("case2b_random_single")
    c3 = payload.get("case3_random_average")
    det = payload.get("deterministic")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    labels, vals, colors = [], [], []
    if det and det.get("eta_ff") is not None:
        labels.append("step1_trace"); vals.append(det["eta_ff"] * 100.0); colors.append("tab:purple")
    if ref and ref.get("eta_ff") is not None:
        labels.append("incoh_ref"); vals.append(ref["eta_ff"] * 100.0); colors.append("tab:blue")
    if c2a and c2a.get("eta_ff") is not None:
        labels.append("same_phase"); vals.append(c2a["eta_ff"] * 100.0); colors.append("tab:red")
    if c2b and c2b.get("eta_ff") is not None:
        labels.append("rand1"); vals.append(c2b["eta_ff"] * 100.0); colors.append("tab:orange")
    for cp in (c3 or {}).get("checkpoints", []):
        labels.append(f"avg_{cp['n_trials']}tr"); vals.append(cp["eta_ff"] * 100.0)
        colors.append("tab:green")
    if vals:
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("eta_ff [%]")
    ax.set_title(f"Coherence comparison (denominator: {payload['setup']['eta_denominator']})")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    cps = (c3 or {}).get("checkpoints", [])
    if cps and all(cp.get("rel_eta_error_vs_incoherent") is not None for cp in cps):
        ax.plot([cp["n_trials"] for cp in cps],
                [cp["rel_eta_error_vs_incoherent"] * 100.0 for cp in cps], "o-", lw=2)
        ax.axhline(1.0, color="green", ls="--", lw=1, label="1%")
        ax.axhline(5.0, color="orange", ls="--", lw=1, label="5%")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "case3 not run", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("random-phase trial count")
    ax.set_ylabel("relative diff vs incoherent [%]")
    ax.set_title("Case 3 convergence")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    mism, mlab = [], []
    for tag, case in (("same_phase", c2a), (f"rand1_seed{CASE2_RANDOM_SEED}", c2b)):
        if case and case.get("rel_eta_error_vs_incoherent") is not None:
            mism.append(case["rel_eta_error_vs_incoherent"] * 100.0)
            mlab.append(tag)
    if mism:
        ax.bar(np.arange(len(mism)), mism, color=["tab:red", "tab:orange"][:len(mism)])
        ax.set_xticks(np.arange(len(mism)))
        ax.set_xticklabels(mlab, rotation=20, ha="right")
    else:
        ax.text(0.5, 0.5, "case2 vs incoherent unavailable",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("relative diff vs incoherent [%]")
    ax.set_title("Case 2 coherent mismatch")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    per_trial = (c3 or {}).get("per_trial", [])
    if per_trial:
        ids = [r["trial"] for r in per_trial]
        eta = [r["eta_ff"] * 100.0 for r in per_trial]
        run = np.cumsum([r["eta_ff"] for r in per_trial]) / np.arange(1, len(per_trial) + 1)
        ax.scatter(ids, eta, s=25, alpha=0.7, label="single coherent realization")
        ax.plot(ids, run * 100.0, lw=2, label="running average")
        if ref and ref.get("eta_ff") is not None:
            ax.axhline(ref["eta_ff"] * 100.0, color="tab:blue", ls="--", lw=1.5,
                       label="incoherent ref")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "case3 not run", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("trial")
    ax.set_ylabel("eta_ff [%]")
    ax.set_title("Case 3 trial-by-trial behavior")
    ax.grid(alpha=0.3)

    fig.suptitle("Lumerical port of the Meep bare-LED reference "
                 "(step1 Method A + step2b case1/2a/2b/3)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# 9. Cases
# =============================================================================

def _record(r, extra=None):
    """Strip the array payloads so the run dict can go into results.json."""
    out = {k: v for k, v in r.items()
           if k not in ("theta", "spectrum", "ukx", "uky", "phases_deg")}
    if extra:
        out.update(extra)
    return out


def run_trace(ctx):
    """step1 Method A: 9 unique runs, multiplicity 4, power-weighted average."""
    print(f"\n{'=' * 72}")
    print(f"Method A (step1) - deterministic {SOURCE_GRID_N}x{SOURCE_GRID_N} grid, "
          f"{len(TRACE_SPECS)} unique runs")
    print(f"{'=' * 72}")
    runs, results = [], []
    t_total = 0.0
    for i, spec in enumerate(TRACE_SPECS):
        prefix = (f"  [{i}] ({spec['x_centered']:.3f}, {spec['y_centered']:.3f}) ...")
        print(prefix, end="", flush=True)
        r = run_once(ctx, [spec], f"trace_{i:02d}")
        log_run(r)
        r["mult"] = spec["mult"]
        t_total += r["t_fdtd"] + r["t_ff"]
        runs.append(r)
        results.append(_record(r, {
            "label": spec["label"],
            "x_centered": spec["x_centered"],
            "y_centered": spec["y_centered"],
            "mult": spec["mult"],
        }))

    w_total = sum(r["mult"] for r in runs)
    P_src_avg = sum(r["mult"] * (r["P_src"] or 0.0) for r in runs) / w_total
    P_up_avg = sum(r["mult"] * (r["P_up_ff"] or 0.0) for r in runs) / w_total
    P_src_box_avg = sum(r["mult"] * (r["P_src_box"] or 0.0) for r in runs) / w_total
    P_src_dip_avg = sum(r["mult"] * (r["P_src_dipolepower"] or 0.0) for r in runs) / w_total
    eta_ff = safe_ratio(P_up_avg, P_src_avg)
    print(f"\n  Deterministic result: eta_ff = {eta_ff * 100:.3f}%")
    print(f"  Total wall time: {t_total:.0f}s ({len(runs)} runs)")

    png = os.path.join(DESIGN_DIR, "trace_deterministic_radiation.png")
    mean_radiation_figure(png, runs,
                          "step1 Method A - multiplicity-weighted mean pattern",
                          npz_path=os.path.join(DESIGN_DIR, "trace_deterministic_radiation.npz"))
    return {
        "eta_ff": eta_ff,
        "P_src_avg": P_src_avg,
        "P_up_ff_avg": P_up_avg,
        "P_src_box_avg": P_src_box_avg,
        "P_src_dipolepower_avg": P_src_dip_avg,
        "eta_ff_box": safe_ratio(P_up_avg, P_src_box_avg),
        "eta_ff_dipolepower": safe_ratio(P_up_avg, P_src_dip_avg),
        "n_runs": len(runs),
        "t_total": t_total,
        "radiation_png": png,
        "results": results,
    }, runs


def run_case1(ctx):
    """step2b case1: 36 individual runs, incoherent (per-source) average."""
    print(f"\n{'=' * 72}")
    print(f"Case 1 - source-wise incoherent average "
          f"({len(SOURCE_SPECS)} individual runs)")
    print(f"{'=' * 72}")
    runs = []
    for idx, spec in enumerate(SOURCE_SPECS, start=1):
        print(f"  [{idx:02d}/{len(SOURCE_SPECS):02d}] {spec['label']} ...", end="", flush=True)
        r = run_once(ctx, [spec], f"case1_{idx:02d}")
        log_run(r)
        runs.append(r)

    P_src_avg = float(np.mean([r["P_src"] or 0.0 for r in runs]))
    P_up_avg = float(np.mean([r["P_up_ff"] or 0.0 for r in runs]))
    P_src_box_avg = float(np.mean([r["P_src_box"] or 0.0 for r in runs]))
    P_src_dip_avg = float(np.mean([r["P_src_dipolepower"] or 0.0 for r in runs]))
    eta_ff = safe_ratio(P_up_avg, P_src_avg)
    png = os.path.join(DESIGN_DIR, "reference_incoherent_radiation.png")
    mean_radiation_figure(png, runs, "Incoherent reference radiation pattern",
                          npz_path=os.path.join(DESIGN_DIR,
                                                "reference_incoherent_radiation.npz"))
    print(f"\n  Incoherent reference: eta_ff = {eta_ff * 100:.3f}%")
    return {
        "eta_ff": eta_ff,
        "P_src": P_src_avg,
        "P_up_ff": P_up_avg,
        "P_src_box": P_src_box_avg,
        "P_src_dipolepower": P_src_dip_avg,
        "eta_ff_box": safe_ratio(P_up_avg, P_src_box_avg),
        "eta_ff_dipolepower": safe_ratio(P_up_avg, P_src_dip_avg),
        "n_runs": len(runs),
        "radiation_png": png,
        "is_external_reference": False,
        "per_source": [
            _record(r, {"label": s["label"],
                        "x_centered": s["x_centered"],
                        "y_centered": s["y_centered"],
                        "pol_label": s["pol_label"]})
            for s, r in zip(SOURCE_SPECS, runs)
        ],
    }, runs


def summarize_against_reference(r, reference):
    eta = r["eta_ff"]
    ref_eta = (reference or {}).get("eta_ff")
    out = {
        "P_src": r["P_src"],
        "P_src_box": r["P_src_box"],
        "P_src_dipolepower": r["P_src_dipolepower"],
        "P_src_dipolepower_positive_only": r["P_src_dipolepower_positive_only"],
        "P_up_ff": r["P_up_ff"],
        "eta_ff": eta,
        "eta_ff_box": r["eta_ff_box"],
        "eta_ff_dipolepower": r["eta_ff_dipolepower"],
        "t_fdtd": r["t_fdtd"],
        "t_ff": r["t_ff"],
        "top_monitor_transmission": r["top_monitor_transmission"],
        "src_box_min_clearance_um": r["src_box_min_clearance_um"],
    }
    if eta is not None and ref_eta is not None:
        out["delta_eta_vs_incoherent"] = eta - ref_eta
        out["rel_eta_error_vs_incoherent"] = abs(eta - ref_eta) / max(abs(ref_eta), EPS_RATIO)
    else:
        out["delta_eta_vs_incoherent"] = None
        out["rel_eta_error_vs_incoherent"] = None
    return out


def run_case2a(ctx, reference):
    print(f"\n{'=' * 72}")
    print("Case 2a - same-phase coherent simultaneous emission")
    print(f"{'=' * 72}")
    phases = np.zeros(len(SOURCE_SPECS))
    print("  all phases = 0 ...", end="", flush=True)
    r = run_once(ctx, SOURCE_SPECS, "case2a_same_phase", phases_rad=phases)
    log_run(r)
    png = os.path.join(DESIGN_DIR, "case2a_same_phase_radiation.png")
    npz = os.path.join(DESIGN_DIR, "case2a_same_phase_radiation.npz")
    save_radiation_figure(png, r["theta"], r["spectrum"], r["ukx"], r["uky"],
                          "Case 2a - same-phase coherent realization",
                          npz_path=npz, phases_rad=phases)
    out = summarize_against_reference(r, reference)
    out.update({
        "description": "simultaneous coherent emission, all phases are zero",
        "radiation_png": png,
        "radiation_npz": npz,
        "phases_rad": phases.tolist(),
        "phases_deg": r["phases_deg"].tolist(),
    })
    return out


def run_case2b(ctx, reference):
    print(f"\n{'=' * 72}")
    print("Case 2b - one random-phase coherent simultaneous emission")
    print(f"{'=' * 72}")
    phases = make_case2_random_phase(len(SOURCE_SPECS), seed=CASE2_RANDOM_SEED)
    print(f"  random_seed{CASE2_RANDOM_SEED} ...", end="", flush=True)
    r = run_once(ctx, SOURCE_SPECS, "case2b_random_single", phases_rad=phases)
    log_run(r)
    png = os.path.join(DESIGN_DIR, "case2b_random_single_radiation.png")
    npz = os.path.join(DESIGN_DIR, "case2b_random_single_radiation.npz")
    save_radiation_figure(png, r["theta"], r["spectrum"], r["ukx"], r["uky"],
                          f"Case 2b - single random-phase realization "
                          f"(seed {CASE2_RANDOM_SEED})",
                          npz_path=npz, phases_rad=phases)
    out = summarize_against_reference(r, reference)
    out.update({
        "description": "simultaneous coherent emission, one random phase draw",
        "seed": CASE2_RANDOM_SEED,
        "radiation_png": png,
        "radiation_npz": npz,
        "phases_rad": phases.tolist(),
        "phases_deg": r["phases_deg"].tolist(),
    })
    return out


def run_case3(ctx, reference):
    print(f"\n{'=' * 72}")
    print(f"Case 3 - random-phase coherent trials (N up to {MAX_TRIALS})")
    print(f"{'=' * 72}")
    trial_dir = os.path.join(DESIGN_DIR, "case3_trial_fields")
    os.makedirs(trial_dir, exist_ok=True)

    all_phases = make_case3_random_phases(MAX_TRIALS, len(SOURCE_SPECS),
                                          seed=CASE3_RANDOM_SEED)
    np.savetxt(os.path.join(DESIGN_DIR, "case3_phases_deg.txt"),
               phases_to_degrees(all_phases))

    trial_results, checkpoints = [], []
    P_src_cum = P_up_cum = 0.0
    spec_cum = None
    ref_eta = (reference or {}).get("eta_ff")

    for t in range(MAX_TRIALS):
        phases = all_phases[t]
        print(f"  trial {t + 1:02d}/{MAX_TRIALS:02d} ...", end="", flush=True)
        r = run_once(ctx, SOURCE_SPECS, f"case3_trial_{t + 1:03d}", phases_rad=phases)
        log_run(r)

        png = os.path.join(trial_dir, f"trial_{t + 1:03d}_radiation_field.png")
        npz = os.path.join(trial_dir, f"trial_{t + 1:03d}_radiation_field.npz")
        save_radiation_figure(png, r["theta"], r["spectrum"], r["ukx"], r["uky"],
                              f"Case 3 trial {t + 1}", npz_path=npz, phases_rad=phases)

        trial_results.append(_record(r, {
            "trial": t + 1,
            "phases_rad": phases.tolist(),
            "phases_deg": r["phases_deg"].tolist(),
            "radiation_png": png,
            "radiation_npz": npz,
        }))

        P_src_cum += r["P_src"] or 0.0
        P_up_cum += r["P_up_ff"] or 0.0
        if r["spectrum"] is not None:
            spec_cum = (np.asarray(r["spectrum"]).copy() if spec_cum is None
                        else spec_cum + np.asarray(r["spectrum"]))

        n_done = t + 1
        if n_done in CASE3_TRIAL_COUNTS or n_done == MAX_TRIALS:
            P_src_avg = P_src_cum / n_done
            P_up_avg = P_up_cum / n_done
            eta_avg = safe_ratio(P_up_avg, P_src_avg)
            cp_png = os.path.join(DESIGN_DIR, f"case3_checkpoint_{n_done:03d}_radiation.png")
            if spec_cum is not None:
                save_radiation_figure(cp_png, r["theta"], spec_cum / n_done,
                                      r["ukx"], r["uky"],
                                      f"Case 3 ensemble average after {n_done} trials")
            checkpoints.append({
                "n_trials": n_done,
                "P_src": P_src_avg,
                "P_up_ff": P_up_avg,
                "eta_ff": eta_avg,
                "delta_eta_vs_incoherent": (eta_avg - ref_eta) if ref_eta is not None else None,
                "rel_eta_error_vs_incoherent": (abs(eta_avg - ref_eta) / max(abs(ref_eta), EPS_RATIO)
                                                if ref_eta is not None else None),
                "radiation_png": cp_png,
            })
            print(f"    checkpoint {n_done:02d}: eta_ff = {eta_avg * 100:.3f}%")

    return {"checkpoints": checkpoints, "per_trial": trial_results}


# =============================================================================
# 10. results.json
# =============================================================================

def json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return None


def setup_block():
    ff_lat_h = z_top_mon - z_ff_lat_bot
    src_box_h = src_box_z_hi - src_box_z_lo
    return {
        "code": "Lumerical port of step1_trace_comparison.py (Method A) + "
                "step2b_coherence_case2abc_36src.py",
        "engine": "lumerical-fdtd",
        "resolution": resolution,
        "wavelength_um": target_wvl,
        "fcen": fcen,
        "meep_fwidth": fwidth,
        "cell_xy": CELL_XY,
        "sz": sz,
        "sz_raw": sz_raw,
        "air_pad_top": air_pad_top,
        "channel_w": channel_w,
        "channel_l": channel_l,
        "n_GaN": n_GaN,
        "n_Sub": n_Sub,
        "n_Poly": n_Poly,
        "pml_th": pml_th,
        "pml_layers": PML_LAYERS,
        "pixel": PIXEL,
        "mon_pml_gap": mon_pml_gap,
        "edge_inset": EDGE_INSET,
        "cells": [int(round(CELL_XY * resolution)), int(round(CELL_XY * resolution)),
                  int(round(sz * resolution))],
        "z": {
            "z_bot": z_bot, "z_sub_bot": z_sub_bot, "z_sub_top": z_sub_top,
            "z_gan_c": z_gan_c, "z_mqw_c": z_mqw_c, "src_z": src_z,
            "z_gan_top": z_gan_top, "z_top_mon": z_top_mon,
            "z_ff_lat_bot": z_ff_lat_bot,
        },
        "monitors": {
            "ap_half": ap_half,
            "ff_lat_h": ff_lat_h,
            "ff_lat_zc": 0.5 * (z_top_mon + z_ff_lat_bot),
            "top_monitor": TOP_MONITOR,
            "ff_up_faces": [f"{FF_PREFIX}_{s}" for s in ("xp", "xm", "yp", "ym", "zp")],
            "src_box_half": src_box_half,
            "src_box_z_lo": src_box_z_lo,
            "src_box_z_hi": src_box_z_hi,
            "src_box_h": src_box_h,
            "src_box_inset": src_box_inset,
        },
        "source_layout": f"{SOURCE_GRID_N}x{SOURCE_GRID_N} endpoint grid, Ex-only "
                         f"(matches original step2)",
        "num_sources": len(SOURCE_SPECS),
        "source_specs": SOURCE_SPECS,
        "trace_specs": TRACE_SPECS,
        "case2_random_seed": CASE2_RANDOM_SEED,
        "case3_random_seed": CASE3_RANDOM_SEED,
        "case3_trial_counts": list(CASE3_TRIAL_COUNTS),
        "case3_max_trials": MAX_TRIALS,
        "dipole_layout_png": os.path.join(DESIGN_DIR, "dipole_layout.png"),
        "eta_denominator": ETA_DENOM,
        "phase_sign": PHASE_SIGN,
        "n2f_points": N2F_POINTS,
        "ff_map_shape": [FF_N_THETA, FF_N_PHI],
        "simulation_time_fs": SIM_TIME_FS,
        "meep_max_time_fs_equivalent": MEEP_MAX_TIME_FS,
        "auto_shutoff_min": 1e-4,
        "keep_fsp": bool(KEEP_FSP),
    }


def load_payload(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        print(f"[results] warning: could not read {path}: {exc}")
        return None


def save_payload(path, payload):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=json_default)
        fp.write("\n")


# =============================================================================
# 11. Main
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Lumerical port of the Meep bare-LED reference scripts "
                    "(step1 Method A + step2b case1/2a/2b/3).")
    # The lab launcher (msopt.cli.run) forwards no script arguments, so every
    # CLI switch also has an env fallback that works under `run OLED_meep_reference.py`.
    p.add_argument("--case", default=os.environ.get("MSOPT_MEEPREF_CASE", "all"),
                   choices=list(ALL_CASES) + ["all"],
                   help="which case to run (default: all; env MSOPT_MEEPREF_CASE)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the resolved geometry table and exit WITHOUT "
                        "starting a Lumerical session")
    p.add_argument("--outdir", default=None,
                   help="output directory (default: $EIDL_RUN_DIR/A)")
    p.add_argument("--case1-json", default=None,
                   help="results.json holding an existing case1 reference "
                        "(default: <outdir>/results.json). Pass '' to force "
                        "case2/3 to run without a reference.")
    return p.parse_args(argv)


def resolve_reference(payload, args):
    """case2a/2b/3 report deltas against case1; reuse a stored one if present."""
    ref = payload.get("reference_incoherent")
    if ref and ref.get("eta_ff") is not None:
        return ref
    path = args.case1_json
    if path is None:
        return None
    if path == "":
        return None
    external = load_payload(path)
    if not external:
        return None
    ref = external.get("reference_incoherent") or external.get("case1_incoherent")
    if not ref or ref.get("eta_ff") is None:
        return None
    print(f"[reference] reusing case1 from {path}: eta_ff = {ref['eta_ff'] * 100:.3f}%")
    out = dict(ref)
    out["loaded_from_json"] = path
    out["is_external_reference"] = True
    return out


def main(argv=None):
    global DESIGN_DIR
    args = parse_args(argv)
    if args.outdir:
        DESIGN_DIR = os.path.abspath(args.outdir) + os.sep

    lines = geometry_lines()
    for line in lines:
        print(line)

    if args.dry_run or env_flag("SESSION_TEST", "0"):
        print("\n[dry-run] geometry resolved; no Lumerical session was started.")
        return 0

    os.makedirs(DESIGN_DIR, exist_ok=True)
    results_path = os.path.join(DESIGN_DIR, "results.json")
    if args.case1_json is None:
        args.case1_json = results_path

    cases = list(ALL_CASES) if args.case == "all" else [args.case]
    print(f"\n[run] cases = {cases}")
    print(f"[run] output = {DESIGN_DIR}")

    with open(os.path.join(DESIGN_DIR, "geometry_table.txt"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    save_dipole_layout(os.path.join(DESIGN_DIR, "dipole_layout.png"))

    payload = load_payload(results_path) or {}
    payload["version"] = ("lumerical port of meep step1 Method A + "
                          "step2b case1/case2a/case2b/case3")
    payload["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["setup"] = setup_block()

    ctx = build_sim()
    t_start = time.time()
    try:
        if "trace" in cases:
            det, _ = run_trace(ctx)
            payload["deterministic"] = det
            save_payload(results_path, payload)

        if "case1" in cases:
            ref, _ = run_case1(ctx)
            payload["reference_incoherent"] = ref
            save_payload(results_path, payload)

        reference = resolve_reference(payload, args)

        if "case2a" in cases:
            payload["case2a_same_phase"] = run_case2a(ctx, reference)
            save_payload(results_path, payload)

        if "case2b" in cases:
            payload["case2b_random_single"] = run_case2b(ctx, reference)
            save_payload(results_path, payload)

        if "case3" in cases:
            payload["case3_random_average"] = run_case3(ctx, reference)
            save_payload(results_path, payload)
    finally:
        try:
            ctx["sim"].fdtd.close()
        except Exception:
            pass

    c2a = payload.get("case2a_same_phase") or {}
    c2b = payload.get("case2b_random_single") or {}
    c3 = payload.get("case3_random_average") or {}
    payload["comparison"] = {
        "case2a_same_phase_rel_error_vs_incoherent": c2a.get("rel_eta_error_vs_incoherent"),
        "case2b_random_single_rel_error_vs_incoherent": c2b.get("rel_eta_error_vs_incoherent"),
        "case3_final_rel_error_vs_incoherent": (
            c3["checkpoints"][-1]["rel_eta_error_vs_incoherent"]
            if c3.get("checkpoints") else None),
        "trace_vs_incoherent_rel_error": (
            abs(payload["deterministic"]["eta_ff"] - payload["reference_incoherent"]["eta_ff"])
            / max(abs(payload["reference_incoherent"]["eta_ff"]), EPS_RATIO)
            if payload.get("deterministic") and payload.get("reference_incoherent") else None),
    }
    save_payload(results_path, payload)

    summary_png = os.path.join(DESIGN_DIR, "coherence_case23_summary.png")
    try:
        save_summary_plot(summary_png, payload)
    except Exception as exc:
        print(f"[summary] warning: summary plot failed: {exc}")

    print(f"\nSaved -> {DESIGN_DIR}")
    print("  dipole_layout.png")
    print("  geometry_table.txt")
    print("  results.json")
    print(f"  {os.path.basename(summary_png)}")
    print(f"Total wall time: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
