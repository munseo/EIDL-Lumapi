# OLED outcoupling

Everything shared lives in **`oled_common.py`**: geometry/config, the layer
stacks, Lumerical helpers, the k-space FoM basis, and the incoherent-dipole
postprocess. The optimizers are thin scripts on top of it.

## Files

| file | role |
|---|---|
| `oled_common.py` | shared config, stacks, sim helpers, FoM basis, postprocess |
| `OLED_opt.py` | **the optimizer to use.** Reciprocal ramp-target engine, defaults to the optimized microcavity stack |
| `OLED_Min.py` | same reciprocal engine, EML-layer **uniformity** emphasis (IPR factor on) |
| `OLED_lens.py` | representative single dipoles, 2-D k-ring ramp FoM |
| `OLED_layered_dipole.py` | **analytic** dipole-in-multilayer solver (transfer matrix + k-parallel integral). Exact LEE + channel split (outcoupled / waveguide / SPP / absorption). 22 self-validations, runs in seconds |
| `OLED_stack_design.py` | searches stack thicknesses with that solver (this produced the optimized stack) |
| `OLED_meep_reference.py` | Lumerical port of the Meep bare-LED reference, for cross-checking the method |
| `OLED_pp_retarget_plots.py` | re-plot a finished postprocess against the design's own optimization target |
| `step1_trace_comparison.py`, `step2b_coherence_case2abc_36src.py` | the original Meep reference scripts (specification for the port) |
| `legacy/` | superseded scripts, kept for reference only |
| `results_analytic/` | analytic solver + stack-design outputs |

## Run parameters

Each optimizer has an editable block at the top (`RESOLUTION`, `DESIGN_H_UM`,
`DESIGN_X_UM/Y_UM`, and in `OLED_opt.py` also `STACK`, `MC_COLOR`,
`MC_STACK_KIND`). Values there are exported as environment DEFAULTS, so an
explicitly set `MSOPT_OLED_*` still wins and sweeps need no file edits.

## Wavelength and design material

**Every run is monochromatic.** The postprocess raises on more than one
wavelength (transmission, dipole power and angular spectra would each have to be
accumulated per wavelength), so `MSOPT_OLED_WAVELENGTHS` carries exactly one
value. This is a real limitation of the current LEE/angular numbers: they are
the response at one line, not a spectrally averaged device efficiency.

`oc.select_stack(STACK, MC_COLOR, MC_STACK_KIND)` is the single place a run
picks its stack, and it **pins that wavelength to the stack's own design line**
(red 0.62 / green 0.53 / blue 0.46 um). All three optimizers use it. Running a
microcavity at the 0.55 um default would evaluate it off its own resonance.

The design region is two constant indices, `DESIGN_N` (rho=1) and `DESIGN_LOW_N`
(rho=0), set at the top of each optimizer. They are dispersionless by
construction, which costs nothing while the run is monochromatic. `DESIGN_N`
must match what the layer is physically made of: 1.45 (SiO2) on the legacy
stack, 2.2 on the microcavity where the design sits on the n=2.2 CPL -- 1.45
against air there is a weak-contrast pattern.

## Postprocess figures

`PP_summary.png` is the radiance, twice. **Left** = the polar lobe (signed, vs
the design's own target). **Right** = the identical curve unrolled onto a linear
theta axis and referenced to normal incidence, so "how bright is 45 deg
compared to 0 deg" is read off directly instead of estimated from the lobe. It
carries the numbers at 30/45/60 deg and the LEE read-out. Lambertian is a flat
100 % line there, since Lambertian means constant radiance.

Radiance is per unit solid angle, so neither panel is an efficiency -- for that
use the cumulative curve.

`OLED_postprocess_emission.png` keeps the 3-panel view (radiance / per-order
share / cumulative extraction) and `OLED_postprocess_cumulative_extraction.png`
the standalone <=10..80 deg curve.

## Two things worth knowing before trusting a number

**LEE from the FDTD postprocess is a lower bound.** `near2far` (`farfield3d`)
projects the field *on the monitor*; light that exits through the lateral PML
before reaching the monitor plane is absent from its input and cannot be
recovered. Measured on one stack: 47.1 % at 2.72 um lateral vs 58.3 % at
5.65 um. The postprocess now sizes the domain from
`MSOPT_OLED_PP_CAPTURE_ANGLE_DEG` (reference plane = top of the stack) and
prints the ACHIEVED capture angle plus a warning when it falls short. For a
*planar* stack the exact answer is cheap — use `OLED_layered_dipole.py`, which
has no truncation at all.

**Thin stacks are meshed in z only.** msopt installs one uniform global mesh, so
a 10 nm layer would force ~2 nm everywhere. `add_stack` instead adds a
z-only mesh override over the under-resolved layers, derived from that stack's
own thinnest layer: x/y keep the optimization step. On the microcavity stack
this is 2.3 M cells instead of 395 M.
