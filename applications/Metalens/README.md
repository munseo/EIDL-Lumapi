# Metalens Example

Reference setup:

- Wavelength: 0.63 um
- Resolution: 80 grids/um
- Aperture/design window: 5 um x 5 um
- Simulation height: 2.0 um
- Focal distance: 1.5 um above the design top
- Design region: 5.0 um x 5.0 um x 0.05 um
- Design refractive-index interpolation: 1.0 to 2.0
- Objective: x-polarized single-wavelength plane-wave focusing intensity,
  computed as mean `|E|^2` over the central `0.1 um x 0.1 um` focal-plane
  window, normalized by incident field intensity.

Run:

```bash
cd /home/eidl/EIDL-Lumapi/applications/Metalens
run Metalens.py -th 30 -GPU 0 --profile
```

Outputs are written under the run directory:

```text
<selected output root>/<timestamp>_Metalens_gpu*_th*/A/
```

Key outputs:

- `target_focus_profile.png`
- `norm_vertical_profile.png`
- `metalens_density_initial_*.png`
- `metalens_density_final_*.png`
- `result_fom.png`
- `benchmark_summary.txt`
