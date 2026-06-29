# Dipole-based LDOS OLED Optimization

## Overview
This script implements dipole-based Local Density of States (LDOS) optimization for OLED grating couplers using Lumerical FDTD simulation with adjoint-based inverse design.

## Key Features

### 1. Dipole Configuration
- **3 x-polarized dipole sources** at radial positions:
  - Position 0: Center (0, 0, z_EML)
  - Position 1: 0.4R radial offset
  - Position 2: 0.8R radial offset
- Each dipole independently optimizes coupling to the grating structure

### 2. Efficiency Curve Support
- **Format**: `"theta1:eff1,theta2:eff2,..."` (degrees : efficiency ratio)
- **Example**: `"0:1.0,45:0.85"` means:
  - 0° emission: target efficiency = 1.0
  - 45° emission: target efficiency = 0.85
  - Linear interpolation between points
- Parsed via: `parse_efficiency_curve(curve_str)`

### 3. Target Field Visualization
- **Diffraction Orders**: Automatically computes grating equation sin(θ_m) = m·λ/period
- **Target Spectrum**: Linear combination of diffraction orders weighted by √efficiency
- **Output**: `LDOS_target_field_info.png` + `LDOS_target_field_summary.txt`
  - Shows efficiency curve overlaid with valid diffraction orders
  - Bar chart of target spectrum amplitudes

### 4. LDOS-based Figure of Merit (FoM)
- **Field Matching Score**: Measures field confinement to EML region
- **Efficiency Score**: Compares radiated power against target baseline efficiency
- **Combined FoM**: Weighted average across 3 dipole positions

## Usage

### Basic Run
```bash
cd /home/eidl/EIDL-Lumapi/applications/OLED
python "OLED_Min copy.py"
```

### With Custom Efficiency Curve
```bash
export MSOPT_OLED_TARGET_EFFICIENCY_CURVE="0:1.0,45:0.85"
python "OLED_Min copy.py"
```

### Environment Variables

#### Efficiency Configuration
- `MSOPT_OLED_TARGET_EFFICIENCY_CURVE` (default: `"0:1.0"`)
  - Specifies radiation efficiency targets at different angles
  - Format: `"angle1_deg:eff1,angle2_deg:eff2,..."`

#### FoM Weights
- `MSOPT_OLED_LDOS_FIELD_MATCH_WEIGHT` (default: `1.0`)
  - Weight for field confinement component
- `MSOPT_OLED_LDOS_EFFICIENCY_WEIGHT` (default: `1.0`)
  - Weight for radiated power component

#### Geometry Parameters
- `MSOPT_OLED_WINDOW_X`, `MSOPT_OLED_WINDOW_Y` (default: `2.5` μm each)
  - Grating period in x and y directions
- `MSOPT_OLED_ACTIVE_X`, `MSOPT_OLED_ACTIVE_Y` (default: `2.0` μm each)
  - Active design region size

#### Simulation Control
- `MSOPT_OLED_RESOLUTION` (default: `0.05` μm)
  - FDTD mesh grid spacing
- `MSOPT_OLED_POSTPROCESS` (default: `"1"`)
  - Enable post-processing (limited for LDOS approach)

### Example Workflows

#### 1. Baseline Optimization (0° target)
```bash
export MSOPT_OLED_TARGET_EFFICIENCY_CURVE="0:1.0"
export MSOPT_OLED_LDOS_FIELD_MATCH_WEIGHT="1.0"
export MSOPT_OLED_LDOS_EFFICIENCY_WEIGHT="1.0"
python "OLED_Min copy.py"
```

#### 2. Multi-angle Optimization (0° to 45°)
```bash
export MSOPT_OLED_TARGET_EFFICIENCY_CURVE="0:1.0,30:0.9,45:0.75"
python "OLED_Min copy.py"
```

#### 3. Higher Resolution, Finer Grid
```bash
export MSOPT_OLED_RESOLUTION="0.025"
export MSOPT_OLED_WINDOW_X="2.0"
export MSOPT_OLED_WINDOW_Y="2.0"
python "OLED_Min copy.py"
```

## Output Files

### Immediately After Optimization Start
- `LDOS_target_field_info.png` - Efficiency curve and target spectrum visualization
- `LDOS_target_field_summary.txt` - Detailed target field data (diffraction orders, angles, efficiencies)

### After Optimization Completes
- `LDOS_optimized_fom_curve.png` - FoM progression during optimization
- `lastdesign.txt` - Final optimized design parameters
- Design visualization images (x-y, x-z, y-z sections)

## Key Functions

### Efficiency Curve Functions
```python
parse_efficiency_curve("0:1.0,45:0.85")
→ [(0.0, 1.0), (45.0, 0.85)]

interpolate_efficiency_at_angle(30.0, curve)
→ 0.925  # Linear interpolation at 30°
```

### Diffraction Order Analysis
```python
diffraction_orders = compute_diffraction_orders(0.55, 2.5, max_order=5)
# Returns: list of (order_m, theta_deg) tuples with |sin(theta)| ≤ 1
```

### Target Field Computation
```python
target_field_info = compute_target_field_with_diffraction_orders(
    wavelength_um=0.55,
    period_um=2.5,
    efficiency_curve=[(0.0, 1.0), (45.0, 0.85)]
)
# Returns dict with:
# - diffraction_orders
# - angle_efficiencies
# - target_spectrum (normalized amplitudes)
```

### Visualization
```python
visualize_target_field_info(target_field_info, design_directory)
# Generates: LDOS_target_field_info.png, LDOS_target_field_summary.txt
```

## Physical Interpretation

### Efficiency Curve Example: "0:1.0,45:0.85"
This means:
- **0° (normal emission)**: Optimize for full efficiency
- **45° (oblique emission)**: Target reduced efficiency (85% of baseline)
- **Linear interpolation**: Smooth transition between points

The optimization then:
1. Computes all valid diffraction orders at the grating period
2. Weights each order's amplitude by its interpolated efficiency
3. Maximizes coupling to this weighted target spectrum
4. Does this for all 3 dipole positions simultaneously

### LDOS vs Reciprocal Approach
- **LDOS approach** (current): Couples dipole source → grating → field analysis
  - More direct modeling of OLED emission
  - Independent sources at multiple spatial positions
- **Reciprocal approach** (old): Used plane wave targets → time-reversal symmetry
  - Works but less accurate for localized source effects

## Troubleshooting

### "Cannot import" Errors at Runtime
→ Install required packages:
```bash
conda install numpy autograd matplotlib
# Or use the workspace's miniconda3 environment
```

### No LDOS_target_field_info.png After Starting
→ Check that efficiency curve string is valid:
```bash
export MSOPT_OLED_TARGET_EFFICIENCY_CURVE="0:1.0,45:0.85"
```

### FoM Not Improving
→ Adjust efficiency curve to be more realistic:
- Try: `"0:1.0"` (simple case first)
- Check diffraction orders in LDOS_target_field_summary.txt
- Verify minimum/maximum angles are within physical limits

## Notes

- The 3 dipole positions are fixed by design (0, 0.4R, 0.8R)
- Efficiency curve is applied uniformly to all dipoles
- Gradients averaged across dipoles via Jacobian
- Each dipole runs independent FDTD simulation
- Total simulation time scales with: 3 × (wavelength points) × (design iterations)

