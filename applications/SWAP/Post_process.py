"""
Lumerical full-scale post-process for SWAP.

This script is the Lumerical-side counterpart of:
  applications/SWAP_gate/Optimization.py, seq == 1 (full_scale)

Assumption:
  - SWAP.py already completed the seq==0 optimization stage
  - <run_dir>/A/lastdesign.txt exists

Outputs are written with the same legacy names consumed by OptField.m:
  - Real_flux_<m>_to_<n>.txt
  - Real_purity_<m>_to_<n>.txt
  - Real_noise_<m>_to_<n>.txt
  - Real_freqs_<m>_to_<n>.txt
  - Ex/Ey/Ez_te00_field.h5
  - Ex/Ey/Ez_te10_field.h5
  - Ex/Ey/Ez_<m>_to_<n>_field.h5
  - Ex/Ey/Ez_<m>_to_<n>_top.h5
  - Half_L_1.txt, Half_R_1.txt
"""

import os
import traceback

import autograd.numpy as npa
import h5py
import numpy as np
from mpi4py import MPI

import msopt as ms


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.environ.get("EIDL_RUN_DIR", os.getcwd()))
design_dir = os.path.join(RUN_DIR, 'A')
post_dir = os.path.join(RUN_DIR, 'Post_process')
data_dir = os.path.join(RUN_DIR, 'Data')
os.makedirs(post_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


def _print(msg):
    if rank == 0:
        print(msg)


def _scalar(value):
    return float(np.real(np.asarray(value).squeeze()))


def _ensure_3d(component):
    component = np.squeeze(np.array(component, dtype=np.complex128))
    if component.ndim == 0:
        component = component.reshape(1, 1, 1)
    elif component.ndim == 1:
        component = component[:, np.newaxis, np.newaxis]
    elif component.ndim == 2:
        component = component[:, :, np.newaxis]
    elif component.ndim != 3:
        raise ValueError(f'Expected a 3D field component, got shape {component.shape}.')
    return component


def _extract_field_slices(result_dict, field_key):
    all_fields = np.array(result_dict[field_key], dtype=np.complex128)
    if all_fields.shape[-1] != 3:
        raise ValueError(f'Unexpected {field_key} shape {all_fields.shape}: last axis must be 3.')

    if all_fields.ndim == 3:
        return [[
            _ensure_3d(all_fields[..., 0]),
            _ensure_3d(all_fields[..., 1]),
            _ensure_3d(all_fields[..., 2]),
        ]]

    n_freq = all_fields.shape[-2]
    slices = []
    for freq_idx in range(n_freq):
        slices.append([
            _ensure_3d(all_fields[..., freq_idx, 0]),
            _ensure_3d(all_fields[..., freq_idx, 1]),
            _ensure_3d(all_fields[..., freq_idx, 2]),
        ])
    return slices


def _export_h5_component(file_path, dataset_prefix, component):
    component_2d = np.squeeze(np.array(component, dtype=np.complex128))
    if component_2d.ndim == 0:
        component_2d = component_2d.reshape(1, 1)
    elif component_2d.ndim == 1:
        component_2d = component_2d[:, np.newaxis]
    elif component_2d.ndim != 2:
        raise ValueError(f'Cannot export {file_path}: expected 2D field, got {component_2d.shape}.')

    with h5py.File(file_path, 'w') as h5f:
        h5f.create_dataset(f'{dataset_prefix}_0.r', data=np.asarray(np.real(component_2d), dtype=np.float64))
        h5f.create_dataset(f'{dataset_prefix}_0.i', data=np.asarray(np.imag(component_2d), dtype=np.float64))


def _export_field_triplet(stem, fields):
    names = [('Ex', 'ex', fields[0]), ('Ey', 'ey', fields[1]), ('Ez', 'ez', fields[2])]
    for prefix, dataset_prefix, component in names:
        _export_h5_component(
            os.path.join(data_dir, f'{prefix}_{stem}.h5'),
            dataset_prefix,
            component,
        )
def _export_field_triplet_h(stem, fields):
    names = [('Hx', 'hx', fields[0]), ('Hy', 'hy', fields[1]), ('Hz', 'hz', fields[2])]
    for prefix, dataset_prefix, component in names:
        _export_h5_component(
            os.path.join(data_dir, f'{prefix}_{stem}.h5'),
            dataset_prefix,
            component,
        )

# =============================================================================
# Same core geometry/physics settings as SWAP.py seq==0
# =============================================================================
Wavelengths = 1.55
mode = [1, 3]

# Important:
# mode[0] is the role equivalent of TE00 in the original Meep script.
# mode[1] is the role equivalent of TE10 in the original Meep script.
MODE_LABELS = {0: 'te00', 1: 'te10'}

resolution = 50
design_region_x = round(0.3, 2)
design_region_y = round(6.0, 2)
design_region_z = round(10.0, 2)
design_region_resolution = int(resolution)

SiO2_h = round(0.65, 2)
LNsub_h = round(0.2, 2)
LNwg_h = design_region_x

min_g = round(2 * design_region_x / 2.0, 2)
input_w_top = round(2.0, 2)
input_w_bot = round(input_w_top + min_g, 2)
w_top = round(2.0, 2)
w_bot = round(w_top + min_g, 2)

Lpml = round(10.0 / resolution, 2)
waveguide_length_I = round(0.5, 2)
waveguide_length_O = round(5.0, 2)
pml_2_src = round(2.0 / resolution, 2)
mon_2_pml = round(2.0 / resolution, 2)

Sy = design_region_y + 2.0
Sz = round(waveguide_length_I + design_region_z + waveguide_length_O, 2)
Sx = round(SiO2_h + LNsub_h + LNwg_h + SiO2_h, 2)

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

source_center = [0, 0, Z_min + pml_2_src]
source_size = [Sx, Sy, 0]
Input_monitor_center = [0, 0, Z_min + waveguide_length_I]
Output_monitor_center = [0, 0, Z_max - mon_2_pml]
dft_monitor_size = [Sx, Sy, 0]
Top_cen = [X_min + SiO2_h + LNsub_h + LNwg_h / 2, 0, 0]
Top_size = [0, Sy, Sz]

half_left_center = [0, -w_top * 0.5, Z_max - mon_2_pml]
half_right_center = [0, w_top * 0.5, Z_max - mon_2_pml]
half_size = [LNsub_h + LNwg_h + SiO2_h, w_top, 0]

LN_eps = [4.8855, 4.5836, 4.8855]
LN_n = [np.sqrt(v) for v in LN_eps]
SiO2_n = [1.44, 1.44, 1.44]


# =============================================================================
# Same broadband definition as Optimization.py full_scale branch
# =============================================================================
fcen = 1.0 / Wavelengths
bandwidth = 0.2 * Wavelengths
fmax = 1.0 / (Wavelengths - 0.5 * bandwidth)
fmin = 1.0 / (Wavelengths + 0.5 * bandwidth)
fwidth = fmax - fmin
df = 0.5
nf = 99

freqs = np.linspace(fcen - 0.5 * fwidth * df, fcen + 0.5 * fwidth * df, nf + 1)
broadband_wavelengths = (1.0 / freqs).tolist()


def _load_design_density():
    design_file = os.path.join(design_dir, 'lastdesign.txt')
    if not os.path.exists(design_file):
        raise RuntimeError(f'Missing optimized design file: {design_file}')

    density = np.loadtxt(design_file)
    if not os.path.exists(os.path.join(design_dir, 'output.gds')):
        os.chdir(design_dir)
        ms.Lumerical_module.GDS_converter(density, Nx, Ny, Nz, True)
        os.chdir('..')
    expected_size = Nx * Ny * Nz
    if density.size != expected_size:
        raise RuntimeError(
            f'Unexpected lastdesign.txt size {density.size}. Expected {expected_size} (= {Nx}x{Ny}x{Nz}).'
        )
    return npa.reshape(density, (Nx, Ny, Nz))


def _add_common_substrate(simulator, sim_length):
    simulator.add_geo(
        center=[X_min + 0.5 * SiO2_h, 0, 0],
        size=[SiO2_h, Sy, sim_length],
        index=[1.44],
        name='sio2_lower',
    )
    simulator.add_geo(
        center=[X_min + SiO2_h + 0.5 * LNsub_h, 0, 0],
        size=[LNsub_h, Sy, sim_length],
        index=LN_n,
        name='ln_sub',
    )


def _build_waveguide_input_sim(src_wavelengths, source_mode_idx, monitor_center, sim_length=Sz):
    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[Sx, Sy, sim_length],
        resolution=resolution,
        unit=1e-6,
        background_index=1.44,
        center_wl=Wavelengths,
        N_f=len(src_wavelengths),
    )

    _add_common_substrate(sim, sim_length)
    sim.add_waveguide(
        center=[X_min + SiO2_h + LNsub_h + 0.5 * LNwg_h, 0, 0],
        length=sim_length,
        height=LNwg_h,
        top_width=input_w_top,
        bottom_width=input_w_bot,
        index=LN_n,
        name='input_wg',
        prop_axis='z',
    )
    sim.add_source(
        mode='eigen',
        name='source',
        center=source_center if sim_length == Sz else [0, 0, -0.5 * sim_length + pml_2_src],
        size=source_size,
        direction='forward',
        src_wl=list(src_wavelengths),
        bandwidth=0.0,
        mode_num=int(mode[source_mode_idx]),
    )
    sim.add_monitor(
        name='field_monitor',
        center=monitor_center,
        size=dft_monitor_size,
        N_f=len(src_wavelengths),
    )
    sim.add_monitor(
        name='field_monitor_H',
        center=monitor_center,
        size=dft_monitor_size,
        N_f=len(src_wavelengths),
    )
    return sim


def _build_optimized_device_sim(src_wavelengths, source_mode_idx, include_top=False, include_half=False):
    density = _load_design_density()

    sim = ms.Lumerical_utill.LumericalFDTDSimulator(
        sim_size=[Sx, Sy, Sz],
        resolution=resolution,
        unit=1e-6,
        background_index=1.44,
        center_wl=Wavelengths,
        N_f=len(src_wavelengths),
    )

    _add_common_substrate(sim, Sz)
    sim.add_waveguide(
        center=[X_min + SiO2_h + LNsub_h + 0.5 * LNwg_h, 0, Z_min + 0.5 * waveguide_length_I],
        length=waveguide_length_I,
        height=LNwg_h,
        top_width=input_w_top,
        bottom_width=input_w_bot,
        index=LN_n,
        name='input_wg',
        prop_axis='z',
    )
    sim.add_waveguide(
        center=[X_min + SiO2_h + LNsub_h + 0.5 * LNwg_h, 0, Z_max - 0.5 * waveguide_length_O],
        length=waveguide_length_O,
        height=LNwg_h,
        top_width=w_top,
        bottom_width=w_bot,
        index=LN_n,
        name='output_wg',
        prop_axis='z',
    )
    sim.add_design_grid(
        name='design',
        center=[X_min + SiO2_h + LNsub_h + 0.5 * LNwg_h, 0, Z_min + waveguide_length_I + 0.5 * design_region_z],
        size=[design_region_x, design_region_y, design_region_z],
        index1=LN_n,
        index2=SiO2_n,
        design_grids=design_grids,
        density=density,
    )
    sim.add_source(
        mode='eigen',
        name='source',
        center=source_center,
        size=source_size,
        direction='forward',
        src_wl=list(src_wavelengths),
        bandwidth=0.0,
        mode_num=int(mode[source_mode_idx]),
    )
    sim.add_monitor(
        name='output_monitor',
        center=Output_monitor_center,
        size=dft_monitor_size,
        N_f=len(src_wavelengths),
    )
    sim.add_monitor(
        name='output_monitor_H',
        center=Output_monitor_center,
        size=dft_monitor_size,
        N_f=len(src_wavelengths),
    )

    if include_top:
        sim.add_monitor(
            name='top_monitor',
            center=Top_cen,
            size=Top_size,
            N_f=len(src_wavelengths),
        )

    if include_half:
        sim.add_monitor(
            name='half_left_monitor',
            center=half_left_center,
            size=half_size,
            N_f=len(src_wavelengths),
        )
        sim.add_monitor(
            name='half_left_monitor_H',
            center=half_left_center,
            size=half_size,
            N_f=len(src_wavelengths),
        )
        sim.add_monitor(
            name='half_right_monitor',
            center=half_right_center,
            size=half_size,
            N_f=len(src_wavelengths),
        )
        sim.add_monitor(
            name='half_right_monitor_H',
            center=half_right_center,
            size=half_size,
            N_f=len(src_wavelengths),
        )

    return sim


def _run_waveguide_reference(role_idx, broadband=True):
    wavelengths = broadband_wavelengths if broadband else [Wavelengths]
    sim = _build_waveguide_input_sim(
        src_wavelengths=wavelengths,
        source_mode_idx=role_idx,
        monitor_center=Input_monitor_center,
        sim_length=Sz,
    )
    run_name = f'ref_{MODE_LABELS[role_idx]}_{"bb" if broadband else "cen"}'
    _print(f"[Post] Running reference simulation: {run_name}")

    try:
        sim.run(name=run_name, save=True)
        e_fields = _extract_field_slices(sim.fdtd.getresult('field_monitor', 'E'), 'E')
        h_fields = _extract_field_slices(sim.fdtd.getresult('field_monitor_H', 'H'), 'H')
        powers = [_scalar(ms.Opt_MS2.Cross_product(e_field, h_field)) for e_field, h_field in zip(e_fields, h_fields)]
        return e_fields, h_fields, powers
    finally:
        try:
            sim.fdtd.switchtolayout()
        except Exception:
            pass


def _run_optimized_output(source_idx, broadband=True):
    wavelengths = broadband_wavelengths if broadband else [Wavelengths]
    include_top = not broadband
    include_half = broadband and source_idx == 0
    sim = _build_optimized_device_sim(
        src_wavelengths=wavelengths,
        source_mode_idx=source_idx,
        include_top=include_top,
        include_half=include_half,
    )
    run_name = f'opt_{source_idx}_{"bb" if broadband else "cen"}'
    _print(f"[Post] Running optimized-device simulation: {run_name}")

    try:
        sim.run(name=run_name, save=True)

        output_e = _extract_field_slices(sim.fdtd.getresult('output_monitor', 'E'), 'E')
        output_h = _extract_field_slices(sim.fdtd.getresult('output_monitor_H', 'H'), 'H')

        result = {
            'output_e': output_e,
            'output_h': output_h,
        }

        if include_top:
            result['top_e'] = _extract_field_slices(sim.fdtd.getresult('top_monitor', 'E'), 'E')[0]
            result['top_h'] = _extract_field_slices(sim.fdtd.getresult('top_monitor', 'H'), 'H')[0]

        if include_half:
            result['half_left_e'] = _extract_field_slices(sim.fdtd.getresult('half_left_monitor', 'E'), 'E')
            result['half_left_h'] = _extract_field_slices(sim.fdtd.getresult('half_left_monitor_H', 'H'), 'H')
            result['half_right_e'] = _extract_field_slices(sim.fdtd.getresult('half_right_monitor', 'E'), 'E')
            result['half_right_h'] = _extract_field_slices(sim.fdtd.getresult('half_right_monitor_H', 'H'), 'H')

        return result
    finally:
        try:
            sim.fdtd.switchtolayout()
        except Exception:
            pass


def _save_summary(mode_roles, input_powers, output_metrics):
    summary_path = os.path.join(post_dir, 'fullscale_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write('Lumerical full-scale post-process summary\n')
        handle.write(f'Wavelength center: {Wavelengths} um\n')
        handle.write(f'Frequency samples: {len(freqs)}\n\n')

        for role_idx in mode_roles:
            handle.write(f'{MODE_LABELS[role_idx]} reference mode number: {mode[role_idx]}\n')
            handle.write(f'{MODE_LABELS[role_idx]} input power center: {input_powers[role_idx][0]:.8f}\n')
        handle.write('\n')

        for source_idx, metrics in output_metrics.items():
            target_idx = abs(source_idx - 1)
            handle.write(f'{source_idx}_to_{target_idx}\n')
            handle.write(f'  source role: {MODE_LABELS[source_idx]}\n')
            handle.write(f'  target role: {MODE_LABELS[target_idx]}\n')
            handle.write(f'  flux max: {np.max(metrics["flux"]):.8f}\n')
            handle.write(f'  purity max: {np.max(metrics["purity"]):.8f}\n')
            handle.write(f'  noise min: {np.min(metrics["noise"]):.8f}\n\n')


if __name__ == '__main__':
    _print('=' * 80)
    _print('SWAP Lumerical Full-Scale Post-Processing')
    _print('=' * 80)

    try:
        if not os.path.exists(os.path.join(design_dir, 'lastdesign.txt')):
            raise RuntimeError(
                f'Missing {os.path.join(design_dir, "lastdesign.txt")}. '
                'Run SWAP.py first so seq==0 optimization writes the design.'
            )

        # Straight-waveguide references, equivalent to the full_scale branch.
        ref_broadband_e = {}
        ref_broadband_h = {}
        ref_powers = {}

        for role_idx in (0, 1):
            e_fields, h_fields, powers = _run_waveguide_reference(role_idx, broadband=True)
            ref_broadband_e[role_idx] = e_fields
            ref_broadband_h[role_idx] = h_fields
            ref_powers[role_idx] = np.asarray(powers, dtype=float)

            center_e, _, _ = _run_waveguide_reference(role_idx, broadband=False)
            _export_field_triplet(f'{MODE_LABELS[role_idx]}_field', center_e[0])

        output_metrics = {}
        for source_idx in (0, 1):
            target_idx = abs(source_idx - 1)

            bb_result = _run_optimized_output(source_idx, broadband=True)
            center_result = _run_optimized_output(source_idx, broadband=False)

            flux = np.zeros(nf + 1, dtype=float)
            purity = np.zeros(nf + 1, dtype=float)
            noise = np.zeros(nf + 1, dtype=float)

            for freq_idx in range(nf + 1):
                output_e = bb_result['output_e'][freq_idx]
                output_h = bb_result['output_h'][freq_idx]

                flux[freq_idx] = _scalar(ms.Opt_MS2.Cross_product(output_e, output_h)) / ref_powers[source_idx][freq_idx]
                purity[freq_idx] = _scalar(
                    ms.Opt_MS2.Overlap_intg(ref_broadband_e[target_idx][freq_idx], output_e, normalization=True)
                )
                noise[freq_idx] = _scalar(
                    ms.Opt_MS2.Overlap_intg(ref_broadband_e[source_idx][freq_idx], output_e, normalization=True)
                )

            output_metrics[source_idx] = {
                'flux': flux,
                'purity': purity,
                'noise': noise,
            }

            np.savetxt(os.path.join(data_dir, f'Real_flux_{source_idx}_to_{target_idx}.txt'), flux)
            np.savetxt(os.path.join(data_dir, f'Real_purity_{source_idx}_to_{target_idx}.txt'), purity)
            np.savetxt(os.path.join(data_dir, f'Real_noise_{source_idx}_to_{target_idx}.txt'), noise)
            np.savetxt(os.path.join(data_dir, f'Real_freqs_{source_idx}_to_{target_idx}.txt'), freqs)

            _export_field_triplet(f'{source_idx}_to_{target_idx}_field', center_result['output_e'][0])
            _export_field_triplet(f'{source_idx}_to_{target_idx}_top', center_result['top_e'])
            _export_field_triplet_h(f'{source_idx}_to_{target_idx}_top', center_result['top_h'])

            if source_idx == 0:
                half_left = np.zeros(nf + 1, dtype=float)
                half_right = np.zeros(nf + 1, dtype=float)
                for freq_idx in range(nf + 1):
                    half_left[freq_idx] = (
                        _scalar(
                            ms.Opt_MS2.Cross_product(
                                bb_result['half_left_e'][freq_idx],
                                bb_result['half_left_h'][freq_idx],
                            )
                        ) / ref_powers[source_idx][freq_idx]
                    )
                    half_right[freq_idx] = (
                        _scalar(
                            ms.Opt_MS2.Cross_product(
                                bb_result['half_right_e'][freq_idx],
                                bb_result['half_right_h'][freq_idx],
                            )
                        ) / ref_powers[source_idx][freq_idx]
                    )

                np.savetxt(os.path.join(data_dir, 'Half_L_1.txt'), half_left)
                np.savetxt(os.path.join(data_dir, 'Half_R_1.txt'), half_right)

        _save_summary((0, 1), ref_powers, output_metrics)

        _print('[Done] Full-scale outputs generated.')
        _print(f'  Legacy outputs : {data_dir}')
        _print(f'  Summary        : {post_dir}')

    except Exception as exc:
        print(f'\n[Error] {exc}', file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
