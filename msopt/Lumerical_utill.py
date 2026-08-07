import numpy as np
from tqdm import tqdm
import time
import shutil
import subprocess
import os
import sys
import configparser
import xml.etree.ElementTree as ET
from pathlib import Path

from . import Opt_MS2


def _discover_lumapi_path():
    explicit = os.environ.get("LUMERICAL_PYTHONPATH")
    if explicit:
        p = Path(explicit).expanduser()
        if (p / "lumapi.py").exists():
            return str(p)

    root = os.environ.get("LUMERICAL_ROOT")
    if root:
        p = Path(root).expanduser() / "api" / "python"
        if (p / "lumapi.py").exists():
            return str(p)

    candidates = sorted(Path("/opt/lumerical").glob("v*/api/python/lumapi.py")) if Path("/opt/lumerical").exists() else []
    if candidates:
        return str(candidates[-1].parent)
    return None


try:
    import lumapi
except ModuleNotFoundError:
    lumapi_dir = _discover_lumapi_path()
    if lumapi_dir and lumapi_dir not in sys.path:
        sys.path.insert(0, lumapi_dir)
    import lumapi


def _is_lumerical_messaging_error(exc):
    text = str(exc)
    markers = (
        "Failed to start messaging",
        "Failed to set up Ansys license sharing",
        "ANSYSLI exited or could not read server port",
        "Could not bind socket on port",
    )
    return any(marker in text for marker in markers)


def _open_lumerical_fdtd():
    session_hide = os.environ.get("LUMERICAL_SESSION_HIDE", "true").lower() == "true"
    session_platform = os.environ.get("LUMERICAL_SESSION_PLATFORM", "offscreen").strip()
    retries = int(os.environ.get("LUMERICAL_SESSION_OPEN_RETRIES", "6"))
    retry_delay = float(os.environ.get("LUMERICAL_SESSION_OPEN_RETRY_DELAY", "12"))

    attempts = []
    if session_platform:
        attempts.append({"hide": session_hide, "serverArgs": {"platform": session_platform}})
    attempts.append({"hide": session_hide})
    attempts.append({"hide": True})
    attempts.append({})

    last_error = None
    retries = max(1, retries)
    for retry_idx in range(retries):
        retryable_messaging_error = False
        for kwargs in attempts:
            try:
                fdtd = lumapi.FDTD(**kwargs)
                return fdtd
            except Exception as exc:
                last_error = exc
                if _is_lumerical_messaging_error(exc):
                    retryable_messaging_error = True
                    break
                continue

        if retryable_messaging_error and retry_idx < retries - 1:
            print(
                "Failed to open Lumerical FDTD session; "
                f"retrying in {retry_delay:g}s ({retry_idx + 1}/{retries - 1}). "
                f"Last error: {last_error}"
            )
            time.sleep(retry_delay)
            continue
        break

    raise RuntimeError(
        "Failed to open a Lumerical FDTD session. "
        f"Tried platform={session_platform!r} with fallback combinations. "
        f"Set LUMERICAL_ROOT/LUMERICAL_PYTHONPATH or adjust LUMERICAL_SESSION_PLATFORM. "
        f"Last error: {last_error}"
    )


def interpolate_index(index, wavelength):
    """
    Resolve scalar/anisotropic/dispersive index data at wavelength in um.

    Accepted forms:
        [n], [nx, ny, nz]
        complex n or complex anisotropic list
        callable: f(wavelength_um) -> n
        dict with wavelength/n/optional k arrays
    """
    if callable(index):
        return interpolate_index(index(wavelength), wavelength)

    if isinstance(index, dict):
        wl_data = index.get("wavelength", index.get("wavelengths", index.get("wl")))
        if wl_data is None:
            raise ValueError("Dispersive index dict requires wavelength/wavelengths/wl.")
        wl = np.asarray(wl_data, dtype=float)

        n_data = np.asarray(index["n"], dtype=float)
        k_data = np.asarray(index.get("k", np.zeros_like(n_data)), dtype=float)

        order = np.argsort(wl)
        wl = wl[order]
        n_data = n_data[order]
        k_data = k_data[order]

        if n_data.ndim == 1:
            n_val = np.interp(wavelength, wl, n_data)
            k_val = np.interp(wavelength, wl, k_data)
            return np.array([n_val + 1j * k_val], dtype=np.complex128)

        vals = []
        for comp in range(n_data.shape[1]):
            n_val = np.interp(wavelength, wl, n_data[:, comp])
            k_val = np.interp(wavelength, wl, k_data[:, comp])
            vals.append(n_val + 1j * k_val)
        return np.asarray(vals, dtype=np.complex128)

    arr = np.asarray(index, dtype=np.complex128).reshape(-1)
    if arr.size not in (1, 3):
        raise ValueError("Index must be scalar, length-1, length-3, callable, or dispersion dict.")
    return arr


def _dispersion_table(index):
    if not isinstance(index, dict):
        return None
    wl_data = index.get("wavelength", index.get("wavelengths", index.get("wl")))
    if wl_data is None or "n" not in index:
        return None

    wl = np.asarray(wl_data, dtype=float).reshape(-1)
    n_data = np.asarray(index["n"], dtype=float)
    k_data = np.asarray(index.get("k", np.zeros_like(n_data)), dtype=float)
    if n_data.ndim == 1:
        n_data = n_data[:, None]
    if k_data.ndim == 1:
        k_data = k_data[:, None]
    if n_data.shape != k_data.shape or n_data.shape[0] != wl.size:
        raise ValueError(
            "Dispersive material table shape mismatch. Expected wavelength length "
            "to match first dimension of n and k."
        )

    if n_data.shape[1] not in (1, 3):
        raise ValueError("Dispersive material n/k table must be isotropic or 3-axis anisotropic.")

    order = np.argsort(wl)
    return wl[order], n_data[order], k_data[order]


class LumericalFDTDSimulator:
    def __init__(
        self, 
        sim_size=[0,0,0],
        resolution: int = None,                 # [cells per unit], e.g., per 'unit'
        points_per_wavelength: int = None,      # ppw at center_wl in background_index
        bc_x: str = "PML",
        bc_y: str = "PML",
        bc_z: str = "PML",
        unit: float = 1e-6,                     # 1.0 => meters; 1e-6 => inputs in um
        background_index: float = 1.0,
        center_wl: float = 1.0,
        N_f: int = 1,
    ):
        self.dims = ['x', 'y', 'z']
        self.kw_to_idx = {kw: i for i, kw in enumerate(self.dims)}
        self.idx_to_kw = {i: kw for i, kw in enumerate(self.dims)}
        self.fdtd = _open_lumerical_fdtd()
        self.fdtd.switchtolayout()

        self.c=299792458
        # keep commonly used attrs
        self.unit = unit
        self.N_f = N_f
        self.center_wl = center_wl*unit
        self.center_wl_um = center_wl
        self.n_bg = background_index
        self.resolution = resolution
        self.ppw = points_per_wavelength
        self.bc = {'x': bc_x, 'y': bc_y, 'z': bc_z}
        self.material_cache = {}
        self.src_wl = np.asarray([center_wl], dtype=float).reshape(-1) * self.unit
        self.src_bw = 0.0

        # FDTD region
        self.fdtd.addfdtd()
        self.fdtd.set("index", background_index)
        self.fdtd.set("auto shutoff min", 1e-4)

        self.sim_center, self.sim_size = self.unit_scaling(center=[0,0,0], size=sim_size)
        nz_dims, n_axis = self.dimension_check(sim_size)
        if len(nz_dims) == 2:  
            self.fdtd.set("dimension", "2D")
        elif len(nz_dims) == 3:
            self.fdtd.set("dimension", "3D")
        else:
            raise ValueError("Simulation must span at least 2 dimensions")

        # region size + BC
        for dim in nz_dims:
            idx = self.kw_to_idx[dim]
            self.fdtd.set(f"{dim}", self.sim_center[idx])
            self.fdtd.set(f"{dim} span", self.sim_size[idx])
            self.fdtd.set(f"{dim} min bc", self.bc[dim])
            self.fdtd.set(f"{dim} max bc", self.bc[dim])

        if len(nz_dims) == 2:
            # set the normal axis to something consistent (e.g., Bloch or PML).
            # In Lumerical's 2D mode (X-Y plane) the out-of-plane (z) boundary
            # is inactive, so guard against "property inactive" errors.
            try:
                self.fdtd.set(f"{n_axis} min bc", "Bloch")
                self.fdtd.set(f"{n_axis} max bc", "Bloch")
            except Exception as _bc_exc:
                print(f"[FDTD] skip out-of-plane '{n_axis}' BC (inactive in 2D): {_bc_exc}")

        # --- MESH CONTROL ---
        # If resolution or ppw specified -> create a single mesh override that spans the whole domain
        if self.resolution or self.ppw:
            # compute target step
            if self.resolution:
                # resolution: cells per 'unit' (e.g., per micron if unit=1e-6)
                self.sim_grid = self.unit / float(self.resolution)
                mode_desc = f"uniform dx=dy=dz={self.sim_grid:.3e} m (resolution={self.resolution} per {self.unit:g} m)"
            else:
                # ppw: points per wavelength in background
                self.sim_grid = self.center_wl / (float(self.ppw))
                mode_desc = f"uniform dx=dy=dz={self.sim_grid:.3e} m (ppw={self.ppw} @ λ0={self.center_wl:g} m, n={self.n_bg})"

            # add one big mesh override to enforce dx,dy,dz
            self.fdtd.addmesh()
            self.fdtd.set("name", "global_uniform_mesh")
            for dim in nz_dims:
                idx = self.kw_to_idx[dim]
                self.fdtd.set(dim, self.sim_center[idx])
                self.fdtd.set(f"{dim} span", self.sim_size[idx])

            # override per-axis and set steps
            self.fdtd.set("override x mesh", 1)
            self.fdtd.set("override y mesh", 1)
            self.fdtd.set("override z mesh", 1)
            self.fdtd.set("dx", float(self.sim_grid))
            self.fdtd.set("dy", float(self.sim_grid))
            self.fdtd.set("dz", float(self.sim_grid))
            print(f"[FDTD] Mesh override enabled: {mode_desc}")
        else:
            # fallback: use mesh accuracy only
            self.fdtd.set("mesh accuracy", 2)
            print("[FDTD] Mesh: default mesh accuracy = 2 (no override)")

    def dimension_check(self, size):
        nonzero_dims = [d for d, s in zip(self.dims, size) if s > 0]
        if len(nonzero_dims) == 2:
            normal_axis = [d for d in self.dims if d not in nonzero_dims][0]
        else:
            normal_axis = None
        return nonzero_dims, normal_axis

    def unit_scaling(self, center, size):
        center=[e * self.unit for e in center]
        size=[e * self.unit for e in size]
        return center, size
    
    def add_source(
            self, 
            mode="custom", 
            name="source", 
            center=[0,0,0], 
            size=[1,1,0], 
            direction="forward", 
            src_wl=[1.0], 
            bandwidth=0.2, 
            Fields=None,
            pol=45,
            theta=0.0,
            phi=0.0,
            single=False,
            mode_num=0,
            broadband=False,
        ):
        self.src_wl = np.asarray(src_wl, dtype=float).reshape(-1) * self.unit
        self.src_bw=0.5*bandwidth
        wl_min = float(np.min(self.src_wl))
        wl_max = float(np.max(self.src_wl))
        nz_dims, n_axis = self.dimension_check(size)
        center, size = self.unit_scaling(center, size=size)
        idx=self.kw_to_idx[n_axis]
        if mode == "custom":
            if Fields is None:
                raise ValueError("Fields required for custom mode")
            dJEx, dJEy, dJEz = Fields
            u_ax, v_ax  = self.kw_to_idx[nz_dims[0]], self.kw_to_idx[nz_dims[1]]
            Su, Sv = size[u_ax], size[v_ax]
            n_ax=self.kw_to_idx[n_axis]

            def full_field_array(arr, nfreq=None):
                if arr is None:
                    return None
                arr = np.asarray(arr, dtype=np.complex128)
                if arr.ndim == 2:
                    shape3 = [1, 1, 1]
                    shape3[u_ax] = arr.shape[0]
                    shape3[v_ax] = arr.shape[1]
                    out = np.zeros(shape3, dtype=np.complex128)
                    if (u_ax, v_ax) == (0, 1):
                        out[:, :, 0] = arr
                    elif (u_ax, v_ax) == (0, 2):
                        out[:, 0, :] = arr
                    elif (u_ax, v_ax) == (1, 2):
                        out[0, :, :] = arr
                    else:
                        raise ValueError("Unexpected source plane axes.")
                    return out
                if arr.ndim == 3 and nfreq is not None and arr.shape[-1] == nfreq and len(nz_dims) == 2:
                    shape4 = [1, 1, 1, nfreq]
                    shape4[u_ax] = arr.shape[0]
                    shape4[v_ax] = arr.shape[1]
                    out = np.zeros(shape4, dtype=np.complex128)
                    if (u_ax, v_ax) == (0, 1):
                        out[:, :, 0, :] = arr
                    elif (u_ax, v_ax) == (0, 2):
                        out[:, 0, :, :] = arr
                    elif (u_ax, v_ax) == (1, 2):
                        out[0, :, :, :] = arr
                    else:
                        raise ValueError("Unexpected source plane axes.")
                    return out
                if arr.ndim == 3:
                    return arr
                if arr.ndim == 4:
                    return arr
                raise ValueError(f"Unexpected imported field shape {arr.shape}.")

            nfreq = len(self.src_wl)
            if broadband and nfreq <= 1:
                broadband = False

            dJEx = full_field_array(dJEx, nfreq=nfreq if broadband else None)
            dJEy = full_field_array(dJEy, nfreq=nfreq if broadband else None)
            dJEz = full_field_array(dJEz, nfreq=nfreq if broadband else None)
            base = next(arr for arr in [dJEx, dJEy, dJEz] if arr is not None)
            if broadband:
                if base.ndim != 4 or base.shape[-1] != nfreq:
                    raise ValueError(
                        "Broadband custom source requires field arrays with final wavelength axis "
                        f"of length {nfreq}; got {base.shape}."
                    )
            Nu, Nv = base.shape[u_ax], base.shape[v_ax]

            du, dv = Su / float(Nu-1), Sv / float(Nv-1)
            u0 = center[u_ax] - 0.5*Su
            v0 = center[v_ax] - 0.5*Sv
            u_grids = u0 + np.arange(Nu)*du
            v_grids = v0 + np.arange(Nv)*dv


            self.u_dim=nz_dims[0]
            self.v_dim=nz_dims[1]
            self.nor_dim=n_axis
            self.norm_pt=center[n_ax]

            def add_imported_source_object(source_name):
                self.fdtd.addimportedsource()
                self.fdtd.set("name", source_name)
                self.fdtd.set("injection axis", self.nor_dim)
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("z", center[2])
                try:
                    self.fdtd.set("wavelength start", wl_min if bandwidth == 0 else wl_min * (1 - self.src_bw))
                    self.fdtd.set("wavelength stop", wl_max if bandwidth == 0 else wl_max * (1 + self.src_bw))
                except Exception:
                    pass

                pts=[0]*3
                pts[u_ax]=u_grids
                pts[v_ax]=v_grids
                pts[n_ax]=self.norm_pt
                self.fdtd.putv('field',self.fdtd.rectilineardataset("field", pts[0],pts[1],pts[2]))

            if broadband:
                add_imported_source_object(name)
                self.fdtd.putv('Ex', dJEx)
                self.fdtd.putv('Ey', dJEy)
                self.fdtd.putv('Ez', dJEz)
                self.fdtd.putv('src_lambdas', self.src_wl)
                self.fdtd.putv('src_freqs', self.c / self.src_wl)
                self.EM_dataset = 'field.addparameter("lambda",src_lambdas,"f",src_freqs);'
                self.EM_dataset += 'field.addattribute("E",Ex,Ey,Ez);'
                self.EM_dataset += 'importdataset(field);'
                self.fdtd.eval(f"{self.EM_dataset}\n")
            else:
                for iidx in range(len(self.src_wl)):
                    source_name = name if len(self.src_wl) == 1 else f"{name}_{iidx}"
                    add_imported_source_object(source_name)
                    self.fdtd.putv('Ex', dJEx[:, :, :, iidx] if dJEx is not None and dJEx.ndim == 4 else dJEx)
                    self.fdtd.putv('Ey', dJEy[:, :, :, iidx] if dJEy is not None and dJEy.ndim == 4 else dJEy)
                    self.fdtd.putv('Ez', dJEz[:, :, :, iidx] if dJEz is not None and dJEz.ndim == 4 else dJEz)
                    self.EM_dataset=f"field.addparameter(\"lambda\",{self.src_wl[iidx]},\"f\",{self.c / self.src_wl[iidx]});"
                    self.EM_dataset+=f"field.addattribute(\"E\",Ex,Ey,Ez);"
                    self.EM_dataset+=f"importdataset(field);"
                    self.fdtd.eval(f"{self.EM_dataset}\n")
                

        elif mode =="eigen":
            self.fdtd.addmode()
            self.fdtd.set('mode selection', 'user select')
            self.fdtd.set('selected mode number', int(mode_num))
            self.fdtd.set("name", name)
            self.fdtd.set("injection axis", n_axis)
            self.fdtd.set("direction", direction)
            self.fdtd.set("x", center[0])
            self.fdtd.set("y", center[1])
            self.fdtd.set("z", center[2])
            if size[0] !=0:
                self.fdtd.set("x span", size[0])
            if size[1] !=0:
                self.fdtd.set("y span", size[1])
            if size[2] !=0:
                self.fdtd.set("z span", size[2])
            if bandwidth == 0:
                self.fdtd.eval(f"set(\"wavelength start\", {wl_min});\nset(\"wavelength stop\", {wl_max});")
            else:
                self.fdtd.eval(
                    f"set(\"wavelength start\", {wl_min*(1-self.src_bw)});\n"
                    f"set(\"wavelength stop\", {wl_max*(1+self.src_bw)});"
                )
        else:
            if single:
                self.fdtd.addplane()
                self.fdtd.set("name", name)
                self.fdtd.set("injection axis", n_axis)
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("z", center[2])
                if size[0] !=0:
                    self.fdtd.set("x span", size[0])
                if size[1] !=0:
                    self.fdtd.set("y span", size[1])
                if size[2] !=0:
                    self.fdtd.set("z span", size[2])
                self.fdtd.set("polarization angle",pol)
                self._set_plane_wave_angles(theta, phi)
                self.fdtd.eval(f"set(\"wavelength start\", {np.min(self.src_wl)});")
                self.fdtd.eval(f"set(\"wavelength stop\", {np.max(self.src_wl)});")
            else:
                for idx in range(len(src_wl)):
                    self.fdtd.addplane()
                    if len(src_wl)==1:
                        self.fdtd.set("name", name)
                    else:
                        self.fdtd.eval(f"addtogroup(\"{name}\");")
                    self.fdtd.set("injection axis", n_axis)
                    self.fdtd.set("direction", direction)
                    self.fdtd.set("x", center[0])
                    self.fdtd.set("y", center[1])
                    self.fdtd.set("z", center[2])
                    if size[0] !=0:
                        self.fdtd.set("x span", size[0])
                    if size[1] !=0:
                        self.fdtd.set("y span", size[1])
                    if size[2] !=0:
                        self.fdtd.set("z span", size[2])
                    self.fdtd.set("polarization angle",pol)
                    self._set_plane_wave_angles(theta, phi)
                    self.fdtd.eval(f"set(\"wavelength start\", {self.src_wl[idx]*(1-self.src_bw)});")
                    self.fdtd.eval(f"set(\"wavelength stop\", {self.src_wl[idx]*(1+self.src_bw)});")
            # self.fdtd.set('optimize for short pulse', False)

    def _set_plane_wave_angles(self, theta=0.0, phi=0.0):
        for prop, value in (
            ("angle theta", theta),
            ("angle phi", phi),
            ("theta", theta),
            ("phi", phi),
        ):
            try:
                self.fdtd.set(prop, float(value))
            except Exception:
                pass
            
    def add_monitor(
            self, 
            name="field_monitor",
            center=[0,0,0], 
            size=[1,1,1],
            N_f=1, 
        ):
        center, size = self.unit_scaling(center=center, size=size)
        nz_dims, n_axis = self.dimension_check(size)

        if len(nz_dims) == 0:
            monitor_type = f"Point"
        elif len(nz_dims) == 2:
            monitor_type = f"2D {n_axis.upper()}-normal"
        elif len(nz_dims) == 3:
            monitor_type = "3D"
        else:
            raise ValueError("Monitor must span at least 1 dimension")
        
        self.fdtd.adddftmonitor()
        self.fdtd.set("name", name)
        self.fdtd.set("monitor type", monitor_type)

        for dim in self.dims:
            idx=self.kw_to_idx[dim]
            self.fdtd.set(f"{dim}", center[idx])
            if dim !=n_axis and len(nz_dims) != 0:
                self.fdtd.set(f"{dim} span", size[idx])

        self.fdtd.set("override global monitor settings", True)
        self.fdtd.set("spatial interpolation", "nearest mesh cell")
        # self.fdtd.set("spatial interpolation", "specified position")
        # self.fdtd.set("spatial interpolation", "none")
        if N_f>len(self.src_wl):
            self.fdtd.set("frequency points",N_f)
        else:
            self.fdtd.set("frequency points",len(self.src_wl))
            self.fdtd.set("sample spacing","custom")
            self.fdtd.set("custom frequency samples",np.array(self.c/np.array(self.src_wl)))

    def _create_sampled_material(self, index, material_name=None, max_coefficients=6):
        table = _dispersion_table(index)
        if table is None:
            raise ValueError("Sampled material requires a dispersive index dict.")

        wl_um, n_data, k_data = table
        if material_name is None:
            material_name = index.get("name", "sampled_material")

        if material_name in self.material_cache:
            return self.material_cache[material_name]

        f = self.c / (wl_um * self.unit)
        eps = (n_data + 1j * k_data) ** 2
        if eps.shape[1] == 1:
            sampled_data = np.column_stack([f, eps[:, 0]])
        else:
            sampled_data = np.column_stack([f, eps[:, 0], eps[:, 1], eps[:, 2]])

        temp = self.fdtd.addmaterial(index.get("material_type", "Sampled data"))
        self.fdtd.setmaterial(temp, "name", material_name)
        try:
            self.fdtd.setmaterial(material_name, "max coefficients", int(index.get("max_coefficients", max_coefficients)))
        except Exception:
            pass
        self.fdtd.setmaterial(material_name, "sampled data", sampled_data)
        try:
            self.fdtd.setmaterial(material_name, "tolerance", float(index.get("tolerance", 0.0)))
        except Exception:
            pass

        self.material_cache[material_name] = material_name
        return material_name

    def _set_object_index(self, index, object_name=None, material_name=None, wavelength=None, dispersive=True):
        if isinstance(index, str):
            self.fdtd.set("material", index)
            return

        dispersion = _dispersion_table(index) if isinstance(index, dict) else None
        if isinstance(index, dict) and dispersive and dispersion is not None and dispersion[0].size >= 2:
            if material_name is None:
                material_name = index.get("name", object_name + "_mat" if object_name else "sampled_material")
            mat = self._create_sampled_material(index, material_name=material_name)
            self.fdtd.set("material", mat)
            return

        wl = self.center_wl_um if wavelength is None else wavelength
        idx = interpolate_index(index, wl)
        n_real = np.real(idx)
        k_imag = np.imag(idx)

        if idx.size == 1 and np.max(np.abs(k_imag)) < 1e-15:
            self.fdtd.set("index", float(n_real[0]))
            return

        mat = self.fdtd.addmaterial("(n,k) Material")
        if material_name is None:
            material_name = object_name + "_mat" if object_name else "dispersive_mat"
        try:
            self.fdtd.setmaterial(mat, "name", material_name)
            mat = material_name
        except Exception:
            pass

        if idx.size == 3:
            self.fdtd.setmaterial(mat, "Anisotropy", 1)
            self.fdtd.setmaterial(mat, "Refractive Index", np.asarray(n_real, dtype=float))
            try:
                self.fdtd.setmaterial(mat, "Imaginary Refractive Index", np.asarray(k_imag, dtype=float))
            except Exception:
                if np.max(np.abs(k_imag)) > 1e-15:
                    print(f"[material] Warning: could not set anisotropic k for {material_name}.")
        else:
            self.fdtd.setmaterial(mat, "Refractive Index", float(n_real[0]))
            try:
                self.fdtd.setmaterial(mat, "Imaginary Refractive Index", float(k_imag[0]))
            except Exception:
                if abs(k_imag[0]) > 1e-15:
                    print(f"[material] Warning: could not set k for {material_name}.")
        self.fdtd.set("material", mat)

    def add_geo(
            self,
            center=[0,0,0],
            size=[1,1,1],
            index=[1.0],
            name=None,
            wavelength=None,
            material_name=None,
            dispersive=True,
        ):
        center, size = self.unit_scaling(center=center, size=size)
        self.fdtd.addrect()
        if name:
            self.fdtd.set("name", name)
        for dim in self.dims:
            idx=self.kw_to_idx[dim]
            self.fdtd.set(f"{dim}", center[idx])
            self.fdtd.set(f"{dim} span", size[idx])

        self._set_object_index(
            index,
            object_name=name,
            material_name=material_name,
            wavelength=wavelength,
            dispersive=dispersive,
        )


    def add_waveguide(
        self,
        center=[0, 0, 0],
        length=1.0,
        top_width=0.5,
        bottom_width=0.7,
        height=0.3,
        prop_axis="z",
        width_axis="y",
        index=[4.8855, 4.5836, 4.8855],
        name="wg",
        wavelength=None,
        material_name=None,
        dispersive=True,
    ):
        # units
        center_s, _ = self.unit_scaling(center=center, size=[0, 0, 0])
        length_s = length * self.unit
        top_w_s = top_width * self.unit
        bottom_w_s = bottom_width * self.unit
        height_s = height * self.unit

        x0, y0, z0 = center_s

        # Lumerical waveguide object uses base angle instead of separate top/bottom widths
        # angle = sidewall angle from horizontal base in degrees
        half_diff = max(bottom_w_s - top_w_s, 0.0) * 0.5
        if half_diff == 0:
            base_angle_deg = 90.0
        else:
            base_angle_deg = np.degrees(np.arctan(height_s / half_diff))

        self.fdtd.addwaveguide()
        self.fdtd.set("name", name)
        self._set_object_index(
            index,
            object_name=name,
            material_name=material_name,
            wavelength=wavelength,
            dispersive=dispersive,
        )
        self.fdtd.set("base width", float(bottom_w_s))
        self.fdtd.set("base height", float(height_s))
        self.fdtd.set("base angle", float(base_angle_deg))

        # straight waveguide centerline
        if prop_axis == "z":
            poles = np.array([
                [x0, z0 - 0.5 * length_s],
                [x0, z0 + 0.5 * length_s],
            ])
            if width_axis == "y":
                self.fdtd.set("first axis", "x")
                self.fdtd.set("rotation 1", 90)
                self.fdtd.set("third axis", "z")
                self.fdtd.set("rotation 3", 90)
            self.fdtd.set("x", x0)
            self.fdtd.set("y", y0)
            self.fdtd.set("z", 0)


        elif prop_axis == "y":
            poles = np.array([
                [x0, y0 - 0.5 * length_s],
                [x0, y0 + 0.5 * length_s],
            ])
            self.fdtd.set("x", x0)
            self.fdtd.set("y", 0)
            self.fdtd.set("z", z0)

        elif prop_axis == "x":
            poles = np.array([
                [x0 - 0.5 * length_s, y0],
                [x0 + 0.5 * length_s, y0],
            ])
            self.fdtd.set("x", x0)
            self.fdtd.set("y", y0)
            self.fdtd.set("z", z0)

        else:
            raise ValueError("prop_axis='z' or 'y' or 'x'.")
        self.fdtd.set("poles", poles)


    def _require_uniform_mesh(self):
        if not hasattr(self, "sim_grid") or self.sim_grid is None:
            raise RuntimeError(
                "Exact design-field alignment requires a uniform mesh override. "
                "Set resolution=... or points_per_wavelength=... when creating the simulator."
            )

    def _snap_to_mesh(self, value):
        self._require_uniform_mesh()
        return np.round(np.asarray(value, dtype=float) / self.sim_grid) * self.sim_grid

    def _node_axis(self, center, N, d):
        self._require_uniform_mesh()
        target_center = float(center)

        if N <= 1:
            return np.array([float(self._snap_to_mesh(target_center))], dtype=float)

        # Keep the grid symmetric about the snapped center. For even N, this
        # intentionally places nodes on the half-cell phase used by Lumerical
        # DFT monitor sampling instead of snapping the first node to a full cell.
        return target_center + (np.arange(N, dtype=float) - 0.5 * (N - 1)) * d


    def _add_or_update_design_mesh_override(self):
        """
        Make the design region itself use the same uniform mesh as the import grid.
        """
        self._require_uniform_mesh()
        name = f"{self.design_name}_mesh"

        if self.fdtd.getnamednumber(name) == 0:
            self.fdtd.addmesh()
            self.fdtd.set("name", name)
        else:
            self.fdtd.eval(f'select("{name}");')

        extents = [
            self._monitor_extent_from_design_axis(self.design_x, self.design_dx, 0),
            self._monitor_extent_from_design_axis(self.design_y, self.design_dy, 1),
            self._monitor_extent_from_design_axis(self.design_z, self.design_dz, 2),
        ]
        for dim, (vmin, vmax) in zip(self.dims, extents):
            self.fdtd.set(f"{dim} min", float(vmin))
            self.fdtd.set(f"{dim} max", float(vmax))

        self.fdtd.set("override x mesh", 1)
        self.fdtd.set("override y mesh", 1)
        self.fdtd.set("override z mesh", 1)
        self.fdtd.set("dx", float(self.design_dx))
        self.fdtd.set("dy", float(self.design_dy))
        self.fdtd.set("dz", float(self.design_dz))

    def _monitor_extent_from_design_axis(self, axis, d, axis_idx):
        axis = np.atleast_1d(np.asarray(axis, dtype=float))
        return float(np.min(axis)), float(np.max(axis))

    def _assert_design_monitor_alignment(self, monitor_name=None, atol=None):
        """
        Check that the monitor grid returned by Lumerical matches the design grid.
        """
        if monitor_name is None:
            monitor_name = self.design_monitor_name
        if atol is None:
            atol = 1e-12

        Eres = self.fdtd.getresult(monitor_name, "E")
        xm = np.atleast_1d(np.squeeze(np.array(Eres["x"], dtype=float)))
        ym = np.atleast_1d(np.squeeze(np.array(Eres["y"], dtype=float)))
        zm = np.atleast_1d(np.squeeze(np.array(Eres["z"], dtype=float)))

        self._last_design_monitor_slices = [
            self._design_monitor_axis_slice("x", xm),
            self._design_monitor_axis_slice("y", ym),
            self._design_monitor_axis_slice("z", zm),
        ]
        xm = xm[self._last_design_monitor_slices[0]]
        ym = ym[self._last_design_monitor_slices[1]]
        zm = zm[self._last_design_monitor_slices[2]]

        self._adopt_close_monitor_axis("x", xm)
        self._adopt_close_monitor_axis("y", ym)
        self._adopt_close_monitor_axis("z", zm)

        if xm.size != self.design_x.size or not np.allclose(xm, self.design_x, atol=atol, rtol=0):
            raise RuntimeError(
                f"x-grid mismatch: monitor={xm.shape}/{xm[:3]}..., design={self.design_x.shape}/{self.design_x[:3]}..."
            )
        if ym.size != self.design_y.size or not np.allclose(ym, self.design_y, atol=atol, rtol=0):
            raise RuntimeError(
                f"y-grid mismatch: monitor={ym.shape}/{ym[:3]}..., design={self.design_y.shape}/{self.design_y[:3]}..."
            )
        if zm.size != self.design_z.size or not np.allclose(zm, self.design_z, atol=atol, rtol=0):
            raise RuntimeError(
                f"z-grid mismatch: monitor={zm.shape}/{zm[:3]}..., design={self.design_z.shape}/{self.design_z[:3]}..."
            )

    def _design_monitor_axis_slice(self, axis_name, monitor_axis):
        design_axis = np.atleast_1d(np.asarray(getattr(self, f"design_{axis_name}"), dtype=float))
        monitor_axis = np.atleast_1d(np.asarray(monitor_axis, dtype=float))
        if monitor_axis.size == design_axis.size:
            return slice(None)
        if monitor_axis.size != design_axis.size + 1:
            return slice(None)

        d = float(getattr(self, f"design_d{axis_name}"))
        adopt_atol = max(1e-9, 0.6 * abs(d))
        head = monitor_axis[: design_axis.size]
        tail = monitor_axis[-design_axis.size :]
        head_delta = float(np.max(np.abs(head - design_axis)))
        tail_delta = float(np.max(np.abs(tail - design_axis)))
        if head_delta <= adopt_atol and head_delta <= tail_delta:
            print(f"[design monitor] cropping extra trailing {axis_name}-sample (no interpolation)")
            return slice(0, design_axis.size)
        if tail_delta <= adopt_atol:
            print(f"[design monitor] cropping extra leading {axis_name}-sample (no interpolation)")
            return slice(1, None)
        return slice(None)

    def _adopt_close_monitor_axis(self, axis_name, monitor_axis):
        design_attr = f"design_{axis_name}"
        d_attr = f"design_d{axis_name}"
        axis_idx = self.kw_to_idx[axis_name]

        design_axis = np.atleast_1d(np.asarray(getattr(self, design_attr), dtype=float))
        monitor_axis = np.atleast_1d(np.asarray(monitor_axis, dtype=float))
        if monitor_axis.size != design_axis.size:
            return

        if np.allclose(monitor_axis, design_axis, atol=1e-12, rtol=0):
            return

        d = float(getattr(self, d_attr))
        adopt_atol = max(1e-9, 0.6 * abs(d))
        max_delta = float(np.max(np.abs(monitor_axis - design_axis)))
        if max_delta > adopt_atol:
            return

        print(
            f"[design monitor] adopting Lumerical {axis_name}-grid "
            f"(max shift {max_delta:.3e} m, no field interpolation)"
        )
        setattr(self, design_attr, monitor_axis.copy())
        if monitor_axis.size > 1:
            setattr(self, d_attr, float(np.mean(np.diff(monitor_axis))))
        self.design_center[axis_idx] = float(0.5 * (monitor_axis[0] + monitor_axis[-1]))
        self.design_size[axis_idx] = float(monitor_axis[-1] - monitor_axis[0])

    def set_spatial_interp(self, monitor_name, setting="specified position"):
        script = f'select("{monitor_name}"); set("spatial interpolation","{setting}");'
        self.fdtd.eval(script)


    def _configure_design_region_objects(self):
        """
        Design region monitor / index monitor / mesh override를
        design_x, design_y, design_z와 정확히 같은 좌표계로 맞춘다.
        """
        xmin, xmax = self._monitor_extent_from_design_axis(self.design_x, self.design_dx, 0)
        ymin, ymax = self._monitor_extent_from_design_axis(self.design_y, self.design_dy, 1)
        zmin, zmax = self._monitor_extent_from_design_axis(self.design_z, self.design_dz, 2)

        # DFT monitor
        if self.fdtd.getnamednumber(self.design_monitor_name) == 0:
            self.fdtd.adddftmonitor()
            self.fdtd.set("name", self.design_monitor_name)

        nz_dims, n_axis = self.dimension_check([xmax - xmin, ymax - ymin, zmax - zmin])
        if len(nz_dims) == 1:
            monitor_type = "1D"
        elif len(nz_dims) == 2:
            monitor_type = f"2D {n_axis.upper()}-normal"
        elif len(nz_dims) == 3:
            monitor_type = "3D"
        else:
            raise ValueError("Monitor must span at least 1 dimension")

        script = (
            f'select("{self.design_monitor_name}");'
            f'set("monitor type","{monitor_type}");'
            f'set("x min",{xmin});'
            f'set("x max",{xmax});'
            f'set("y min",{ymin});'
            f'set("y max",{ymax});'
            f'set("z min",{zmin});'
            f'set("z max",{zmax});'
            f'set("override global monitor settings",1);'
        )
        self.fdtd.eval(script)
        # self.set_spatial_interp(self.design_monitor_name, "nearest mesh cell")
        self.set_spatial_interp(self.design_monitor_name, "none")
        # self.set_spatial_interp(self.design_monitor_name, "specified position")

        try:
            self.fdtd.setnamed(self.design_monitor_name, "down sample x", 1)
            self.fdtd.setnamed(self.design_monitor_name, "down sample y", 1)
            self.fdtd.setnamed(self.design_monitor_name, "down sample z", 1)
        except Exception:
            pass

        self._configure_design_monitor_spectral_settings()
        self._add_or_update_design_mesh_override()
        self._configure_design_index_monitor()
        self.fdtd.eval(
            f'setnamed("FDTD","min mesh step",{min(self.design_dx, self.design_dy, self.design_dz)});'
        )

    def _configure_design_monitor_spectral_settings(self):
        if not hasattr(self, "src_wl") or self.src_wl is None:
            return
        if self.fdtd.getnamednumber(self.design_monitor_name) == 0:
            return
        
        try:
            self.fdtd.setnamed(self.design_monitor_name, "override global monitor settings", True)
        except Exception:
            pass

        try:
            self.fdtd.setnamed(self.design_monitor_name, "use source limits", False)
        except Exception:
            pass

        try:
            self.fdtd.setnamed(self.design_monitor_name, "sample spacing", "custom")
        except Exception:
            pass

        try:
            self.fdtd.setnamed(self.design_monitor_name, "custom frequency samples",
                            np.array(self.c / np.array(self.src_wl)))
        except Exception:
            try:
                self.fdtd.setnamed(self.design_monitor_name, "frequency points", len(self.src_wl))
            except Exception:
                pass

    def _configure_design_index_monitor(self):
        """
        Create/update an index monitor that matches the design grid extent exactly.
        """
        xmin, xmax = self._monitor_extent_from_design_axis(self.design_x, self.design_dx, 0)
        ymin, ymax = self._monitor_extent_from_design_axis(self.design_y, self.design_dy, 1)
        zmin, zmax = self._monitor_extent_from_design_axis(self.design_z, self.design_dz, 2)

        name = self.design_index_monitor_name

        if self.fdtd.getnamednumber(name) == 0:
            self.fdtd.addindex()
            self.fdtd.set("name", name)

        nz_dims, n_axis = self.dimension_check([xmax - xmin, ymax - ymin, zmax - zmin])
        if len(nz_dims) == 1:
            monitor_type = "linear " + nz_dims[0]
        elif len(nz_dims) == 2:
            monitor_type = f"2D {n_axis.upper()}-normal"
        elif len(nz_dims) == 3:
            monitor_type = "3D"
        else:
            raise ValueError("Index monitor must span at least 1 dimension")

        script = (
            f'select("{name}");'
            f'set("monitor type","{monitor_type}");'
            f'set("x min",{xmin});'
            f'set("x max",{xmax});'
            f'set("y min",{ymin});'
            f'set("y max",{ymax});'
            f'set("z min",{zmin});'
            f'set("z max",{zmax});'
        )
        self.fdtd.eval(script)

        try:
            self.fdtd.setnamed(name, "spatial interpolation", "none")
        except Exception:
            pass

        try:
            self.fdtd.setnamed(name, "down sample x", 1)
            self.fdtd.setnamed(name, "down sample y", 1)
            self.fdtd.setnamed(name, "down sample z", 1)
        except Exception:
            pass

    def density2idx(
            self,
            density
        ):
        d_copy = np.clip(np.array(density.copy()).flatten(), 0.0, 1.0)
        rho=[0]*3
        for ax in range(3):
            n_idx=np.array((d_copy*(self.index1[ax]**2-self.index2[ax]**2)))
            n_idx+=(self.index2[ax]**2)*np.ones(self.design_grids[0]*self.design_grids[1]*self.design_grids[2])
            rho[ax]=np.array(np.sqrt(n_idx))
        return rho[0], rho[1], rho[2]

    def add_design_grid(
            self, 
            name="design", 
            center=[0, 0, 0],           
            size=[1, 1, 1],             
            index1=[1.5],
            index2=[1.0],
            design_grids=[50, 50, 50],
            density=None,
            wavelength=None,
        ):
        self._require_uniform_mesh()

        self.design_name = name
        self.design_monitor_name = self.design_name + "_monitor"
        self.design_index_monitor_name = self.design_name + "_index_monitor"

        self.design_grids = list(design_grids)
        Nx, Ny, Nz = self.design_grids

        wl = self.center_wl_um if wavelength is None else wavelength
        index1_resolved = interpolate_index(index1, wl)
        index2_resolved = interpolate_index(index2, wl)
        if index1_resolved.size == 1:
            index1_resolved = np.repeat(index1_resolved, 3)
        if index2_resolved.size == 1:
            index2_resolved = np.repeat(index2_resolved, 3)

        if np.max(np.abs(np.imag(index1_resolved))) > 1e-15 or np.max(np.abs(np.imag(index2_resolved))) > 1e-15:
            print(
                "[design] Warning: lossy design endpoints are resolved for dε/dρ, "
                "but imported design geometry currently uses real(n) only."
            )

        self.index1_complex = index1_resolved
        self.index2_complex = index2_resolved
        self.index1 = [float(np.real(index1_resolved[0])), float(np.real(index1_resolved[1])), float(np.real(index1_resolved[2]))]
        self.index2 = [float(np.real(index2_resolved[0])), float(np.real(index2_resolved[1])), float(np.real(index2_resolved[2]))]

        if hasattr(self, "src_wl") and self.src_wl is not None:
            src_wl_um = np.asarray(self.src_wl, dtype=float).reshape(-1) / self.unit
        else:
            src_wl_um = np.array([wl], dtype=float)
        self.index1_spectrum = np.zeros((src_wl_um.size, 3), dtype=np.complex128)
        self.index2_spectrum = np.zeros_like(self.index1_spectrum)
        for i, wl_i in enumerate(src_wl_um):
            idx1_i = interpolate_index(index1, float(wl_i))
            idx2_i = interpolate_index(index2, float(wl_i))
            if idx1_i.size == 1:
                idx1_i = np.repeat(idx1_i, 3)
            if idx2_i.size == 1:
                idx2_i = np.repeat(idx2_i, 3)
            self.index1_spectrum[i] = idx1_i
            self.index2_spectrum[i] = idx2_i

        self.design_dx = float(self.sim_grid)
        self.design_dy = float(self.sim_grid)
        self.design_dz = float(self.sim_grid)

        center_scaled, _ = self.unit_scaling(center=center, size=[0, 0, 0])
        cx, cy, cz = self._snap_to_mesh(center_scaled)

        self.design_center = [cx, cy, cz]

        # node axis
        self.design_x = self._node_axis(cx, Nx, self.design_dx)
        self.design_y = self._node_axis(cy, Ny, self.design_dy)
        self.design_z = self._node_axis(cz, Nz, self.design_dz)

        self.design_size = [
            float(np.max(self.design_x) - np.min(self.design_x)) if Nx > 1 else 0.0,
            float(np.max(self.design_y) - np.min(self.design_y)) if Ny > 1 else 0.0,
            float(np.max(self.design_z) - np.min(self.design_z)) if Nz > 1 else 0.0,
        ]

        self._configure_design_region_objects()

        if density is None:
            density = 0.5 * np.ones(Nx * Ny * Nz, dtype=float)

        self.update_design_density(density)

    def update_design_density(self, density):
        Nx, Ny, Nz = self.design_grids

        density = np.asarray(density, dtype=float).copy()

        if density.ndim != 3:
            if density.size != Nx * Ny * Nz:
                raise ValueError(f"density size mismatch: got {density.size}, expected {Nx*Ny*Nz}")
            density = density.reshape(Nx, Ny, Nz)
        elif density.shape != (Nx, Ny, Nz):
            raise ValueError(f"density shape mismatch: got {density.shape}, expected {(Nx, Ny, Nz)}")
        self.design_density = density.copy()

        rho1, rho2, rho3 = self.density2idx(density)
        rho1 = np.asarray(rho1, dtype=float).reshape(Nx, Ny, Nz)
        rho2 = np.asarray(rho2, dtype=float).reshape(Nx, Ny, Nz)
        rho3 = np.asarray(rho3, dtype=float).reshape(Nx, Ny, Nz)

        n_ani = np.ones((Nx, Ny, Nz, 3), dtype=float)
        n_ani[:, :, :, 0] = np.clip(rho1, 1.0, 100.0)
        n_ani[:, :, :, 1] = np.clip(rho2, 1.0, 100.0)
        n_ani[:, :, :, 2] = np.clip(rho3, 1.0, 100.0)

        self.design_n = n_ani.copy()

        # importnk2 requires the z axis to have >= 2 layers even when the design
        # (and its monitors) are a single 2D z-plane.  Duplicate the single
        # design layer into a thin 2-layer slab centred on the design z-plane
        # for the IMPORT ONLY; design_grids / design_z / monitors stay Nz=1.
        z_geo_imp = np.asarray(self.design_z, dtype=float)
        n_geo_imp = np.ascontiguousarray(n_ani)
        if z_geo_imp.size == 1:
            _dz = float(self.design_dz) if float(self.design_dz) > 0 else float(self.sim_grid)
            z_geo_imp = np.array([z_geo_imp[0] - 0.5 * _dz,
                                  z_geo_imp[0] + 0.5 * _dz], dtype=float)
            n_geo_imp = np.ascontiguousarray(np.repeat(n_ani, 2, axis=2))

        self.fdtd.putv("x_geo", np.asarray(self.design_x, dtype=float))
        self.fdtd.putv("y_geo", np.asarray(self.design_y, dtype=float))
        self.fdtd.putv("z_geo", z_geo_imp)
        self.fdtd.putv("n_geo", n_geo_imp)

        if self.fdtd.getnamednumber(self.design_name) == 0:
            script = (
                f'addimport;'
                f'set("name","{self.design_name}");'
                f'importnk2(n_geo, x_geo, y_geo, z_geo);'
            )
            self.fdtd.eval(script)
        else:
            try:
                self.fdtd.eval(
                    f'select("{self.design_name}");'
                    f'importnk2(n_geo, x_geo, y_geo, z_geo);'
                )
            except Exception:
                # Fallback for cases where the import object cannot be updated in place.
                self.fdtd.eval(
                    f'select("{self.design_name}");'
                    f'delete;'
                    f'addimport;'
                    f'set("name","{self.design_name}");'
                    f'importnk2(n_geo, x_geo, y_geo, z_geo);'
                )

    def add_design_monitor(self):
        self.design_monitor_name = self.design_name + "_monitor"
        if not hasattr(self, "design_index_monitor_name"):
            self.design_index_monitor_name = self.design_name + "_index_monitor"
        self._configure_design_region_objects()


    def _configured_session_resource_names(self, resource_type="GPU"):
        names = []
        config_paths = []
        xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
        if xdg_config_home:
            config_paths.append(
                Path(xdg_config_home).expanduser()
                / "Lumerical"
                / "FDTD Solutions.ini"
            )
        for home_value in (os.environ.get("HOME"), os.environ.get("EIDL_REAL_HOME")):
            if home_value:
                config_paths.append(Path(home_value).expanduser() / ".config" / "Lumerical" / "FDTD Solutions.ini")
        config_paths.append(Path.home() / ".config" / "Lumerical" / "FDTD Solutions.ini")
        config_path = next((path for path in config_paths if path.exists()), None)
        if config_path is None:
            return names
        parser = configparser.RawConfigParser()
        parser.optionxform = str
        try:
            parser.read(config_path)
            xml_text = parser.get("jobmanager", "FDTD_v2", fallback="")
            if not xml_text:
                return names
            root = ET.fromstring(xml_text)
        except Exception as exc:
            print(f"[FDTD] could not read Lumerical resource config: {exc}")
            return names

        requested = str(resource_type).upper()
        for engine in root.findall("engine"):
            name = (engine.findtext("name") or "").strip()
            device_type = (engine.findtext("DeviceType") or "").strip().upper()
            if not name:
                continue
            if requested == "GPU":
                if device_type.startswith("GPU"):
                    names.append(name)
            elif requested == "CPU":
                if device_type == "CPU":
                    names.append(name)
            else:
                names.append(name)
        return names

    def _session_resource_names(self, resource_type="GPU"):
        explicit = os.environ.get("LUMERICAL_SESSION_RESOURCE_NAME", "").strip()
        names = []
        if explicit:
            names.append(explicit)
        names.extend(self._configured_session_resource_names(resource_type))
        if str(resource_type).upper() == "GPU":
            names.extend([
                "Local GPU",
                "local GPU",
                "Local Host",
                "localhost",
                "Localhost",
                "Local Computer",
                "local host",
            ])
        else:
            names.extend([
                "Local Host",
                "localhost",
                "Localhost",
                "Local Computer",
                "local host",
            ])
        unique = []
        for name in names:
            if name and name not in unique:
                unique.append(name)
        return unique

    def _run_session_only(self, solver="FDTD", resource_type="GPU", run_name=""):
        errors = []
        for resource_name in self._session_resource_names(resource_type):
            try:
                print(
                    f"[FDTD] session run: name={run_name}, solver={solver}, "
                    f"resource_type={resource_type}, resource_name={resource_name}"
                )
                self.fdtd.run(solver, resource_type, resource_name)
                return resource_name
            except Exception as exc:
                errors.append(f"{resource_name}: {exc}")
        raise RuntimeError(
            "Lumerical session run failed for all configured resource names. "
            "External solver fallback is disabled. Set LUMERICAL_SESSION_RESOURCE_NAME "
            "to the exact resource name shown in the Lumerical Resource Manager. "
            "Tried: " + " | ".join(errors)
        )

    def _configure_session_resources(self):
        gpu_device = os.environ.get("LUMERICAL_SESSION_GPU_DEVICE", "GPU 0")
        threads = os.environ.get("FDTD_THREADS", "1")
        try:
            self.fdtd.setresource("FDTD", 1, "active", 1)
            self.fdtd.setresource("FDTD", 1, "processes", "1")
            self.fdtd.setresource("FDTD", 1, "threads", str(threads))
        except Exception:
            pass
        try:
            self.fdtd.setresource("FDTD", 2, "active", 1)
            self.fdtd.setresource("FDTD", 2, "processes", "1")
            self.fdtd.setresource("FDTD", 2, "threads", str(threads))
            self.fdtd.setresource("FDTD", 2, "device type", gpu_device)
            self.fdtd.setresource("FDTD", 2, "solver extra command line options", "-gpu")
        except Exception:
            pass

    def _run_log_tail(self, run_name, max_chars=3000):
        log_path = Path(f"{run_name}_p0.log")
        if not log_path.exists():
            return ""
        try:
            text = log_path.read_text(errors="replace")
        except Exception:
            return ""
        return text[-max_chars:]

    def run(self,name="fdtd_tutorial",save=True):
        fsp_path = os.path.abspath(f"{name}.fsp")
        self._last_run_fsp_path = fsp_path
        if save:
            self.fdtd.save(fsp_path)
        self.fdtd.switchtolayout()
        self.fdtd.eval("select(\"FDTD\");")

        self._configure_session_resources()

        try:
            dimension = str(self.fdtd.getnamed("FDTD", "dimension"))
        except Exception:
            dimension = ""
        if dimension == "3D":
            retries = max(1, int(os.environ.get("LUMERICAL_SESSION_RUN_RETRIES", "3")))
            retry_delay = float(os.environ.get("LUMERICAL_SESSION_RUN_RETRY_DELAY", "5"))
            last_error = None
            for attempt in range(retries):
                try:
                    self._configure_session_resources()
                    self._run_session_only("FDTD", "GPU", run_name=name)
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt >= retries - 1:
                        break
                    print(
                        "[FDTD] session run failed; reloading saved project and "
                        f"retrying in {retry_delay:g}s ({attempt + 1}/{retries - 1}). "
                        f"Last error: {exc}"
                    )
                    time.sleep(retry_delay)
                    try:
                        self.fdtd.switchtolayout()
                        self.fdtd.load(fsp_path)
                    except Exception:
                        try:
                            self.fdtd.close()
                        except Exception:
                            pass
                        self.fdtd = _open_lumerical_fdtd()
                        self.fdtd.switchtolayout()
                        self.fdtd.load(fsp_path)
            raise last_error
        else:
            self.fdtd.run()

import tempfile, os
import scipy.io as sio
from autograd import jacobian
import os

class LumericalOptimizationProblem:
    """Define the optimization module"""
    def __init__(
        self,
        sim,
        objective_functions: list,
        objective_arguments: list,
        FoM_size=[5, 5, 0],
        FoM_center=[0, 0, 0],
        adj_fwd=False,
        opt_idx=0,
        broadband_adjoint=False,
        Incoherent=False,
    ):
        self.sim = sim
        self.objective_functions = objective_functions
        self.objective_arguments = objective_arguments
        self.forward_adj=adj_fwd
        self.opt_idx=opt_idx
        self.broadband_adjoint = broadband_adjoint
        # INCOHERENT MODE
        # Default (False) keeps the historical behaviour: every objective's
        # dJ/dE is summed into ONE adjoint source and one adjoint run produces
        # one gradient. That is correct only when the objectives are meant to be
        # added coherently.
        #
        # With Incoherent=True each objective gets its OWN adjoint source and its
        # OWN adjoint run, and the per-objective gradients are summed afterwards:
        #     g = sum_j ( fwd x adj_j )
        # The forward is still run once. Wavelength stays a SUB-level of the
        # objective, so a multi-J multi-wavelength problem is executed in the
        # order J1_lam1, J1_lam2, ..., JN_lam1, ... -- the existing broadband
        # machinery is reused unchanged inside each objective.
        self.Incoherent = bool(Incoherent)
        self.adjoint_fields_per_J = []
        self.f0_per_J = []
        self.gradient_per_J = []

        self.H_field = any(arg >= 3 for arg in self.objective_arguments)
        self.num_components = 6 if self.H_field else 3

        self.tangential_only = False
        self.adj_complex_prefactor = 1.0 + 0.0j

        self.FoM_center, self.FoM_size = self.sim.unit_scaling(FoM_center, FoM_size)
        self.boundary_overlap_beta=0.6
        self.f0 = None
        self.gradient = None
        self.forward_fields = None
        self.adjoint_fields = None
        self.FoM_fields = None
        self.last_forward_had_nonfinite = False
        self.forward_result_hook = None
        self.src_spectrum = None 
        self.iter=0
        self.rs_cnt=1

        self.u_grids =None 
        self.v_grids =None 

        self.avg_fwd=[0,0]
        self.avg_adj=[0,0]

        self.sim.fdtd.switchtolayout()
        if self.sim.fdtd.getnamednumber(self.sim.design_name) == 0:
            raise ValueError(f"Should define the design region first: sim.add_design_grid()")

        if self.sim.fdtd.getnamednumber("FoM_monitor") == 0:
            self.sim.add_monitor(
                name="FoM_monitor",
                center=FoM_center,
                size=FoM_size
            )
        if self.sim.fdtd.getnamednumber(self.sim.design_name+"_monitor") == 0:    
            self.sim.add_design_monitor()

        self.dedr=np.array(self.sim.index1)**2-np.array(self.sim.index2)**2
        if hasattr(self.sim, "index1_spectrum") and hasattr(self.sim, "index2_spectrum"):
            self.dedr_spectrum = self.sim.index1_spectrum ** 2 - self.sim.index2_spectrum ** 2
        else:
            self.dedr_spectrum = None
        self.g_norm=np.zeros(len(self.objective_functions))
        self.current_state = "INIT"
        self.sim_copy=self.sim
        self.base_fsp_path = os.path.abspath(f"base_{self.opt_idx}.fsp")
        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.save(self.base_fsp_path)
        self.sim.fdtd.close()
        self.sim.fdtd = _open_lumerical_fdtd()
        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.load(self.base_fsp_path)




    """Run unit iterations"""
    def __call__(self, rho_vector=None, need_value=True, need_gradient=True):
        start = time.time()
        if rho_vector is not None:
            self.sim.fdtd.switchtolayout()
            self.sim.update_design_density(density=rho_vector)
            self.design=rho_vector
            self.current_state = "INIT"
            design_upt = time.time()
            print(f"Design update time: {design_upt - start:.2f} seconds")

        start = time.time()
        if need_value and self.current_state == "INIT":
            self.forward_run()
            fwd = time.time()
            print(f"Forward run time: {fwd - start:.2f} seconds")
            self.avg_fwd[0] +=(fwd - start)
            self.avg_fwd[1] += 1

        if need_gradient:
            if self.current_state == "INIT":
                self.forward_run()
                fwd = time.time()
                print(f"Forward run time: {fwd - start:.2f} seconds")
                self.avg_fwd[0] +=(fwd - start)
                self.avg_fwd[1] += 1

                self.adjoint_dipole_run()
                adj = time.time()
                print(f"Adjoint total run time: {adj - fwd:.2f} seconds")
                self.avg_adj[0] +=(adj-fwd)
                self.avg_adj[1] += 1
                
                self.calculate_gradient()
                grad = time.time()
                print(f"Gradient calculation time: {grad - adj:.2f} seconds")
            elif self.current_state == "FWD":
                start = time.time()
                self.adjoint_dipole_run()
                adj = time.time()
                print(f"Adjoint total run time: {adj - start:.2f} seconds")
                self.avg_adj[0] +=(adj-start)
                self.avg_adj[1] += 1

                self.calculate_gradient()
                grad = time.time()
                print(f"Gradient calculation time: {grad - adj:.2f} seconds")
                print(f"Average Gradient: {np.mean(np.abs(self.gradient))}\nMaximum Gradient: {np.max(np.abs(self.gradient))}")
            else:
                raise ValueError(
                    f"Incorrect solver state detected: {self.current_state}"
                )
            print(f"\nForward cnt: {self.avg_fwd[1]}, Average time: {self.avg_fwd[0]/self.avg_fwd[1]} sec")
            print(f"Adjoint cnt: {self.avg_adj[1]}, Average time: {self.avg_adj[0]/self.avg_adj[1]} sec\n")
        if self.rs_cnt%(4)==0:
            self.sim.fdtd.close()
            self.sim.fdtd = self.sim_copy
            self.sim.fdtd = _open_lumerical_fdtd()
            self.sim.fdtd.switchtolayout()
            self.sim.fdtd.load(self.base_fsp_path)
            self.rs_cnt+=1
        else:
            self.rs_cnt+=1
        return self.f0, self.gradient

    """get EM field as array"""
    def get_field(self, monitor_name, H_field=False, check_design_alignment=False):
        Eres = self.sim.fdtd.getresult(monitor_name, "E")

        if check_design_alignment and monitor_name == self.sim.design_monitor_name:
            self.sim._assert_design_monitor_alignment(monitor_name)

        Eall = np.array(Eres["E"], dtype=np.complex128)
        if (
            check_design_alignment
            and monitor_name == self.sim.design_monitor_name
            and hasattr(self.sim, "_last_design_monitor_slices")
        ):
            sx, sy, sz = self.sim._last_design_monitor_slices
            Eall = Eall[sx, sy, sz, ...]
        if Eall.shape[-1] != 3:
            raise ValueError(f"Unexpected E field shape {Eall.shape}: last axis is not 3.")

        Ex = Eall[..., 0]
        Ey = Eall[..., 1]
        Ez = Eall[..., 2]

        if H_field:
            Hres = self.sim.fdtd.getresult(monitor_name, "H")
            Hall = np.array(Hres["H"], dtype=np.complex128)
            if Hall.shape[-1] != 3:
                raise ValueError(f"Unexpected H field shape {Hall.shape}: last axis is not 3.")

            Hx = Hall[..., 0]
            Hy = Hall[..., 1]
            Hz = Hall[..., 2]
            return np.stack([Ex, Ey, Ez, Hx, Hy, Hz], axis=0)

        return np.stack([Ex, Ey, Ez], axis=0)

    def assemble_boundary_tangent_gradient_mask(self, gx, gy, gz):
        """
        gx, gy, gz: shape (Nx, Ny, Nz, Nf)

        Component-wise scaling rule:
        For each component g_self:

        - if no axis is on boundary: scale = 1.0
        - if only self axis is on boundary: scale = 1.0
        - if self axis is not on boundary and exactly one other axis is boundary: scale = 0.5
        - if self axis is on boundary and one more axis is boundary: scale = 0.5
        - if self axis is the only non-boundary axis (i.e. two others are boundary): scale = 0.125
        - if all three axes are boundary: scale = 0.125
        """
        if not (gx.shape == gy.shape == gz.shape):
            raise ValueError(f"gx, gy, gz shape mismatch: {gx.shape}, {gy.shape}, {gz.shape}")

        Nx, Ny, Nz, Nf = gx.shape

        bx = np.zeros((Nx, Ny, Nz, 1), dtype=bool)
        by = np.zeros((Nx, Ny, Nz, 1), dtype=bool)
        bz = np.zeros((Nx, Ny, Nz, 1), dtype=bool)
        end_x= np.zeros((Nx, Ny, Nz, 1), dtype=bool)
        end_y= np.zeros((Nx, Ny, Nz, 1), dtype=bool)
        end_z= np.zeros((Nx, Ny, Nz, 1), dtype=bool)

        bx[0, :, :, :] = True
        bx[Nx - 1, :, :, :] = True
        end_x[Nx - 1, :, :, :] = True


        by[:, 0, :, :] = True
        by[:, Ny - 1, :, :] = True
        end_y[:, Ny - 1, :, :] = True

        bz[:, :, 0, :] = True
        bz[:, :, Nz - 1, :] = True
        end_z[:, :, Nz - 1, :] = True

        bcount = bx.astype(np.int32) + by.astype(np.int32) + bz.astype(np.int32)

        def component_scale(self_boundary, end_mask):
            s = np.empty((Nx, Ny, Nz, 1), dtype=float)
            
            # bcount == 0 -> 1.0
            s[bcount == 0] = 1.0/2

            # bcount == 1
            mask = (bcount == 1)
            s[mask] = np.where(self_boundary[mask], 1.0/2, 0.5/2)

            # bcount == 2
            mask = (bcount == 2)
            s[mask] = np.where(self_boundary[mask], 0.5/2, 0.125/2)

            # bcount == 3 -> 0.125
            s[bcount == 3] = 0.125/2

            s=np.where(end_mask, 0, s)

            return s

        sx = component_scale(bx,end_x)
        sy = component_scale(by,end_y)
        sz = component_scale(bz,end_z)

        return sx * gx, sy * gy, sz * gz 


    """ Forward run"""
    def forward_run(self):
        self.iter += 1

        retries = max(1, int(os.environ.get("LUMERICAL_FORWARD_RESULT_RETRIES", "6")))
        retry_delay = float(os.environ.get("LUMERICAL_FORWARD_RESULT_RETRY_DELAY", "10"))
        last_error = None
        for attempt in range(retries):
            self.sim.fdtd.switchtolayout()
            self.sim.fdtd.setnamed('source', 'enabled', True)
            self.sim.fdtd.setnamed('design_monitor', 'enabled', True)
            self.sim.fdtd.setnamed('FoM_monitor', 'enabled', True)

            try:
                self.sim.run(name="Forward_run", save=True)
                self.forward_fields = self.get_field(
                    self.sim.design_monitor_name,
                    H_field=False,
                    check_design_alignment=True,
                )
                try:
                    spec = self.sim.fdtd.getresult("source", "spectrum")
                    self.src_freqs = np.asarray(spec["f"], dtype=float).reshape(-1)
                    self.src_spectrum = np.asarray(spec["spectrum"], dtype=float).reshape(-1)
                except Exception:
                    wl = np.asarray(getattr(self.sim, "src_wl", []), dtype=float).reshape(-1)
                    if wl.size:
                        c = getattr(self.sim, "c", 299792458.0)
                        self.src_freqs = np.asarray(c / wl, dtype=float).reshape(-1)
                    else:
                        self.src_freqs = np.asarray([float(getattr(self.sim, "center_wl", 0.0))], dtype=float)
                    self.src_spectrum = np.ones_like(self.src_freqs, dtype=float)

                self.FoM_fields = self.get_field("FoM_monitor", H_field=self.H_field)

                Eres = self.sim.fdtd.getresult("FoM_monitor", "E")
                self.xg = np.atleast_1d(np.squeeze(np.array(Eres["x"])))
                self.yg = np.atleast_1d(np.squeeze(np.array(Eres["y"])))
                self.zg = np.atleast_1d(np.squeeze(np.array(Eres["z"])))

                try:
                    self.dt = float(np.squeeze(np.array(self.sim.fdtd.getresult("source", "dt"))))
                    self.time_sn = np.asarray(self.sim.fdtd.getresult("source", "time_signal"))
                except Exception:
                    self.dt = 1.0
                    self.time_sn = np.zeros_like(self.src_freqs, dtype=float)
                if self.forward_result_hook is not None:
                    self.forward_result_hook(self)
                break
            except Exception as exc:
                last_error = exc
                log_tail = self.sim._run_log_tail("Forward_run")
                if attempt >= retries - 1:
                    raise RuntimeError(
                        "Forward session run finished without readable monitor results. "
                        f"Missing/failed provider: {self.sim.design_monitor_name} or FoM_monitor. "
                        "The saved FDTD run log often contains the real solver-side cause. "
                        "External solver fallback is disabled. Check GPU availability with "
                        "`resource` and use a GPU with no active jobs. "
                        f"Last Python error: {exc}\n"
                        f"Forward_run_p0.log tail:\n{log_tail}"
                    ) from exc
                print(
                    "Forward session run returned without readable monitor results; "
                    f"retrying in {retry_delay:g}s ({attempt + 1}/{retries - 1}). "
                    f"Last error: {exc}"
                )
                if log_tail:
                    print(f"Forward_run_p0.log tail:\n{log_tail}")
                time.sleep(retry_delay)
                try:
                    self.sim.fdtd.switchtolayout()
                    self.sim.fdtd.load(getattr(self.sim, "_last_run_fsp_path", os.path.abspath("Forward_run.fsp")))
                except Exception:
                    try:
                        self.sim.fdtd.close()
                    except Exception:
                        pass
                    self.sim.fdtd = _open_lumerical_fdtd()
                    self.sim.fdtd.switchtolayout()
                    self.sim.fdtd.load(getattr(self.sim, "_last_run_fsp_path", os.path.abspath("Forward_run.fsp")))
        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.setnamed('design_monitor', 'enabled', False)
        self.sim.fdtd.setnamed('FoM_monitor', 'enabled', False)
        self.sim.fdtd.setnamed('source', 'enabled', False)

        args = []

        self.last_forward_had_nonfinite = False
        for arg in self.objective_arguments:
            field_arg = np.asarray(self.FoM_fields[arg], dtype=np.complex128)
            bad_count = np.count_nonzero(~np.isfinite(field_arg))
            if bad_count:
                self.last_forward_had_nonfinite = True
                print(f"[forward_run] replaced non-finite FoM field component {arg}: {bad_count}")
                field_arg = np.nan_to_num(field_arg, nan=0.0, posinf=0.0, neginf=0.0)
                self.FoM_fields[arg] = field_arg
            args.append(field_arg)
        raw_f0 = [J(*args) for J in self.objective_functions]
        self.last_forward_had_nonfinite = self.last_forward_had_nonfinite or any(
            not np.isfinite(float(np.real(value))) if np.ndim(value) == 0 else False
            for value in raw_f0
        )
        if self.last_forward_had_nonfinite:
            print("[forward_run] candidate marked unstable; optimizer should reject/backtrack this step.")
        self.f0 = raw_f0
        self.f0 = [
            float(np.nan_to_num(np.real(value), nan=0.0, posinf=0.0, neginf=0.0))
            if np.ndim(value) == 0
            else value
            for value in self.f0
        ]
        self.current_state = "FWD"

    def _run_adjoint_with_result_retry(self, source_name):
        """Run one adjoint solve and require a readable design-monitor result.

        A session solve can return normally even when the external engine failed
        to acquire an HPC license. In that case no monitor result exists. The
        forward path retries this condition; adjoint solves must do the same.
        """
        retries = max(1, int(os.environ.get("LUMERICAL_ADJOINT_RESULT_RETRIES", "6")))
        retry_delay = float(
            os.environ.get("LUMERICAL_ADJOINT_RESULT_RETRY_DELAY", "10")
        )
        last_error = None
        for attempt in range(retries):
            self.sim.fdtd.switchtolayout()
            self.sim.fdtd.setnamed(source_name, "enabled", True)
            self.sim.fdtd.setnamed("design_monitor", "enabled", True)
            try:
                self.sim.run(name="Adjoint_run", save=True)
                return self.get_field(
                    self.sim.design_monitor_name,
                    H_field=False,
                    check_design_alignment=True,
                )
            except Exception as exc:
                last_error = exc
                log_tail = self.sim._run_log_tail("Adjoint_run")
                if attempt >= retries - 1:
                    raise RuntimeError(
                        "Adjoint session run finished without readable design-monitor "
                        f"results after {retries} attempt(s). "
                        "The saved FDTD log often contains the solver-side cause. "
                        f"Last Python error: {exc}\n"
                        f"Adjoint_run_p0.log tail:\n{log_tail}"
                    ) from exc
                print(
                    "Adjoint session run returned without readable monitor results; "
                    f"retrying in {retry_delay:g}s ({attempt + 1}/{retries - 1}). "
                    f"Last error: {exc}"
                )
                if log_tail:
                    print(f"Adjoint_run_p0.log tail:\n{log_tail}")
                time.sleep(retry_delay)
                fsp_path = getattr(
                    self.sim,
                    "_last_run_fsp_path",
                    os.path.abspath("Adjoint_run.fsp"),
                )
                try:
                    self.sim.fdtd.switchtolayout()
                    self.sim.fdtd.load(fsp_path)
                except Exception:
                    try:
                        self.sim.fdtd.close()
                    except Exception:
                        pass
                    self.sim.fdtd = _open_lumerical_fdtd()
                    self.sim.fdtd.switchtolayout()
                    self.sim.fdtd.load(fsp_path)
        raise last_error


    """ Adjoint source update"""
    def update_adjoint_dipole(self, dJ):
        self._adjoint_source_inserted = False
        self.adj_wl = self.sim.src_wl
        self.adj_bw = self.sim.src_bw

        dJEx_r = dJ[0] if len(dJ) > 0 else None
        dJEy_r = dJ[1] if len(dJ) > 1 else None
        dJEz_r = dJ[2] if len(dJ) > 2 else None
        dJHx_r = dJ[3] if len(dJ) > 3 else None
        dJHy_r = dJ[4] if len(dJ) > 4 else None
        dJHz_r = dJ[5] if len(dJ) > 5 else None
        if self.src_freqs is None or self.src_spectrum is None:
            raise RuntimeError("Source spectrum not initialized. Run forward first.")

        amps_sq = 0
        for arr in [dJEx_r, dJEy_r, dJEz_r, dJHx_r, dJHy_r, dJHz_r]:
            if arr is not None:
                arr = np.nan_to_num(np.asarray(arr, dtype=np.complex128), nan=0.0, posinf=0.0, neginf=0.0)
                amps_sq += np.abs(arr) ** 2

        max_amp = np.max(np.sqrt(amps_sq))
        if not np.isfinite(max_amp) or max_amp == 0:
            print("[update_adjoint_dipole] all-zero dJ, no source added.")
            return

        if np.sum(self.FoM_size) <= 0:
            raise ValueError("Point/dipole adjoint path is not updated here. Use planar FoM monitor.")

        nz_dims, n_axis = self.sim.dimension_check(self.FoM_size)
        n_ax=self.sim.kw_to_idx[n_axis]
        self.nor_dim = n_axis
        axis_map = {"x": "x-axis", "y": "y-axis", "z": "z-axis"}

        shape3 = (self.xg.size, self.yg.size, self.zg.size)

        def source_step_size():
            if shape3[0] > 1 and (np.max(self.xg) - np.min(self.xg)) != 0:
                return (np.max(self.xg) - np.min(self.xg)) / (shape3[0] - 1)
            if shape3[1] > 1 and (np.max(self.yg) - np.min(self.yg)) != 0:
                return (np.max(self.yg) - np.min(self.yg)) / (shape3[1] - 1)
            if shape3[2] > 1 and (np.max(self.zg) - np.min(self.zg)) != 0:
                return (np.max(self.zg) - np.min(self.zg)) / (shape3[2] - 1)
            return 1.0

        step_size = source_step_size()

        def scaled_slice(arr, iidx):
            omega_i = 2 * np.pi * self.sim.c / self.adj_wl[iidx]
            scale_i = 1j * omega_i * step_size
            if arr is None:
                return np.zeros(shape3, dtype=np.complex128)
            arr_i = np.array(arr[:, :, :, iidx], dtype=np.complex128)
            arr_i = np.nan_to_num(arr_i, nan=0.0, posinf=0.0, neginf=0.0)
            return arr_i * scale_i

        if self.broadband_adjoint and len(self.adj_wl) > 1:
            self.sim.fdtd.addimportedsource()
            self.sim.fdtd.set("name", "adjoint_source")
            self.sim.fdtd.set("injection axis", axis_map[self.nor_dim])
            self.sim.fdtd.set("direction", "forward" if self.forward_adj else "backward")
            self.sim.fdtd.set("x", self.FoM_center[0])
            self.sim.fdtd.set("y", self.FoM_center[1])
            self.sim.fdtd.set("z", self.FoM_center[2])
            self.sim.fdtd.putv(
                "field",
                self.sim.fdtd.rectilineardataset("field", self.xg, self.yg, self.zg)
            )

            Ex_all = np.zeros(shape3 + (len(self.adj_wl),), dtype=np.complex128)
            Ey_all = np.zeros_like(Ex_all)
            Ez_all = np.zeros_like(Ex_all)
            Hx_all = Hy_all = Hz_all = None
            use_imported_H = self.H_field and (dJHx_r is not None or dJHy_r is not None or dJHz_r is not None)
            if use_imported_H:
                Hx_all = np.zeros_like(Ex_all)
                Hy_all = np.zeros_like(Ex_all)
                Hz_all = np.zeros_like(Ex_all)

            for iidx in range(len(self.adj_wl)):
                Ex_all[:, :, :, iidx] = scaled_slice(dJEx_r, iidx)
                Ey_all[:, :, :, iidx] = scaled_slice(dJEy_r, iidx)
                Ez_all[:, :, :, iidx] = scaled_slice(dJEz_r, iidx)
                if use_imported_H:
                    Hx_all[:, :, :, iidx] = scaled_slice(dJHx_r, iidx)
                    Hy_all[:, :, :, iidx] = scaled_slice(dJHy_r, iidx)
                    Hz_all[:, :, :, iidx] = scaled_slice(dJHz_r, iidx)

            if self.tangential_only:
                if self.nor_dim == "x":
                    Ex_all[:] = 0
                elif self.nor_dim == "y":
                    Ey_all[:] = 0
                elif self.nor_dim == "z":
                    Ez_all[:] = 0

            self.sim.fdtd.putv("Ex", Ex_all)
            self.sim.fdtd.putv("Ey", Ey_all)
            self.sim.fdtd.putv("Ez", Ez_all)
            self.sim.fdtd.putv("adj_lambdas", self.adj_wl)
            self.sim.fdtd.putv("adj_freqs", self.sim.c / self.adj_wl)
            self.sim.fdtd.eval('field.addparameter("lambda",adj_lambdas,"f",adj_freqs);')
            self.sim.fdtd.eval('field.addattribute("E",Ex,Ey,Ez);')
            if use_imported_H:
                self.sim.fdtd.putv("Hx", Hx_all)
                self.sim.fdtd.putv("Hy", Hy_all)
                self.sim.fdtd.putv("Hz", Hz_all)
                self.sim.fdtd.eval('field.addattribute("H",Hx,Hy,Hz);')
            self.sim.fdtd.eval('importdataset(field);')
            self.sim.fdtd.setnamed("adjoint_source", "enabled", False)
            self._adjoint_source_inserted = True
            return

        inserted_any = False
        for iidx in range(len(self.adj_wl)):
            self.sim.fdtd.addimportedsource()
            self.sim.fdtd.set("name", f"adjoint_source_{iidx}")
            self.sim.fdtd.set("injection axis", axis_map[self.nor_dim])
            self.sim.fdtd.set("direction", "forward" if self.forward_adj else "backward")
            self.sim.fdtd.set("x", self.FoM_center[0])
            self.sim.fdtd.set("y", self.FoM_center[1])
            self.sim.fdtd.set("z", self.FoM_center[2])

            self.sim.fdtd.putv(
                "field",
                self.sim.fdtd.rectilineardataset("field", self.xg, self.yg, self.zg)
            )
            freq_cur = self.sim.c / self.adj_wl[iidx]
            # nearest frequency index
            idx = int(np.argmin(np.abs(self.src_freqs - freq_cur)))
            S_i = self.src_spectrum[idx]
            idx0 = int(np.argmin(np.abs(self.src_freqs - self.sim.c /  np.max(self.adj_wl))))
            idx1 = int(np.argmin(np.abs(self.src_freqs - self.sim.c /  np.mean(self.adj_wl))))
            idx2 = int(np.argmin(np.abs(self.src_freqs - self.sim.c /  np.min(self.adj_wl))))
            S_ref=np.max(abs(self.src_spectrum))
            S_ref0=(abs(self.src_spectrum[idx0]))
            S_ref1=(abs(self.src_spectrum[idx1]))
            S_ref2=(abs(self.src_spectrum[idx2]))

            S_mean=(S_ref0+S_ref1+S_ref2)/3

            # scale_i=1j*omega_i*(S_ref**2/(S_mean*np.abs(S_i)))**0.5
            # scale_i=1j*omega_i*(S_ref/(S_mean))**0.5
            def get_slice(arr):
                return scaled_slice(arr, iidx)

            # E source
            Ex_i = get_slice((dJEx_r))
            Ey_i = get_slice((dJEy_r))
            Ez_i = get_slice((dJEz_r))

            if self.tangential_only:
                if self.nor_dim == "x":
                    Ex_i[:] = 0
                elif self.nor_dim == "y":
                    Ey_i[:] = 0
                elif self.nor_dim == "z":
                    Ez_i[:] = 0

            self.sim.fdtd.putv("Ex", Ex_i)
            self.sim.fdtd.putv("Ey", Ey_i)
            self.sim.fdtd.putv("Ez", Ez_i)

            # # H source
            use_imported_H = False
            if self.H_field and (dJHx_r is not None or dJHy_r is not None or dJHz_r is not None):
                Hx_i = get_slice(dJHx_r)
                Hy_i = get_slice(dJHy_r)
                Hz_i = get_slice(dJHz_r)
                use_imported_H = True



            self.sim.fdtd.eval(f'field.addparameter("lambda",{self.adj_wl[iidx]});')
            self.sim.fdtd.eval('field.addattribute("E",Ex,Ey,Ez);')

            if use_imported_H:
                self.sim.fdtd.putv("Hx", Hx_i)
                self.sim.fdtd.putv("Hy", Hy_i)
                self.sim.fdtd.putv("Hz", Hz_i)
                self.sim.fdtd.eval('field.addattribute("H",Hx,Hy,Hz);')

            self.sim.fdtd.eval('importdataset(field);')
            # self.sim.fdtd.set('optimize for short pulse', False)
            self.sim.fdtd.setnamed(f"adjoint_source_{iidx}", "enabled", False)
            inserted_any = True
        self._adjoint_source_inserted = inserted_any

    def _dJ_fields(self, J, args):
        """dJ/dE (and dJ/dH) of ONE objective on the FoM plane -- the adjoint
        source amplitude for that objective."""
        ncomp = 6 if self.H_field else 3
        Fields = [np.zeros_like(self.FoM_fields[c], dtype=np.complex128)
                  for c in range(ncomp)]
        for local_i, field_comp in enumerate(self.objective_arguments):
            dJ = np.array(jacobian(J, argnum=local_i)(*args), dtype=np.complex128)
            Fields[field_comp] += dJ
        return Fields

    def _run_adjoint_for_fields(self, Fields):
        """Insert the adjoint source for one dJ/dE set, run it, return the
        adjoint fields. Wavelength handling is untouched: broadband_adjoint
        still decides between one multi-wavelength source and the per-wavelength
        loop, so wavelength remains a sub-level of the objective."""
        self.update_adjoint_dipole(Fields)
        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.setnamed('FoM_monitor', 'enabled', False)
        adjoint_fields = np.zeros_like(self.forward_fields, dtype=np.complex128)
        if not getattr(self, "_adjoint_source_inserted", False):
            self.sim.fdtd.setnamed('design_monitor', 'enabled', False)
            print("[adjoint_dipole_run] zero adjoint source; using zero adjoint fields.")
            return adjoint_fields
        self.sim.fdtd.setnamed('design_monitor', 'enabled', True)
        if self.broadband_adjoint and len(self.adj_wl) > 1:
            adjoint_fields = self._run_adjoint_with_result_retry(source_name="adjoint_source")
            self.sim.fdtd.switchtolayout()
            self.sim.fdtd.eval('select("adjoint_source"); delete;')
        else:
            for iidx in range(len(self.adj_wl)):
                res = self._run_adjoint_with_result_retry(source_name=f"adjoint_source_{iidx}")
                adjoint_fields[:, :, :, :, iidx] = res[:, :, :, :, iidx]
                self.sim.fdtd.switchtolayout()
                self.sim.fdtd.eval(f'select("adjoint_source_{iidx}"); delete;')
        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.setnamed('design_monitor', 'enabled', False)
        self.sim.fdtd.setnamed('FoM_monitor', 'enabled', False)
        return adjoint_fields

    def adjoint_dipole_run_incoherent(self):
        """One adjoint run PER objective, in the order J1_lam*, J2_lam*, ...

        Each objective's adjoint fields are kept separately so calculate_gradient
        can form fwd x adj_j for each and only then add them up. The forward run
        is NOT repeated -- self.forward_fields and self.FoM_fields from the single
        forward are reused for every objective.
        """
        start = time.time()
        self.sim.fdtd.switchtolayout()
        args = [self.FoM_fields[k] for k in self.objective_arguments]
        nJ, nlam = len(self.objective_functions), max(len(self.sim.src_wl), 1)
        print(f"[incoherent] {nJ} objective(s) x {nlam} wavelength(s); "
              f"order J1_lam1..J1_lam{nlam}, ..., J{nJ}_lam{nlam}")
        self.adjoint_fields_per_J = []
        self.f0_per_J = [float(np.real(np.sum(J(*args)))) for J in self.objective_functions]
        for jobj, J in enumerate(self.objective_functions):
            t0 = time.time()
            Fields = self._dJ_fields(J, args)
            self.adjoint_fields_per_J.append(self._run_adjoint_for_fields(Fields))
            print(f"[incoherent] J{jobj + 1}: f0={self.f0_per_J[jobj]:.6e}, "
                  f"adjoint {time.time() - t0:.2f} s")
        # keep the summed field so anything downstream that still reads
        # self.adjoint_fields sees the coherent-equivalent result
        self.adjoint_fields = np.sum(np.asarray(self.adjoint_fields_per_J), axis=0)
        self.current_state = "Adj"
        print(f"[incoherent] total adjoint time: {time.time() - start:.2f} s")

    """ Adjoint run"""
    def adjoint_dipole_run(self):
        if getattr(self, "Incoherent", False):
            return self.adjoint_dipole_run_incoherent()
        start = time.time()
        self.sim.fdtd.switchtolayout()
        args = [self.FoM_fields[k] for k in self.objective_arguments]

        ncomp = 6 if self.H_field else 3
        Fields = [None] * ncomp
        for comp in range(ncomp):
            Fields[comp] = np.zeros_like(self.FoM_fields[comp], dtype=np.complex128)

        for jobj, J in enumerate(self.objective_functions):
            for local_i, field_comp in enumerate(self.objective_arguments):
                dJ_darg_i = jacobian(J, argnum=local_i)(*args)
                dJ_darg_i = np.array(dJ_darg_i, dtype=np.complex128)
                Fields[field_comp] += dJ_darg_i

        Jacob = time.time()

        self.update_adjoint_dipole(Fields)

        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.setnamed('FoM_monitor', 'enabled', False)
        d_arr = time.time()
        self.adjoint_fields = np.zeros_like(self.forward_fields, dtype=np.complex128)
        if not getattr(self, "_adjoint_source_inserted", False):
            self.sim.fdtd.setnamed('design_monitor', 'enabled', False)
            self.current_state = "Adj"
            print("[adjoint_dipole_run] zero adjoint source; using zero adjoint fields.")
            print(f"Jacobian time: {Jacob - start:.2f} seconds")
            print(f"Source insertion time: {d_arr - Jacob:.2f} seconds")
            print("Adjoint run time: 0.00 seconds")
            return

        self.sim.fdtd.setnamed('design_monitor', 'enabled', True)

        if self.broadband_adjoint and len(self.adj_wl) > 1:
            adj_res = self._run_adjoint_with_result_retry(
                source_name="adjoint_source",
            )
            self.adjoint_fields = adj_res
            self.sim.fdtd.switchtolayout()
            self.sim.fdtd.eval('select("adjoint_source"); delete;')
        else:
            for iidx in range(len(self.adj_wl)):
                adj_res = self._run_adjoint_with_result_retry(
                    source_name=f"adjoint_source_{iidx}",
                )
                self.adjoint_fields[:, :, :, :, iidx] = adj_res[:, :, :, :, iidx]
                self.sim.fdtd.switchtolayout()
                self.sim.fdtd.eval(f'select("adjoint_source_{iidx}"); delete;')
        adj = time.time()

        self.sim.fdtd.switchtolayout()
        self.sim.fdtd.setnamed('design_monitor', 'enabled', False)
        self.sim.fdtd.setnamed('FoM_monitor', 'enabled', False)
        self.current_state = "Adj"

        print(f"Jacobian time: {Jacob - start:.2f} seconds")
        print(f"Source insertion time: {d_arr - Jacob:.2f} seconds")
        print(f"Adjoint run time: {adj - d_arr:.2f} seconds")



    """Gradient calculation"""
    def calculate_gradient(self, debug_mode: bool = False):
        # Incoherent: form fwd x adj_j for EVERY objective and only then sum.
        # Going through the same code path per objective (rather than adding the
        # adjoint fields first) is what makes the decomposition meaningful: the
        # per-objective gradients are kept in self.gradient_per_J so the caller
        # can see which objective is driving the design.
        if getattr(self, "Incoherent", False) and self.adjoint_fields_per_J:
            saved = self.adjoint_fields
            self.gradient_per_J, total = [], None
            try:
                for jobj, adj_j in enumerate(self.adjoint_fields_per_J):
                    self.adjoint_fields = adj_j
                    g_j = np.asarray(self._calculate_gradient_single(debug_mode), dtype=float)
                    self.gradient_per_J.append(g_j)
                    total = g_j.copy() if total is None else total + g_j
                    print(f"[incoherent] J{jobj + 1} |grad|={np.linalg.norm(g_j):.6e}")
            finally:
                self.adjoint_fields = saved
            self.gradient = np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"[incoherent] summed |grad|={np.linalg.norm(self.gradient):.6e} "
                  f"over {len(self.gradient_per_J)} objective(s)")
            self.current_state = "INIT"
            return self.gradient.flatten()
        return self._calculate_gradient_single(debug_mode)

    def _calculate_gradient_single(self, debug_mode: bool =False):
        fwd = np.asarray(self.forward_fields, dtype=np.complex128)   # (3, Nx, Ny, Nz, Nf) raw node/corner
        adj = np.asarray(self.adjoint_fields, dtype=np.complex128)   # (3, Nx, Ny, Nz, Nf) raw node/corner
        bad_fwd = np.count_nonzero(~np.isfinite(fwd))
        bad_adj = np.count_nonzero(~np.isfinite(adj))
        if bad_fwd or bad_adj:
            print(f"[gradient_check] replaced non-finite fields: forward={bad_fwd}, adjoint={bad_adj}")
            fwd = np.nan_to_num(fwd, nan=0.0, posinf=0.0, neginf=0.0)
            adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

        if fwd.shape != adj.shape:
            raise RuntimeError(f"Forward/adjoint shape mismatch: fwd={fwd.shape}, adj={adj.shape}")

        Nx, Ny, Nz = self.sim.design_grids
        Nf = fwd.shape[-1]

        expected_shape = (3, Nx, Ny, Nz, Nf)
        if fwd.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected field shape. Expected {expected_shape}, got {fwd.shape}"
            )
        
        import scipy as sp
        # eff_vol=(Nx/(Nx-1))*(Ny/(Ny-1))*(Nz/(Nz-1))
        # voxel_vol = self.sim.design_dx * self.sim.design_dy #* self.sim.design_dz
        # scale_f =sp.constants.epsilon_0 #*voxel_vol / (self.sim.unit ** 2)
        scale_f =2*1e-9#self.sim.design_dx#voxel_vol / (self.sim.unit ** 2)
        # print("res info:", self.sim.design_dx, self.sim.design_dy, self.sim.design_dz)

        # print("max|adj_x| =", np.max(np.abs(adj[0])))
        # print("max|adj_y| =", np.max(np.abs(adj[1])))
        # print("max|adj_z| =", np.max(np.abs(adj[2])))

        # print("max|fwd_x| =", np.max(np.abs(fwd[0])))
        # print("max|fwd_y| =", np.max(np.abs(fwd[1])))
        # print("max|fwd_z| =", np.max(np.abs(fwd[2])))

        # print("max|Re(adj_x*fwd_x)| =", np.max(np.abs(np.real(adj[0] * fwd[0]))))
        # print("max|Re(adj_y*fwd_y)| =", np.max(np.abs(np.real(adj[1] * fwd[1]))))
        # print("max|Re(adj_z*fwd_z)| =", np.max(np.abs(np.real(adj[2] * fwd[2]))))
        self.SSSSSSSSSSF=scale_f

        # # 1) field interaction first on raw node/corner grid
        if self.dedr_spectrum is not None and self.dedr_spectrum.shape[0] == Nf:
            dedr_x = self.dedr_spectrum[:, 0].reshape((1, 1, 1, Nf))
            dedr_y = self.dedr_spectrum[:, 1].reshape((1, 1, 1, Nf))
            dedr_z = self.dedr_spectrum[:, 2].reshape((1, 1, 1, Nf))
        else:
            dedr_x = self.dedr[0]
            dedr_y = self.dedr[1]
            dedr_z = self.dedr[2]

        gx_eff = np.real(dedr_x * adj[0] * fwd[0])
        gy_eff = np.real(dedr_y * adj[1] * fwd[1])
        gz_eff = np.real(dedr_z * adj[2] * fwd[2])
        gx_eff = np.nan_to_num(gx_eff, nan=0.0, posinf=0.0, neginf=0.0)
        gy_eff = np.nan_to_num(gy_eff, nan=0.0, posinf=0.0, neginf=0.0)
        gz_eff = np.nan_to_num(gz_eff, nan=0.0, posinf=0.0, neginf=0.0)
        gx_eff, gy_eff, gz_eff =self.assemble_boundary_tangent_gradient_mask(gx_eff, gy_eff, gz_eff)

        gx_node= self._pair_gather_backward(gx_eff, axis=0)
        gy_node= self._pair_gather_backward(gy_eff, axis=1)
        gz_node= self._pair_gather_backward(gz_eff, axis=2)
        # g_direct=self.assemble_boundary_tangent_gradient_mask(gx_node, gy_node, gz_node)
        # self.gssssx=gx_node
        # self.gssssy=gy_node
        # self.gssssz=gz_node
        grad_total = (gx_node + gy_node + gz_node) * scale_f
        # grad_total = g_direct*scale_f
        self.current_state = "INIT"
        dJ_dus = np.transpose(grad_total, (3, 0, 1, 2))  # (Nf, Nx, Ny, Nz)
        print("[gradient_check] grad_3d_by_f.shape =", dJ_dus.shape)

        if debug_mode:
            self.gradient = dJ_dus#.fletten()
        else:
            if len(self.objective_functions) > 1 and not getattr(self, "Incoherent", False):
                # coherent multi-objective: msopt's historical Minimax combination
                self.gradient = Opt_MS2.Minimax(self.f0, dJ_dus)
            else:
                # incoherent: this call handles ONE objective, so the only sum left
                # is over frequency. The objective sum happens in calculate_gradient.
                self.gradient = np.sum(dJ_dus, axis=0).flatten()
        self.gradient = np.nan_to_num(self.gradient, nan=0.0, posinf=0.0, neginf=0.0)
        print("[gradient_check] grad_3d_by_f.shape =", self.gradient.shape)
        return self.gradient.flatten()

    
    def Born_validity(self, dJ_du):
        outlier_th= np.percentile(np.abs(dJ_du), 99.9)
        dJ_du= np.where(np.abs(dJ_du)> outlier_th, 0, dJ_du)
        return dJ_du

    def fd_ad(
            self, 
            v=None, 
            mapping=None, 
            Load_FD : bool = False,
            step_size=0.01,
            fd_stride=16,
        ):
        import matplotlib.pyplot as plt
        Nx, Ny, Nz = self.sim.design_grids

        def build_region_masks(Nx, Ny, Nz):
            ii, jj, kk = np.meshgrid(
                np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij"
            )

            on_xmin = (ii == 0)
            on_xmax = (ii == Nx - 1)
            on_ymin = (jj == 0)
            on_ymax = (jj == Ny - 1)
            on_zmin = (kk == 0)
            on_zmax = (kk == Nz - 1)

            on_xb = on_xmin | on_xmax
            on_yb = on_ymin | on_ymax
            on_zb = on_zmin | on_zmax

            n_bnd = on_xb.astype(int) + on_yb.astype(int) + on_zb.astype(int)

            masks_3d = {
                "all": np.ones((Nx, Ny, Nz), dtype=bool),
                "interior": (n_bnd == 0),
                "face_only": (n_bnd == 1),
                "edge_only": (n_bnd == 2),
                "corner_only": (n_bnd == 3),
                "boundary": (n_bnd >= 1),
                "x_face_only": on_xb & (~on_yb) & (~on_zb),
                "y_face_only": on_yb & (~on_xb) & (~on_zb),
                "z_face_only": on_zb & (~on_xb) & (~on_yb),
                "xy_edge_only": on_xb & on_yb & (~on_zb),
                "xz_edge_only": on_xb & on_zb & (~on_yb),
                "yz_edge_only": on_yb & on_zb & (~on_xb),
            }

            masks_2d = {k: v.reshape(-1) for k, v in masks_3d.items()}
            return masks_3d, masks_2d
        
        def build_region_masks_2d(Nx, Ny):
            ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

            on_xmin = (ii == 0)
            on_xmax = (ii == Nx - 1)
            on_ymin = (jj == 0)
            on_ymax = (jj == Ny - 1)

            on_xb = on_xmin | on_xmax
            on_yb = on_ymin | on_ymax
            boundary = on_xb | on_yb

            ring1 = (
                ((ii == 1) | (ii == Nx - 2) | (jj == 1) | (jj == Ny - 2))
                & (~boundary)
            )

            core_xy = (
                (ii >= 2) & (ii <= Nx - 3) &
                (jj >= 2) & (jj <= Ny - 3)
            )

            masks_2d = {
                "all": np.ones((Nx, Ny), dtype=bool),
                "boundary": boundary,
                "ring1": ring1,
                "core_xy": core_xy,

                "xmin_face_only": on_xmin & (~on_yb),
                "xmax_face_only": on_xmax & (~on_yb),
                "ymin_face_only": on_ymin & (~on_xb),
                "ymax_face_only": on_ymax & (~on_xb),
            }

            masks_flat = {k: v.reshape(-1) for k, v in masks_2d.items()}
            return masks_2d, masks_flat
        
        def compute_stats(fd_flat, ad_flat, mask_flat, rel_th=0.05):
            if np.count_nonzero(mask_flat) == 0:
                return None

            fd_sel = fd_flat[mask_flat]
            ad_sel = ad_flat[mask_flat]

            if fd_sel.size == 0:
                return None

            abs_th = rel_th * np.max(np.abs(fd_sel)) if np.max(np.abs(fd_sel)) > 0 else 0.0
            valid = np.abs(fd_sel) > abs_th

            fd_v = fd_sel[valid]
            ad_v = ad_sel[valid]

            if fd_v.size == 0:
                return None

            nmse_num = np.sum((fd_v - ad_v) ** 2)
            nmse_den = np.sum(fd_v ** 2)

            nmse = None
            if nmse_den > 0:
                nmse = 1.0 - nmse_num / nmse_den

            good_ad = np.abs(ad_v) > 1e-30
            scales = fd_v[good_ad] / ad_v[good_ad] if np.any(good_ad) else np.array([])

            corr = None
            if fd_v.size >= 2 and np.std(fd_v) > 0 and np.std(ad_v) > 0:
                corr = np.corrcoef(fd_v, ad_v)[0, 1]

            sign_match = np.mean(np.sign(fd_v) == np.sign(ad_v))

            return {
                "count_total": int(fd_sel.size),
                "count_valid": int(fd_v.size),
                "nmse": nmse,
                "median_scale": float(np.median(scales)) if scales.size else None,
                "mean_scale": float(np.mean(scales)) if scales.size else None,
                "corr": float(corr) if corr is not None else None,
                "sign_match": float(sign_match),
                "abs_th": float(abs_th),
            }
        Nf = len(self.objective_functions)

        if mapping is None:
            X = np.asarray(v, dtype=float).copy()
        else:
            X=mapping(v, 1.0)

        self.sim.fdtd.switchtolayout()
        self.sim.update_design_density(density=X)
        self.forward_run()
        self.adjoint_dipole_run()
        self.calculate_gradient(debug_mode=True)
        f_init=self.f0.copy()
        dJ_du_init = self.gradient.sum(axis=3)
        np.savetxt("dJ_du.txt", self.gradient.reshape(self.gradient.shape[0], -1))

        idx_list = np.arange(0, Nx * Ny, fd_stride)
        NMSE = [0 for _ in range(Nf)]
        Variance = [0 for _ in range(Nf)]
        if Load_FD:
            fds = np.loadtxt(f"FD_LUM_{step_size}.txt")
            fds = np.reshape(fds, (Nf, Nx * Ny))
            valid_counts = [0 for _ in range(Nf)]
            scales_all = [[] for _ in range(Nf)]

            # rel_th = 0.05
            # abs_th = [rel_th * np.max(np.abs(fds[i])) for i in range(Nf)]
            rel_ths = [0.05, 0.05, 0.05]
            abs_th = [rel_ths[i] * np.max(np.abs(fds[i])) for i in range(Nf)]

            for idx in idx_list:
                for i in range(Nf):
                    fd_val = fds[i][idx]
                    ad_val = dJ_du_init[i, :, :].flatten()[idx] * step_size

                    if np.abs(fd_val) < abs_th[i]:
                        continue
                    if np.abs(ad_val) < 1e-30:
                        continue

                    NMSE[i] += (fd_val - ad_val) ** 2
                    Variance[i] += fd_val ** 2
                    scales_all[i].append(fd_val / ad_val)
                    valid_counts[i] += 1

            for i in range(Nf):
                print(f"\nChannel {i}")
                if Variance[i] > 0 and valid_counts[i] > 0:
                    NMSE[i] = 1 - (NMSE[i] / Variance[i])**2
                    print(f"valid count   : {valid_counts[i]}")
                    print(f"NMSE accuracy : {NMSE[i]}")
                    print(f"median scale  : {np.median(scales_all[i])}")
                    print(f"mean scale    : {np.mean(scales_all[i])}")
                else:
                    NMSE[i] = None
                    print("No valid samples after thresholding.")
                fd_flat = fds[i].reshape(-1)
                ad_flat = (dJ_du_init[i] * step_size).reshape(-1)

                mask = np.abs(fd_flat) > 0.05 * np.max(np.abs(fd_flat))
                if np.any(mask):
                    corr = np.corrcoef(fd_flat[mask], ad_flat[mask])[0, 1]
                    print(f"corr         : {corr}")

                mask_corr = np.abs(fd_flat) > abs_th[i]
                if np.any(mask_corr):
                    sign_match = np.mean(np.sign(fd_flat[mask_corr]) == np.sign(ad_flat[mask_corr]))
                    print(f"sign match   : {sign_match}")
                # region-wise diagnostics
                # _, region_masks_2d = build_region_masks(Nx, Ny, Nz)
                _, region_masks_2d = build_region_masks_2d(Nx, Ny)

                print("---- region wise ----")
                # for region_name in [
                #     "all",
                #     "interior",
                #     "boundary",
                #     "face_only",
                #     "edge_only",
                #     "corner_only",
                #     "x_face_only",
                #     "y_face_only",
                #     "z_face_only",
                #     "xy_edge_only",
                #     "xz_edge_only",
                #     "yz_edge_only",
                # ]:
                # for region_name in [
                #     "all",
                #     "interior",
                #     "boundary",
                #     "face_only",
                #     "corner_only",
                #     "x_face_only",
                #     "y_face_only",
                # ]:
                for region_name in [
                    "all",
                    "boundary",
                    "ring1",
                    "core_xy",
                    "xmin_face_only",
                    "xmax_face_only",
                    "ymin_face_only",
                    "ymax_face_only",
                ]:
                    stats = compute_stats(
                        fd_flat=fd_flat,
                        ad_flat=ad_flat,
                        mask_flat=region_masks_2d[region_name],
                        rel_th=0.7,
                    )

                    if stats is None:
                        print(f"{region_name:12s}: no data")
                        continue

                    print(
                        f"{region_name:12s}: "
                        f"valid={stats['count_valid']:4d}, "
                        f"NMSE={stats['nmse'] if stats['nmse'] is not None else None}, "
                        f"median_scale={stats['median_scale']}, "
                        f"mean_scale={stats['mean_scale']}, "
                        f"corr={stats['corr']}, "
                        f"sign={stats['sign_match']}"
                    )

        else:
            fds = np.zeros((Nf, Nx*Ny))
            v0 = v0 = np.asarray(v, dtype=float).copy().reshape(-1)

            for idx in idx_list:
                print(f"[FD] idx {idx}/{Nx*Ny}")

                v0[idx] += step_size
                if mapping is None:
                    x1=v0.copy()
                else:
                    x1=mapping(v0, 1.0)

                self.sim.fdtd.switchtolayout()
                self.sim.update_design_density(density=x1)
                self.forward_run()

                for i in range(Nf):
                    fds[i][idx] += self.f0[i] - f_init[i]
                    ad_val = dJ_du_init[i, :, :].flatten()[idx] * step_size
                    fd_val = fds[i][idx]

                    NMSE[i] += (fd_val - ad_val) ** 2
                    Variance[i] += fd_val ** 2

                    if np.abs(ad_val) > 1e-30:
                        print(f"  ch {i} scale: {fd_val / ad_val}")

                v0[idx] -= step_size

            for i in range(Nf):
                if Variance[i] > 0:
                    NMSE[i] = 1 - (NMSE[i] / Variance[i])**2
                else:
                    NMSE[i] = None
            self.sim.fdtd.switchtolayout()
            self.sim.update_design_density(density=X)
            np.savetxt(f"FD_LUM_{step_size}.txt", fds)


        norm = np.max(np.abs(fds))
        norm2 = np.max(np.abs(dJ_du_init))
        for i in range(Nf):
            plt.subplot(2, Nf, i + 1)
            plt.imshow(np.asarray(fds[i]).reshape(Nx, Ny), cmap="seismic", vmin=-norm, vmax=norm)
            plt.colorbar()
            plt.axis("off")
            title_txt = f"NMSE={round(NMSE[i]*100, 2)}%" if NMSE[i] is not None else "NMSE=None"
            plt.title(title_txt)

            plt.subplot(2, Nf, i + 1 + Nf)
            plt.imshow(dJ_du_init[i], cmap="seismic", vmin=-norm2, vmax=norm2)
            plt.colorbar()
            plt.axis("off")

        plt.savefig(f"dJ_du_Lum.png")
        plt.cla()
        plt.clf()
        plt.close()


        # index_idx = (Nx // 2, Ny // 2, Nz // 2)
        index_idx = (Nx // 2, 0, 0)

        print("\n=== index monitor check ===")
        self.check_design_index_monitor(interior_only=False)

        print("\n=== single-node index stencil ===")
        index_debug_result = self.debug_single_node_index_stencil(
            density=X,
            delta_rho=step_size,
            idx=index_idx,
            thresh=1e-12,
        )

        def save_index_debug_plots(index_debug_result, sim, prefix="idx_debug"):
            ix, iy, iz = index_debug_result["idx"]

            x = np.asarray(sim.design_x)
            y = np.asarray(sim.design_y)
            z = np.asarray(sim.design_z)

            comps = {
                "x": index_debug_result["delta_read"]["nx"],
                "y": index_debug_result["delta_read"]["ny"],
                "z": index_debug_result["delta_read"]["nz"],
            }

            # target 값 (중앙 기준)
            target_all = index_debug_result["delta_target"]

            fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)

            for row, (comp, arr) in enumerate(comps.items()):
                vmax = np.max(np.abs(arr))
                if vmax == 0:
                    vmax = 1e-30

                target = target_all[f"n{comp}"][ix, iy, iz]
                read_val = arr[ix, iy, iz]
                ratio = read_val / target if target != 0 else 0.0

                txt = (
                    f"input Δn_{comp}: {target:.5f}\n"
                    f"updated Δn_{comp}: {read_val:.5f}\n"
                    f"scale: {ratio:.5f}"
                )

                # === xy ===
                im0 = axes[row, 0].imshow(
                    arr[:, :, iz].T,
                    cmap="seismic",
                    vmin=-vmax,
                    vmax=vmax,
                    origin="lower",
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    aspect="equal",
                )
                axes[row, 0].set_title(f"{comp}: xy @ z={iz}")
                axes[row, 0].text(
                    0.02, 0.98, txt,
                    transform=axes[row, 0].transAxes,
                    ha="left", va="top",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7)
                )
                fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

                # === yz ===
                im1 = axes[row, 1].imshow(
                    arr[ix, :, :].T,
                    cmap="seismic",
                    vmin=-vmax,
                    vmax=vmax,
                    origin="lower",
                    extent=[y.min(), y.max(), z.min(), z.max()],
                    aspect="equal",
                )
                axes[row, 1].set_title(f"{comp}: yz @ x={ix}")
                fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

                # === xz ===
                im2 = axes[row, 2].imshow(
                    arr[:, iy, :].T,
                    cmap="seismic",
                    vmin=-vmax,
                    vmax=vmax,
                    origin="lower",
                    extent=[x.min(), x.max(), z.min(), z.max()],
                    aspect="equal",
                )
                axes[row, 2].set_title(f"{comp}: xz @ y={iy}")
                fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

            fig.savefig(f"{prefix}_all.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

        def save_index_debug_plots_for_key_points(sim, density, delta_rho=0.01, prefix="idx_debug"):
            """
            Save index-debug plots for 4 representative locations:

            1. corner : (0, 0, 0)
            2. edge   : (Nx//2, 0, 0)
            3. center : (Nx//2, Ny//2, Nz//2)
            4. ring1  : (Nx-2, Ny-2, Nz-2)
            5. ring2  : (Nx//2, Ny//2, Nz-2)
            6. ring3  : (Nx//2, Ny-2, Nz-2)
            7. face   : (Nx//2, Ny//2, Nz-1)

            Requires:
            - sim.debug_single_node_index_stencil(...)
            - save_index_debug_plots(index_debug_result, sim, prefix=...)
            """
            Nx, Ny, Nz = sim.design_grids

            probe_points = {
                "corner": (0, 0, 0),
                "edge": (Nx // 2, 0, 0),
                "center": (Nx // 2, Ny // 2, Nz // 2),
                # "ring1": (Nx-2, Ny-2, Nz-2),
                # "ring2": (Nx//2, Ny//2, Nz-2),
                # "ring3": (Nx//2, Ny-2, Nz-2),
                "ring1": (1, 1, 1),
                "ring2": (Nx//2, Ny//2, 1),
                "ring3": (Nx//2, 1, 1),
                "face": (Nx // 2, Ny // 2, Nz - 1),
                "face0": (Nx // 2, Ny // 2, 0),
            }

            results = {}

            for label, idx in probe_points.items():
                print(f"\n=== {label.upper()} {idx} ===")
                index_debug_result = self.debug_single_node_index_stencil(
                    density=density,
                    delta_rho=delta_rho,
                    idx=idx,
                    thresh=1e-12,
                )

                save_index_debug_plots(
                    index_debug_result=index_debug_result,
                    sim=sim,
                    prefix=f"{prefix}_{label}"
                )

                results[label] = index_debug_result

            return results


        # results = save_index_debug_plots_for_key_points(
        #     sim=self.sim,
        #     density=X,
        #     delta_rho=step_size,
        #     prefix="idx_debug"
        # )
        # save_index_debug_plots(index_debug_result, self.sim, prefix="idx_debug")


        return {
            "f0": f_init,
            "L_dJ_du": self.gradient,
            "L_dJ_du0": dJ_du_init,
            "fds": fds,
            "NMSE": NMSE,
        }


    # -------------------------
    # index monitor helpers
    # -------------------------
    def get_design_index_arrays(self):
        if self.sim.fdtd.getnamednumber(self.sim.design_index_monitor_name) == 0:
            raise RuntimeError(
                f"Index monitor '{self.sim.design_index_monitor_name}' does not exist."
            )

        res = self.sim.fdtd.getresult(self.sim.design_index_monitor_name, "index")

        nx = np.array(res["index_x"], dtype=np.complex128).real
        ny = np.array(res["index_y"], dtype=np.complex128).real
        nz = np.array(res["index_z"], dtype=np.complex128).real

        if nx.ndim == 4:
            nx = nx[:, :, :, 0]
            ny = ny[:, :, :, 0]
            nz = nz[:, :, :, 0]

        return nx**2, ny**2, nz**2

    def check_design_index_monitor(self, atol=1e-6, rtol=0.0, interior_only=True):
        if self.sim.fdtd.getnamednumber(self.sim.design_index_monitor_name) == 0:
            raise RuntimeError(
                f"Index monitor '{self.sim.design_index_monitor_name}' does not exist."
            )

        res = self.sim.fdtd.getresult(self.sim.design_index_monitor_name, "index")

        xm = np.atleast_1d(np.squeeze(np.array(res["x"], dtype=float)))
        ym = np.atleast_1d(np.squeeze(np.array(res["y"], dtype=float)))
        zm = np.atleast_1d(np.squeeze(np.array(res["z"], dtype=float)))

        if xm.size != self.sim.design_x.size or not np.allclose(
            xm, self.sim.design_x, atol=1e-12, rtol=0
        ):
            raise RuntimeError("Index monitor x-grid does not match design_x")
        if ym.size != self.sim.design_y.size or not np.allclose(
            ym, self.sim.design_y, atol=1e-12, rtol=0
        ):
            raise RuntimeError("Index monitor y-grid does not match design_y")
        if zm.size != self.sim.design_z.size or not np.allclose(
            zm, self.sim.design_z, atol=1e-12, rtol=0
        ):
            raise RuntimeError("Index monitor z-grid does not match design_z")

        nx_chk, ny_chk, nz_chk = self.get_design_index_arrays()

        tx = self.sim.design_n[:, :, :, 0]**2
        ty = self.sim.design_n[:, :, :, 1]**2
        tz = self.sim.design_n[:, :, :, 2]**2

        if interior_only and min(tx.shape) > 2:
            sl = np.s_[1:-1, 1:-1, 1:-1]
            nx_cmp, ny_cmp, nz_cmp = nx_chk[sl], ny_chk[sl], nz_chk[sl]
            tx_cmp, ty_cmp, tz_cmp = tx[sl], ty[sl], tz[sl]
        else:
            nx_cmp, ny_cmp, nz_cmp = nx_chk, ny_chk, nz_chk
            tx_cmp, ty_cmp, tz_cmp = tx, ty, tz

        err_x = np.max(np.abs(nx_cmp - tx_cmp))
        err_y = np.max(np.abs(ny_cmp - ty_cmp))
        err_z = np.max(np.abs(nz_cmp - tz_cmp))

        ok_x = np.allclose(nx_cmp, tx_cmp, atol=atol, rtol=rtol)
        ok_y = np.allclose(ny_cmp, ty_cmp, atol=atol, rtol=rtol)
        ok_z = np.allclose(nz_cmp, tz_cmp, atol=atol, rtol=rtol)

        print("[index monitor check]")
        print("x-grid match: True")
        print("y-grid match: True")
        print("z-grid match: True")
        print(f"max |nx-target| = {err_x:.3e}")
        print(f"max |ny-target| = {err_y:.3e}")
        print(f"max |nz-target| = {err_z:.3e}")
        print(f"allclose nx: {ok_x}")
        print(f"allclose ny: {ok_y}")
        print(f"allclose nz: {ok_z}")

        return {
            "nx_err_max": err_x,
            "ny_err_max": err_y,
            "nz_err_max": err_z,
            "nx_ok": ok_x,
            "ny_ok": ok_y,
            "nz_ok": ok_z,
        }

    def debug_single_node_index_stencil(self, density, delta_rho=0.01, idx=None, thresh=1e-12):
        Nx, Ny, Nz = self.sim.design_grids

        rho0 = np.asarray(density, dtype=float).copy()
        if rho0.ndim != 3:
            if rho0.size != Nx * Ny * Nz:
                raise ValueError(f"density size mismatch: got {rho0.size}, expected {Nx*Ny*Nz}")
            rho0 = rho0.reshape(Nx, Ny, Nz)
        elif rho0.shape != (Nx, Ny, Nz):
            raise ValueError(f"density shape mismatch: got {rho0.shape}, expected {(Nx, Ny, Nz)}")

        if idx is None:
            ix, iy, iz = Nx // 2, Ny // 2, Nz // 2
        else:
            ix, iy, iz = idx

        self.sim.fdtd.switchtolayout()
        self.sim.update_design_density(rho0)
        nx0, ny0, nz0 = self.get_design_index_arrays()

        rho1 = rho0.copy()
        rho1[ix, iy, iz] += delta_rho

        self.sim.fdtd.switchtolayout()
        self.sim.update_design_density(rho1)
        nx1, ny1, nz1 = self.get_design_index_arrays()

        dnx = nx1 - nx0
        dny = ny1 - ny0
        dnz = nz1 - nz0

        rho1x, rho1y, rho1z = self.sim.density2idx(rho1)
        rho0x, rho0y, rho0z = self.sim.density2idx(rho0)

        tx0 = np.asarray(rho0x**2, dtype=float).reshape(Nx, Ny, Nz)
        ty0 = np.asarray(rho0y**2, dtype=float).reshape(Nx, Ny, Nz)
        tz0 = np.asarray(rho0z**2, dtype=float).reshape(Nx, Ny, Nz)

        tx1 = np.asarray(rho1x**2, dtype=float).reshape(Nx, Ny, Nz)
        ty1 = np.asarray(rho1y**2, dtype=float).reshape(Nx, Ny, Nz)
        tz1 = np.asarray(rho1z**2, dtype=float).reshape(Nx, Ny, Nz)

        dtx = tx1 - tx0
        dty = ty1 - ty0
        dtz = tz1 - tz0

        print("[single-node perturbation]")
        print(f"idx = {(ix, iy, iz)}")
        print(f"delta_rho = {delta_rho}")
        print("[center response]")
        print("target dx,dy,dz:", dtx[ix, iy, iz], dty[ix, iy, iz], dtz[ix, iy, iz])
        print("read   dx,dy,dz:", dnx[ix, iy, iz], dny[ix, iy, iz], dnz[ix, iy, iz])
        print("org   x,y,z:", nx0[ix, iy, iz], ny0[ix, iy, iz], nz0[ix, iy, iz])

        maskx = np.abs(dnx) > thresh
        masky = np.abs(dny) > thresh
        maskz = np.abs(dnz) > thresh

        idx_x = np.argwhere(maskx)
        idx_y = np.argwhere(masky)
        idx_z = np.argwhere(maskz)

        print("[stencil size]")
        print("nx nonzero:", len(idx_x))
        print("ny nonzero:", len(idx_y))
        print("nz nonzero:", len(idx_z))

        if len(idx_x) > 0:
            print("[first 20 nx stencil entries]")
            for p in idx_x[:20]:
                i, j, k = p
                print((int(i), int(j), int(k)), dnx[i, j, k])

        if len(idx_y) > 0:
            print("[first 20 ny stencil entries]")
            for p in idx_y[:20]:
                i, j, k = p
                print((int(i), int(j), int(k)), dny[i, j, k])

        if len(idx_z) > 0:
            print("[first 20 nz stencil entries]")
            for p in idx_z[:20]:
                i, j, k = p
                print((int(i), int(j), int(k)), dnz[i, j, k])

        return {
            "idx": (ix, iy, iz),
            "delta_rho": delta_rho,
            "baseline": {"nx": nx0, "ny": ny0, "nz": nz0},
            "new": {"nx": nx1, "ny": ny1, "nz": nz1},
            "delta_read": {"nx": dnx, "ny": dny, "nz": dnz},
            "delta_target": {"nx": dtx, "ny": dty, "nz": dtz},
            "idx_x": idx_x,
            "idx_y": idx_y,
            "idx_z": idx_z,
        }






    def _boundary_weight_1d(self, N, beta=None):
        """
        1D transverse boundary weight.

        interior -> 1
        boundary -> beta
        """
        
        if beta is None:
            beta = self.boundary_overlap_beta

        w = np.ones(N, dtype=float)
        if N >= 1:
            w[0] = beta
            w[-1] = beta
        return w


    def _pair_gather_backward(self, g, axis):
        """
        Generic transpose of backward pair-average operator.

        Backprop through:
            m[i] = 0.5*u[i] + 0.5*u[i+1]
            m[last] = background (ignored)
        then transpose gather is:
            grad_u(i) = 0.5 * grad_m(i) + 0.5 * grad_m(i-1)

        axis:
            0 -> x
            1 -> y
            2 -> z
        """

        out = np.zeros_like(g)

        sl_out0 = [slice(None)] * g.ndim  # u[i]
        sl_out1 = [slice(None)] * g.ndim  # u[i+1]
        sl_g    = [slice(None)] * g.ndim  # m[i] (valid: 0 ~ -2)

        sl_g[axis]    = slice(0, -1)
        sl_out0[axis] = slice(0, -1)
        sl_out1[axis] = slice(1, None)

        out[tuple(sl_out0)] += 1.0 * g[tuple(sl_g)]
        out[tuple(sl_out1)] += 1.0 * g[tuple(sl_g)]

        return out
    
    def gradient_check(self):
        grad_total = np.zeros(np.shape(self.adjoint_fields[0]), dtype=np.complex128)
        for arg in [0,1,2]:
            # grad_contrib = (self.adjoint_fields[arg])*(self.forward_fields[arg])
            grad_contrib = self.dedr[arg]*((self.adjoint_fields[arg]))*(self.forward_fields[arg])
            grad_total += grad_contrib#*self.sim.n_cur[:,:,:,:,arg]
        grad_total = np.real(grad_total)
        self.current_state = "INIT"
        dJ_dus_norm=[]
        dJ_dus=[]
        for i in range(len(self.objective_functions)):
            dJ_du= grad_total[:,:,:,i].flatten()
            print(np.max((dJ_du)))
            print(np.min((dJ_du)))
            print(dJ_du)
            # norm=np.max(np.abs(dJ_du))
            # self.g_norm[i]=max(self.g_norm[i], norm)
            # dJ_dus_norm.append(dJ_du/self.g_norm[i])
            dJ_dus.append(grad_total[:,:,:,i].flatten())

        self.gradient=Opt_MS2.Minimax(self.f0, dJ_dus)
        np.savetxt("dJ_du.txt", dJ_dus)
        # np.savetxt("dJ_du_norm.txt", dJ_dus_norm)
        np.savetxt("minimax.txt", self.gradient)

        return self.gradient
