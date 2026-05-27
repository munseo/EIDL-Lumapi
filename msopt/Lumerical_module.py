import sys
import math
import numpy as np
import autograd.numpy as npa
import h5py
from mpi4py import MPI
from matplotlib import pyplot as plt
import scipy.io as sio
from tqdm import tqdm
from scipy import special

def GDS_converter(design, Nx, Ny, Nz, is_3D):
    import gdspy

    structure_weight = design
    if is_3D:
        structure_weight = structure_weight.reshape(Nx, Ny * Nz)[Nx - 1]
    data = npa.rot90(structure_weight.reshape(Ny, Nz))

    cellz = gdspy.Cell("Top")
    width = 1
    height = 1
    rectangles = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] == 1.0:
                rect = gdspy.Rectangle(
                    (j * width, -i * height),
                    ((j + 1) * width, -(i + 1) * height),
                )
                rectangles.append(rect)
    outline = gdspy.fast_boolean(rectangles, None, "or", precision=0.1)
    cellz.add(outline)

    gdspy.write_gds("output.gds")
    cellz.write_svg("output.svg")

def Xpol_v(PHI, THETA):
    return np.cos(THETA) * np.cos(PHI), -np.sin(PHI)

def Ypol_v(PHI, THETA):
    return np.cos(THETA) * np.sin(PHI), np.cos(PHI)

def LCP_v(PHI, THETA):
    return np.cos(PHI) -1j* np.sin(PHI), np.sin(PHI) +1j* np.cos(PHI)

def RCP_v(PHI, THETA):
    return np.cos(PHI) +1j* np.sin(PHI), np.sin(PHI) -1j* np.cos(PHI)

def DW_intg(
    x, y, z,
    k=1.0,
    NA=0.95,
    l=4,
    polarization=0,
    theta_sample=100,
    phi_sample=100
):
    """
    Returns:
        complex scalar amplitude at the specified point
    """
    theta_max = np.arcsin(NA)

    # Sampling points
    theta = np.linspace(0, theta_max, theta_sample)
    phi = np.linspace(0, 2 * np.pi, phi_sample)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    # 2D meshgrid
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')

    sin_t = np.sin(THETA)
    cos_t = np.cos(THETA)
    sqrt_cos_t = np.sqrt(cos_t)
    amp = sqrt_cos_t * sin_t

    # OAM phase
    oam_phase = np.exp(1j * l * PHI)

    # Wavevector components
    kx = k * sin_t * np.cos(PHI)
    ky = k * sin_t * np.sin(PHI)
    kz = k * cos_t
    phase = np.exp(1j * (kx * x + ky * y + kz * z))

    # Polarization unit vectors
    Pv=[Xpol_v, Ypol_v, LCP_v, RCP_v]
    e_theta, e_phi=Pv[polarization](PHI,THETA)
    # print(e_theta)
    
    # Vector components in Cartesian coordinates
    ex = e_theta * cos_t * np.cos(PHI) - e_phi * np.sin(PHI)
    ey = e_theta * cos_t * np.sin(PHI) + e_phi * np.cos(PHI)
    ez = -e_theta * sin_t

    # Integrand (weight) and sum
    weight = amp * oam_phase * phase * sin_t * dtheta * dphi
    Ex = np.sum(weight * ex)
    Ey = np.sum(weight * ey)
    Ez = np.sum(weight * ez)

    return Ex, Ey, Ez

def dimension_check(size):
    dims = ['x', 'y', 'z']
    nonzero_dims = [d for d, s in zip(dims, size) if s > 0]
    if len(nonzero_dims) == 2:
        normal_axis = [d for d in dims if d not in nonzero_dims][0]
    else:
        normal_axis = None
    return nonzero_dims, normal_axis


def Beam_save(
        filename,
        Size,
        Center,
        resolution,
        mode,
        lamb,
        NA,
        z0,
        Pol='X',
        Type=0,
        ):
    k0 = 2 * np.pi / lamb

    if Pol == 'X':
        px, py, pz = 1, 0, 0
    elif Pol == 'Y':
        px, py, pz = 0, 1, 0
    elif Pol == 'LCP':
        px, py, pz = 1, -1.0j, 0
    elif Pol == 'RCP':
        px, py, pz = 1,  1.0j, 0
    else:
        raise ValueError("polarization must be 'X', 'Y', 'LCP', or 'RCP'")

    dims = ['x', 'y', 'z']
    kw_to_idx = {kw: i for i, kw in enumerate(dims)}

    nz_dims, n_axis = dimension_check(Size)
    u_ax, v_ax = kw_to_idx[nz_dims[0]], kw_to_idx[nz_dims[1]]
    n_ax = kw_to_idx[n_axis]

    Su, Sv = Size[u_ax], Size[v_ax]
    Nu, Nv = int(Size[u_ax] * resolution) + 1, int(Size[v_ax] * resolution) + 1

    du, dv = Su / float(Nu - 1), Sv / float(Nv - 1)
    u0 = Center[u_ax] - 0.5 * Su
    v0 = Center[v_ax] - 0.5 * Sv

    u_grids = u0 + np.arange(Nu) * du
    v_grids = v0 + np.arange(Nv) * dv
    n_grid = Center[n_ax]

    uc = Center[u_ax]
    vc = Center[v_ax]
    R_ap = 0.5 * min(Su, Sv)

    N_grids = [0, 0, 0]
    N_grids[u_ax] = Nu
    N_grids[v_ax] = Nv
    N_grids[n_ax] = 1

    Au = np.zeros(N_grids, dtype=np.complex128)
    Av = np.zeros(N_grids, dtype=np.complex128)
    An = np.zeros(N_grids, dtype=np.complex128)

    uv_list = [(i, j, k) for i in range(N_grids[0]) for j in range(N_grids[1]) for k in range(N_grids[2])]
    circ = []

    for i, j, k in uv_list:
        temp_idx = [i, j, k]
        u = u_grids[temp_idx[u_ax]]
        v = v_grids[temp_idx[v_ax]]

        du_c = u - uc
        dv_c = v - vc
        r = np.sqrt(du_c**2 + dv_c**2)

        if r < R_ap:
            circ.append((i, j, k))

    if Type == 0:
        for i, j, k in tqdm(circ, desc="Gaussian Field calculation"):
            temp_idx = [i, j, k]
            u = u_grids[temp_idx[u_ax]]
            v = v_grids[temp_idx[v_ax]]

            du_c = u - uc
            dv_c = v - vc
            r = np.sqrt(du_c**2 + dv_c**2)

            w0 = lamb / (np.pi * NA)
            amp = lg(mode, w0, lamb, du_c, dv_c, z0, r)

            Au[i, j, k] += amp * px
            Av[i, j, k] += amp * py
            An[i, j, k] += amp * pz

    else:
        for i, j, k in tqdm(circ, desc="Bessel Field calculation"):
            temp_idx = [i, j, k]
            u = u_grids[temp_idx[u_ax]]
            v = v_grids[temp_idx[v_ax]]

            du_c = u - uc
            dv_c = v - vc
            r = np.sqrt(du_c**2 + dv_c**2)
            phi = np.arctan2(dv_c, du_c)

            kt = k0 * NA
            amp = special.jv(mode, kt * r) * np.exp(-1j * mode * phi)

            Au[i, j, k] += amp * px
            Av[i, j, k] += amp * py
            An[i, j, k] += amp * pz

    A_all = [0, 0, 0]
    A_all[u_ax] = Au
    A_all[v_ax] = Av
    A_all[n_ax] = An

    np.savez(filename, ax=A_all[0], ay=A_all[1], az=A_all[2])
    print(f"[rank 0] Saved field data to {filename}")



def vectoral_beam_load(
    filename: str,
) -> list:
    # Step 1: rank 0 loads field data
    data = np.load(filename+'.npz')
    ax = data["ax"]
    ay = data["ay"]
    az = data["az"]

    fields=[0]*3
    fields[0]=ax
    fields[1]=ay
    fields[2]=az
    # print(fields)
    return np.array(fields)


def _radiation_pattern_components(pattern, theta, phi, basis="spherical"):
    field = pattern(theta, phi)

    if isinstance(field, tuple) or isinstance(field, list):
        if len(field) == 2:
            a0, a1 = field
            if basis == "spherical":
                e_theta_u = np.cos(theta) * np.cos(phi)
                e_theta_v = np.cos(theta) * np.sin(phi)
                e_theta_n = -np.sin(theta)
                e_phi_u = -np.sin(phi)
                e_phi_v = np.cos(phi)
                e_phi_n = np.zeros_like(phi)

                au = a0 * e_theta_u + a1 * e_phi_u
                av = a0 * e_theta_v + a1 * e_phi_v
                an = a0 * e_theta_n + a1 * e_phi_n
                return au, av, an
            if basis == "cartesian":
                return a0, a1, np.zeros_like(a0, dtype=np.complex128)
        if len(field) == 3:
            return field[0], field[1], field[2]
        raise ValueError("pattern must return scalar, (Etheta, Ephi), or (Eu, Ev, En).")

    if basis not in ("scalar", "spherical"):
        raise ValueError("Scalar pattern is only valid with basis='scalar' or basis='spherical'.")
    return field, np.zeros_like(field, dtype=np.complex128), np.zeros_like(field, dtype=np.complex128)


def theta_intensity_pattern(
        theta_points=None,
        intensity_points=None,
        intensity=None,
        phase=None,
        polarization="theta",
        normalize=True,
        ):
    """
    Build a phi-invariant angular-spectrum pattern from I(theta).

    theta_points/intensity_points define a table in radians. Alternatively,
    intensity may be a callable intensity(theta). The returned pattern is
    compatible with arbitrary_radiation_field().
    """
    if intensity is None:
        if theta_points is None or intensity_points is None:
            raise ValueError("Provide either intensity(theta) callable or theta_points/intensity_points.")
        theta_points = np.asarray(theta_points, dtype=float).reshape(-1)
        intensity_points = np.asarray(intensity_points, dtype=float).reshape(-1)
        if theta_points.size != intensity_points.size:
            raise ValueError("theta_points and intensity_points must have the same length.")
        order = np.argsort(theta_points)
        theta_points = theta_points[order]
        intensity_points = intensity_points[order]

        def intensity(theta):
            return np.interp(theta, theta_points, intensity_points, left=0.0, right=0.0)

    def pattern(theta, phi):
        I = np.asarray(intensity(theta), dtype=float)
        I = np.maximum(I, 0.0)
        if normalize:
            I_max = np.max(I)
            if I_max > 0:
                I = I / I_max
        ph = 0.0 if phase is None else phase(theta)
        amp = np.sqrt(I) * np.exp(1j * ph)
        zero = np.zeros_like(amp, dtype=np.complex128)

        if polarization in ("theta", "s", "TE"):
            return amp, zero
        if polarization in ("phi", "p", "TM"):
            return zero, amp
        if polarization in ("circular+", "rcp"):
            return amp / np.sqrt(2), 1j * amp / np.sqrt(2)
        if polarization in ("circular-", "lcp"):
            return amp / np.sqrt(2), -1j * amp / np.sqrt(2)
        if callable(polarization):
            return polarization(theta, phi, amp)
        raise ValueError("Unknown polarization. Use theta, phi, circular+/-, or a callable.")

    return pattern


def arbitrary_radiation_field(
        Size,
        Center,
        resolution,
        lamb,
        pattern,
        NA=1.0,
        theta_max=None,
        n=1.0,
        theta_sample=181,
        phi_sample=361,
        basis="spherical",
        propagation_distance=0.0,
        normalize=True,
        aperture=True,
        ):
    """
    Build a Lumerical imported-source field from an arbitrary radiation pattern.

    pattern(theta, phi) may return scalar amplitude, (Etheta, Ephi), or
    (Eu, Ev, En). The local u/v axes are the two nonzero source-plane axes in
    Size; n is the injection-axis direction.
    """
    dims = ['x', 'y', 'z']
    kw_to_idx = {kw: i for i, kw in enumerate(dims)}

    nz_dims, n_axis = dimension_check(Size)
    u_ax, v_ax = kw_to_idx[nz_dims[0]], kw_to_idx[nz_dims[1]]
    n_ax = kw_to_idx[n_axis]

    Su, Sv = Size[u_ax], Size[v_ax]
    Nu, Nv = int(Size[u_ax] * resolution) + 1, int(Size[v_ax] * resolution) + 1

    du, dv = Su / float(Nu - 1), Sv / float(Nv - 1)
    u0 = Center[u_ax] - 0.5 * Su
    v0 = Center[v_ax] - 0.5 * Sv
    u_grids = u0 + np.arange(Nu) * du
    v_grids = v0 + np.arange(Nv) * dv

    k = 2 * np.pi * n / lamb
    if theta_max is None:
        theta_max = np.arcsin(min(NA / n, 1.0))
    theta = np.linspace(0, theta_max, theta_sample)
    phi = np.linspace(0, 2 * np.pi, phi_sample, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    dtheta = theta[1] - theta[0] if theta_sample > 1 else theta_max
    dphi = phi[1] - phi[0] if phi_sample > 1 else 2 * np.pi
    weight = np.sin(THETA) * dtheta * dphi

    Au_k, Av_k, An_k = _radiation_pattern_components(pattern, THETA, PHI, basis=basis)
    ku = k * np.sin(THETA) * np.cos(PHI)
    kv = k * np.sin(THETA) * np.sin(PHI)
    kn = k * np.cos(THETA)

    N_grids = [0, 0, 0]
    N_grids[u_ax] = Nu
    N_grids[v_ax] = Nv
    N_grids[n_ax] = 1

    Au = np.zeros(N_grids, dtype=np.complex128)
    Av = np.zeros(N_grids, dtype=np.complex128)
    An = np.zeros(N_grids, dtype=np.complex128)

    uc = Center[u_ax]
    vc = Center[v_ax]
    R_ap = 0.5 * min(Su, Sv)

    uv_list = [(i, j, kidx) for i in range(N_grids[0]) for j in range(N_grids[1]) for kidx in range(N_grids[2])]
    for i, j, kidx in tqdm(uv_list, desc="Radiation inverse field calculation"):
        temp_idx = [i, j, kidx]
        u = u_grids[temp_idx[u_ax]]
        v = v_grids[temp_idx[v_ax]]
        du_c = u - uc
        dv_c = v - vc

        if aperture and np.sqrt(du_c**2 + dv_c**2) > R_ap:
            continue

        phase = np.exp(-1j * (ku * du_c + kv * dv_c + kn * propagation_distance))
        Au[i, j, kidx] = np.sum(Au_k * phase * weight)
        Av[i, j, kidx] = np.sum(Av_k * phase * weight)
        An[i, j, kidx] = np.sum(An_k * phase * weight)

    A_all = [0, 0, 0]
    A_all[u_ax] = Au
    A_all[v_ax] = Av
    A_all[n_ax] = An

    if normalize:
        amp_max = max(np.max(np.abs(A_all[0])), np.max(np.abs(A_all[1])), np.max(np.abs(A_all[2])))
        if amp_max > 0:
            A_all[0] /= amp_max
            A_all[1] /= amp_max
            A_all[2] /= amp_max

    return np.array(A_all)


def arbitrary_radiation_save(
        filename,
        Size,
        Center,
        resolution,
        lamb,
        pattern,
        NA=1.0,
        theta_max=None,
        n=1.0,
        theta_sample=181,
        phi_sample=361,
        basis="spherical",
        propagation_distance=0.0,
        normalize=True,
        aperture=True,
        ):
    fields = arbitrary_radiation_field(
        Size,
        Center,
        resolution,
        lamb,
        pattern,
        NA=NA,
        theta_max=theta_max,
        n=n,
        theta_sample=theta_sample,
        phi_sample=phi_sample,
        basis=basis,
        propagation_distance=propagation_distance,
        normalize=normalize,
        aperture=aperture,
    )
    np.savez(filename, ax=fields[0], ay=fields[1], az=fields[2])
    print(f"[rank 0] Saved radiation field data to {filename}")


def arbitrary_radiation_broadband_field(
        Size,
        Center,
        resolution,
        lambdas,
        pattern,
        NA=1.0,
        theta_max=None,
        n=1.0,
        theta_sample=181,
        phi_sample=361,
        basis="spherical",
        propagation_distance=0.0,
        normalize=True,
        aperture=True,
        ):
    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
    fields = []
    for lamb in lambdas:
        fields.append(
            arbitrary_radiation_field(
                Size,
                Center,
                resolution,
                float(lamb),
                pattern,
                NA=NA,
                theta_max=theta_max,
                n=n,
                theta_sample=theta_sample,
                phi_sample=phi_sample,
                basis=basis,
                propagation_distance=propagation_distance,
                normalize=normalize,
                aperture=aperture,
            )
        )
    Ex = np.stack([field[0] for field in fields], axis=-1)
    Ey = np.stack([field[1] for field in fields], axis=-1)
    Ez = np.stack([field[2] for field in fields], axis=-1)
    return np.array([Ex, Ey, Ez])


def arbitrary_radiation_broadband_save(
        filename,
        Size,
        Center,
        resolution,
        lambdas,
        pattern,
        NA=1.0,
        theta_max=None,
        n=1.0,
        theta_sample=181,
        phi_sample=361,
        basis="spherical",
        propagation_distance=0.0,
        normalize=True,
        aperture=True,
        ):
    fields = arbitrary_radiation_broadband_field(
        Size,
        Center,
        resolution,
        lambdas,
        pattern,
        NA=NA,
        theta_max=theta_max,
        n=n,
        theta_sample=theta_sample,
        phi_sample=phi_sample,
        basis=basis,
        propagation_distance=propagation_distance,
        normalize=normalize,
        aperture=aperture,
    )
    np.savez(filename, ax=fields[0], ay=fields[1], az=fields[2], lambdas=np.asarray(lambdas, dtype=float))
    print(f"[rank 0] Saved broadband radiation field data to {filename}")


def balanced_xy_dipole_fom(E_x, E_y, target_ratio=1.0, eps=1e-30):
    """
    FoM for unpolarized or arbitrary in-plane dipoles in an emitting layer.

    It maximizes total in-plane electric-field intensity while penalizing
    imbalance between Ex and Ey coupling. target_ratio = Ix / Iy.
    """
    Ix = npa.sum(npa.abs(E_x) ** 2)
    Iy = npa.sum(npa.abs(E_y) ** 2)
    total = Ix + Iy
    balance = 4 * target_ratio * Ix * Iy / ((Ix + target_ratio * Iy) ** 2 + eps)
    return total * balance


def uniform_incoherent_xy_dipole_fom(E_x, E_y, target_ratio=1.0, eps=1e-30):
    """
    Incoherent OLED ensemble FoM for a 2D emitting plane.

    The score is evaluated locally from intensities, then averaged with a
    uniformity factor so the optimizer cannot win only by making a hotspot.
    """
    Ix = npa.abs(E_x) ** 2
    Iy = npa.abs(E_y) ** 2
    I = Ix + Iy
    pol_balance = 4 * target_ratio * Ix * Iy / ((Ix + target_ratio * Iy) ** 2 + eps)
    local_coupling = I * pol_balance
    mean_coupling = npa.mean(local_coupling)
    uniformity = mean_coupling ** 2 / (npa.mean(local_coupling ** 2) + eps)
    return mean_coupling * uniformity



def lg(l,w,lamb,x,y,z,r):
    #Rayleigh range
    z_r=np.pi*w**2/lamb
    #Beam waist
    w_z=w*np.sqrt((z**2+z_r**2)/z_r**2)
    #Gouy phase
    psi=(l+1)*np.arctan2(z/z_r,1)
    #Normalizing factor
    lg_func = np.sqrt(2/(np.pi*math.factorial(np.abs(l))))
    lg_func *= 1/w_z*(r*np.sqrt(2)/w_z)**(np.abs(l))*np.exp(-r**2/w_z**2)*np.exp(-1j*l*np.arctan2(y,x))
    #Focusing
    lg_func *= np.exp(-1j*psi)*np.exp(1j*2*np.pi/lamb*(r**2)*z/(2*(z**2+z_r**2)))
    #Associate Laguerre
    #lg_func *= scipy.special.assoc_laguerre(2*r**2/w_z**2,0,k=abs(l)) #if p=0 LG_l0(x)=1
    return lg_func




def vectoral_beam_save(filename, Size, Center, resolution, mode, freq, NA, z0, Pol='X'):

    if Pol == 'X':
        p_idx=0
    elif Pol == 'Y':
        p_idx=1
    elif Pol == 'LCP':
        p_idx=2
    elif Pol == 'RCP':
        p_idx=3
    else:
        raise ValueError("polarization must be 'X', 'Y', 'LCP', or 'RCP'")

    wavelength=1/freq
    dx = 1 / resolution

    dims = ['x', 'y', 'z']
    kw_to_idx = {kw: i for i, kw in enumerate(dims)}
    idx_to_kw = {i: kw for i, kw in enumerate(dims)}

    nz_dims, n_axis = dimension_check(Size)
    u_ax, v_ax  = kw_to_idx[nz_dims[0]], kw_to_idx[nz_dims[1]]
    Su, Sv = Size[u_ax], Size[v_ax]
    n_ax= kw_to_idx[n_axis]
    Nu, Nv = int(Size[u_ax]*resolution)+1, int(Size[v_ax]*resolution)+1       

    du, dv = Su / float(Nu-1), Sv / float(Nv-1)
    u0 = Center[u_ax] - 0.5*Su
    v0 = Center[v_ax] - 0.5*Sv
    u_grids = u0 + np.arange(Nu)*du
    v_grids = v0 + np.arange(Nv)*dv
    n_grid= Center[n_ax]

    N_grids=[0, 0, 0]
    N_grids[u_ax]=Nu
    N_grids[v_ax]=Nv
    N_grids[n_ax]=1

    Au = np.zeros(N_grids, dtype=np.complex128)
    Av = np.zeros(N_grids, dtype=np.complex128)
    An = np.zeros(N_grids, dtype=np.complex128)
    uv_list = [(i, j, k) for i in range(N_grids[0]) for j in range(N_grids[1]) for k in range(N_grids[2])]
    circ=[]
    for i, j, k in uv_list:
        temp_idx=[i,j,k]
        u = u_grids[temp_idx[u_ax]]
        v = v_grids[temp_idx[v_ax]]
        r= np.sqrt(u**2+v**2)
        if r < Su/2:
            circ.append((i, j, k))
    for i, j, k in tqdm(circ, desc="Field calculation"):
        temp_idx=[i,j,k]
        au, av, an = DW_intg(u_grids[temp_idx[u_ax]], v_grids[temp_idx[v_ax]], z0, wavelength, NA, mode, p_idx)
        # print(ax)
        Au[i,j,k]+=au
        Av[i,j,k]+=av
        An[i,j,k]+=an
        # print(Ax[i,j])
    A_all=[0,0,0]
    A_all[u_ax]=Au
    A_all[v_ax]=Av
    A_all[n_ax]=An
    np.savez(filename, ax=A_all[0], ay=A_all[1], az=A_all[2])
    print(f"[rank 0] Saved field data to {filename}")
