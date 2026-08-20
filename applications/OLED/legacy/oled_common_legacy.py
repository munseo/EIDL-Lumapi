"""Parts of oled_common.py that only the retired OLED scripts used.

Moved out on 2026-08-19 when the application was reduced to the OLED_rec chain
(OLED_rec -> k_mapping -> OLED_layered_dipole, plus oled_common and
oled_rec_plots). Nothing OLED_rec reaches, directly or transitively, is in here:
the split was computed as a reachability closure from every `oc.NAME` the live
files use, then checked by hand for the two functions the closure got wrong --
save_current_design_sections appears only in a COMMENT, and
oled_constrained_score is still exercised by test_oled_common so it stayed.

Kept rather than deleted because the retired scripts in this directory import
them; they are not maintained and are not imported by anything live.
"""
import numpy as np
import matplotlib.pyplot as plt

from oled_common import *          # noqa: F401,F403 -- these bodies call back into it


def env_list_str(name, default):
    raw = os.environ.get(name, "").replace(";", ",").replace(" ", ",")
    vals = [v.strip().lower() for v in raw.split(",") if v.strip()]
    return vals if vals else list(default)


def broadcast(values, size, name):
    if len(values) == size:
        return list(values)
    if len(values) == 1:
        return list(values) * size
    raise ValueError(f"{name} must have length 1 or {size}, got {len(values)}")


def set_fdtd_background_index(fdtd, index_value):
    try:
        fdtd.setnamed("FDTD", "index", float(index_value))
    except Exception:
        fdtd.eval(f'select("FDTD"); set("index", {float(index_value):.16g});')


def choose_norm_power(G, incident_power, bulk_power, source_power, current_power):
    for label, value in (
        ("design_incident_power", incident_power),
        ("bulk_reference_power", bulk_power),
        ("source_power", source_power),
        ("current_dipole_power", current_power),
    ):
        value = valid_power(value)
        if value is not None:
            return max(value, G.channel_power_floor), label
    return G.channel_power_floor, "floor"


def save_target_orders(G, info):
    path = os.path.join(G.design_dir, "OLED_target_orders.csv")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("m,n,kx_over_k0,ky_over_k0,theta_deg,phi_deg,target_efficiency\n")
        for o in info["orders"]:
            fp.write(
                f"{o['m']},{o['n']},{o['ux']:.8f},{o['uy']:.8f},"
                f"{o['theta_deg']:.8f},{o['phi_deg']:.8f},{o['efficiency']:.8f}\n"
            )
    print(f"[target] saved {len(info['orders'])} propagating orders: {path}")
    save_target_orders_figure(G, info)


def save_target_orders_figure(G, info):
    orders = info["orders"]
    if not orders:
        return
    theta_max = max((o["theta_deg"] for o in orders), default=0.0)
    angle_range = np.linspace(0.0, max(theta_max + 5.0, 1.0), 181)
    eff_range = np.asarray([G.interp_curve(a) for a in angle_range], dtype=float)
    signed_angles = np.concatenate((-angle_range[1:][::-1], angle_range))
    signed_eff = np.concatenate((eff_range[1:][::-1], eff_range))

    fig = plt.figure(figsize=(6.5, 3.9))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.deg2rad(signed_angles), signed_eff, "b-", linewidth=2, label="target efficiency")
    for o in orders:
        if o["efficiency"] <= 0.0:
            continue
        angle = o["theta_deg"]
        order_angles = [0.0] if abs(angle) <= 1e-12 else [-angle, angle]
        ax.scatter(np.deg2rad(order_angles), [o["efficiency"]] * len(order_angles), c="tab:orange", s=60, zorder=5)
        ax.text(np.deg2rad(angle), min(float(o["efficiency"]) + 0.05, 1.0), f"({o['m']},{o['n']})", ha="center", va="bottom", fontsize=7)
    ax.scatter([], [], c="tab:orange", s=60, label="propagating orders")
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0.0, 1.0)
    ax.set_title(f"Angular target (lambda={info['wavelength_um']}um, period={info['period_x_um']}x{info['period_y_um']}um)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    path = os.path.join(G.design_dir, "OLED_target_field_info.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[target] saved target field info figure: {path}")


def dft_matrix(n):
    i = np.arange(int(n), dtype=float)
    return np.exp(-2j * np.pi * np.outer(i, i) / float(n))


def order_mask(ukx, uky, ux, uy, sigma_deg, propagating):
    sigma = max(float(np.sin(np.deg2rad(max(float(sigma_deg), 1e-9)))), 1e-9)
    w = np.exp(-0.5 * (((ukx - ux) ** 2 + (uky - uy) ** 2) ** 0.5 / sigma) ** 2)
    w = np.where(propagating, w, 0.0)
    peak = float(np.max(w))
    return w / peak if peak > 0.0 else w


def build_ramp_target(G, angle_thetas):
    """Piecewise-linear target ratio CURVE over the available emission angles.

    The specified (angle, ratio) pairs are control points; between them the target ratio
    varies linearly, so the intermediate emission angles also follow the ratio (e.g.
    0:1, 45:0.85 -> 1.0 at 0 deg ramping to 0.85 at the 45-deg mode). Energy is packed
    into [min control angle, max control angle] with that shape; angles outside the range
    are leakage (target 0). Control angles snap to the nearest available emission mode.
    Returns (target_profile[sum=1 in-range], in_range, spec_idx, ctrl_a, ctrl_r).
    """
    ctrl = sorted(((float(a), float(r)) for a, r in G.target_angle_pairs), key=lambda ar: ar[0])
    ctrl_a = np.asarray([float(angle_thetas[int(np.argmin(np.abs(angle_thetas - a)))]) for a, _ in ctrl], dtype=float)
    ctrl_r = np.asarray([r for _, r in ctrl], dtype=float)
    order = np.argsort(ctrl_a)
    ctrl_a, ctrl_r = ctrl_a[order], ctrl_r[order]
    a_lo, a_hi = float(ctrl_a[0]), float(ctrl_a[-1])
    in_range = (angle_thetas >= a_lo - 1e-6) & (angle_thetas <= a_hi + 1e-6)
    target_curve = np.where(in_range, np.interp(angle_thetas, ctrl_a, ctrl_r), 0.0)
    target_profile = target_curve / max(float(np.sum(target_curve)), 1e-30)
    spec_idx = [int(np.argmin(np.abs(angle_thetas - a))) for a in ctrl_a]
    return target_profile, in_range.astype(float), spec_idx, ctrl_a, ctrl_r


def _angular_target_1d(G, x, y, k0):
    # phi = 0 radial line: 1D-DFT along x at y = 0. Emission angles are the (m,0) orders,
    # sin(theta) = kx/k0 = m*lambda/P. +-kx of the same |theta| grouped into one mode.
    dx = abs(float(np.mean(np.diff(x))))
    iy0 = int(np.argmin(np.abs(y)))
    kx = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    u = kx / k0
    propagating = np.abs(u) <= 1.0 + 1e-9
    theta = np.round(np.rad2deg(np.arcsin(np.clip(np.abs(u), 0.0, 1.0))), 3)
    angle_thetas = np.asarray(sorted({float(t) for t, p in zip(theta, propagating) if p}), dtype=float)
    if angle_thetas.size == 0:
        angle_thetas = np.zeros(1, dtype=float)
    angle_select = np.stack([((np.abs(theta - th) < 1e-3) & propagating).astype(float) for th in angle_thetas])
    target_profile, in_range, spec_idx, ctrl_a, ctrl_r = build_ramp_target(G, angle_thetas)
    print(f"[target] mode=radial_1d, phi=0 (m,0) angles {np.round(angle_thetas, 1).tolist()} deg; "
          f"in-range {np.round(angle_thetas[in_range > 0.5], 1).tolist()}, target {np.round(target_profile, 3).tolist()}")
    return {
        "mode": "radial_1d",
        "x_size": x.size, "y_size": y.size, "iy0": iy0,
        "dft_x": dft_matrix(x.size),
        "angle_select": angle_select,
        "angle_thetas": angle_thetas, "in_range": in_range, "target_profile": target_profile,
        "spec_idx": spec_idx, "spec_snapped": ctrl_a.tolist(), "ctrl_ratios": ctrl_r.tolist(),
    }


def _angular_target_2d(G, x, y, k0):
    # Full 2D monitor: 2D-DFT to (kx,ky), then azimuthally integrate the flux over each
    # polar-angle ring. Captures every (m,n) diffraction order, not just (m,0), so it is
    # valid for the non-cylindrical / dual-polarization case.
    dx, dy = abs(float(np.mean(np.diff(x)))), abs(float(np.mean(np.diff(y))))
    kx = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(y.size, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    u = np.sqrt(KX ** 2 + KY ** 2) / k0
    propagating = u <= 1.0 + 1e-9
    theta = np.round(np.rad2deg(np.arcsin(np.clip(u, 0.0, 1.0))), 3)
    angle_thetas = np.asarray(sorted({float(t) for t, p in zip(theta.ravel(), propagating.ravel()) if p}), dtype=float)
    if angle_thetas.size == 0:
        angle_thetas = np.zeros(1, dtype=float)
    # ring_select[k] gathers every (m,n) bin whose polar angle == angle_thetas[k]
    # (azimuthal integration over all orders on that ring).
    ring_select = np.stack([((np.abs(theta - th) < 1e-3) & propagating).astype(float) for th in angle_thetas])
    target_profile, in_range, spec_idx, ctrl_a, ctrl_r = build_ramp_target(G, angle_thetas)
    print(f"[target] mode=kspace_2d, (m,n) angles {np.round(angle_thetas, 1).tolist()} deg; "
          f"in-range {np.round(angle_thetas[in_range > 0.5], 1).tolist()}, target {np.round(target_profile, 3).tolist()}")
    return {
        "mode": "kspace_2d",
        "x_size": x.size, "y_size": y.size,
        "dft_x": dft_matrix(x.size), "dft_y": dft_matrix(y.size),
        "ring_select": ring_select,
        "angle_thetas": angle_thetas, "in_range": in_range, "target_profile": target_profile,
        "spec_idx": spec_idx, "spec_snapped": ctrl_a.tolist(), "ctrl_ratios": ctrl_r.tolist(),
    }


def angular_target_from_monitor_grid(G, x_axis, y_axis, target_info):
    """Angle basis for the FoM. Dispatches on G.fom_mode (radial_1d / kspace_2d)."""
    x = np.ravel(np.asarray(x_axis, dtype=float))
    y = np.ravel(np.asarray(y_axis, dtype=float))
    k0 = 2.0 * np.pi / (float(target_info["wavelength_um"]) * 1e-6)
    builder = _angular_target_2d if G.fom_mode == "kspace_2d" else _angular_target_1d
    return builder(G, x, y, k0)


def _ramp_fom(G, profile, target):
    """Shared T * M**ratio_emphasis overlap against the linear-ramp target profile."""
    in_range = npa.asarray(target["in_range"])
    tgt = npa.asarray(target["target_profile"])
    throughput = npa.sum(profile * in_range)
    q = profile / (throughput + 1e-30)
    match = npa.sum(npa.minimum(q * in_range, tgt))
    fom = throughput * (match ** G.ratio_emphasis)
    fracs = [profile[int(k)] for k in target["spec_idx"]]
    return fom, fracs, throughput, match


def angular_powers(G, Ex, Ey, Hx, Hy, target, flux_sign=1.0):
    """Emission-angle power profile -> ramp-overlap FoM. Dispatches on target['mode']."""
    if target.get("mode") == "kspace_2d":
        dft_x, dft_y = npa.asarray(target["dft_x"]), npa.asarray(target["dft_y"])

        def spectrum2d(f):
            if f.shape[0] != target["x_size"] or f.shape[1] != target["y_size"]:
                raise ValueError(f"monitor shape {f.shape} does not match angular target")
            # Two sequential 1D DFTs (O(N^3)); a single 3-operand einsum is O(N^4) under
            # autograd (no path optimization) and was ~85x slower on a 126x126 monitor.
            g = npa.einsum("ia,ab...->ib...", dft_x, f)    # DFT along x
            return npa.einsum("jb,ib...->ij...", dft_y, g)  # DFT along y -> (kx,ky)

        Ex_k, Ey_k, Hx_k, Hy_k = [spectrum2d(f) for f in (Ex, Ey, Hx, Hy)]
        flux = 0.5 * float(flux_sign) * npa.real(Ex_k * npa.conj(Hy_k) - Ey_k * npa.conj(Hx_k))
        if flux.ndim > 2:
            flux = npa.sum(flux, axis=tuple(range(2, flux.ndim)))     # sum over wavelengths
        flux = npa.maximum(npa.where(npa.isfinite(flux), flux, 0.0), 0.0)
        # Azimuthal integration: sum Sz(kx,ky) over every (m,n) bin on each theta ring.
        power = npa.einsum("kij,ij->k", npa.asarray(target["ring_select"]), flux)
    else:
        iy0 = int(target["iy0"])
        dft = npa.asarray(target["dft_x"])

        def line_spectrum(f):
            if f.shape[0] != target["x_size"] or f.shape[1] != target["y_size"]:
                raise ValueError(f"monitor shape {f.shape} does not match angular target")
            return npa.einsum("ia,a...->i...", dft, f[:, iy0])         # phi=0 line -> kx

        Ex_k, Ey_k, Hx_k, Hy_k = [line_spectrum(f) for f in (Ex, Ey, Hx, Hy)]
        flux = 0.5 * float(flux_sign) * npa.real(Ex_k * npa.conj(Hy_k) - Ey_k * npa.conj(Hx_k))
        if flux.ndim > 1:
            flux = npa.sum(flux, axis=tuple(range(1, flux.ndim)))      # sum over wavelengths
        flux = npa.maximum(npa.where(npa.isfinite(flux), flux, 0.0), 0.0)
        power = npa.einsum("ki,i->k", npa.asarray(target["angle_select"]), flux)

    profile = power / (npa.sum(power) + 1e-30)
    fom, fracs, throughput, match = _ramp_fom(G, profile, target)
    return npa.sum(power), fom, fracs, profile, throughput, match


def optimization_polarizations():
    # Multi-objective route: "x,y" -> N_fom=2, each polarization is one coherent-grid
    # objective and the optimizer optimizes their mean (see combine_fom). Default single.
    raw = os.environ.get(
        "MSOPT_OLED_OPT_POLARIZATIONS",
        os.environ.get("MSOPT_OLED_OPT_DIPOLE_POLARIZATION", "x, y"),
    )
    pols = [p.strip().lower() for p in raw.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    for p in pols:
        if p not in ("x", "y", "z"):
            raise ValueError(f"Unsupported optimization dipole polarization {p!r}.")
    return pols or ["x"]


def build_evenly_spaced_dipoles(G, pol="x"):
    """A compact, uniformly spaced grid of same-polarization dipoles = one coherent
    optimization objective (all fired together in a single sim)."""
    count = max(1, env_int("MSOPT_OLED_OPT_DIPOLE_COUNT", 25))
    grid_n = max(1, int(np.ceil(np.sqrt(count))))
    xs = np.linspace(-G.active_radius, G.active_radius, grid_n)
    ys = np.linspace(-G.active_radius, G.active_radius, grid_n)
    pol = str(pol).strip().lower()
    if pol not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported optimization dipole polarization {pol!r}.")

    dipoles = []
    for i in range(count):
        row, col = divmod(i, grid_n)
        x = float(xs[col])
        y = float(ys[row])
        dipoles.append(
            {
                "name": f"opt_dipole_{i}_{pol}",
                "dipole_idx": i,
                "dipole_x": x,
                "dipole_y": y,
                "dipole_z": float(G.eml_c[2]),
                "polarization": pol,
                "weight": 1.0,
            }
        )
    return dipoles


def channel_fom_terms(G, Ex, Ey, Hx, Hy, channel):
    target = channel["angular_target"]
    flux_sign = float(channel.get("last_top_flux_sign", 1.0))
    total, fom, spec_p, profile, throughput, match = angular_powers(G, Ex, Ey, Hx, Hy, target, flux_sign)
    return {"fom": fom, "spec_fracs": spec_p, "profile": profile, "total": total,
            "throughput": throughput, "match": match}


def combine_fom(G, vals):
    vals = npa.where(npa.isfinite(vals), vals, 0.0)
    vals = npa.maximum(vals, G.fom_floor)
    vals = npa.reshape(vals, (-1,))
    return npa.mean(vals) if vals.size else G.fom_floor


def clean_fom_values(G, values):
    return np.asarray(
        [
            max(float(np.nan_to_num(np.real(v[0] if isinstance(v, (list, tuple, np.ndarray)) else v), nan=G.fom_floor, posinf=G.score_cap, neginf=G.fom_floor)), G.fom_floor)
            for v in values
        ],
        dtype=float,
    )


def measure_design_incident_reference(G, channel):
    # Reference power incident on the design region from the dipole, measured with
    # the design region AND everything above it filled with design material. That
    # material (n=1.45) is index-matched to the SiO2 directly below, so there is no
    # reflecting interface at or above the design's lower face: the net upward flux
    # there is the pure incident power (no back-reflection from the design/superstrate).
    ref = make_sim(G, [G.Sx, G.Sy, G.Sz])
    # A channel may carry a whole coherent grid; the incident reference has to be
    # measured with the SAME excitation or the normalization does not correspond
    # to the run it normalizes.
    pts = channel.get("dipoles")
    if pts:
        for k, (dx, dy, dz) in enumerate(pts):
            add_dipole(G, ref, dx, dy, dz, channel["polarization"],
                       name=f"opt_dipole_{k}", enabled=True, group_name="source")
    else:
        add_dipole(G, ref, channel["dipole_x"], channel["dipole_y"], channel["dipole_z"], channel["polarization"])
    add_stack(G, ref)
    fill_bottom = G.design_incident_monitor_c[2]
    fill_h = G.Z_max - fill_bottom
    fill_c = [0.0, 0.0, fill_bottom + 0.5 * fill_h]
    ref.add_geo(fill_c, [G.Sx, G.Sy, fill_h], G.design_high_index, "design_incident_fill", float(np.mean(G.visible_wavelengths)))
    ref.add_monitor(G.design_incident_monitor_name, G.design_incident_monitor_c, G.design_incident_monitor_s)
    ref.run(name=f"design_incident_reference_{channel['dipole_idx']}", save=True)
    load_run_results(ref)
    freqs = source_freqs(G, ref)
    source_power = read_source_power(ref.fdtd, freqs)
    T_incident = read_transmission(ref.fdtd, G.design_incident_monitor_name)
    incident_power = abs(T_incident) * source_power if valid_power(source_power) is not None else None
    try:
        ref.fdtd.close()
    except Exception:
        pass
    incident_power = valid_power(incident_power)
    print(
        f"[incident-ref] channel {channel['dipole_idx']}: T={T_incident:.6e}, "
        f"source_power={valid_power(source_power) or float('nan'):.6e}, "
        f"incident_power={incident_power if incident_power is not None else float('nan'):.6e}"
    )
    return incident_power


def raw_monitor_flux(fdtd, monitor_name, target, flux_sign):
    try:
        E = np.asarray(fdtd.getresult(monitor_name, "E")["E"], dtype=np.complex128)
        H = np.asarray(fdtd.getresult(monitor_name, "H")["H"], dtype=np.complex128)
        pz = 0.5 * np.real(E[..., 0] * np.conj(H[..., 1]) - E[..., 1] * np.conj(H[..., 0]))
        return max(float(flux_sign) * float(np.sum(pz)) * float(target["monitor_cell_area"]), 0.0)
    except Exception:
        return None


def format_design_plot_status(f0_vals=None, last_plot_state=None):
    lines = []
    if f0_vals is not None:
        vals = np.asarray(f0_vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lines.append(f"FoM mean={np.mean(vals):.3e}, min={np.min(vals):.3e}, max={np.max(vals):.3e}")
    metrics = last_plot_state.get("last_fom_metrics") if isinstance(last_plot_state, dict) else None
    if isinstance(metrics, dict):
        for label, key in (("throughput", "throughput"), ("match", "match")):
            val = metrics.get(key)
            if val is not None and np.isfinite(val):
                lines.append(f"{label}={float(val):.3f}")
    return "\n".join(lines)


def save_current_design_sections(G, design, f0_vals=None, mapping=None, last_plot_state=None):
    rho = design_to_grid(G, npa.clip(design, 0.0, 1.0), mapping)
    Nx, Ny, Nz = G.design_grids
    x_axis = np.linspace(-0.5 * G.design_s[0], 0.5 * G.design_s[0], Nx)
    y_axis = np.linspace(-0.5 * G.design_s[1], 0.5 * G.design_s[1], Ny)
    z_axis = np.linspace(G.design_c[2] - 0.5 * G.design_s[2], G.design_c[2] + 0.5 * G.design_s[2], Nz)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].imshow(rho[:, :, Nz // 2].T, origin="lower", extent=(x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]), cmap="binary", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("y (um)")
    axes[0].set_title("x-y section at z=center")

    axes[1].imshow(rho[:, Ny // 2, :].T, origin="lower", extent=(x_axis[0], x_axis[-1], z_axis[0], z_axis[-1]), cmap="binary", vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("z (um)")
    axes[1].set_title("x-z section at y=0")

    status_text = format_design_plot_status(f0_vals, last_plot_state)
    fig.suptitle("Current design sections")
    if status_text:
        fig.text(0.5, 0.02, status_text, ha="center", va="bottom", fontsize=8.5)
        fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.92))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    path = os.path.join(G.design_dir, "design_iter_temp.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def adjoint_loop(G, opts, iter_plot_fn=None, combine=None):
    """Adjoint-loop factory for ms.Opt_MS2.OPT_Ms, ported from OLED_new.py.

    iter_plot_fn(X, vals), if given, is called once per non-string forward
    evaluation at exactly the point where OLED_new.py saved its per-iteration
    design-section / field-snapshot / emission plots; the calling script
    supplies a callback that performs its own plotting there.  combine, if
    given, replaces the default mean combine_fom (OLED_lens uses a
    channel-weighted combiner); it is also the function differentiated by
    autograd in the Case == 3 branch.
    """
    if combine is None:
        def combine(vals):
            return combine_fom(G, vals)

    N_fom = len(opts)

    def loop(X, N_cases, Case=True):
        if Case == 3:
            vals = npa.maximum(npa.asarray(clean_fom_values(G, N_cases)), G.fom_floor)
            coeffs = npa.where(npa.isfinite(ag_jacobian(combine)(vals)), ag_jacobian(combine)(vals), 0.0)
            grad = 0.0
            for c, g in zip(coeffs, X[0]):
                grad = grad + c * npa.where(npa.isfinite(npa.array(g)), npa.array(g), 0.0)
            print(f"[outcoupling] combined grad max={np.max(np.abs(grad)):.6e}")
            return npa.where(npa.isfinite(grad), grad, 0.0)

        f0s, grads = [0] * N_fom, [0] * N_fom
        for i, opt in enumerate(opts):
            if isinstance(X, str):
                f0s[i], grads[i] = opt(need_gradient=Case)
            else:
                f0s[i], grads[i] = opt(rho_vector=[npa.clip(X, 0.0, 1.0)], need_gradient=Case)

        if any(getattr(opt, "last_forward_had_nonfinite", False) for opt in opts):
            zero_grads = [np.zeros_like(g, dtype=float) if not isinstance(g, (int, float)) else np.zeros(G.design_cells) for g in grads]
            return (G.unstable_candidate_fom, [G.fom_floor] * N_fom, zero_grads) if Case and not isinstance(X, str) else zero_grads

        vals = clean_fom_values(G, f0s)

        if not isinstance(X, str) and iter_plot_fn is not None:
            try:
                iter_plot_fn(X, vals)
            except Exception as exc:
                print(f"[outcoupling] skipped per-iteration plots: {exc}")

        combined = combine(vals)
        print(f"[outcoupling] combined FoM={combined:.6e}, mean={np.mean(vals):.6e}, min={np.min(vals):.6e}, max={np.max(vals):.6e}")
        if Case:
            return grads if isinstance(X, str) else (combined, f0s, grads)
        return combined, f0s

    return loop


def save_xz_monitor_field_snapshot(G, problem=None):
    try:
        sim = getattr(problem, "sim", None) if problem is not None else None
        if sim is None or not hasattr(sim, "fdtd"):
            return None
        return render_xz_field_image(
            sim.fdtd.getresult(G.xyz_monitor_name, "E"),
            os.path.join(G.design_dir, "xz_monitor_field.png"),
            "XZ monitor plane |E| (full simulation cross-section)",
        )
    except Exception:
        return None


def save_fom_monitor_field_snapshot(G, problem=None):
    try:
        if problem is None:
            return None
        fdtd = getattr(problem, "sim", None)
        if fdtd is None or not hasattr(fdtd, "fdtd"):
            return None
        fdtd_obj = fdtd.fdtd
        res = fdtd_obj.getresult(G.target_monitor_name, "E")
        E = np.asarray(res["E"], dtype=np.complex128)
        if E.size == 0:
            return None
        field_mag = np.sqrt(np.sum(np.abs(E) ** 2, axis=-1))
        if field_mag.ndim >= 2:
            field_mag = np.squeeze(field_mag)
        x = np.asarray(res.get("x", np.arange(field_mag.shape[1])), dtype=float).reshape(-1)
        y = np.asarray(res.get("y", np.arange(field_mag.shape[0])), dtype=float).reshape(-1)
        if field_mag.ndim == 1:
            field_mag = field_mag.reshape(1, -1)
        if field_mag.shape[0] != y.size or field_mag.shape[1] != x.size:
            field_mag = np.reshape(field_mag, (y.size, x.size))
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        im = ax.imshow(field_mag.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="viridis", aspect="equal")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_title("FoM monitor plane |E|")
        fig.colorbar(im, ax=ax, pad=0.03)
        fig.tight_layout()
        path = os.path.join(G.design_dir, "fom_monitor_field.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path
    except Exception:
        return None


def save_angular_target_preview(G, angular_target, file_prefix="OLED_angular_target"):
    # The FoM basis: the continuous linear-ramp target across the in-range emission
    # angles (control points marked), normalized to sum 1 in-range.
    thetas = np.asarray(angular_target["angle_thetas"], dtype=float)
    target = np.asarray(angular_target["target_profile"], dtype=float)
    inr = np.asarray(angular_target["in_range"], dtype=float) > 0.5
    ctrl_a = np.asarray(angular_target["spec_snapped"], dtype=float)
    ctrl_r = np.asarray(angular_target["ctrl_ratios"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(thetas[inr], target[inr], width=1.8, color="tab:red", alpha=0.75, label="target profile (in-range)")
    ax.plot(thetas, np.zeros_like(thetas), "k|", markersize=14, label="available angles")
    # Show the underlying ramp (control ratios) on a twin axis for reference.
    axr = ax.twinx()
    axr.plot(ctrl_a, ctrl_r, "o--", color="tab:blue", lw=1.5, label="control ratio ramp")
    axr.set_ylabel("control ratio (relative)", color="tab:blue")
    axr.set_ylim(0.0, 1.1 * max(float(np.max(ctrl_r)), 1e-9))
    for th, t in zip(thetas[inr], target[inr]):
        ax.annotate(f"{th:.1f}\n{t:.2f}", (th, t), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xlabel("emission angle theta (deg)")
    ax.set_ylabel("target power fraction (in-range, sums to 1)")
    basis = "2D k-space (all m,n orders)" if angular_target.get("mode") == "kspace_2d" else "phi=0 line (m,0 orders)"
    ax.set_title(f"Angular target profile [{basis}]: linear ratio ramp")
    ax.set_xlim(-3.0, 92.0)
    ax.set_ylim(0.0, max(1.35 * float(np.max(target)) if target.size else 1.0, 0.05))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    path = os.path.join(G.design_dir, f"{file_prefix}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    np.savez(os.path.join(G.design_dir, f"{file_prefix}.npz"), angle_thetas=thetas, target_profile=target, in_range=inr)
    print(f"[target] saved angular target profile: {path}")


def save_optimization_emission_plot(G, target_channels=None, last_plot_state=None):
    # Save the single-channel emission performance versus angle as a PNG.
    ch = target_channels[0] if target_channels else None
    prof = None
    at = None
    if isinstance(ch, dict):
        prof = ch.get("last_angle_profile")
        at = ch.get("angular_target")
    if prof is None and isinstance(last_plot_state, dict):
        prof = last_plot_state.get("last_angle_profile")
        at = last_plot_state.get("angular_target")
    if prof is None and at is None:
        fig, ax = plt.subplots(figsize=(8.5, 4))
        ax.text(0.5, 0.5, "No emission profile available", ha="center", va="center")
        ax.set_axis_off()
        path = os.path.join(G.design_dir, "OLED_opt_emission.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    if prof is not None and at is not None:
        achieved = np.asarray(prof, dtype=float).reshape(-1)
        thetas = np.asarray(at["angle_thetas"], dtype=float).reshape(-1)
        inr = np.asarray(at["in_range"], dtype=float).reshape(-1) > 0.5
        tgt = np.asarray(at["target_profile"], dtype=float).reshape(-1)
        thru = float(np.sum(achieved[inr]))
        target = tgt * thru
        q = achieved / max(thru, 1e-30)
        match = float(np.sum(np.minimum(q[inr], tgt[inr])))
        fom = thru * match ** G.ratio_emphasis
    else:
        achieved = np.asarray([0.0], dtype=float)
        thetas = np.asarray([0.0], dtype=float)
        inr = np.asarray([True], dtype=bool)
        tgt = np.asarray([0.0], dtype=float)
        target = np.asarray([0.0], dtype=float)
        q = np.asarray([0.0], dtype=float)
        thru = 0.0
        match = 0.0
        fom = 0.0

    fig, ax = plt.subplots(figsize=(8.5, 4))
    xpos = np.arange(thetas.size)
    bw = 0.4
    ax.bar(xpos - 0.5 * bw, target, width=bw, label=r"target $t_k\cdot T$ (linear ramp)", color="tab:orange")
    ax.bar(xpos + 0.5 * bw, achieved, width=bw, label="achieved (of total)", color="tab:blue")
    for idx in np.where(inr)[0]:
        ax.annotate(f"q={q[idx]:.2f}", (xpos[idx] + 0.5 * bw, achieved[idx]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color="tab:blue")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{t:.1f}" for t in thetas])
    ax.set_xlabel("emission angle theta (deg)")
    ax.set_ylabel("angular mode-power fraction (of total)")
    basis = "2D k-space" if (at.get("mode") == "kspace_2d") else "phi=0 line"
    ax.set_title(f"Angular mode power [{basis}], opt dipoles - FoM={fom:.3f}  thru={thru:.3f}  match={match:.3f}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(G.design_dir, "OLED_opt_emission.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    np.savez(os.path.join(G.design_dir, "OLED_opt_emission.npz"),
             angle_thetas=thetas, achieved=achieved, target=target, q=q, throughput=thru, in_range=inr.astype(float),
             fom=np.asarray([fom]), throughput_metric=np.asarray([thru]), match_metric=np.asarray([match]))
    return path
