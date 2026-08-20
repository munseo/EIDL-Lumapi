"""Figures for OLED_rec.py -- presentation only, no physics.

Split out of OLED_rec.py to keep that file about the FoM and the optimization.
Nothing here is imported by the objective functions; deleting this module would
cost pictures, not results.

CONFIGURATION IS INJECTED, not imported. OLED_rec is a configured experiment
script whose settings live in its module globals, and these figures read sixteen
of them. Importing OLED_rec from here would be circular, and threading sixteen
arguments through five plotting calls would be worse than the coupling it
removes, so bind() adopts the caller's namespace once instead. The code below is
byte-identical to what lived in OLED_rec, which is the point: moving it must not
be able to change a figure.

Only bind()'s names are shared, and every one of them is read-only here except
_FOM_TRACE, which is mutated IN PLACE by OLED_rec._plot_state -- so both modules
see the same list, which is what the history plot needs.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import oled_common as oc

_NEEDS = ("G", "LAM", "MULTIOBJ", "N_EML", "N_ORG", "OBJ_SPECS", "PERIOD",
          "SOURCE_POL", "SYMMETRY", "TARGET_MAX_ANGLE", "_FOM_TRACE", "_SM",
          "_ST", "eml_c", "mapping", "target_modes")


def bind(ns):
    """Adopt the caller's configuration. Call once, after its globals exist."""
    missing = [k for k in _NEEDS if k not in ns]
    if missing:
        raise KeyError(f"oled_rec_plots.bind is missing {missing}")
    globals().update({k: ns[k] for k in _NEEDS})


def _draw_stack(ax, x0, x1, show_names=True):
    """The layer stack in cross-section, drawn from the geometry actually built."""
    for L in G.stack_layers:
        zc, dz = L["center"][2], L["size"][2]
        nr = float(L["index"]["n"][0])
        kr = float(L["index"].get("k", [0.0])[0])
        metal = kr > 0.5
        ax.add_patch(plt.Rectangle((x0, zc - 0.5 * dz), x1 - x0, dz,
                                   facecolor="#8a8a8a" if metal else
                                   plt.cm.Blues(0.15 + 0.55 * (nr - 1.7) / 0.6),
                                   edgecolor="none", zorder=1))
        if show_names and dz > 0.02 and L["name"] != "EML":
            ax.text(x1 - 0.02 * (x1 - x0), zc, f"{L['name']}  n={nr:.2f}",
                    fontsize=6.0, ha="right", va="center",
                    color="w" if metal else "#123", zorder=6)
    dz = G.design_s[2]
    ax.add_patch(plt.Rectangle((x0, G.design_c[2] - 0.5 * dz), x1 - x0, dz,
                               facecolor="#ffd9a0", edgecolor="#e08b00",
                               hatch="///", lw=1.2, zorder=2))
    ax.text(x0 + 0.02 * (x1 - x0), G.design_c[2], "design (out-coupler)",
            fontsize=7, va="center", color="#7a4a00", zorder=6)
    ax.axhline(eml_c[2], color="#b00020", lw=1.4, ls="--", zorder=4)
    ax.text(x0 + 0.02 * (x1 - x0), eml_c[2] - 0.035,
            f"EML plane  n={N_EML:.2f}", fontsize=7, color="#b00020", zorder=6)


def plot_kmap_geometry(path=None):
    """What the pitch k_mapping chose actually does, in real space.

    Left  the forward run this script optimizes: one normal-incidence plane wave
          from air, diffracted by the design into the discrete internal angles.
    Right the emission it stands for. By reciprocity the same coefficient governs
          a ray leaving the EML at that internal angle and exiting into air at the
          target angle -- which is the pairing the FoM is really requesting.
    """
    path = path or os.path.join(G.design_dir, "OLED_rec_kmap_geometry.png")
    P = float(PERIOD)
    x0, x1 = -0.62 * P, 0.62 * P
    z_des_top = G.design_c[2] + 0.5 * G.design_s[2]
    z_des_bot = G.design_c[2] - 0.5 * G.design_s[2]
    # Air is trimmed to what the outgoing rays need. The panels are set to EQUAL
    # aspect further down so the drawn angles ARE the angles -- with the default
    # auto aspect a 23 deg ray in a 0.4 um stack under a 2.2 um period renders
    # almost vertical, which is exactly the thing this figure exists to show.
    z_top = z_des_top + 0.62
    z_bot = G.Z_min
    z_anode = max(L["center"][2] + 0.5 * L["size"][2] for L in G.stack_layers
                  if "anode" in L["name"].lower()) if any(
        "anode" in L["name"].lower() for L in G.stack_layers) else z_bot

    tgt, seen = [], set()
    for mo in target_modes:
        if mo["suppress"] or round(mo["u"], 9) in seen:
            continue
        seen.add(round(mo["u"], 9))
        tgt.append(mo)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, max(len(tgt), 1)))

    fig, (aL, aR) = plt.subplots(1, 2, figsize=(14.0, 6.0), sharey=True)
    for ax in (aL, aR):
        _draw_stack(ax, x0, x1, show_names=(ax is aR))
        ax.set_xlim(x0, x1)
        ax.set_ylim(z_bot, z_top)
        ax.set_aspect("equal")                    # so the drawn angles are the angles
        ax.set_xlabel(r"x ($\mu$m)")
        for xb in (-0.5 * P, 0.5 * P):
            ax.axvline(xb, color="#e08b00", lw=0.9, ls=":", zorder=3)
        zp = z_des_top + 0.07
        ax.annotate("", xy=(0.5 * P, zp), xytext=(-0.5 * P, zp),
                    arrowprops=dict(arrowstyle="<->", color="#e08b00", lw=1.4), zorder=6)
        ax.text(0.0, zp + 0.02, f"P = {P:.4f} $\\mu$m", ha="center", fontsize=8.5,
                color="#7a4a00", zorder=6)
    aL.set_ylabel(r"z ($\mu$m)")

    # ---- left: the forward probe ---------------------------------------------
    for xs in np.linspace(-0.5 * P, 0.5 * P, 7):
        aL.annotate("", xy=(xs, z_des_top), xytext=(xs, z_top - 0.06),
                    arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=1.5), zorder=5)
    aL.text(0.0, z_top - 0.04, r"normal-incidence probe, $\theta_{air}=0^\circ$",
            ha="center", va="top", fontsize=9, color="#1f4e79", zorder=6)
    # Carry the diffracted rays all the way to the anode mirror, not just to the
    # EML: over the 0.14 um from the design to the EML even a 23 deg ray shifts by
    # 0.06 um and reads as vertical.
    run = z_des_bot - z_anode
    for c, mo in zip(cols, tgt):
        t = np.deg2rad(mo["theta_eml_deg"])
        dx = run * np.tan(t)
        for sgn in ((1, -1) if dx > 1e-9 else (1,)):
            aL.annotate("", xy=(sgn * dx, z_anode), xytext=(0.0, z_des_bot),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8), zorder=5)
            xe = (z_des_bot - eml_c[2]) * np.tan(t) * sgn
            aL.plot([xe], [eml_c[2]], "o", ms=4.5, color=c, mec="w", mew=0.6, zorder=7)
        aL.plot([], [], "-", color=c, lw=1.8,
                label=f"m={max(mo['m'], mo['n'])}: $u$={mo['u']:.3f}, "
                      f"$\\theta_{{EML}}$={mo['theta_eml_deg']:.1f}$^\\circ$")
    aL.set_title("forward: 0$^\\circ$ plane wave $\\rightarrow$ target diffraction "
                 "angles in the EML", fontsize=10)
    aL.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    # ---- right: the emission it is equivalent to ------------------------------
    rise = z_des_bot - eml_c[2]
    out = z_top - z_des_top - 0.10
    for c, mo in zip(cols, tgt):
        ti, ta = np.deg2rad(mo["theta_eml_deg"]), np.deg2rad(mo["theta_air_deg"])
        xs = -0.30 * P                            # common launch point on the EML
        xm = xs + rise * np.tan(ti)
        aR.annotate("", xy=(xm, z_des_bot), xytext=(xs, eml_c[2]),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8), zorder=5)
        aR.plot([xm, xm], [z_des_bot, z_des_top], "-", color=c, lw=1.0,
                alpha=0.5, zorder=5)
        xe = xm + out * np.tan(ta)
        aR.annotate("", xy=(xe, z_des_top + out), xytext=(xm, z_des_top),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8), zorder=5)
        aR.text(xe, z_des_top + out + 0.03, f"{mo['theta_air_deg']:.0f}$^\\circ$",
                color=c, fontsize=9, ha="center", va="bottom", zorder=6,
                fontweight="bold")
        aR.plot([], [], "-", color=c, lw=1.8,
                label=f"{mo['theta_eml_deg']:.1f}$^\\circ$ in EML "
                      f"$\\rightarrow$ {mo['theta_air_deg']:.1f}$^\\circ$ in air")
    aR.plot([-0.30 * P], [eml_c[2]], "*", ms=13, color="#b00020", zorder=7)
    aR.set_title("emission (reciprocal): each target order $\\rightarrow$ its air angle",
                 fontsize=10)
    aR.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    fig.suptitle(f"k-mapping result: pitch {P:.4f} $\\mu$m "
                 f"($\\lambda$={LAM:g} $\\mu$m, $\\lambda/P$={LAM / P:.4f}), "
                 f"{len(tgt)} target order(s), {SYMMETRY} symmetry", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] k-map geometry -> {path}")


def plot_angle_mapping(path=None):
    """Free-space emission angle vs the EML-internal angle it is fed by.

    The whole formulation lives in u = k_par/k0 because that is what a planar stack
    conserves; this is the picture of what that means in angles. u = sin(theta_air)
    = n_org*sin(theta_EML), so the internal angles are compressed by refraction --
    the entire escape cone is theta_EML < asin(1/n_org), and everything past it is
    trapped no matter what the out-coupler does.
    """
    path = path or os.path.join(G.design_dir, "OLED_rec_angle_mapping.png")
    n = float(N_ORG)
    th_air = np.linspace(0.0, 90.0, 721)
    u = np.sin(np.deg2rad(th_air))
    th_org = np.rad2deg(np.arcsin(np.clip(u / n, 0.0, 1.0)))
    crit = float(np.rad2deg(np.arcsin(min(1.0 / n, 1.0))))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))
    ax.plot(th_air, th_org, "-", color="#1f4e79", lw=2.0,
            label=f"$u=\\sin\\theta_{{air}}=n\\,\\sin\\theta_{{EML}}$, n={n:.3f}")
    ax.axhline(crit, color="#b00020", ls="--", lw=1.4,
               label=f"escape cone limit  {crit:.2f}$^\\circ$")
    ax.fill_between([0, 90], crit, 90, color="#b00020", alpha=0.07)
    tgt = [m for m in target_modes if not m["suppress"]]
    sup = [m for m in target_modes if m["suppress"]]
    seen = set()
    for mo in tgt:
        key = round(mo["u"], 9)
        if key in seen:
            continue
        seen.add(key)
        ax.plot([mo["theta_air_deg"]], [mo["theta_org_deg"]], "o", ms=8,
                color="#e08b00", zorder=5)
        ax.annotate(f"m={max(mo['m'], mo['n'])}\n{mo['theta_air_deg']:.1f}$^\\circ$"
                    f"$\\to${mo['theta_org_deg']:.1f}$^\\circ$",
                    (mo["theta_air_deg"], mo["theta_org_deg"]),
                    textcoords="offset points", xytext=(6, -22), fontsize=8)
    for mo in sup:
        ax.plot([mo["theta_air_deg"]], [mo["theta_org_deg"]], "x", ms=9,
                color="#b00020", mew=2, zorder=5)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, max(90.0 / 3.0, crit * 1.35))
    ax.set_xlabel(r"free-space emission angle  $\theta_{air}$  (deg)")
    ax.set_ylabel(r"matching angle inside the EML  $\theta_{EML}$  (deg)")
    ax.set_title("angle mapping: what the design actually has to steer")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    # Same information as momentum, which is where the orders are evenly spaced.
    ax2.axhline(1.0, color="#b00020", ls="--", lw=1.4, label="air light line  u=1")
    ax2.axhline(n, color="#444", ls=":", lw=1.4, label=f"$n_{{org}}$={n:.3f}")
    ax2.fill_between([-0.5, len(tgt) - 0.5], 1.0, n, color="#b00020", alpha=0.07)
    ax2.text(len(tgt) / 2.0 - 0.5, (1.0 + n) / 2.0, "trapped\n(waveguide / SPP)",
             ha="center", va="center", fontsize=9, color="#b00020")
    for k, mo in enumerate(tgt):
        ax2.plot([k], [mo["u"]], "o", ms=8, color="#e08b00", zorder=5)
        ax2.annotate(f"({mo['m']},{mo['n']})\nu={mo['u']:.3f}", (k, mo["u"]),
                     textcoords="offset points", xytext=(0, 9), fontsize=7.5,
                     ha="center", linespacing=1.15)
    ax2.set_xticks(range(len(tgt)))
    ax2.set_xticklabels([f"{m['theta_air_deg']:.0f}$^\\circ$\n$\\phi$={m['phi_deg']:.0f}"
                         for m in tgt], fontsize=8)
    ax2.set_ylim(0, n * 1.08)
    ax2.set_xlim(-0.5, len(tgt) - 0.5)
    ax2.set_ylabel(r"in-plane momentum  $u=k_\parallel/k_0$")
    ax2.set_title(f"targets in momentum, pitch {PERIOD:.4f} $\\mu$m "
                  f"($\\lambda/P$={LAM / PERIOD:.4f})")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] angle mapping -> {path}")


def plot_iteration_state(rho, purities, level, match, it, path=None, beta=None,
                         fom=float("nan"), trial=False):
    """Structure and per-order purity side by side, once per iteration.

    Plotted together on purpose: the question being asked of every iteration is
    which feature of the structure is feeding which order, and that is unreadable
    from either panel alone.
    """
    # One rolling file, overwritten each iteration -- the iteration number lives in
    # the title, not in a pile of filenames.
    path = path or os.path.join(G.design_dir, "OLED_rec_current.png")
    try:
        # oled_common.design_to_grid already accepts all three input sizes OPT_Ms
        # can hand over -- the full voxel vector, the mapped-parameter vector, or an
        # Nx*Ny sheet -- so use it instead of calling mapping() again. Doing that by
        # hand is what fed 204304 values to a radial map wanting its 912 parameters
        # ("cannot reshape array of size 204304 into shape (57,16)").
        vol = oc.design_to_grid(G, np.asarray(rho, float).ravel(), mapping)
    except Exception as exc:                          # never kill a run over a plot
        print(f"[plot] structure panel skipped: {exc}")
        vol = None

    tgt = [(k, mo) for k, mo in enumerate(target_modes) if not mo["suppress"]]
    # Column 2 is three stacked x-z cross-sections instead of one panel. The
    # earlier x-z-at-y=0 / y-z-at-x=0 pairing was redundant -- Sym_geo_C8
    # imposes D4/C4v (fliplr + flipud + transpose), so those two cuts are
    # identical by construction -- so this instead shows the SAME kind of cut
    # (x-z, i.e. constant y) at three different y, which is actually new
    # information: how the profile changes moving across the cell.
    #
    # The design region is wide and thin (period ~um across, a fraction of
    # that tall), so an aspect="equal" cut wants to render far wider than one
    # unit-width column can hold -- aspect="auto" (the original choice) fills
    # the column but draws the cross-section at the WRONG aspect ratio
    # instead, which is the actual bug being fixed here. Sizing the column
    # from the real x:z ratio (rather than guessing a fixed width_ratio) is
    # what stops the aspect-correct image (and its colorbar) from spilling
    # into column 3 -- the failure mode the first attempt at this had, at a
    # column width picked before checking whether it was wide enough.
    aspect_xz = float(G.design_s[0]) / max(float(G.design_s[2]), 1e-9)
    BASE_COL_W = 4.6            # approx. inches one width_ratio=1.0 unit renders at
    row_h_est = 3.3 / 3.0       # rough usable plot height per row, inches
    col2_ratio = max(row_h_est * aspect_xz / BASE_COL_W, 1.0)
    width_ratios = [1.0, col2_ratio, 1.05, 1.2]
    fig = plt.figure(figsize=(BASE_COL_W * sum(width_ratios), 4.6))
    outer_gs = fig.add_gridspec(1, 4, width_ratios=width_ratios)
    a1 = fig.add_subplot(outer_gs[0, 0])
    # A narrow SECOND inner column holds one colorbar axis spanning all three
    # rows -- carved from the same subgridspec as the image axes, so it is
    # guaranteed to stay inside column 2's own bounds instead of relying on
    # fig.colorbar's automatic ax-shrinking (matplotlib's make_axes), which is
    # what let the previous version's colorbar drift past the column boundary.
    inner_gs = outer_gs[0, 1].subgridspec(3, 2, width_ratios=[1.0, 0.05],
                                          hspace=0.75, wspace=0.25)
    a1b = fig.add_subplot(inner_gs[0, 0])
    a1c = fig.add_subplot(inner_gs[1, 0])
    a1d = fig.add_subplot(inner_gs[2, 0])
    cax = fig.add_subplot(inner_gs[:, 1])
    a2 = fig.add_subplot(outer_gs[0, 2])
    a3 = fig.add_subplot(outer_gs[0, 3])
    # Same sections oled_common.save_current_design_sections uses, and the same
    # binary colormap: a z-averaged single panel hid whether the design had actually
    # binarized or was still grey, and hid any z structure entirely.
    if vol is not None:
        Nx, Ny, Nz = G.design_grids
        xa = np.linspace(-0.5 * G.design_s[0], 0.5 * G.design_s[0], Nx)
        ya = np.linspace(-0.5 * G.design_s[1], 0.5 * G.design_s[1], Ny)
        za = np.linspace(G.design_c[2] - 0.5 * G.design_s[2],
                         G.design_c[2] + 0.5 * G.design_s[2], Nz)
        gray = float(np.mean((vol > 1e-3) & (vol < 1 - 1e-3)))
        a1.imshow(vol[:, :, Nz // 2].T, origin="lower", cmap="binary",
                  extent=(xa[0], xa[-1], ya[0], ya[-1]), vmin=0.0, vmax=1.0,
                  aspect="equal", interpolation="nearest")
        a1.set_ylabel(r"y ($\mu$m)")
        a1.set_title(f"x-y at z=center   grey {gray*100:.1f}%")

        y_max = float(ya[-1])
        rows = [(a1b, 0.8 * y_max, r"0.8\,y_{max}"),
                (a1c, 0.5 * y_max, r"0.5\,y_{max}"),
                (a1d, 0.0, r"0")]
        im = None
        for ax, y_target, y_lbl in rows:
            iy = int(np.argmin(np.abs(ya - y_target)))
            im = ax.imshow(vol[:, iy, :].T, origin="lower", cmap="binary",
                           extent=(xa[0], xa[-1], za[0], za[-1]), vmin=0.0, vmax=1.0,
                           aspect="equal", interpolation="nearest")
            ax.set_ylabel(r"z ($\mu$m)")
            ax.set_title(f"x-z at $y={y_lbl}={ya[iy]:.3f}\\,\\mu$m", fontsize=9)
        fig.colorbar(im, cax=cax, label=r"$\rho$")
    a1.set_xlabel(r"x ($\mu$m)")
    a1d.set_xlabel(r"x ($\mu$m)")

    got = np.array([purities[k] for k, _mo in tgt], dtype=float)
    want = np.array([target_modes[k]["weight"] for k, _mo in tgt], dtype=float)
    want = want / max(want.sum(), 1e-30) * max(got.sum(), 1e-30)   # same total
    x = np.arange(len(tgt))
    a2.bar(x - 0.2, got, 0.4, color="#1f4e79", label="achieved")
    a2.bar(x + 0.2, want, 0.4, color="#e08b00", alpha=0.85,
           label="requested ratio (rescaled to the same total)")
    a2.set_xticks(x)
    a2.set_xticklabels([f"{mo['theta_air_deg']:.0f}$^\\circ$\n({mo['m']},{mo['n']})"
                        for _k, mo in tgt], fontsize=8)
    a2.set_ylabel("modal purity  $p_i$")
    a2.set_title(f"order purity   "
                 f"$\\Sigma$={got.sum():.4f}   level={level:.3f}   match={match:.3f}")
    a2.grid(True, alpha=0.3, axis="y")
    a2.legend(fontsize=8)

    # Third panel: the trend. FoM, level and match are ALL bounded to [0,1] now
    # (see level_score / match_J), so they share one axis instead of the old
    # FoM-vs-twin-axis-T*S split, which needed two scales because raw T*S was
    # not bounded. Trial points (msopt's line search) are marked apart from
    # accepted ones, because a run that looks like it is going backwards is
    # usually just showing the rejected probes alongside the accepted steps.
    if _FOM_TRACE:
        idx = np.array([r[0] for r in _FOM_TRACE], float)
        f = np.array([r[1] for r in _FOM_TRACE], float)
        lv = np.array([r[2] for r in _FOM_TRACE], float)
        mt = np.array([r[3] for r in _FOM_TRACE], float)
        a3.plot(idx, f, "-o", ms=4.5, color="#1f4e79", lw=1.6, label="FoM")
        a3.plot(idx, mt, "-", color="#7a4a00", lw=1.2, alpha=0.85, label="match")
        a3.plot(idx, lv, "-", color="#e08b00", lw=1.2, alpha=0.85, label="level")
        a3.axhline(f.max(), color="#2e7d32", ls="--", lw=1.0,
                   label=f"best FoM {f.max():.4f}")
        a3.set_xlabel("msopt iteration")
        a3.set_ylabel("score (all bounded 0-1)")
        a3.set_ylim(-0.02, 1.05)
        a3.set_title("FoM / match / level history (accepted steps)")
        a3.grid(True, alpha=0.3)
        a3.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(f"OLED_rec  --  iteration {it}"
                 f"{'  (line-search trial)' if trial else ''}   "
                 f"FoM={fom:.4f}   level={level:.3f}   match={match:.3f}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def iter_plot(state):
    """Per-iteration record of which mode is carrying the score."""
    hist = state.get("history") or []
    if not hist:
        return
    arr = np.asarray(hist, dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for k, s in enumerate(OBJ_SPECS):
        if k >= arr.shape[1]:
            break
        if s["kind"] == "merged":
            ax.plot(arr[:, k], lw=2.0, color="#1f4e79",
                    label=f"level x match, unweighted ({len(s['idx'])} order(s))")
            continue
        if s["kind"] == "trans":
            ax.plot(arr[:, k], lw=2.0, color="k",
                    label=f"captured level, weighted (_ST={_ST:.2f})")
            continue
        if s["kind"] == "match":
            ax.plot(arr[:, k], lw=2.0, color="#7a4a00",
                    label=f"distribution match, weighted (_SM={_SM:.2f}, "
                          f"{len(s['idx'])} order(s))")
            continue
        lead = target_modes[s["idx"][0]]
        lbl = (f"SUPPRESS >{TARGET_MAX_ANGLE:g} deg ({len(s['idx'])} order(s))"
               if s["kind"] == "suppress" else
               f"{lead['theta_air_deg']:g} deg air / {lead['theta_org_deg']:.1f} deg EML "
               f"(m,n)=({lead['m']},{lead['n']})")
        ax.plot(arr[:, k], lw=1.4, ls="--" if s["kind"] == "suppress" else "-",
                label=lbl)
    # Whatever _scalar_fom actually reports -- the MIN under minimax, not the
    # sum. The distinction was easy to overlook while three objectives were
    # plotted alongside it; with the merged FoM there can be as few as one
    # curve, and a "total" that sits above every curve on the plot would be
    # nothing but a misreading waiting to happen.
    mm = MULTIOBJ == "minimax"
    ax.plot(arr.min(axis=1) if mm else arr.sum(axis=1), "k--", lw=1.8,
            label="FoM (min)" if mm else "total FoM")
    ax.set_xlabel("FoM evaluation")
    ax.set_ylabel("per-objective score (weighted contribution)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"OLED_rec: EML momentum content, pitch {PERIOD:g} um, pol {SOURCE_POL}")
    fig.tight_layout()
    fig.savefig(os.path.join(G.design_dir, "OLED_rec_mode_history.png"), dpi=150)
    plt.close(fig)
