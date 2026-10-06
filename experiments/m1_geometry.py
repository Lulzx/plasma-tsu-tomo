"""M1: grid, D-shaped mask, chord fans, geometry-matrix statistics, Siddon disk validation, phantom library."""
from __future__ import annotations

from _common import *  # noqa: F401,F403  (sets sys.path, Agg backend)
from _common import (CMAP_FIELD, OI, build_problem, field_grid, log, make_parser, md_table, plt, save_fig,
                     save_json, setup, np)

SCRIPT = "m1_geometry"


def siddon_disk_validation(n=256, n_chords=400, seed=3):
    """Aggregate Siddon line integral of a uniform disk vs the analytic chord length (random chords)."""
    from tomo.geometry import Chords, geometry_matrix, make_grid
    g = make_grid(n)
    cx, cy, r = 0.5, 0.5, 0.4
    img = (((g.X - cx) ** 2 + (g.Y - cy) ** 2) <= r ** 2).astype(float).ravel()
    rng = np.random.default_rng(seed)
    p0, p1 = rng.uniform(0, 1, (n_chords, 2)), rng.uniform(0, 1, (n_chords, 2))
    T = geometry_matrix(Chords(p0, p1, np.zeros(n_chords, int)), g, dtype=np.float64)
    num, ana = T @ img, np.zeros(n_chords)
    for i in range(n_chords):
        d = p1[i] - p0[i]
        L = float(np.hypot(*d))
        u = d / L
        w = p0[i] - [cx, cy]
        bq, cq = w @ u, w @ w - r * r
        disc = bq * bq - cq
        if disc > 0:
            s0, s1 = max(-bq - np.sqrt(disc), 0), min(-bq + np.sqrt(disc), L)
            ana[i] = max(s1 - s0, 0)
    hit = ana > 0.05
    rel = np.abs(num[hit] / ana[hit] - 1)
    return {"grid_n": n, "n_chords_hit": int(hit.sum()), "sum_ratio_minus_1": float(num[hit].sum() / ana[hit].sum() - 1),
            "median_abs_rel_err": float(np.median(rel)), "max_abs_rel_err": float(rel.max()),
            "note": "uniform disk r=0.4 at (0.5,0.5), 256^2 grid, random chords; pixel-discretised disk vs analytic"}


def main():
    ap = make_parser(SCRIPT, __doc__)
    args = ap.parse_args()
    cfg, out = setup(args, SCRIPT)
    p = build_problem(cfg, "peaked", cfg["seed"])
    g, mask, T, ch = p.grid, p.mask, p.T, p.chords
    cam_names = [c["name"] for c in cfg["cameras"]]
    cam_cols = [OI["orange"], OI["sky"], OI["green"], OI["verm"], OI["purple"]]

    # --- figure 1: grid, mask, chord fans
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    ax = axs[0]
    ax.imshow(mask, origin="lower", extent=(g.xmin, g.xmax, g.ymin, g.ymax), cmap="Greys", alpha=0.35)
    for x in np.linspace(g.xmin, g.xmax, g.n + 1):
        ax.axvline(x, color="0.8", lw=0.3)
        ax.axhline(x, color="0.8", lw=0.3)
    ax.set_title(f"{g.n}x{g.n} grid, D-mask: {int(mask.sum())} active pixels")
    ax = axs[1]
    ax.imshow(mask, origin="lower", extent=(g.xmin, g.xmax, g.ymin, g.ymax), cmap="Greys", alpha=0.25)
    for ci, name in enumerate(cam_names):
        sel = ch.camera == ci
        for a, b in zip(ch.p0[sel], ch.p1[sel]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=cam_cols[ci], lw=0.5, alpha=0.7)
        ax.plot([], [], color=cam_cols[ci], label=f"{name} ({int(sel.sum())})")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title(f"{len(ch.p0)} chords in {len(cam_names)} fans")
    ax = axs[2]
    cnt = (np.asarray(T) > 0).sum(0)
    from tomo.metrics import to_img
    im = ax.imshow(to_img(cnt, mask, fill=np.nan), origin="lower", extent=(g.xmin, g.xmax, g.ymin, g.ymax), cmap=CMAP_FIELD)
    fig.colorbar(im, ax=ax, label="chords crossing pixel (count)")
    ax.set_title("coverage per pixel")
    for ax in axs:
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
        ax.set_xlim(g.xmin, g.xmax)
        ax.set_ylim(g.ymin, g.ymax)
    save_fig(fig, out, "geometry_fans.png")

    # --- geometry matrix stats
    Tn = np.asarray(T, float)
    sv = np.linalg.svd(Tn, compute_uv=False)
    cnt_row = (Tn > 0).sum(1)
    stats = {"M_chords": int(Tn.shape[0]), "N_active": int(Tn.shape[1]), "n_grid": int(g.n),
             "nnz": int((Tn > 0).sum()), "density": float((Tn > 0).mean()),
             "pixels_per_chord_mean": float(cnt_row.mean()), "pixels_per_chord_max": int(cnt_row.max()),
             "chords_per_pixel_mean": float(cnt.mean()), "chords_per_pixel_max": int(cnt.max()),
             "frac_pixels_uncrossed": float(np.mean(cnt == 0)),
             "chord_length_mean_m": float(Tn.sum(1).mean()), "rank": int((sv > sv[0] * 1e-10).sum()),
             "sv_max": float(sv[0]), "sv_min": float(sv[-1]), "cond": float(sv[0] / sv[-1]),
             "T_bytes_float32": int(T.nbytes), "ill_posed_ratio_N_over_M": float(Tn.shape[1] / Tn.shape[0])}
    # --- Siddon validation
    val = siddon_disk_validation()
    log("Siddon disk validation:", val)
    save_json({"geometry_matrix": stats, "siddon_disk_validation": val}, f"{out}/geometry_stats.json")
    with open(f"{out}/geometry_stats.md", "w") as f:
        f.write(md_table([{"quantity": k, "value": v} for k, v in stats.items()], ["quantity", "value"]))
        f.write("\nSiddon validation: " + ", ".join(f"{k}={v}" for k, v in val.items()) + "\n")
    # singular values
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.semilogy(sv, color=OI["blue"])
    ax.set_xlabel("index")
    ax.set_ylabel("singular value of T (m)")
    ax.set_title("geometry matrix spectrum")
    save_fig(fig, out, "geometry_singular_values.png")

    # --- phantom library: fine and coarse truth
    names = cfg["experiments"]["phantoms"] + ["random"]
    rows = []
    for nm in names:
        pp = build_problem(cfg, nm, cfg["seed"] if nm != "random" else cfg["experiments"]["seed0"])
        rows.append((pp.fine_img, pp.eps_true_img, pp))
    fig, axs = plt.subplots(2, len(names), figsize=(2.6 * len(names), 5.2), squeeze=False)
    for j, (nm, (fine, coarse, pp)) in enumerate(zip(names, rows)):
        vmax = float(fine.max())
        for i, (img, gg, lab) in enumerate([(np.where(pp.fine_mask, fine, np.nan), pp.fine_grid, "fine 128x128"),
                                            (np.where(pp.mask, coarse, np.nan), pp.grid, "coarse truth 32x32")]):
            im = axs[i, j].imshow(img, origin="lower", extent=(gg.xmin, gg.xmax, gg.ymin, gg.ymax), vmin=0, vmax=vmax, cmap=CMAP_FIELD)
            axs[i, j].set_title(f"{nm}: {lab}", fontsize=8)
            axs[i, j].set_xlabel("R (m)")
            axs[i, j].set_ylabel("Z (m)")
            fig.colorbar(im, ax=axs[i, j], fraction=0.046, label="emissivity (a.u.)")
    fig.tight_layout()
    save_fig(fig, out, "phantoms.png")
    log("done")


if __name__ == "__main__":
    main()
