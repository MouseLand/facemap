import matplotlib.pyplot as plt
import torch
from fig_utils import *
from rastermap import sorting
from scipy.stats import wilcoxon, zscore

from facemap.neural_prediction import prediction_utils
from facemap.utils import bin1d, compute_varexp

yratio = 12 / 7.8
colors = [[0.5, 0.5, 0.5], [0.75, 0.75, 0.25], [0.9, 0.6, 0.25]]
lbls = ["keypoints", "motion PCs", "movie PCs"]


def panel_wavelets(data_path, db, ax):
    state_dict = torch.load(
        f"{data_path}/proc/models/model_{db['mname']}_{db['datexp']}_{db['blk']}.pth"
    )
    wavelets = state_dict["core.features.wavelet0.weight"].cpu().numpy().squeeze()
    cc = (zscore(wavelets, axis=1) @ zscore(wavelets, axis=1).T) / wavelets.shape[-1]
    cc_sort, isort = sorting.travelling_salesman(cc)[:2]
    wavelets = wavelets[isort]
    dyk = 0
    cmap = plt.get_cmap("copper")(np.linspace(0.2, 0.9, 10))
    for k in range(len(wavelets)):
        wv = wavelets[k].copy()
        wv -= wv.min()
        ax.plot(wv + dyk, color="k")  # cmap[9-k])
        dyk += wv.max() * 0.8
    ax.plot([0, 50], np.zeros(2) - 0.25, color="k")
    ax.text(0, 1, "convolution filters", transform=ax.transAxes, fontsize=12)
    ax.text(0, -0.01, "1 sec.", transform=ax.transAxes)
    ax.axis("off")


def panels_varexp(
    data_path, dbs, grid, trans, il, iexps=[2, 10], subsample=1, compute_binned=False
):
    mstrs = [f"{db['mname']}_{db['datexp']}_{db['blk']}" for db in dbs]
    ve_kps, ve_svds, ve_expls, xypos = [], [], [], []
    ve_overall = np.zeros((len(dbs), 3))
    ve_all = np.zeros((len(dbs), 3))
    ve_binned = np.zeros((len(dbs), 3))
    for iexp, mstr in enumerate(mstrs):
        ve_expl = np.load(f"{data_path}/proc/neuralpred/{mstr}_spks_test.npz")[
            "varexp_expl_neurons"
        ]
        ve_kp = np.load(f"{data_path}/proc/neuralpred/{mstr}_kp_pred_test.npz")[
            "varexp_neurons"
        ]
        ve_svd = np.load(f"{data_path}/proc/neuralpred/{mstr}_svd_pred_test.npz")[
            "varexp_neurons"
        ]

        if compute_binned:
            spks_test = np.load(f"{data_path}/proc/neuralpred/{mstr}_spks_test.npz")[
                "spks_test"
            ].T
            bin_size = 4
            spks_test_bin = bin1d(spks_test, bin_size, axis=0)
            print(spks_test.shape)

            for j in range(3):
                if j == 2:
                    spks_pred_test = np.load(
                        f"{data_path}/proc/neuralpred/{mstr}_kp_pred_test.npz"
                    )["spks_pred_test"].T
                else:
                    spks_pred_test = np.load(
                        f"{data_path}/proc/neuralpred/{mstr}_svd_pred_test.npz"
                    )["spks_pred_test"][1 - j]
                spks_pred_test_bin = bin1d(spks_pred_test, bin_size, axis=0)
                ve_binned[iexp, j] = compute_varexp(
                    spks_test_bin, spks_pred_test_bin
                ).mean()

        igood = ve_expl > 1e-3
        ve_n = ve_kp / ve_expl
        ve_n[~igood] = np.nan
        ve_kps.append(ve_n * 100)
        ve_n = ve_svd[:, -1] / ve_expl
        ve_n[~igood] = np.nan
        ve_svds.append(ve_n * 100)
        ve_expls.append(ve_expl.mean() * 100)
        ve_overall[iexp, 2] = (ve_kp.mean() / ve_expl.mean()) * 100
        ve_overall[iexp, :2] = (
            ve_svd[:, ::-1].mean(axis=0) / ve_expl[igood, np.newaxis].mean(axis=0)
        ) * 100
        ve_all[iexp, 2] = (ve_kp.mean()) * 100
        ve_all[iexp, :2] = (ve_svd[:, ::-1].mean(axis=0)) * 100

        xpos = np.load(f"{data_path}/proc/neuralpred/{mstr}_clust_kl_ve.npz")["xpos"]
        ypos = np.load(f"{data_path}/proc/neuralpred/{mstr}_clust_kl_ve.npz")["ypos"]
        xypos.append([xpos, ypos])
    ve_expls = np.array(ve_expls)
    ax = plt.subplot(grid[2, 0])
    ax.axis("off")
    pos = ax.get_position()
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = grid.figure.add_axes([pos[0], pos[1] - 0.05, pos[2], pos[3]])
    il = plot_label(ltr, il, ax, trans, fs_title)
    ax.axis("off")
    iexp = iexps[0]
    ypos, xpos = -xypos[iexp][1], xypos[iexp][0]
    ylim = np.array([ypos.min(), ypos.max()])
    xlim = np.array([xpos.min(), xpos.max()])
    ylr = 1
    ax = grid.figure.add_axes(
        [pos[0] - 0.02, pos[1] - 0.04, pos[2] * 1.2, pos[2] * 1.2 / ylr * yratio]
    )
    im = ax.scatter(
        ypos[::subsample],
        xpos[::subsample],
        s=1,
        c=ve_kps[iexp][::subsample],
        cmap="magma",
        vmin=0,
        vmax=75,
        rasterized=True,
    )
    # ax.set_xlim(ylim)
    # ax.set_ylim(xlim)
    ax.axis("off")
    add_apml(ax, xpos, ypos)
    ax.axis("square")

    cbar = plt.colorbar(
        im,
        cax=grid.figure.add_axes(
            [pos[0] + pos[2] * 0.85, pos[1] + pos[2] * 0.8 / ylr * yratio, 0.01, 0.1]
        ),
    )
    cbar.ax.set_ylabel("% normalized variance\nexplained (test data)", fontsize="small")
    cbar.ax.set_yticks([0, 25, 50, 75])

    ax = plt.subplot(grid[2, 1])
    ax.axis("off")
    pos = ax.get_position()
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    iexp = iexps[1]
    ypos, xpos = -xypos[iexp][1], xypos[iexp][0]
    ylim = np.array([ypos.min(), ypos.max()])
    xlim = np.array([xpos.min(), xpos.max()])
    ylr = np.diff(ylim)[0] / np.diff(xlim)[0]
    ax = grid.figure.add_axes(
        [pos[0] - 0.01, pos[1] - 0.05, pos[2] * 1.1, pos[2] * 1.1 / ylr * yratio]
    )
    il = plot_label(ltr, il, ax, trans, fs_title)
    ax.scatter(
        ypos[::subsample],
        xpos[::subsample],
        s=1,
        c=ve_kps[iexp][::subsample],
        cmap="magma",
        vmin=0,
        vmax=75,
        rasterized=True,
    )
    ax.set_xlim(ylim)
    ax.set_ylim(xlim)
    ax.axis("off")

    ax = plt.subplot(grid[1:, 2])
    ax.axis("off")
    pos = ax.get_position()
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = grid.figure.add_axes(
        [pos[0] + 0.0, pos[1] + 0.0, pos[2] - 0.03, pos[3] - 0.03]
    )
    il = plot_label(ltr, il, ax, trans, fs_title)
    t = 0
    vis = np.array([db["visual"] for db in dbs])
    ax.plot(ve_overall[vis].T, color=viscol, lw=0.5)
    ax.plot(ve_overall[~vis].T, color=smcol, lw=0.5)
    ax.errorbar(
        np.arange(0, 3),
        ve_overall[vis].mean(axis=0),
        ve_overall[vis].std(axis=0) / vis.sum() ** 0.5,
        color=viscol,
        lw=3,
        zorder=5,
    )
    ax.errorbar(
        np.arange(0, 3),
        ve_overall[~vis].mean(axis=0),
        ve_overall[~vis].std(axis=0) / (~vis).sum() ** 0.5,
        color=smcol,
        lw=3,
        zorder=5,
    )
    ax.text(
        0.03,
        0.91,
        "visual",
        color=viscol,
        transform=ax.transAxes,
        ha="left",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.96,
        "sensorimotor",
        color=smcol,
        transform=ax.transAxes,
        ha="left",
        fontweight="bold",
    )
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        ["movie\nPCs", "motion  \nPCs  ", "  keypoints"],  # rotation=45,
        ha="center",
        rotation_mode="anchor",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("% normalized variance explained (test data)")
    ymax = 62
    ymin = 0
    ax.set_ylim([ymin, ymax])
    ymax = 15
    ax.plot([0.0, 1], (ymax) * np.ones(2), color="k", lw=1, marker=2, markersize=3)

    ax.text(0.5, ymax, "n.s.", ha="center", va="bottom")
    ax.plot([0, 2], (ymax * 0.9) * np.ones(2), color="k", lw=1, marker=2, markersize=3)
    ax.text(1, ymax * 0.9, "***", ha="center", va="center")
    ax.plot([1, 2], (ymax * 0.8) * np.ones(2), color="k", lw=1, marker=2, markersize=3)
    ax.text(1.5, ymax * 0.8, "***", ha="center", va="center")

    for k, inds in enumerate([vis, ~vis]):
        if k == 0:
            print("visual")
        else:
            print("sensorimotor")
        print("explainable variance: ", ve_expls[inds].mean())
        print("raw varexp: ", ve_all[inds].mean(axis=0))
        if compute_binned:
            print(f"raw varexp (bin_size={bin_size}): ", ve_binned[inds].mean(axis=0))
        print("normalized varexp: ", ve_overall[inds].mean(axis=0))
    print("wilcoxon tests of kps vs pcs")
    print(wilcoxon(ve_overall[:, 0], ve_overall[:, 1]))
    print(wilcoxon(ve_overall[:, 0], ve_overall[:, 2]))
    print(wilcoxon(ve_overall[:, 1], ve_overall[:, 2]))
    return il, xypos, ve_kps, ve_svds


def panels_scaling(data_path, dbs, grid, trans, il):
    mstrs = [f"{db['mname']}_{db['datexp']}_{db['blk']}" for db in dbs]
    ves = [np.zeros((len(dbs), 6, 3)), np.zeros((len(dbs), 5, 3))]
    nns = [np.zeros((len(dbs), 6)), np.zeros((len(dbs), 5))]
    for iexp, mstr in enumerate(mstrs):
        d = np.load(f"{data_path}/proc/neuralpred/{mstr}_scaling.npz")
        ve_expl = np.load(f"{data_path}/proc/neuralpred/{mstr}_spks_test.npz")[
            "varexp_expl_neurons"
        ].mean()
        ves[0][iexp] = (d["varexps_neurons"] / ve_expl) * 100
        nns[0][iexp] = d["nlengths"]
        ves[1][iexp] = (d["varexps_time"] / ve_expl) * 100
        nns[1][iexp] = d["tlengths"] / 3 / 60

    vis = np.array([db["visual"] for db in dbs])

    for k in range(2):
        for j, inds in enumerate([vis, ~vis]):
            ax = plt.subplot(grid[k, j])
            vem = ves[k][inds].mean(axis=0)
            # vem = vem0.copy() # swap movie and motion
            # vem[:,1:] = vem0[:,[2,1]]
            x = nns[k][inds].mean(axis=0)
            for i in range(3):
                ax.plot(x, vem[:, i], color=colors[i], zorder=3 - i)
                if k == 0 and j == 1:
                    ax.text(
                        1.0,
                        0.3 - i * 0.12,
                        lbls[i],
                        color=colors[i],
                        transform=ax.transAxes,
                        ha="right",
                    )
            # ax.set_xscale('log')
            if j == 0:
                il = plot_label(ltr, il, ax, trans, fs_title)
                ax.set_ylabel("% normalized variance\nexplained (test data)")
                ax.set_title(
                    "visual", fontweight="bold", color=viscol, fontsize="medium"
                )
            else:
                ax.set_title(
                    "sensorimotor", fontweight="bold", color=smcol, fontsize="medium"
                )
            if k == 0:
                ax.set_xlabel("# of neurons")
                ax.set_xscale("log")
                ax.set_xticks([100, 1000, 10000])
            else:
                ax.set_xlabel("# of timepoints (minutes)")
            ax.set_ylim([0, 52])
            ax.set_yticks([0, 25, 50])

    # ax = plt.subplot(grid[0,1])
    # il = plot_label(ltr, il, ax, trans, fs_title)

    return il


def panels_cum_varexp(data_path, dbs, axs):
    try:
        ve = np.load(f"{data_path}proc/neuralpred/ve_cum.npy")
    except:
        mstrs = [f"{db['mname']}_{db['datexp']}_{db['blk']}" for db in dbs]
        ve_overall = np.zeros((len(dbs), 3, 128))
        for iexp, mstr in enumerate(mstrs):
            U = np.load(f"{data_path}/proc/neuralpred/{mstr}_spks_test.npz")["U"]
            sp = np.load(f"{data_path}/proc/neuralpred/{mstr}_spks_test.npz")[
                "spks_test"
            ]
            V = sp.T @ U
            kp_pc = np.load(f"{data_path}/proc/neuralpred/{mstr}_kp_pred_test.npz")[
                "Y_pred_test"
            ]
            svd_pc = np.load(f"{data_path}/proc/neuralpred/{mstr}_svd_pred_test.npz")[
                "Y_pred_test"
            ]

            Vvar = V.var(axis=0)
            ve_overall[iexp, 0, :] = (
                (Vvar - ((V - kp_pc) ** 2).mean(axis=0)) / Vvar.sum() * 100
            )
            for j in range(2):
                ve_overall[iexp, j + 1, :] = (
                    (Vvar - ((V - svd_pc[j]) ** 2).mean(axis=0)) / Vvar.sum() * 100
                )
        ve = np.cumsum(ve_overall, axis=-1)
        np.save(f"{data_path}proc/neuralpred/ve_cum.npy", ve)
    vis = np.array([db["visual"] for db in dbs])
    for k, ax in enumerate(axs):
        inds = vis if k == 0 else ~vis
        for j in range(3):
            vem = ve[inds, j].mean(axis=0)
            ves = ve[inds, j].std(axis=0) / (len(ve) ** 0.5)
            ax.semilogx(np.arange(1, 129), vem.T, color=colors[j])
            ax.fill_between(
                np.arange(1, 129), vem + ves, vem - ves, color=colors[j], alpha=0.25
            )
            print(lbls[j], "1st pc= ", vem[0])
        xt = 2 ** np.arange(0, 10, 2)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(x) for x in xt])
        ax.set_ylim([0, 45])
        ax.set_xlim([1, 128])
        ax.set_title(
            f"{['visual','sensorimotor'][k]}",
            color=smcol if k else viscol,
            fontsize="medium",
            fontweight="bold",
        )
        if k == 0:
            ax.set_ylabel("% cumulative variance \n explained (test data)")
        ax.set_xlabel("# of neural PCs")


def fig3(data_path, dbs, save_fig=False, compute_binned=False):
    fig = plt.figure(figsize=(12, 7.8))
    trans = mtransforms.ScaledTranslation(-25 / 72, 7 / 72, fig.dpi_scale_trans)
    grid = plt.GridSpec(
        3,
        5,
        figure=fig,
        left=0.05,
        right=0.97,
        top=0.95,
        bottom=0.07,
        wspace=0.5,
        hspace=0.5,
    )
    il = 0
    ax = plt.subplot(grid[:1, :3])
    il = plot_label(ltr, il, ax, trans, fs_title)
    pos = ax.get_position()
    ax.axis("off")
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = fig.add_axes([pos[0] - 0.06, pos[1] - 0.095, pos[2] + 0.09, pos[3] + 0.09])
    ax.imshow(plt.imread(f"{data_path}figs/keypoints_net.png"))
    ax.text(
        0, 1.05, "     neural network model", fontsize="large", transform=ax.transAxes
    )
    ax.axis("off")

    ax = plt.subplot(grid[1, 0])
    trans2 = mtransforms.ScaledTranslation(-25 / 72, -15 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, trans2, fs_title)
    pos = ax.get_position()
    ax.axis("off")
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = fig.add_axes([pos[0] - 0.01, pos[1] - 0.08, pos[2] - 0.03, pos[3] + 0.05])
    panel_wavelets(data_path, dbs[2], ax)

    ax = plt.subplot(grid[1, 1])
    trans2 = mtransforms.ScaledTranslation(-25 / 72, -15 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, trans2, fs_title)
    ax.axis("off")
    pos = ax.get_position()
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = fig.add_axes([pos[0] - 0.01, pos[1] + 0.04, pos[2] - 0.04, pos[3] - 0.04])
    img = plt.imread(f"{data_path}figs/brain_windows.png")
    ax.imshow(img)
    ax.axis("off")

    il, xypos, ve_kps, ve_svds = panels_varexp(
        data_path, dbs, grid, trans, il, subsample=5, compute_binned=compute_binned
    )

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(
        3, 2, subplot_spec=grid[:, -2:], wspace=0.5, hspace=0.5
    )
    trans2 = mtransforms.ScaledTranslation(-50 / 72, 7 / 72, fig.dpi_scale_trans)
    il = panels_scaling(data_path, dbs, grid1, trans2, il)

    axs = [plt.subplot(grid1[-1, 0]), plt.subplot(grid1[-1, 1])]
    il = plot_label(ltr, il, axs[0], trans2, fs_title)
    panels_cum_varexp(data_path, dbs, axs)

    if save_fig:
        fig.savefig(f"{data_path}figs/fig3_draft.pdf")
