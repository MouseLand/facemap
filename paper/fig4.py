"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import matplotlib.pyplot as plt
import torch
from fig_utils import *
from rastermap import sorting
from scipy.stats import zscore

from facemap.utils import bin1d

yratio = 9.5 / 7.8


def panels_activity(data_path, db, grid1, trans, il, tmin=0, running=False):
    run = np.load(
        f"{data_path}/neural_data/spont_{db['mname']}_{db['datexp']}_{db['blk']}.npz"
    )["run"]
    spks_test = np.load(
        f"{data_path}/proc/neuralpred/{db['mname']}_{db['datexp']}_{db['blk']}_spks_test.npz"
    )["spks_test"]
    itest = np.load(
        f"{data_path}/proc/neuralpred/{db['mname']}_{db['datexp']}_{db['blk']}_spks_test.npz"
    )["itest"]
    run = run[itest]
    spks_pred_test_net = np.load(
        f"{data_path}/proc/neuralpred/{db['mname']}_{db['datexp']}_{db['blk']}_net_pred_test.npz"
    )["spks_pred_test"][0]
    spks_pred_test_lin = np.load(
        f"{data_path}/proc/neuralpred/{db['mname']}_{db['datexp']}_{db['blk']}_rrr_pred_test.npz"
    )["spks_pred_test"][0].T
    clust_kl_ve = np.load(
        f"{data_path}/proc/neuralpred/{db['mname']}_{db['datexp']}_{db['blk']}_clust_kl_ve.npz"
    )
    
    isort = clust_kl_ve["isort"]
    ve_clust = clust_kl_ve["varexp_clust"]
    clust_test = clust_kl_ve["clust_test"]
    clust_pred_test = clust_kl_ve["clust_pred_test"]
    clust_pred_test -= clust_test.mean(axis=0)
    clust_pred_test /= clust_test.std(axis=0)
    clust_test -= clust_test.mean(axis=0)
    clust_test /= clust_test.std(axis=0)

    nbin = 25
    spks_rm = bin1d(spks_test[isort], nbin)
    spks_pred_rm = bin1d(spks_pred_test_net[isort], nbin)
    spks_pred_lin_rm = bin1d(spks_pred_test_lin[isort], nbin)
    spks_pred_rm -= spks_rm.mean(axis=1, keepdims=True)
    spks_pred_rm /= spks_rm.std(axis=1, keepdims=True)
    spks_pred_lin_rm -= spks_rm.mean(axis=1, keepdims=True)
    spks_pred_lin_rm /= spks_rm.std(axis=1, keepdims=True)
    spks_rm -= spks_rm.mean(axis=1, keepdims=True)
    spks_rm /= spks_rm.std(axis=1, keepdims=True)

    tmax = tmin + 400
    trange = tmax - tmin
    titles = [
        f"activity in {['sensorimotor','visual'][db['visual']]} neurons (test data)",
        "network prediction",
        "linear prediction",
    ]
    for k, (sp, ttl) in enumerate(
        zip([spks_rm, spks_pred_rm, spks_pred_lin_rm], titles)
    ):
        ax = plt.subplot(grid1[k, 0])
        ax.imshow(sp[:, tmin:tmax], vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
        ax.axis("off")
        if not running or k > 0:
            ax.set_title(ttl, pad=0)
        ax.plot(-2 * np.ones(2), sp.shape[0] - np.array([0, 5000 / 25]), color="k")
        ax.plot([0, 15], (sp.shape[0] + 25) * np.ones(2), color="k")
        ax.set_xlim([-2.5, trange])
        if k == 0:
            ax.text(
                -2, sp.shape[0], "5000 neurons", rotation=90, ha="right", va="bottom"
            )
            ax.text(0, sp.shape[0] + 50, "5 sec.", ha="left", va="top")
            il = plot_label(ltr, il, ax, trans, fs_title)
            if running:
                axin = ax.inset_axes([0, 1, 1, 0.15])
                axin.fill_between(
                    np.arange(0, trange), np.abs(run[tmin:tmax]), color=[0, 0.3, 0]
                )
                axin.text(
                    1,
                    1,
                    "running speed",
                    va="bottom",
                    ha="right",
                    transform=axin.transAxes,
                    color=[0, 0.3, 0],
                )
                axin.set_xlim([-2.5, trange])
                axin.set_ylim([0, 8])
                axin.axis("off")

    ax = plt.subplot(grid1[3, 0])
    ax.axis("off")
    pos = ax.get_position()
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = grid1.figure.add_axes([pos[0], pos[1] - 0.03, pos[2], pos[3] + 0.02])
    il = plot_label(ltr, il, ax, trans, fs_title)
    np.random.seed(0)
    inds = ve_clust.argsort()[::-1][1:14:2][np.random.permutation(7)]
    dy = 4
    for i, ind in enumerate(inds):
        ax.plot(
            clust_test[tmin:tmax, ind] - dy * i,
            color=viscol if db["visual"] else smcol,
            lw=1,
            zorder=10 - i,
        )
        ax.plot(
            clust_pred_test[tmin:tmax, ind] - dy * i,
            color=[0.5, 0.5, 0.5],
            lw=2,
            zorder=10 - i,
        )
        cc = (zscore(clust_test[:, ind]) * zscore(clust_pred_test[:, ind])).mean()
        ax.text(
            -2,
            -dy * i - dy / 5,
            f"$r$={cc:.2f}",
            ha="right",
            va="bottom",
            fontsize="small",
        )
    ax.text(
        0.75,
        1,
        "network prediction",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        color=[0.5, 0.5, 0.5],
    )
    ax.text(
        0,
        1.03,
        "example neural activity clusters",
        transform=ax.transAxes,
        fontsize="large",
    )
    # ax.set_title('example neural activity clusters')
    ax.set_xlim([-2.5, trange])
    ax.plot(
        [0, 15],
        (-dy * (i + 0.2) + clust_test[tmin:tmax, inds[-1]].min()) * np.ones(2),
        color="k",
    )
    ax.set_ylim(
        [
            -dy * (i + 0.3) + clust_test[tmin:tmax, inds[-1]].min(),
            clust_test[tmin:tmax, inds[0]].max(),
        ]
    )
    ax.axis("off")

    return il


def panels_kl_cc(data_path, dbs, grid2, trans, il, subsample=1):
    mstrs = [f"{db['mname']}_{db['datexp']}_{db['blk']}" for db in dbs]
    kls = np.zeros(0)
    ves = np.zeros(0)
    ccs = np.zeros(0)
    rs = np.zeros(0)
    for iexp, mstr in enumerate(mstrs):
        kl = np.load(f"{data_path}/proc/neuralpred/{mstr}_clust_kl_ve.npz")["kl_clust"]
        kls = np.append(kls, kl)
        ve = (
            np.load(f"{data_path}/proc/neuralpred/{mstr}_clust_kl_ve.npz")[
                "varexp_clust"
            ]
            * 100
        )
        ct = np.load(f"{data_path}/proc/neuralpred/{mstr}_clust_kl_ve.npz")[
            "clust_test"
        ]
        cp = np.load(f"{data_path}/proc/neuralpred/{mstr}_clust_kl_ve.npz")[
            "clust_pred_test"
        ]
        ves = np.append(ves, ve)
        ccs = np.append(ccs, (zscore(ct, axis=0) * zscore(cp, axis=0)).mean(axis=0))
        rs = np.append(rs, np.corrcoef(ccs[-100:], kl)[0, 1])

    # example mouse
    iexp = 10
    clust_kl_ve = np.load(f"{data_path}/proc/neuralpred/{mstrs[iexp]}_clust_kl_ve.npz")
    labels = clust_kl_ve["labels"]
    ve_clust = clust_kl_ve["varexp_clust"]
    kl_clust = clust_kl_ve["kl_clust"]
    xypos = [clust_kl_ve["xpos"], clust_kl_ve["ypos"]]

    ax = plt.subplot(grid2[0, 0])
    ax.axis("off")
    pos = ax.get_position()
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ypos, xpos = -xypos[1], xypos[0]
    ylim = np.array([ypos.min(), ypos.max()])
    xlim = np.array([xpos.min(), xpos.max()])
    ylr = np.diff(ylim)[0] / np.diff(xlim)[0]
    ax = grid2.figure.add_axes(
        [pos[0] - 0.05, pos[1] - 0.01, pos[2] * 0.9, pos[2] * 0.9 / ylr * yratio]
    )
    trans2 = mtransforms.ScaledTranslation(
        -25 / 72, 15 / 72, grid2.figure.dpi_scale_trans
    )
    il = plot_label(ltr, il, ax, trans2, fs_title)
    ax.scatter(
        ypos[::subsample],
        xpos[::subsample],
        s=1,
        color=[0.9, 0.9, 0.9],
        rasterized=True,
    )
    clcol = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    clcol = np.vstack((clcol[:3], clcol[5:6], clcol[7:]))
    inds = kl_clust.argsort()[[0, 10, 55, 57, 97]]
    for i, ind in enumerate(inds):
        ax.scatter(
            ypos[labels == ind],
            xpos[labels == ind],
            s=3,
            color=clcol[i],
            marker="x",
            lw=2,
            rasterized=True,
        )
    ax.text(0.0, 1.05, "example clusters", fontsize="large", transform=ax.transAxes)
    ax.set_xlim(ylim)
    ax.set_ylim(xlim)
    ax.axis("off")

    colors = [smcol, viscol]
    for k, ii in enumerate([np.arange(100 * 10, 100 * 16), np.arange(0, 100 * 10)]):
        ax = plt.subplot(grid2[k + 1, 0])
        trans2 = mtransforms.ScaledTranslation(
            -50 / 72, 20 / 72, grid2.figure.dpi_scale_trans
        )
        ax.axis("off")
        pos = ax.get_position()
        pos = [pos.x0, pos.y0, pos.width, pos.height]
        ax = grid2.figure.add_axes(
            [pos[0] - 0.01, pos[1] + 0.01, pos[2] + 0.02, pos[3] - 0.05]
        )
        il = plot_label(ltr, il, ax, trans2, fs_title)
        ax.scatter(kls[ii], ccs[ii], s=1, alpha=0.5, color=colors[k], rasterized=True)
        if k == 0:
            for i, ind in enumerate(inds):
                ax.scatter(
                    kls[iexp * 100 + ind],
                    ccs[iexp * 100 + ind],
                    s=300,
                    lw=1,
                    facecolor=clcol[i],
                    marker=".",
                    edgecolor="w",
                )
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 4])
        ax.set_title(f"{['sensorimotor', 'visual'][k]} clusters")
        ax.set_ylabel("correlation (test data)")
        ax.set_xlabel("locality index")
        ax.text(
            0.01,
            0.01,
            f"$r$ = {np.corrcoef(ccs[ii],kls[ii])[0,1]:.2f}",
            transform=ax.transAxes,
            fontweight="bold",
        )
    if 0:
        axin = ax.inset_axes([0.75, 1.1, 0.3, 0.3])
        axin.hist(rs, np.arange(-1.0, -0.4, 0.1), facecolor=[0.75, 0.75, 0.75])
        axin.tick_params(axis="both", which="major", labelsize="small")
        axin.set_xlim([-1.0, -0])
        axin.set_ylim([0, 6])
        axin.set_xlabel("$r$", {"va": "bottom"}, fontsize="small")
        axin.set_ylabel("# of\nrecordings", fontsize="small")


def fig4(data_path, dbs, save_fig=False):
    fig = plt.figure(figsize=(9.5, 7.8))
    trans = mtransforms.ScaledTranslation(-30 / 72, 7 / 72, fig.dpi_scale_trans)
    grid = plt.GridSpec(
        3,
        4,
        figure=fig,
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.06,
        wspace=0.75,
        hspace=0.75,
    )
    il = 0

    grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=grid[:, :3], hspace=0.2
    )
    il = panels_activity(data_path, dbs[2], grid1, trans, il, tmin=1500)

    grid2 = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=grid[:, -1])
    panels_kl_cc(data_path, dbs, grid2, trans, il, subsample=5)
    if save_fig:
        fig.savefig(f"{data_path}figs/fig4_draft.pdf")
