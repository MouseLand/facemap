"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import fig4
import matplotlib.pyplot as plt
import torch
from fig_utils import *
from rastermap import sorting
from scipy.stats import wilcoxon, zscore

from facemap.utils import bin1d


def varexp_ranks(data_path, dbs, evals=None, save_fig=False):
    colors = [[0.5, 0.5, 0.5], [0.75, 0.75, 0.25]]
    lbls = ["keypoints", "movie PCs"]

    fig = plt.figure(figsize=(9,3))
    trans = mtransforms.ScaledTranslation(-50 / 72, 7 / 72, fig.dpi_scale_trans)
    grid = plt.GridSpec(
            1,
            3,
            figure=fig,
            left=0.15,
            right=0.95,
            top=0.9,
            bottom=0.2,
            wspace=0.5,
            hspace=0.25,
    )
        
    mstrs = [f"{db['mname']}_{db['datexp']}_{db['blk']}" for db in dbs]
    ve = np.zeros((len(dbs), 128, 2))
    evals = np.zeros((len(dbs), 500)) if evals is None else evals
    for iexp, mstr in enumerate(mstrs):
        if evals[iexp].sum()==0:
            svd_path = f"{data_path}cam/cam0_{mstr}_proc.npy"
            svds = np.load(svd_path, allow_pickle=True).item()
            ev = (svds["movSVD"][0]**2).sum(axis=0)
            evals[iexp] = ev / ev.sum()
        d = np.load(f"{data_path}/proc/neuralpred/{mstr}_rrr_pred_test.npz")
        ve[iexp,:] = d['varexp'][:128, ::-1] * 100
        #plt.semilogx(np.arange(1, len(d['varexp'])+1), )
        
    il = 0
    ax = plt.subplot(grid[0,0])
    il = plot_label(ltr, il, ax, trans, fs_title)
    vem = evals.mean(axis=0)
    ves = evals.std(axis=0) / (evals.shape[0]-1)**0.5
    ax.loglog(np.arange(1,501), vem, color='k')
    ax.fill_between(
        np.arange(1, 501), vem + ves, vem - ves, color='k', alpha=0.25
    )
    ax.set_ylabel('fraction of variance')
    ax.set_xlabel('PC dimension')
    ax.set_title(
        "face movie PCs", fontweight="bold", fontsize="medium"
    )
            
    colors = [[0.5, 0.5, 0.5], [0.75, 0.75, 0.25]]
    lbls = ["keypoints", "movie PCs"]

    vis = np.array([db["visual"] for db in dbs])
    ranks = np.arange(1,129)
    for j, inds in enumerate([vis, ~vis]):
        ax = plt.subplot(grid[0,j+1])
        if j==0:
            il = plot_label(ltr, il, ax, trans, fs_title)
        for i in range(2):
            vem = ve[inds,:,i].mean(axis=0)
            ves = ve[inds,:,i].std(axis=0) / ((inds.sum()-1)**0.5)
            #print(vem+ves - (vem-ves))
            ax.plot(ranks, vem, color=colors[i])
            ax.fill_between(
                    ranks, vem + ves, vem - ves, color=colors[i], alpha=0.25
                )
            if j == 0:
                x = 0.6
                y = 0.1 + i * 0.12
                ax.text(
                    x, y, lbls[i], color=colors[i], transform=ax.transAxes
                )
                
        if j == 0:
            #il = plot_label(ltr, il, ax, trans, fs_title)
            ax.set_ylabel("% variance explained, \ntop 128 PCs (test data)")
            ax.set_title(
                "visual", fontweight="bold", color=viscol, fontsize="medium"
            )
        else:
            ax.set_title(
                "sensorimotor", fontweight="bold", color=smcol, fontsize="medium"
            )
        ax.set_xlabel("ranks")
        ax.set_xscale("log")
        ax.set_xticks([1,4,16,64])
        ax.set_xticklabels(["1", "4", "16", "64"])
        ax.set_xlim([1,128])
        ax.set_ylim([0, 38])

    if save_fig:
        fig.savefig(f"{data_path}figs/suppfig_veranks.pdf")

    return evals

def example_sm(data_path, db, save_fig=False):
    fig = plt.figure(figsize=(9.5 * 0.75, 7.8))
    trans = mtransforms.ScaledTranslation(-30 / 72, 7 / 72, fig.dpi_scale_trans)
    grid = plt.GridSpec(
        4,
        1,
        figure=fig,
        left=0.1,
        right=0.97,
        top=0.95,
        bottom=0.03,
        wspace=0.75,
        hspace=0.25,
    )
    il = 0

    il = fig4.panels_activity(data_path, db, grid, trans, il, tmin=0)
    if save_fig:
        fig.savefig(f"{data_path}figs/suppfig_examplesm.pdf")


def example_clusters(data_path, dbs, dbs_ex, save_fig=False):
    fig = plt.figure(figsize=(14, 16))
    yratio = 1
    trans = mtransforms.ScaledTranslation(-30 / 72, 7 / 72, fig.dpi_scale_trans)
    grid = plt.GridSpec(
        5,
        1,
        figure=fig,
        left=0.05,
        right=0.97,
        top=0.99,
        bottom=0.01,
        wspace=0.75,
        hspace=0.35,
    )
    subsample = 10
    il = 0
    sc = [0.9, 0.75]

    d = np.load(f"{data_path}proc/neuralpred/n_clusters_analysis.npz")
    nc = d["n_clusters_range"]
    ccs = d["ccs"]
    ax = plt.subplot(grid[0,0])
    pos = ax.get_position()
    ax.axis("off")
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = fig.add_axes([pos[0] +0.07, pos[1], pos[3]*0.75, pos[3]*0.75])
    for k, db in enumerate(dbs):
        ax.plot(nc, ccs[k], color=viscol if db["visual"] else smcol, lw=1)
    ax.plot([100,100], [0,.32],'k--', lw=1)
    ax.set_ylim([0, 0.32])
    ax.set_ylabel("average correlation")
    ax.set_xlabel("number of clusters")
    trans1 = mtransforms.ScaledTranslation(-60 / 72, 7 / 72, fig.dpi_scale_trans)
    il = plot_label(ltr, il, ax, trans1, fs_title)

    for k, db in enumerate(dbs_ex):
        clust_kl_ve = np.load(
            f"{data_path}/proc/neuralpred/{db['mname']}_{db['datexp']}_{db['blk']}_clust_kl_ve.npz"
        )
        labels = clust_kl_ve["labels"]
        ve_clust = clust_kl_ve["varexp_clust"]
        kl_clust = clust_kl_ve["kl_clust"]
        ct = clust_kl_ve["clust_test"]
        cp = clust_kl_ve["clust_pred_test"]
        cc = (zscore(ct, axis=0) * zscore(cp, axis=0)).mean(axis=0)

        xypos = [clust_kl_ve["xpos"], clust_kl_ve["ypos"]]

        grid1 = matplotlib.gridspec.GridSpecFromSubplotSpec(
            5,
            20,
            subplot_spec=grid[2*k+1:2*k+3, 0],
            hspace=[0.2, 1.0][k],
        )
        for i, ind in enumerate(kl_clust.argsort()):
            ax = plt.subplot(grid1[i // 20, i % 20])
            ax.axis("off")
            pos = ax.get_position()
            pos = [pos.x0, pos.y0, pos.width, pos.height]
            ypos, xpos = -xypos[1], xypos[0]
            ylim = np.array([ypos.min(), ypos.max()])
            xlim = np.array([xpos.min(), xpos.max()])
            ylr = np.diff(ylim)[0] / np.diff(xlim)[0]
            ax = grid1.figure.add_axes(
                [pos[0], pos[1], pos[2] * sc[k], pos[2] * sc[k] / ylr * yratio]
            )
            if i == 0:
                il = plot_label(ltr, il, ax, trans, fs_title)
            ax.scatter(
                ypos[::subsample],
                xpos[::subsample],
                s=1,
                color=[0.9, 0.9, 0.9],
                rasterized=True,
            )
            ax.scatter(ypos[labels == ind], xpos[labels == ind], s=3, rasterized=True)
            ax.set_title(f"LI={kl_clust[ind]:.2f}\nr={cc[ind]:.2f}", fontsize="medium")
            ax.set_xlim(ylim)
            ax.set_ylim(xlim)
            ax.axis("off")

    if save_fig:
        fig.savefig(f"{data_path}figs/suppfig_exampleclusters.pdf")

def model_complexity_AP(data_path, dbs, save_fig=False):
    mstrs = [f"{db['mname']}_{db['datexp']}_{db['blk']}" for db in dbs]
    d = np.load(f"{data_path}/proc/neuralpred/{mstrs[0]}_complexity.npz")
    n_latents = d["n_latents"]
    n_filts = d["n_filts"]
    ve_no_param = np.zeros((len(dbs), 4))
    ve_nl_all = np.zeros((len(dbs), 5, 2))
    ve_latents = np.zeros((len(dbs), len(n_latents)))
    ve_filts = np.zeros((len(dbs), len(n_filts)))

    for iexp, mstr in enumerate(mstrs):
        d = np.load(f"{data_path}/proc/neuralpred/{mstr}_complexity.npz")
        ve_expl = np.load(f"{data_path}/proc/neuralpred/{mstr}_spks_test.npz")[
            "varexp_expl_neurons"
        ].mean()
        ve_no_param[iexp] = (
            d["varexps_no_param_neurons"].mean(axis=-1) / ve_expl
        ) * 100
        ve_nl_all[iexp] = (d["varexps_nl_all_neurons"].mean(axis=1).T / ve_expl) * 100
        ve_latents[iexp] = (d["varexps_latents_neurons"].mean(axis=0) / ve_expl) * 100
        ve_filts[iexp] = (d["varexps_filts_neurons"].mean(axis=0) / ve_expl) * 100

    fig = plt.figure(figsize=(12, 4))
    yratio = 12 / 4
    trans = mtransforms.ScaledTranslation(-40 / 72, 20 / 72, fig.dpi_scale_trans)
    grid = plt.GridSpec(
        1,
        6,
        figure=fig,
        left=0.08,
        right=0.9,
        top=0.8,
        bottom=0.35,
        wspace=0.6,
        hspace=1.5,
    )
    il = 0

    vis = np.array([db["visual"] for db in dbs])
    ylim = [38, 50]
    colors = [viscol, smcol]

    ax = plt.subplot(grid[0, 0])
    il = plot_label(ltr, il, ax, trans, fs_title)
    i0 = 4
    for j, inds in enumerate([vis, ~vis]):
        ax.plot(n_latents, ve_latents[inds].mean(axis=0), color=colors[j])
        ax.scatter(
            n_latents[i0],
            ve_latents[inds, i0].mean(axis=0),
            marker="*",
            color=colors[j],
            s=150,
        )
    ax.set_xlabel("# of deep\nbehavioral features")
    ax.set_ylim([0, 52])
    ax.set_xscale("log")
    xts = 2.0 ** np.arange(0, 11, 2)
    ax.set_xticks(xts)
    ax.set_xticklabels(
        ["1", "4", "16 ", "64 ", "256 ", "   1024"], fontsize="small"
    )  # , rotation=45, ha='right')

    ax.set_ylabel("% normalized variance\n explained (test data)")

    i0 = [1, 0]
    lstr = ["core", "readout"]
    for k in range(2):
        ax = plt.subplot(grid[0, 1 + k])
        il = plot_label(ltr, il, ax, trans, fs_title)
        for j, inds in enumerate([vis, ~vis]):
            ax.plot(
                np.arange(1, 6), ve_nl_all[inds, :, k].mean(axis=0), color=colors[j]
            )
            ax.scatter(
                i0[k] + 1,
                ve_nl_all[inds, i0[k], k].mean(axis=0),
                marker="*",
                color=colors[j],
                s=150,
            )
        ax.set_ylim(ylim)
        ax.set_xlim([0.5, 5.5])
        ax.set_xticks(np.arange(1, 6))
        ax.set_xlabel(f"# of {lstr[k]} layers")

    ax = plt.subplot(grid[0, 3])
    il = plot_label(ltr, il, ax, trans, fs_title)
    i0 = 0
    for j, inds in enumerate([vis, ~vis]):
        ax.plot(ve_no_param[inds].mean(axis=0), color=colors[j])
        ax.scatter(
            i0, ve_no_param[inds, i0].mean(axis=0), marker="*", color=colors[j], s=150
        )
    ax.set_ylim(ylim)
    ax.set_xticks(np.arange(0, 4))
    ax.set_xticklabels(
        ["normal", "w/o 1st linear layer", "w/o ReLU conv layer", "w/o ReLU deep beh."],
        rotation=45,
        ha="right",
    )

    ax = plt.subplot(grid[0, 4])
    il = plot_label(ltr, il, ax, trans, fs_title)
    i0 = 2
    for j, inds in enumerate([vis, ~vis]):
        ax.plot(n_filts, ve_filts[inds].mean(axis=0), color=colors[j])
        ax.scatter(
            n_filts[i0],
            ve_filts[inds, i0].mean(axis=0),
            marker="*",
            color=colors[j],
            s=150,
        )
    ax.set_xscale("log")
    ax.set_ylim(ylim)
    ax.set_xlabel("# of convolution \nfilters")
    ax.set_xticks([2, 10, 50])
    ax.set_xticklabels(["2", "10", "50"])

    for iexp, mstr in enumerate(mstrs):
        d = np.load(f"{data_path}/proc/neuralpred/{mstr}_complexity.npz")
        ve_expl = np.load(f"{data_path}/proc/neuralpred/{mstr}_spks_test.npz")[
            "varexp_expl_neurons"
        ].mean()
        ve_no_param[iexp] = (
            d["varexps_no_param_neurons"].mean(axis=-1) / ve_expl
        ) * 100
        ve_nl_all[iexp] = (d["varexps_nl_all_neurons"].mean(axis=1).T / ve_expl) * 100
        ve_latents[iexp] = (d["varexps_latents_neurons"].mean(axis=0) / ve_expl) * 100
        ve_filts[iexp] = (d["varexps_filts_neurons"].mean(axis=0) / ve_expl) * 100

    ax = plt.subplot(grid[0, 5])
    pos = ax.get_position()
    ax.axis("off")
    pos = [pos.x0, pos.y0, pos.width, pos.height]
    ax = fig.add_axes([pos[0] + 0.03, pos[1], pos[2], pos[3]])
    mstrs = [f"{db['mname']}_{db['datexp']}_{db['blk']}" for db in dbs]
    ve_all = []
    for j, mstr in enumerate(mstrs):
        d = np.load(f"{data_path}/proc/neuralpred/{mstr}_kpwo_pred_test.npz")
        ve_expl = np.load(f"{data_path}/proc/neuralpred/{mstr}_spks_test.npz")[
            "varexp_expl_neurons"
        ].mean()
        kpa = d["varexp_neurons"].mean(axis=0)
        kpareas = d["kpareas"]
        vef = np.load(f"{data_path}/proc/neuralpred/{mstr}_net_pred_test.npz")[
            "varexp_neurons"
        ][:, 1]
        kpf = np.array([vef.mean(), *kpa]) / ve_expl * 100
        ve_all.append(kpf)
    ve_all = np.array(ve_all)
    for i, inds in enumerate([vis, ~vis]):
        print(ve_all[inds].mean(axis=0))
        ax.plot(ve_all[inds].T, color=viscol if i == 0 else smcol, lw=1, alpha=0.5)
        plt.errorbar(
            np.arange(0, 4),
            ve_all[inds].mean(axis=0),
            ve_all[inds].std(axis=0) / inds.sum() ** 0.5,
            color=viscol if i == 0 else smcol,
            lw=3,
            zorder=10,
        )

    ax.set_title("Prediction excluding\nkeypoint groups", fontsize="medium")
    ax.set_ylim([0, 64])
    ax.set_xticks(np.arange(0, 4))
    ax.set_xticklabels(["all", "w/o eye", "w/o whisker", "w/o nose"],
            rotation=45,
        ha="right")
    ax.set_ylabel("% normalized variance\nexplained (test data)")
    il = plot_label(ltr, il, ax, trans, fs_title)


    if save_fig:
        fig.savefig(f"{data_path}figs/suppfig_complexity_AP.pdf")
