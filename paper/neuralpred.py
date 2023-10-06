"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os
import sys
import time

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

from facemap import keypoints
from facemap.neural_prediction import keypoints_utils, neural_model, prediction_utils
from facemap.utils import bin1d, compute_varexp, split_traintest

sys.path.insert(0, "/github/rastermap/")
from rastermap import clustering, mapping


def model_complexity(data_path, dbs, n_layers_test=5, device=torch.device("cuda")):
    """quantifying performance of model with different architectures/dims"""
    tic = time.time()
    varexp_complexity = [[], [], [], [], [], [], []]
    for iexp, db in enumerate(dbs):
        mname, datexp, blk, twocam = db["mname"], db["datexp"], db["blk"], db["2cam"]
        cid = 0
        kp_path0 = (
            f"{data_path}proc/keypoints/kpfilt_cam{cid}_{mname}_{datexp}_{blk}.npy"
        )
        if not os.path.exists(kp_path0):
            continue
        neural_file = f"{data_path}neural_data/spont_{mname}_{datexp}_{blk}.npz"
        print(f"{iexp} loading {neural_file}")
        dat = np.load(neural_file)

        spks = dat["spks"]
        tcam = dat["tcam"]
        tneural = dat["tneural"]

        # z-score neural activity
        spks -= spks.mean(axis=1)[:, np.newaxis]
        std = ((spks**2).mean(axis=1) ** 0.5)[:, np.newaxis]
        std[std == 0] = 1
        spks /= std

        Y = PCA(n_components=128).fit_transform(spks.T)
        U = spks @ Y
        U /= (U**2).sum(axis=0) ** 0.5

        d = np.load(kp_path0, allow_pickle=True).item()
        xy, keypoint_labels = d["xy"], d["keypoint_labels"]
        x = xy.reshape(xy.shape[0], -1).copy()
        x = (x - x.mean(axis=0)) / x.std(axis=0)

        d = np.load(
            f"{data_path}proc/neuralpred/{mname}_{datexp}_{blk}_kp_pred_test.npz"
        )
        varexps = d["varexp"]
        varexps_neurons = d["varexp_neurons"]
        print(f"standard, varexp {varexps:.3f}")

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model = neural_model.KeypointsNetwork(
            n_in=x.shape[-1], n_out=Y.shape[-1], identity=True
        ).to(device)
        (
            y_pred_test,
            varexps_identity,
            spks_pred_test,
            varexps_identity_neurons,
            itest,
        ) = model.train_model(x, Y, tcam, tneural, U=U, spks=spks, device=device)
        print(f"identity, varexp {varexps_identity:.3f}")

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model = neural_model.KeypointsNetwork(
            n_in=x.shape[-1], n_out=Y.shape[-1], relu_wavelets=False
        ).to(device)
        (
            y_pred_test,
            varexps_no_relu_wavelets,
            spks_pred_test,
            varexps_no_relu_wavelets_neurons,
            itest,
        ) = model.train_model(x, Y, tcam, tneural, U=U, spks=spks, device=device)
        print(f"no relu_wavelets, varexp {varexps_no_relu_wavelets:.3f}")

        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model = neural_model.KeypointsNetwork(
            n_in=x.shape[-1], n_out=Y.shape[-1], relu_latents=False
        ).to(device)
        (
            y_pred_test,
            varexps_no_relu_latents,
            spks_pred_test,
            varexps_no_relu_latents_neurons,
            itest,
        ) = model.train_model(x, Y, tcam, tneural, U=U, spks=spks, device=device)
        print(f"no relu_latents, varexp {varexps_no_relu_latents:.3f}")

        varexps_no_param = [
            varexps,
            varexps_identity,
            varexps_no_relu_wavelets,
            varexps_no_relu_latents,
        ]
        varexps_no_param_neurons = [
            varexps_neurons,
            varexps_identity_neurons,
            varexps_no_relu_wavelets_neurons,
            varexps_no_relu_latents_neurons,
        ]

        varexps_nl_all = []
        varexps_nl_all_neurons = []
        for core in [1, 0]:
            varexps_nl = np.zeros(n_layers_test)
            varexps_nl_neurons = np.zeros((len(spks), n_layers_test))
            for k, nl in enumerate(np.arange(1, n_layers_test + 1)):
                if (core and nl == 2) or (not core and nl == 1):
                    varexps_nl[k] = varexps
                    varexps_nl_neurons[:, k] = varexps_neurons
                else:
                    np.random.seed(0)
                    torch.manual_seed(0)
                    torch.cuda.manual_seed(0)
                    if core:
                        model = neural_model.KeypointsNetwork(
                            n_in=x.shape[-1], n_out=Y.shape[-1], n_core_layers=nl
                        ).to(device)
                    else:
                        model = neural_model.KeypointsNetwork(
                            n_in=x.shape[-1], n_out=Y.shape[-1], n_out_layers=nl
                        ).to(device)
                    (
                        y_pred_test,
                        varexps_nl[k],
                        spks_pred_test,
                        varexps_nl_neurons[:, k],
                        itest,
                    ) = model.train_model(
                        x, Y, tcam, tneural, U=U, spks=spks, device=device
                    )
                    print(f"n_{['out', 'core'][core]}={nl}, varexp {varexps_nl[k]:.3f}")

            varexps_nl_all.append(varexps_nl)
            varexps_nl_all_neurons.append(varexps_nl_neurons)

        # compute prediction as a function of n_dims and n_latents
        dim_latents = np.hstack((2 ** np.arange(0, 9, 2), 2 ** np.arange(9, 11, 1)))
        dim_filts = np.array([2, 6, 10, 20, 50])

        varexps_dim_all = []
        varexps_dim_all_neurons = []
        dim_all = []
        for latents in [1, 0]:
            dims = dim_latents if latents else dim_filts
            print(dims)
            varexps_dim = np.nan * np.zeros(len(dims))
            varexps_dim_neurons = np.nan * np.zeros((len(spks), len(dims)))
            for k, dim in enumerate(dims):
                if (latents and k == 4) or (not latents and k == 2):
                    varexps_dim[k] = varexps
                    varexps_dim_neurons[:, k] = varexps_neurons
                else:
                    np.random.seed(0)
                    torch.manual_seed(0)
                    torch.cuda.manual_seed(0)
                    if latents:
                        model = neural_model.KeypointsNetwork(
                            n_in=x.shape[-1], n_latents=dim, n_out=Y.shape[-1]
                        ).to(device)
                    else:
                        model = neural_model.KeypointsNetwork(
                            n_in=x.shape[-1], n_filt=dim, n_out=Y.shape[-1]
                        ).to(device)
                    (
                        y_pred_test,
                        varexps_dim[k],
                        spks_pred_test,
                        varexps_dim_neurons[:, k],
                        itest,
                    ) = model.train_model(
                        x, Y, tcam, tneural, U=U, spks=spks, device=device
                    )
                    print(
                        f"n_{['filt', 'latents'][latents]}={dim}, varexp {varexps_dim[k]:.3f}"
                    )
            varexps_dim_all.append(varexps_dim)
            varexps_dim_all_neurons.append(varexps_dim_neurons)
            dim_all.append(dims)

        for k, arr in enumerate(
            [
                varexps_no_param,
                varexps_no_param_neurons,
                varexps_nl_all,
                varexps_nl_all_neurons,
                varexps_dim_all,
                varexps_dim_all_neurons,
                dim_all,
            ]
        ):
            varexp_complexity[k].append(arr)

        np.savez(
            f"{data_path}proc/neuralpred/{mname}_{datexp}_{blk}_complexity.npz",
            varexps_no_param=varexps_no_param,
            varexps_no_param_neurons=varexps_no_param_neurons,
            varexps_nl_all=varexps_nl_all,
            varexps_nl_all_neurons=varexps_nl_all_neurons,
            varexps_latents=varexps_dim_all[0],
            varexps_latents_neurons=varexps_dim_all_neurons[0],
            n_latents=dim_all[0],
            varexps_filts=varexps_dim_all[1],
            varexps_filts_neurons=varexps_dim_all_neurons[1],
            n_filts=dim_all[1],
        )

    return varexp_complexity


def explvar_save_itest(spks, xpos, ypos, U, sv, delay=-1, save_path=None):
    """compute explainable variance and save U, sv and test data"""
    varexp_expl, varexp_expl_neurons, itest = prediction_utils.peer_prediction(
        spks, xpos, ypos
    )
    print(f"explainable variance {varexp_expl:.3f}, {varexp_expl_neurons.mean():.3f}")
    itest = itest.flatten()
    itest -= delay
    spks_test = spks[:, itest]
    if save_path:
        np.savez(
            save_path,
            spks_test=spks_test,
            itest=itest,
            U=U,
            sv=sv,
            varexp_expl=varexp_expl,
            varexp_expl_neurons=varexp_expl_neurons,
        )


def rrr_net_varexp(
    data_path, svd_path, kp_path0, mstr, Y, tcam, tneural, U, spks, delay=-1
):
    mtypes = ["rrr", "net"]
    svds = np.load(svd_path, allow_pickle=True).item()
    movSVD = svds["movSVD"][0].copy()
    x_kp = keypoints_utils.get_normalized_keypoints(kp_path0, exclude_keypoints="paw")
    X = [movSVD, x_kp]  # , svds["movSVD"][0].copy()]
    X[0] -= X[0].mean(axis=0)
    X[0] /= X[0][:, 0].std(axis=0)

    for k, mtype in enumerate(mtypes):
        if k == 0:
            vout = prediction_utils.rrr_varexp(
                X, Y, tcam, tneural, U, spks, rank=128, delay=delay
            )
            varexp, varexp_neurons, Y_pred_test, spks_pred_test, itest = vout

            print(f"{mtype} varexp {varexp[31,0]:.3f}, {varexp[20,1]:.3f}")
        else:
            varexp, varexp_neurons, Y_pred_test, spks_pred_test = [], [], [], []
            for j, x in enumerate(X):
                vout = prediction_utils.get_keypoints_to_neural_varexp(
                    x, Y, tcam, tneural, U=U, spks=spks, delay=delay
                )
                varexp.append(vout[0])
                varexp_neurons.append(vout[1])
                Y_pred_test.append(vout[2])
                spks_pred_test.append(vout[3])
                itest = vout[4]
                if j == 1:
                    latents, model = vout[-2:]
                    torch.save(
                        model.state_dict(),
                        f"{data_path}proc/models/model_{mstr}.pth",
                    )
                    np.save(
                        f"{data_path}proc/neuralpred/latents_{mstr}.npy",
                        latents,
                    )
            print(f"{mtype} varexp {varexp[0]:.3f}, {varexp[1]:.3f}")
        
        np.savez(
            f"{data_path}proc/neuralpred/{mstr}_{mtype}_pred_test.npz",
            spks_pred_test=np.array(spks_pred_test),
            Y_pred_test=np.array(Y_pred_test),
            itest=itest,
            varexp=np.array(varexp),
            varexp_neurons=np.array(varexp_neurons).T,
        )


def cluster_and_sort(spks, Usv, xpos, ypos, spks_test, spks_pred_test, save_path=None):
    """compute clusters varexp and KL"""
    n_clusters = 100
    np.random.seed(0)
    rm = mapping.Rastermap(
        n_clusters=n_clusters, time_lag_window=10, locality=0.5, verbose=False
    ).fit(spks, u=Usv, normalize=False)
    labels = rm.embedding_clust
    isort = rm.isort
    print(labels.max() + 1)

    itest = itest.flatten()
    clust_test = np.zeros((len(itest), n_clusters), "float32")
    clust_pred_test = np.zeros((len(itest), n_clusters), "float32")
    kl_clust = np.zeros(n_clusters)
    for i in range(n_clusters):
        iclust = labels == i
        kl_clust[i] = prediction_utils.KLDiv_discrete(
            np.stack((xpos[iclust], ypos[iclust]), axis=-1),
            np.stack((xpos, ypos), axis=-1),
        )
        clust_test[:, i] = spks_test[iclust].mean(axis=0)
        clust_pred_test[:, i] = spks_pred_test[1][iclust].mean(axis=0)
    varexp_clust = compute_varexp(clust_test, clust_pred_test)

    if save_path is not None:
        np.savez(
            save_path,
            labels=labels,
            isort=isort,
            kl_clust=kl_clust,
            clust_test=clust_test,
            clust_pred_test=clust_pred_test,
            varexp_clust=varexp_clust,
            xpos=xpos,
            ypos=ypos,
        )


def kpareas_varexp(kp_path0, mstr, Y, tcam, tneural, U, spks, delay=-1, save_path=None):
    x_kp = keypoints_utils.get_normalized_keypoints(kp_path0, exclude_keypoints="paw")
    d = np.load(kp_path0, allow_pickle=True).item()
    # remove some keypoints to make areas equal
    labels = d["keypoint_labels"][:-1]
    kpall = np.array([l.split("(")[0] for l in labels])
    kpareas = np.unique(kpall)
    varexp, varexp_neurons = [], []
    for j, area in enumerate(kpareas):
        inds = np.ones(11, "bool")
        inds[kpall==area] = False
        inds = np.tile(inds[:,np.newaxis], (1,2)).flatten()
        vout = prediction_utils.get_keypoints_to_neural_varexp(
            x_kp[:, inds], Y, tcam, tneural, U=U, spks=spks, delay=delay
        )
        itest = vout[4]
        varexp.append(vout[0])
        varexp_neurons.append(vout[1])

        print(f"{area} varexp {varexp[-1]:.3f}")
    if save_path is not None:
        np.savez(
            save_path,
            itest=itest,
            varexp=np.array(varexp),
            varexp_neurons=np.array(varexp_neurons).T,
            kpareas=kpareas,
        )


def kp_svd_analyses(
    data_path,
    dbs,
    compute_explvar=True,
    compute_rrr_net=True,
    compute_clusters=True,
    compute_kpareas=True,
):
    """compute prediction performance from svds and kps to neural activity

    main analyses for figures 3 and 4

    """
    # path for saving results
    os.makedirs(os.path.join(data_path, "proc/"), exist_ok=True)

    delay = -1
    tic = time.time()
    for iexp, db in enumerate(dbs):
        mname, datexp, blk, twocam = db["mname"], db["datexp"], db["blk"], db["2cam"]
        cid = 0
        kp_path0 = (
            f"{data_path}proc/keypoints/kpfilt_cam{cid}_{mname}_{datexp}_{blk}.npy"
        )
        if not os.path.exists(kp_path0):
            print(f"{kp_path0} not found")
            continue
        neural_file = f"{data_path}neural_data/spont_{mname}_{datexp}_{blk}.npz"
        print(f"{iexp} loading {neural_file}")
        dat = np.load(neural_file)

        xpos = dat["xpos"]
        ypos = dat["ypos"]
        spks = dat["spks"]
        tcam = dat["tcam"]
        tneural = dat["tneural"]

        # z-score neural activity
        spks -= spks.mean(axis=1)[:, np.newaxis]
        std = ((spks**2).mean(axis=1) ** 0.5)[:, np.newaxis]
        std[std == 0] = 1
        spks /= std

        ### compute explainable variance
        ev_save_path = (
            f"{data_path}proc/neuralpred/{mname}_{datexp}_{blk}_spks_test.npz"
        )
        if os.path.exists(ev_save_path) and not compute_explvar:
            U = np.load(ev_save_path)["U"]
            sv = np.load(ev_save_path)["sv"]
            Y = spks.T @ U
        else:
            Y = PCA(n_components=128).fit_transform(spks.T)
            U = spks @ Y
            sv = (Y**2).sum(axis=0) ** 0.5
            U /= (U**2).sum(axis=0) ** 0.5

        if compute_explvar:
            explvar_save_itest(
                spks, xpos, ypos, U, sv, delay=delay, save_path=ev_save_path
            )

        ### compute variance explained using RRR and network for SVDs and keypoints
        mstr = f"{mname}_{datexp}_{blk}"
        if compute_rrr_net:
            svd_path = f"{data_path}cam/cam{cid}_{mname}_{datexp}_{blk}_proc.npy"
            rrr_net_varexp(
                data_path,
                svd_path,
                kp_path0,
                mstr,
                Y,
                tcam,
                tneural,
                U,
                spks,
                delay=delay,
            )

        net_save_path = f"{data_path}proc/neuralpred/{mstr}_net_pred_test.npz"

        ### compute clusters varexp and KL
        cluster_save_path = (
            f"{data_path}proc/neuralpred/{mname}_{datexp}_{blk}_clust_kl_ve.npz"
        )
        if compute_clusters:
            spks_test = np.load(ev_save_path)["spks_test"]
            spks_pred_test = np.load(net_save_path)["spks_pred_test"]
            cluster_and_sort(
                spks, U * sv, xpos, ypos, spks_test, spks_pred_test, cluster_save_path
            )

        kpareas_save_path = f"{data_path}proc/neuralpred/{mstr}_kpwo_pred_test.npz"
        if compute_kpareas:
            kpareas_varexp(
                kp_path0,
                mstr,
                Y,
                tcam,
                tneural,
                U,
                spks,
                delay=delay,
                save_path=kpareas_save_path,
            )

        print(f"processed in {time.time()-tic:.2f}s")

    return


def compute_cluster_corr(data_path, dbs):
    from rastermap.clustering import scaled_kmeans
    n_clusters_range = np.array([2, 4, 10, 25, 50, 100, 250, 500])
    ccs = np.zeros((len(dbs), len(n_clusters_range)))

    for iexp, db in enumerate(dbs):
        mname, datexp, blk, twocam = db["mname"], db["datexp"], db["blk"], db["2cam"]
        neural_file = f"{data_path}neural_data/spont_{mname}_{datexp}_{blk}.npz"
        print(f"{iexp} loading {neural_file}")
        dat = np.load(neural_file)

        xpos = dat["xpos"]
        ypos = dat["ypos"]
        spks = dat["spks"]
        tcam = dat["tcam"]
        tneural = dat["tneural"]

        # z-score neural activity
        spks -= spks.mean(axis=1)[:, np.newaxis]
        std = ((spks**2).mean(axis=1) ** 0.5)[:, np.newaxis]
        std[std == 0] = 1
        spks /= std

        # load explainable variance and PCs
        ev_save_path = (
            f"{data_path}proc/neuralpred/{mname}_{datexp}_{blk}_spks_test.npz"
        )
        U = np.load(ev_save_path)["U"]
        sv = np.load(ev_save_path)["sv"]
        Y = spks.T @ U
        V = Y / sv
        
        for i, n_clusters in enumerate(n_clusters_range):
            X_nodes, iclust = scaled_kmeans(U*sv, n_clusters=n_clusters)

            #X_nodes_z = zscore(X_nodes, axis=1)
            Xt = X_nodes @ V.T 
            Xz = zscore(Xt, axis=1)

            cc = (Xz[iclust] * spks).mean(axis=-1)
            print(f"{n_clusters}, {cc.mean():.2f}")
            ccs[iexp, i] = cc.mean()

    np.savez(f"{data_path}proc/neuralpred/n_clusters_analysis.npz", 
            n_clusters_range=n_clusters_range,
            ccs=ccs)


def compute_varexp_small(
    spks,
    X,
    tcam,
    tneural,
    nmin,
    delay,
    device,
    itrain0,
    itest0,
    U=None,
    ineurons=None,
    itime=None,
    verbose=False,
):
    if itime is not None:
        itrain = itrain0.copy()[:, itime]
        itest = itest0.copy()
        inds = np.concatenate((itrain.flatten(), itest.flatten()), axis=0)
        spks_small = spks[:, inds - delay]
    else:
        itrain = itrain0.copy()
        itest = itest0.copy()
        spks_small = spks

    spks_small = spks_small[ineurons] if ineurons is not None else spks_small

    tlen = itrain.size
    nlen = spks_small.shape[0]
    if spks_small.shape[0] > nmin:
        if U is None:
            Ya = PCA(n_components=128).fit_transform(spks_small.T)
            U = spks_small @ Ya
            sv = (Ya**2).sum(axis=0) ** 0.5
            U /= (U**2).sum(axis=0) ** 0.5
            Y = spks[ineurons].T @ U if ineurons is not None else spks.T @ U
            Ui = U
        else:
            Ui = U 
            Y = spks[ineurons].T @ U if ineurons is not None else spks.T @ U
        spksi = spks[ineurons] if ineurons is not None else spks
    else:
        Y = spks[ineurons].T
        Ui, spksi = None, None

    n_iter, learning_rate, weight_decay = 300, 1e-3, 1e-4
    if nlen <= 2000:
        learning_rate /= 10
        weight_decay /= 10
    if tlen < 10000:
        n_iter -= 100
        learning_rate /= 2
        weight_decay /= 2

    ves = np.zeros((spks_small.shape[0], 2))

    ### network
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = neural_model.KeypointsNetwork(n_in=X.shape[-1], n_out=Y.shape[-1]).to(
        device
    )
    Y_pred_tests = []
    y_pred_test, ve, spks_pred_test, varexp_neurons, _ = model.train_model(
        X,
        Y,
        tcam,
        tneural,
        U=Ui,
        spks=spksi,
        delay=delay,
        learning_rate=learning_rate,
        n_iter=n_iter,
        weight_decay=weight_decay,
        itrain=itrain,
        itest=itest,
        device=device,
        verbose=verbose,
    )
    Y_pred_tests.append(y_pred_test)
    ves[:, 0] = varexp_neurons

    ### RRR
    lam = 1e-6 if tlen > 10000 else 1e-5
    X_ds = prediction_utils.resample_data(X, tcam, tneural, crop="linspace")
    if delay < 0:
        Ys = np.vstack((Y[-delay:], np.tile(Y[[-1], :], (-delay, 1))))
    else:
        X_ds = np.vstack((X_ds[delay:], np.tile(X_ds[[-1], :], (delay, 1))))
        Ys = Y
    Y_pred_test = prediction_utils.rrr_prediction(
        X_ds.astype("float32"),
        Ys.astype("float32"),
        rank=32,
        lam=lam,
        itrain=itrain,
        itest=itest,
    )[0]
    # single neuron prediction
    spks_pred_test = Y_pred_test @ Ui.T if spksi is not None else Y_pred_test
    spks_test = (
        spksi[:, itest.flatten() - delay].T
        if spksi is not None
        else Y[itest.flatten() - delay]
    )
    ves[:, 1] = compute_varexp(spks_test, spks_pred_test)
    Y_pred_tests.append(Y_pred_test)

    return ves, Y[itest.flatten() - delay], Y_pred_tests, Ui


def varexp_scaling(data_path, dbs, device=torch.device("cuda")):
    """compute variance explained as a function of number of neurons and timepoints"""
    tfracs = 2.0 ** np.arange(-4, 0)
    nfracs = 2.0 ** np.arange(-8, 0, 2)
    nfracs = np.append(nfracs, 0.5)

    nneurons_all = np.zeros((len(dbs), len(nfracs) + 1), "int")
    ntime_all = np.zeros((len(dbs), len(tfracs) + 1), "int")

    nsamples = 5

    delay = -1
    nmin = 200  # take PCs if greater than nmin neurons

    cid = 0
    tic = time.time()
    for iexp, db in enumerate(dbs):
        mname, datexp, blk = db["mname"], db["datexp"], db["blk"]
        kp_path0 = (
            f"{data_path}proc/keypoints/kpfilt_cam{cid}_{mname}_{datexp}_{blk}.npy"
        )
        neural_file = f"{data_path}neural_data/spont_{mname}_{datexp}_{blk}.npz"
        print(f"{iexp} loading {neural_file}")
        dat = np.load(neural_file)
        spks = dat["spks"]
        tcam = dat["tcam"]
        tneural = dat["tneural"]

        # z-score neural activity
        spks -= spks.mean(axis=1)[:, np.newaxis]
        std = ((spks**2).mean(axis=1) ** 0.5)[:, np.newaxis]
        std[std == 0] = 1
        spks /= std
        ev_save_path = (
            f"{data_path}proc/neuralpred/{mname}_{datexp}_{blk}_spks_test.npz"
        )
        if os.path.exists(ev_save_path):
            U = np.load(ev_save_path)["U"]
            
        nneurons, ntime = spks.shape
        nlengths = (nneurons * nfracs).astype(int)
        tlengths = (ntime * tfracs).astype(int)
        ntime_all[iexp, :-1] = tlengths
        ntime_all[iexp, -1] = int(ntime * 0.75)
        nneurons_all[iexp, :-1] = nlengths
        nneurons_all[iexp, -1] = nneurons
        
        for j,xtype in enumerate(['svd','kp']):
            if j == 1:
                svd_path = f"{data_path}cam/cam{cid}_{mname}_{datexp}_{blk}_proc.npy"
                svds = np.load(svd_path, allow_pickle=True).item()
                X = svds["movSVD"][0]
                X -= X.mean(axis=0)
                X /= X[:, 0].std(axis=0)
            else:
                X = keypoints_utils.get_normalized_keypoints(
                    kp_path0, exclude_keypoints="paw"
                )
            
            # fit full model once
            itrain, itest = split_traintest(len(tneural) - 1)
            vefull = compute_varexp_small(
                spks, X, tcam, tneural, nmin, delay, device, itrain, itest, U=U
            )[0]
            print(vefull.mean(axis=0))

            ves = np.zeros((len(tlengths) + 1, 2))
            for sample in range(nsamples):
                np.random.seed(sample)
                trand = np.random.randint(
                    0, itrain.shape[1] - int((tlengths / itrain.shape[0]).max())
                )
                for k, tlen in enumerate((tlengths / itrain.shape[0]).astype(int)):
                    print(f"tlen={tlen}")
                    itime = np.arange(trand, trand + tlen)
                    vea = compute_varexp_small(
                        spks,
                        X,
                        tcam,
                        tneural,
                        nmin,
                        delay,
                        device,
                        itrain,
                        itest,
                        itime=itime,
                    )[0]
                    veak = vea.mean(axis=0)
                    ves[k] += veak
                    print(veak)
                veak = vefull.mean(axis=0)
                ves[-1] += veak
                print(veak)
            ves /= nsamples
            # print(ves)
            varexps_time0 = ves

            ves = np.zeros((len(nlengths) + 1, 2))
            for sample in range(nsamples):
                np.random.seed(sample)
                nrand = np.random.permutation(nneurons)
                for k, nlen in enumerate(nlengths):
                    print(f"nlen={nlen}")
                    ineurons = nrand[:nlen]
                    itime = np.arange(0, itrain.shape[1])
                    vea = compute_varexp_small(
                        spks,
                        X,
                        tcam,
                        tneural,
                        nmin,
                        delay,
                        device,
                        itrain,
                        itest,
                        ineurons=ineurons,
                    )[0]
                    veak = vea[: nlengths[0]].mean(axis=0)
                    ves[k] += veak
                    print(veak)
                veak = vefull[nrand[: nlengths[0]]].mean(axis=0)
                ves[-1] += veak
                print(veak)
            ves /= nsamples
            # print(ves)
            varexps_neurons0 = ves

            np.savez(
                f"{data_path}proc/neuralpred/{mname}_{datexp}_{blk}_{xtype}_scaling.npz",
                varexps_neurons=varexps_neurons0,
                varexps_time=varexps_time0,
                nlengths=nneurons_all[iexp],
                tlengths=ntime_all[iexp],
            )

        # varexps_neurons[iexp] = varexps_neurons0
        # varexps_time[iexp] = varexps_time0

        print(f"time {time.time()-tic:.2f}sec")
