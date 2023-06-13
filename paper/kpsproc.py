"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from facemap import keypoints
from facemap.neural_prediction import prediction_utils as prediction


def filter_keypoints(data_path, dbs):
    """filter keypoints for outliers and by confidence threshold"""
    for iexp, db in enumerate(dbs):
        mname, datexp, blk, twocam = db["mname"], db["datexp"], db["blk"], db["2cam"]
        cids = [0, 1] if db["2cam"] else [0]
        for cid in cids:
            kp_path0 = f"{data_path}keypoints_new/cam{cid}_{mname}_{datexp}_{blk}_FacemapPose.h5"
            print(f"{iexp} loading {kp_path0}")

            xy, keypoint_labels = keypoints.load_keypoints(
                kp_path0, keypoint_labels=None, confidence_threshold=True
            )
            # x = xy.reshape(xy.shape[0], -1).copy()
            # x = (x - x.mean(axis=0)) / x.std(axis=0)

            np.save(
                f"{data_path}proc/keypoints/kpfilt_cam{cid}_{mname}_{datexp}_{blk}.npy",
                {"xy": xy, "keypoint_labels": keypoint_labels},
            )


def future_prediction(data_path, dbs):
    """predict keypoints into future"""
    varexps = []
    varexps_areas = []

    for iexp, db in enumerate(dbs):
        mname, datexp, blk, twocam = db["mname"], db["datexp"], db["blk"], db["2cam"]
        cid = 0
        kp_path0 = (
            f"{data_path}proc/keypoints/kpfilt_cam{cid}_{mname}_{datexp}_{blk}.npy"
        )
        d = np.load(kp_path0, allow_pickle=True).item()
        xy, keypoint_labels = d["xy"], d["keypoint_labels"]

        x = xy.reshape(xy.shape[0], -1).copy()
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        # remove paw
        x = x[:, :-2]
        keypoint_labels = keypoint_labels[:-1]

        vet, vet_area, tlags, ypred, itest = prediction.predict_future(
            x, keypoint_labels
        )
        vet[vet < 0] = 0
        vet_area[vet_area < 0] = 0
        if iexp == 3:
            ypred_ex = ypred[:, 0]
            x_ex = x[itest][0]

        plt.figure(figsize=(3, 3))
        plt.plot(tlags, np.nanmean(vet, axis=0), color="k")
        plt.plot(tlags, vet_area.T)
        plt.show()

        varexps.append(vet)
        varexps_areas.append(vet_area)

    np.save(
        f"{data_path}proc/keypoints/varexp_future_kpfilt.npy",
        {
            "varexps": varexps,
            "varexps_areas": varexps_areas,
            "tlags": tlags,
            "ypred_ex": ypred_ex,
            "x_ex": x_ex,
            "keypoint_labels": keypoint_labels,
        },
    )
