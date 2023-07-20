"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
"Test Facemap's pose estimation output "
import os

import numpy as np

from facemap.pose import pose
from facemap.utils import load_keypoints

r_tol = 5e-3


def test_pose_model_initialization(video_names):
    video, _ = video_names
    pose_object = pose.Pose(filenames=[[video]], model_name="facemap_model_state")
    pose_object.load_model()

    assert pose_object is not None


def test_pose_estimation_output(data_dir, video_names, bodyparts, expected_output_dir):
    video, _ = video_names
    video_extension = "." + video.split(".")[-1]
    video_abs_path = str(data_dir.joinpath("cam1").joinpath(video))

    # Initialize model
    pose_object = pose.Pose(
        filenames=[[video_abs_path]], model_name="facemap_model_state"
    )
    # Run prediction
    pose_object.run()
    # Get output
    test_h5_path = (
        video_abs_path.split(video_extension)[0] + "_FacemapPose.h5"
    )  # change to remove _hdf
    test_pkl_path = (
        video_abs_path.split(video_extension)[0] + "_FacemapPose_metadata.pkl"
    )
    # Expected output path
    h5_filename = os.path.basename(test_h5_path)
    expected_h5_path = expected_output_dir.joinpath(h5_filename)

    # Compare outputs
    test_output = load_keypoints(bodyparts, test_h5_path)
    expected_output = load_keypoints(bodyparts, expected_h5_path)
    match = np.median(np.abs(test_output - expected_output))
    assert match < r_tol
