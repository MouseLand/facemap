"Test Facemap's pose estimation output "
import os
from pathlib import Path

import numpy as np
import pandas as pd

from facemap.pose import pose

r_tol = 5e-3

def test_pose_model_initialization(video_names):
    video, _ = video_names
    pose_object = pose.Pose(filenames=[[video]], model_name="facemap_model_state")
    pose_object.load_model()

    assert pose_object is not None

def test_pose_estimation_output(data_dir, video_names, expected_output_dir):

    video, _ = video_names
    video_extension = "." + video.split(".")[-1]
    video_abs_path = str(data_dir.joinpath("cam1").joinpath(video))

    # Initialize model
    pose_object = pose.Pose(
        filenames=[[video_abs_path]], model_name="facemap_model_state"
    )
    # Run prediction
    pose_object.run_all(plot=False)
    # Get output
    test_h5_path = video_abs_path.split(video_extension)[0] + "_FacemapPose.h5"
    # Expected output path
    h5_filename = os.path.basename(test_h5_path)
    expected_h5_path = expected_output_dir.joinpath(h5_filename)

    # Compare outputs
    test_output = pd.read_hdf(test_h5_path)
    expected_output = pd.read_hdf(expected_h5_path)
    match = np.median(np.abs(test_output.values - expected_output.values))
    assert match < r_tol
