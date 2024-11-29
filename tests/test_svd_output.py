"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
"Test facemap SVD processing outputs for single and multiple videos"
import os

import numpy as np

from facemap import process

r_tol, a_tol = 1e-2, 1e-1  # 1e-2, 1


def test_output_single_video(data_dir, video_names, expected_output_dir):
    v1, _ = video_names
    test_filenames = [[str(data_dir.joinpath("cam1").joinpath(v1))]]
    save_path = str(data_dir.joinpath("cam1"))
    output_filename, _ = v1.split(".")
    # Process video
    process.run(test_filenames, sbin=7, motSVD=True, movSVD=True, savepath=save_path)
    test_proc_filename = os.path.join(save_path, output_filename + "_proc.npy")
    # Compare outputs
    output = np.load(test_proc_filename, allow_pickle=True).item()
    expected_proc_filename = expected_output_dir.joinpath("single_video_proc.npy")
    expected_output = np.load(expected_proc_filename, allow_pickle=True).item()
    print("test_proc_filename", test_proc_filename)
    print("expected_proc_filename", expected_proc_filename)

    assert is_output_correct(output, expected_output)


def test_output_multivideo(data_dir, video_names, expected_output_dir):
    v1, v2 = video_names
    test1 = str(data_dir.joinpath("cam1").joinpath(v1))
    test2 = str(data_dir.joinpath("cam2").joinpath(v2))

    # For videos recorded simultaneously from multiple cams
    test_filenames = [[test1, test2]]
    save_path = str(data_dir.joinpath("cam2"))
    # Process videos
    process.run(test_filenames, sbin=12, motSVD=True, movSVD=True, savepath=save_path)
    output_filename, _ = v1.split(".")
    test_proc_filename = os.path.join(save_path, output_filename + "_proc.npy")
    print("test_proc_filename", test_proc_filename)

    # Compare output
    output = np.load(test_proc_filename, allow_pickle=True).item()
    expected_proc_filename = expected_output_dir.joinpath("multi_video_proc.npy")
    print("expected_proc_filename", expected_proc_filename)
    expected_output = np.load(expected_proc_filename, allow_pickle=True).item()

    assert is_output_correct(output, expected_output)


def is_output_correct(test_output, expected_output):
    params_match = check_params(test_output, expected_output)
    print("params match", params_match)
    frames_match = check_frames(test_output, expected_output)
    print("frames_match", frames_match)
    motion_match = check_motion(test_output, expected_output)
    print("motion_match", motion_match)
    U_match = check_U(test_output, expected_output)
    print("U_match", U_match)
    # V_match = check_V(test_output, expected_output)
    # print("V_match", V_match)
    return params_match and frames_match and motion_match and U_match  # and V_match


def check_params(test_output, expected_output):
    all_outputs_match = (
        test_output["Ly"] == expected_output["Ly"]
        and test_output["Lx"] == expected_output["Lx"]
        and test_output["sbin"] == expected_output["sbin"]
        and test_output["Lybin"][0] == expected_output["Lybin"][0]
        and test_output["Lxbin"][0] == expected_output["Lxbin"][0]
        and test_output["sybin"][0] == expected_output["sybin"][0]
        and test_output["sxbin"][0] == expected_output["sxbin"][0]
        and test_output["LYbin"] == expected_output["LYbin"]
        and test_output["LXbin"] == expected_output["LXbin"]
    )
    return all_outputs_match


def check_frames(test_output, expected_output):
    avgframes_match = np.allclose(
        test_output["avgframe"][0],
        expected_output["avgframe"][0],
        rtol=r_tol,
        atol=a_tol,
    )
    avgmotion_match = np.allclose(
        test_output["avgmotion"][0],
        expected_output["avgmotion"][0],
        rtol=r_tol,
        atol=a_tol,
    )
    avgframe_reshape_match = np.allclose(
        test_output["avgframe_reshape"][0],
        expected_output["avgframe_reshape"][0],
        rtol=r_tol,
        atol=a_tol,
    )
    avgmotion_reshape_match = np.allclose(
        test_output["avgmotion_reshape"][0],
        expected_output["avgmotion_reshape"][0],
        rtol=r_tol,
        atol=a_tol,
    )
    return (
        avgframes_match
        and avgmotion_match
        and avgframe_reshape_match
        and avgmotion_reshape_match
    )


def check_U(test_output, expected_output):
    nPCs = test_output["motSVD"][0].shape[1]
    motionMask_pos = [np.allclose(test_output["motMask"][0][:,i], 
                                expected_output["motMask"][0][:,i], 
                                rtol=r_tol, atol=a_tol) for i in range(nPCs)]
    motionMask_neg = [np.allclose(test_output["motMask"][0][:,i], 
                                -1 * expected_output["motMask"][0][:,i], 
                                rtol=r_tol, atol=a_tol) for i in range(nPCs)]
    motionMask = np.array(motionMask_pos) | np.array(motionMask_neg)
    motionMask = np.all(motionMask)

    movieMask_pos = [np.allclose(test_output["movMask"][0][:,i], 
                            expected_output["movMask"][0][:,i], 
                            rtol=r_tol, atol=a_tol) for i in range(nPCs)]
    movieMask_neg = [np.allclose(test_output["movMask"][0][:,i],
                                -1 * expected_output["movMask"][0][:,i],
                                rtol=r_tol, atol=a_tol) for i in range(nPCs)]
    movieMask = np.array(movieMask_pos) | np.array(movieMask_neg)
    movieMask = np.all(movieMask)
    motionMask_reshape = test_output["motMask_reshape"][0].shape == expected_output["motMask_reshape"][0].shape
    movieMask_reshape = test_output["movMask_reshape"][0].shape == expected_output["movMask_reshape"][0].shape
    print("motionMask", motionMask)
    print("movieMask", movieMask)
    print("motionMask_reshape", motionMask_reshape)
    print("movMask_reshape", movieMask_reshape)
    return motionMask and movieMask and motionMask_reshape and movieMask_reshape


def check_V(test_output, expected_output):
    motion_V = np.allclose(
        test_output["motSVD"][0], expected_output["motSVD"][0], rtol=r_tol, atol=a_tol
    )
    movie_V = np.allclose(
        test_output["movSVD"][0], expected_output["movSVD"][0], rtol=r_tol, atol=a_tol
    )
    return motion_V and movie_V


def check_motion(test_output, expected_output):
    return (test_output["motion"][0] == expected_output["motion"][0]).all()
