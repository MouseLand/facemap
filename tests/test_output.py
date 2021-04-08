"Test facemap pipeline by comparing outputs"
from facemap import process
import numpy as np
from pathlib import Path
import os

r_tol, a_tol = 1e-2, 1e-2

def test_output_single_video(data_dir, video_names):
    v1, _ = video_names
    data_dir1, _ = data_dir
    test_filenames = [[data_dir1+video for video in v1]]
    process.run(test_filenames, savepath=os.getcwd()+"/tests/")    

    output_filename, _ = os.path.splitext(v1[0])
    test_proc_filename = os.getcwd()+"/tests/"+output_filename+"_proc.npy"
    output = np.load(test_proc_filename,allow_pickle=True).item()
    expected_proc_filename = os.getcwd()+"/tests/expected_output/singlevideo_proc.npy"
    expected_output = np.load(expected_proc_filename,allow_pickle=True).item()
    
    assert is_output_correct(output, expected_output)

def test_output_multivideo(data_dir, video_names):
    v1, v2 = video_names
    data_dir1, data_dir2 = data_dir
    test1 = [data_dir1+video for video in v1]
    test2 = [data_dir2+video for video in v2]
    test_filenames = [test1, test2]
    process.run(test_filenames, savepath=os.getcwd()+"/tests/")    

    output_filename, _ = os.path.splitext(v1[0])
    test_proc_filename = os.getcwd()+"/tests/"+output_filename+"_proc.npy"
    output = np.load(test_proc_filename,allow_pickle=True).item()
    expected_proc_filename = os.getcwd()+"/tests/expected_output/multivideo_proc.npy"
    expected_output = np.load(expected_proc_filename,allow_pickle=True).item()

    assert is_output_correct(output, expected_output)

def is_output_correct(test_output, expected_output):
    params_match = check_params(test_output, expected_output)
    frames_match = check_frames(test_output, expected_output)
    motion_match = check_motion(test_output, expected_output)
    U_match = check_U(test_output, expected_output)
    V_match = check_V(test_output, expected_output)
    return params_match and frames_match and motion_match and U_match and V_match

def check_params(test_output, expected_output):
    all_outputs_match = (test_output['Ly'] == expected_output['Ly'] and
                    test_output['Lx'] == expected_output['Lx'] and
                    test_output['sbin'] ==  expected_output['sbin'] and
                    test_output['Lybin'][0] ==  expected_output['Lybin'][0] and
                    test_output['Lxbin'][0] == expected_output['Lxbin'][0] and 
                    test_output['sybin'][0] == expected_output['sybin'][0] and
                    test_output['sxbin'][0] == expected_output['sxbin'][0] and
                    test_output['LYbin'] == expected_output['LYbin'] and
                    test_output['LXbin'] == expected_output['LXbin']) 
    return all_outputs_match

def check_frames(test_output, expected_output):
    print(test_output['avgframe'][0].shape, expected_output['avgframe'][0].shape)
    avgframes_match = np.allclose(test_output['avgframe'][0], expected_output['avgframe'][0], rtol=r_tol, atol=a_tol) 
    avgmotion_match = np.allclose(test_output['avgmotion'][0], expected_output['avgmotion'][0], rtol=r_tol, atol=a_tol) 
    return avgframes_match and avgmotion_match

def check_U(test_output, expected_output):
    return np.allclose(test_output['motMask'][0], expected_output['motMask'][0], rtol=r_tol, atol=a_tol)

def check_V(test_output, expected_output):
    return np.allclose(test_output['motSVD'][0], expected_output['motSVD'][0], rtol=r_tol, atol=a_tol)

def check_motion(test_output, expected_output):
    return (test_output['motion'][0] == expected_output['motion'][0]).all()