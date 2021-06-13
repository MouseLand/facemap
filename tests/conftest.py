import pytest
import os, sys
from pathlib import Path

@pytest.fixture()
def video_names():
    video1_names = ['cam1_test.avi']
    video2_names = ['cam2_test.avi']
    return video1_names, video2_names

@pytest.fixture()
def data_dir(video_names):
    data_dir = os.path.join(os.getcwd(),'Video_samples/sample_movies/')
    #data_dir2 = os.path.join(os.getcwd(),'Video_samples/sample_movies2/')
    return data_dir#1, data_dir2
