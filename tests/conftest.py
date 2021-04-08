import pytest
import os, sys
from pathlib import Path

@pytest.fixture()
def video_names():
    video1_names = ['cam1.avi']
    video2_names = ['cam1_2.avi']
    return video1_names, video2_names

@pytest.fixture()
def data_dir(video_names):
    data_dir1 = os.path.join(os.getcwd(),'Video_samples/sample_movies2/')
    data_dir2 = os.path.join(os.getcwd(),'Video_samples/sample_movies/')
    return data_dir1, data_dir2