from genericpath import exists
import pytest
import os, sys, tempfile, shutil
from tqdm import tqdm
from pathlib import Path
from urllib.request import urlopen

@pytest.fixture()
def video_names():
    video1_name = 'cam1_test.avi'
    video2_name = 'cam2_test.avi'
    return video1_name, video2_name

@pytest.fixture()
def data_dir(video_names):
    fm_dir = Path.home().joinpath('.facemap')
    fm_dir.mkdir(exist_ok=True)
    data_dir = fm_dir.joinpath('data')
    data_dir.mkdir(exist_ok=True)
    data_dir_cam1 = data_dir.joinpath('cam1')
    data_dir_cam1.mkdir(exist_ok=True)
    data_dir_cam2 = data_dir.joinpath('cam2')
    data_dir_cam2.mkdir(exist_ok=True)

    for i,video_name in enumerate(video_names):
        url = 'https://www.facemappy.org/test_data/' + video_name
        if '1' in video_name:
            cached_file = str(data_dir_cam1.joinpath(video_name))
        else:
            cached_file = str(data_dir_cam2.joinpath(video_name))
        if not os.path.exists(cached_file):
            download_url_to_file(url, cached_file)

    return data_dir

@pytest.fixture()
def expected_output_dir(data_dir):
    expected_output_dir = data_dir.joinpath('expected_output')
    expected_output_dir.mkdir(exist_ok=True)
    # Download expected output files
    """
    download_url_to_file('https://www.facemappy.org/test_data/singlevideo_proc.npy', 
                        expected_output_dir.joinpath('singlevideo_proc.npy'))
    download_url_to_file('https://www.facemappy.org/test_data/multivideo_proc.npy', 
                        expected_output_dir.joinpath('multivideo_proc.npy'))
    """
    return expected_output_dir
    
def download_url_to_file(url, dst, progress=True):
    # Following adapted from https://github.com/MouseLand/cellpose/blob/35c16c94e285a4ec2fa17f148f06bbd414deb5b8/cellpose/utils.py#L45
    """Download object at the given URL to a local path.
            Thanks to torch, slightly modified
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)