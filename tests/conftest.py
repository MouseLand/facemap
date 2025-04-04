"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os
import shutil
import sys
import tempfile
from glob import glob
from pathlib import Path
from urllib.request import urlopen
import zipfile

import pytest
from tqdm import tqdm


@pytest.fixture(scope="session")
def video_names():
    video1_name = "cam1_test.avi"
    video2_name = "cam2_test.avi"
    return video1_name, video2_name


@pytest.fixture(scope="session")
def bodyparts():
    BODYPARTS = [
        "eye(back)",
        "eye(bottom)",
        "eye(front)",
        "eye(top)",
        "lowerlip",
        "mouth",
        "nose(bottom)",
        "nose(r)",
        "nose(tip)",
        "nose(top)",
        "nosebridge",
        "paw",
        "whisker(I)",
        "whisker(III)",
        "whisker(II)",
    ]
    return BODYPARTS

def extract_zip(cached_file, url, data_path):
    download_url_to_file(url, cached_file)        
    with zipfile.ZipFile(cached_file,"r") as zip_ref:
        zip_ref.extractall(data_path)

@pytest.fixture(scope="session")
def data_dir(video_names):
    fm_dir = Path.home().joinpath(".facemap")
    fm_dir.mkdir(exist_ok=True)
    extract_zip(fm_dir.joinpath("data.zip"), "https://osf.io/download/67f025b541d31bf67eb256dd/", fm_dir)
    data_dir = fm_dir.joinpath("data")
    return data_dir


@pytest.fixture(scope="session")
def expected_output_dir(data_dir):
    expected_output_dir = data_dir.joinpath("expected_output")
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
    if hasattr(meta, "getheaders"):
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
        with tqdm(
            total=file_size,
            disable=not progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
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


@pytest.fixture(autouse=True, scope="session")
def test_suite_cleanup(data_dir):
    # setup
    yield
    # teardown - put your command here
    clear_dir_outputs(data_dir)


def clear_dir_outputs(data_dir):
    # Delete test outputs
    folders = ["cam1", "cam2"]
    for folder in folders:
        files = glob(os.path.join(data_dir, folder, "*"))
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext != ".avi":
                os.remove(filename)
