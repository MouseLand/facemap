"""
Facemap model trained for generating pose estimates. Contains functions for: 
- downloading pre-trained models
- Model class 
"""
import os
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

MODEL_PARAMS_URL = "https://www.facemappy.org/models/facemap_model_params.pth"
MODEL_STATE_URL = "https://www.facemappy.org/models/facemap_model_state.pt"

def get_data_dir():
    """
    Get the path to the data directory.
    """
    current_workdir = os.getcwd()
    model_dir = os.path.join(current_workdir, "facemap", "pose")
    # Change model directory to path object
    model_dir = Path(model_dir)
    return model_dir

def get_model_params_path():
    """
    Get the path to the model parameters file.
    """
    model_dir = get_data_dir()
    cached_params_file = str(model_dir.joinpath("facemap_model_params.pth"))
    if not os.path.exists(cached_params_file):
        download_url_to_file(MODEL_PARAMS_URL, cached_params_file)
    return cached_params_file

def get_model_state_path():
    """
    Get the path to the model state file.
    """
    model_dir = get_data_dir()
    cached_state_file = str(model_dir.joinpath("facemap_model_state.pt"))
    if not os.path.exists(cached_state_file):
        download_url_to_file(MODEL_STATE_URL, cached_state_file)
    return cached_state_file

def download_url_to_file(url, filename):
    """
    Download a file from a URL to a local file.
    """
    # Check if file already exists
    if os.path.exists(filename):
        return

    # Download file
    print("Downloading %s to %s" % (url, filename))
    urlretrieve(url, filename)

