"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
"""
Facemap model trained for generating pose estimates. Contains functions for: 
- downloading pre-trained models
- Model class 
"""
import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve

MODEL_PARAMS_URL = "https://osf.io/download/67f00beaba4331d9888b7f36/"
MODEL_STATE_URL = "https://osf.io/download/67f00be8959068ade6cf70f1/"


def get_data_dir():
    """
    Get the path to the data directory.
    """
    current_workdir = os.getcwd()
    model_dir = os.path.join(current_workdir, "facemap", "pose")
    # Change model directory to path object
    model_dir = Path(model_dir)
    return model_dir


def get_models_dir():
    """
    Get the path to the hidden data directory containing model data.
    """
    fm_dir = Path.home().joinpath(".facemap")
    fm_dir.mkdir(exist_ok=True)
    models_dir = fm_dir.joinpath("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir


def copy_to_models_dir(filename):
    """
    Copy a file to the facemap models directory.
    """
    # Check if file already exists
    if not os.path.exists(filename):
        # File does not exist, raise an error
        raise FileNotFoundError("File %s does not exist" % filename)

    # Copy file to hidden directory
    model_dir = get_models_dir()
    # check if filename already exists in model_dir
    if not os.path.exists(model_dir.joinpath(filename)):
        shutil.copy(filename, model_dir)
        print("Copied %s to %s" % (filename, model_dir))
    else:
        print("File %s already exists in %s. Overwriting file." % (filename, model_dir))
        shutil.copy(filename, model_dir)
    return model_dir


def get_model_params_path():
    """
    Get the path to the model parameters file.
    """
    model_dir = get_models_dir()
    cached_params_file = str(model_dir.joinpath("facemap_model_params.pth"))
    if not os.path.exists(cached_params_file):
        download_url_to_file(MODEL_PARAMS_URL, cached_params_file)
    # If the file is not in the models directory, copy it there
    if not os.path.exists(cached_params_file):
        copy_to_models_dir(cached_params_file)
    return cached_params_file


def get_model_states_paths():
    """
    Get the paths to the model state files.
    """
    # Get the path to the models directory
    model_dir = get_models_dir()
    # Get all .pt files in the models directory
    model_files = [
        str(model_dir.joinpath(f)) for f in os.listdir(model_dir) if f.endswith(".pt")
    ]
    return model_files


def get_basemodel_state_path():
    """
    Get the path to the model state file.
    """
    model_dir = get_models_dir()
    cached_state_file = str(model_dir.joinpath("facemap_model_state.pt"))
    if not os.path.exists(cached_state_file):
        download_url_to_file(MODEL_STATE_URL, cached_state_file)
    # If the file is not in the models directory, copy it there
    if not os.path.exists(cached_state_file):
        copy_to_models_dir(cached_state_file)
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


def update_models_data_txtfile(filenames):
    """
    Update the facemap models data text file containing names of files used for training the model from refined keypoints.
    Input:-
        - filenames: list of strings containing the filenames of the refined keypoints files to be added to the text file
    """
    # Get the path to the models directory
    model_dir = get_models_dir()
    # Get the path to the models data text file
    models_data_file = str(model_dir.joinpath("models_data.txt"))
    # Check if the file exists
    if not os.path.exists(models_data_file):
        # Create the file
        open(models_data_file, "w").close()
    # Open the file
    file_readobject = open(models_data_file, "r")
    # Read the file
    with open(models_data_file, "a") as f:  # append mode
        for data_file in filenames:
            # If the filename exists in the file, skip it
            if data_file in file_readobject.read():
                continue
            else:
                # Write the filename to the file
                f.write(data_file + "\n")
    return models_data_file


def get_model_files():
    """
    Get the filenames of the data files used for model files.
    """
    # Get the path to the models directory
    model_dir = get_models_dir()
    # Get the path to the models data text file
    models_data_file = str(model_dir.joinpath("models_data.txt"))
    # Check if the file exists
    if not os.path.exists(models_data_file):
        # Create the file
        open(models_data_file, "w").close()
        return []
    # Open the file
    file_readobject = open(models_data_file, "r")
    # Read the file
    filenames = file_readobject.read().splitlines()
    return filenames

get_model_params_path()
get_basemodel_state_path()
