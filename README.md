[![Downloads](https://static.pepy.tech/badge/facemap)](https://pepy.tech/project/facemap)
[![Downloads](https://static.pepy.tech/badge/facemap/month)](https://pepy.tech/project/facemap)
[![GitHub stars](https://badgen.net/github/stars/Mouseland/facemap)](https://github.com/MouseLand/facemap/stargazers)
[![GitHub forks](https://badgen.net/github/forks/Mouseland/facemap)](https://github.com/MouseLand/facemap/network/members)
[![](https://img.shields.io/github/license/MouseLand/facemap)](https://github.com/MouseLand/facemap/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/facemap.svg)](https://badge.fury.io/py/facemap)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://pypi.org/project/facemap/)
[![GitHub open issues](https://badgen.net/github/open-issues/Mouseland/facemap)](https://github.com/MouseLand/facemap/issues)

# Facemap <img src="https://raw.githubusercontent.com/MouseLand/facemap/main/facemap/mouse.png" width="300" title="facemap" alt="facemap" align="right" vspace = "50">

Facemap is a framework for predicting neural activity from mouse orofacial movements. It includes a pose estimation model for tracking distinct keypoints on the mouse face, a neural network model for predicting neural activity using the pose estimates, and also can be used compute the singular value decomposition (SVD) of behavioral videos.

Please find the detailed documentation at **[facemap.readthedocs.io](https://facemap.readthedocs.io/en/latest/index.html)**.

To learn about Facemap, read the [paper](https://www.nature.com/articles/s41593-023-01490-6) or check out the tweet [thread](https://twitter.com/Atika_Ibrahim/status/1588885329951367168?s=20&t=AhE3vBTnCvW36QiTyhu0qQ). For support, please open an [issue](https://github.com/MouseLand/facemap/issues).

- For latest released version (from PyPI) including svd processing only, run `pip install facemap` for headless version or `pip install facemap[gui]` for using GUI. Note: `pip install facemap` not yet available for latest tracker and neural model, instead install with `pip install git+https://github.com/mouseland/facemap.git`

### CITATION

**If you use Facemap, please cite the Facemap [paper](https://www.nature.com/articles/s41593-023-01490-6):**   
Syeda, A., Zhong, L., Tung, R., Long, W., Pachitariu, M.\*, & Stringer, C.\* (2024). Facemap: a framework for modeling neural activity based on orofacial tracking. <em>Nature Neuroscience</em>, 27(1), 187-195.
[[bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:ckbIvC5D_FsJ:scholar.google.com/&output=citation&scisdr=ClF-mOb-EMjM4mZZ21s:AFWwaeYAAAAAZcpfw1vc6bUrQR0LDQdzaTPXbO8&scisig=AFWwaeYAAAAAZcpfw5Aeocyxj1cWqLJIgPajziE&scisf=4&ct=citation&cd=-1&hl=en)]

**If you use the SVD computation or pupil tracking components, please also cite our previous [paper](https://www.nature.com/articles/s41592-022-01663-4):**  
Stringer, C.\*, Pachitariu, M.\*, Steinmetz, N., Reddy, C. B., Carandini, M., & Harris, K. D. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. <em>Science, 364</em>(6437), eaav7893.
[[bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:DNVOkEas4K8J:scholar.google.com/&output=citation&scisdr=CgXHFLYtEMb9qP1Bt0Q:AAGBfm0AAAAAY3JHr0TJourtY6W2vbjy7opKXX2jOX9Z&scisig=AAGBfm0AAAAAY3JHryiZnvgWM1ySwd_xQ9brvQxH71UM&scisf=4&ct=citation&cd=-1&hl=en&scfhb=1)]

The MATLAB version of the GUI is no longer supported (see old [documentation](https://github.com/MouseLand/facemap/blob/main/docs/svd_matlab_tutorial.md)).

### Disclaimer
The outputs of Facemap have only been tested on macos-12 and earlier versions and newer versions may give different/incorrect output so it's advised to use macos-12 for Facemap until the issue is resolved.

### Logo
Logo was designed by Atika Syeda and [Tzuhsuan Ma](https://github.com/tzhma).

### Video tutorial 
Please follow the [video tutorial](https://www.youtube.com/watch?v=aO_kXkOuadg) for instructions on how to use Facemap or read the instructions below. 

## Installation

If you have an older `facemap` environment you can remove it with `conda env remove -n facemap` before creating a new one.

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

1. Install an [Anaconda](https://www.anaconda.com/products/distribution) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt which has `conda` for **python 3** in the path
3. Create a new environment with `conda create --name facemap python=3.8`. We recommend python 3.8, but python 3.9 and 3.10 will likely work as well.
4. To activate this new environment, run `conda activate facemap`
5. To install the minimal version of facemap, run `python -m pip install facemap`.  
6. To install facemap and the GUI, run `python -m pip install facemap[gui]`. If you're on a zsh server, you may need to use ' ' around the facemap[gui] call: `python -m pip install 'facemap[gui]'.

To upgrade facemap (package [here](https://pypi.org/project/facemap/)), run the following in the environment:

~~~sh
python -m pip install facemap --upgrade
~~~

Note you will always have to run `conda activate facemap` before you run facemap. If you want to run jupyter notebooks in this environment, then also `pip install notebook` and `python -m pip install matplotlib`.

You can also try to install facemap and the GUI dependencies from your base environment using the command

~~~~sh
python -m pip install facemap[gui]
~~~~

If you have **issues** with installation, see the [docs](https://github.com/MouseLand/facemap/blob/dev/docs/installation.md) for more details. You can also use the facemap environment file included in the repository and create a facemap environment with `conda env create -f environment.yml` which may solve certain dependency issues.

If these suggestions fail, open an issue.

### GPU version (CUDA) on Windows or Linux

If you plan on running many images, you may want to install a GPU version of *torch* (if it isn't already installed).

Before installing the GPU version, remove the CPU version:
~~~
pip uninstall torch
~~~

Follow the instructions [here](https://pytorch.org/get-started/locally/) to determine what version to install. The Anaconda install is strongly recommended, and then choose the CUDA version that is supported by your GPU (newer GPUs may need newer CUDA versions > 10.2). For instance this command will install the 11.3 version on Linux and Windows (note the `torchvision` and `torchaudio` commands are removed because facemap doesn't require them):

~~~
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
~~~~

and this will install the 11.7 toolkit

~~~
conda install pytorch pytorch-cuda=11.7 -c pytorch
~~~

## Supported videos
Facemap supports grayscale and RGB movies. The software can process multi-camera videos for pose tracking and SVD analysis. Please see [example movies](https://drive.google.com/open?id=1cRWCDl8jxWToz50dCX1Op-dHcAC-ttto) for testing the GUI. Movie file extensions supported include:

'.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf'

For more details, please refer to the [data acquisition page](https://github.com/MouseLand/facemap/blob/main/docs/data_acquisition.md).

## Support

For any issues or questions about Facemap, please [open an issue](https://github.com/MouseLand/facemap/issues). Please find solutions to some common issues below:

### Download of pretrained models
The models will be downloaded automatically from our website when you first run Facemap for processing keypoints. If download of pretrained models fails, please try the following:

- to resolve certificate error try: ```pip install –upgrade certifi```, or
- download the pretrained model files: [model_params](https://osf.io/download/67f00beaba4331d9888b7f36/), [model_state](https://osf.io/download/67f00be8959068ade6cf70f1/)  and place them in the `models` subfolder of the hidden `facemap` folder located in home directory. Path to the hidden folder is: `C:\Users\your_username\.facemap\models` on Windows and `/home/your_username/.facemap/models` on Linux and Mac. 

# Running Facemap

To get started, run the following command in terminal to open the GUI:

```
python -m facemap
```

Click "File" and load a single video file ("Load video"), or click "Load multiple videos" to choose a folder from which you can select movies to run. The video(s) will pop up in the left side of the GUI. You can zoom in and out with the mouse wheel, and you can drag by holding down the mouse. Double-click to return to the original, full view.

Next you can extract information from the videos like track keypoints, compute movie SVDs, track pupil size etc. Also you can load in neural activity and predict it from these extracted features.

## I. Pose tracking

<img src="https://raw.githubusercontent.com/MouseLand/facemap/main/figs/facemap.gif" width="100%" height="470" title="Tracker" alt="tracker" algin="middle" vspace = "10">

Facemap provides a trained network for tracking distinct keypoints on the mouse face from different camera views (some examples shown below). Check the `keypoints` box then click `process`. Next a bounding box will appear -- focus this on the face as shown below. Then the processed keypoints `*.h5` file will be saved in the output folder along with the corresponding metadata file `*.pkl`.

Keypoints will be predicted in the selected bounding box region so please ensure the bounding box focuses on the face. See example frames [here](figs/mouse_views.png). 

For more details on using the tracker, please refer to the [GUI Instructions](https://github.com/MouseLand/facemap/blob/main/docs/pose_tracking_gui_tutorial.md). Check out the [notebook](https://github.com/MouseLand/facemap/blob/main/docs/notebooks/process_keypoints.ipynb) for processing keypoints in colab.

<p float="middle">
<img src="https://raw.githubusercontent.com/MouseLand/facemap/main/figs/mouse_face1_keypoints.png"  width="310" height="290" title="View 1" alt="view1" align="left" vspace = "10" hspace="30" style="border: 0.5px solid white"  />
<img src="https://raw.githubusercontent.com/MouseLand/facemap/main/figs/mouse_face0_keypoints.png" width="310" height="290" title="View 2" alt="view2" algin="right" vspace = "10" style="border: 0.5px solid white">
</p>

### 📢 User contributions 📹 📷
Facemap aims to provide a simple and easy-to-use tool for tracking mouse orofacial movements. The tracker's performance for new datasets could be further improved by expand our training set. You can contribute to the model by sharing videos/frames on the following email address(es): `asyeda1[at]jh.edu` or `stringerc[at]janelia.hhmi.org`.

## II. ROI and SVD processing

Facemap allows pupil tracking, blink tracking and running estimation, see more details **here**. Also, Facemap can compute the singular value decomposition (SVD) of ROIs on single and multi-camera videos. SVD analysis can be performed across static frames called movie SVD (`movSVD`) to extract the spatial components or over the difference between consecutive frames called motion SVD (`motSVD`) to extract the temporal components of the video. The first 500 principal components from SVD analysis are saved as output along with other variables.

You can draw ROIs to compute the motion/movie SVD within the ROI, and/or compute the full video SVD by checking `multivideo`. Then check `motSVD`  and/or `movSVD` and click `process`. The processed SVD `*_proc.npy` (and optionally `*_proc.mat`) file will be saved in the output folder selected.

For more details see [SVD python tutorial](https://github.com/MouseLand/facemap/blob/main/docs/svd_python_tutorial.md) or [SVD MATLAB tutorial](https://github.com/MouseLand/facemap/blob/main/docs/svd_matlab_tutorial.md).

([video](https://www.youtube.com/watch?v=Rq8fEQ-DOm4) with old install instructions)

<img src="https://github.com/MouseLand/facemap/raw/main/figs/face_fast.gif" width="100%" alt="face gif">

## III. Neural activity prediction

Facemap includes a deep neural network encoding model for predicting neural activity or principal components of neural activity from mouse orofacial pose estimates extracted using the tracker or SVDs. 

The encoding model used for prediction is described as follows:
<p float="middle">
<img src="https://raw.githubusercontent.com/MouseLand/facemap/main/figs/encoding_model.png"  width="70%" height="300" title="neural model" alt="neural model" align="center" vspace = "10" hspace="30" style="border: 0.5px solid white"  />
</p>

Please see neural activity prediction [tutorial](https://github.com/MouseLand/facemap/blob/main/docs/neural_activity_prediction_tutorial.md) for more details.
