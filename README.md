[![Downloads](https://pepy.tech/badge/facemap)](https://pepy.tech/project/facemap)
[![Downloads](https://pepy.tech/badge/facemap/month)](https://pepy.tech/project/facemap)
[![GitHub stars](https://badgen.net/github/stars/Mouseland/facemap)](https://github.com/MouseLand/facemap/stargazers)
[![GitHub forks](https://badgen.net/github/forks/Mouseland/facemap)](https://github.com/MouseLand/facemap/network/members)
[![](https://img.shields.io/github/license/MouseLand/facemap)](https://github.com/MouseLand/facemap/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/facemap.svg)](https://badge.fury.io/py/facemap)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://pypi.org/project/facemap/)
[![GitHub open issues](https://badgen.net/github/open-issues/Mouseland/facemap)](https://github.com/MouseLand/facemap/issues)

# Facemap <img src="facemap/mouse.png" width="200" title="lilmouse" alt="lilmouse" align="right" vspace = "50">

GUI for face pose tracking of rodents from different camera views (python only) and svd processing of videos (python and MATLAB).

## [Installation](https://github.com/MouseLand/facemap/blob/dev/docs/installation.md)

##### For latest released version (from PyPI)

Run the following
~~~
pip install facemap
~~~
for headless version or the following for using GUI:
~~~~
pip install facemap[gui]
~~~~

To upgrade Facemap (package [here](https://pypi.org/project/facemap/)), within the environment run:
~~~~
pip install facemap --upgrade
~~~~

##### Using the environment.yml file (recommended)

1. Download the `environment.yml` file from the repository
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Run `conda env create -f environment.yml`
4. To activate this new environment, run `conda activate facemap`
5. You should see `(facemap)` on the left side of the terminal line. Now run `python -m facemap` and you're all set.

# Pose tracking

The latest python version is integrated with Facemap network for tracking 14 distinct keypoints on mouse face and an additional point for tracking paw. The keypoints can be tracked from different camera views as shown below. 

<p float="middle">
<img src="figs/mouse_face1_keypoints.png"  width="280" height="250" title="View 1" alt="view1" align="left" vspace = "10" hspace="30" style="border: 0.5px solid white"  />
<img src="figs/mouse_face0_keypoints.png" width="280" height="250" title="View 2" alt="view2" algin="right" vspace = "10" style="border: 0.5px solid white">
</p>
  
## [GUI Instructions](docs/pose_tracking_gui_tutorial.md)
For pose tracking, load video and check `keypoints` then click `process` button. A dialog box will appear for selecting a bounding box for the face. The keypoints will be tracked in the selected bounding box. Please ensure that the bouding box is focused on the face where all the keypoints shown above will be visible. See example frames [here](figs/mouse_views.png). 

Use the file menu to set path of output folder. The processed keypoints file will be saved in the output folder with an extension of `.h5` and corresponding metadata file with extension `.pkl`.

## [CLI Instructions](docs/pose_tracking_cli_tutorial.md)

For more examples, please see [tutorial notebooks](https://github.com/MouseLand/facemap/tree/dev/notebooks).

## :mega: User contributions :video_camera: :camera: 
Facemap's goal is to provide a simple way to generate keypoints for rodent face tracking. However, we need a large dataset of images from different camera views to reduce any errors on new mice videos. Hence, we would like to get your help to further expand our dataset. You can contribute by sending us a video or few frames of your mouse on following email address(es): `syedaa[at]janelia.hhmi.org` or `stringerc[at]janelia.hhmi.org`. Please let us know of any issues using the software by sending us an email or [opening an issue on GitHub](https://github.com/MouseLand/facemap/issues).


# SVD processing

Works for grayscale and RGB movies. Can process multi-camera videos. Some example movies to test the GUI on are located [here](https://drive.google.com/open?id=1cRWCDl8jxWToz50dCX1Op-dHcAC-ttto). You can save the output from both the python and matlab versions as a matlab file with a checkbox in the GUI (if you'd like to use the python version - it has a better GUI).

Supported movie files:

'.mj2','.mp4','.mkv','.avi','.mpeg','.mpg','.asf'

### Data acquisition info

IR ILLUMINATION:

For recording in darkness we use [IR illumination](https://www.amazon.com/Logisaf-Invisible-Infrared-Security-Cameras/dp/B01MQW8K7Z/ref=sr_1_12?s=security-surveillance&ie=UTF8&qid=1505507302&sr=1-12&keywords=ir+light) at 850nm, which works well with 2p imaging at 970nm and even 920nm. Depending on your needs, you might want to choose a different wavelength, which changes all the filters below as well. 950nm works just as well, and probably so does 750nm, which still outside of the visible range for rodents.  

If you want to focus the illumination on the mouse eye or face, you will need a different, more expensive system. Here is an example, courtesy of Michael Krumin from the Carandini lab: [driver](https://www.thorlabs.com/thorproduct.cfm?partnumber=LEDD1B), [power supply](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=1710&pn=KPS101#8865), [LED](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=2692&pn=M850L3#4426), [lens](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=259&pn=AC254-030-B#2231), and [lens tube](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=4109&pn=SM1V10#3389), and another [lens tube](https://www.thorlabs.com/thorproduct.cfm?partnumber=SM1L10).

CAMERAS:

We use [ptgrey cameras](https://www.ptgrey.com/flea3-13-mp-mono-usb3-vision-vita-1300-camera). The software we use for simultaneous acquisition from multiple cameras is [BIAS](http://public.iorodeo.com/notes/bias/) software. A basic lens that works for zoomed out views [here](https://www.bhphotovideo.com/c/product/414195-REG/Tamron_12VM412ASIR_12VM412ASIR_1_2_4_12_F_1_2.html). To see the pupil well you might need a better zoom lens [10x here](https://www.edmundoptics.com/imaging-lenses/zoom-lenses/10x-13-130mm-fl-c-mount-close-focus-zoom-lens/#specs).

For 2p imaging, you'll need a tighter filter around 850nm so you don't see the laser shining through the mouse's eye/head, for example [this](https://www.thorlabs.de/thorproduct.cfm?partnumber=FB850-40). Depending on your lenses you'll need to figure out the right adapter(s) for such a filter. For our 10x lens above, you might need all of these:  [adapter1](https://www.edmundoptics.com/optics/optical-filters/optical-filter-accessories/M52-to-M46-Filter-Thread-Adapter/), [adapter2](https://www.thorlabs.de/thorproduct.cfm?partnumber=SM2A53), [adapter3](https://www.thorlabs.de/thorproduct.cfm?partnumber=SM2A6), [adapter4](https://www.thorlabs.de/thorproduct.cfm?partnumber=SM1L03).


## [*HOW TO GUI* (Python)](docs/svd_python_tutorial.md)

([video](https://www.youtube.com/watch?v=Rq8fEQ-DOm4) with old install instructions)

<img src="figs/face_fast.gif" width="100%" alt="face gif">

Run the following command in a terminal
```
python -m facemap
```
Default starting folder is set to wherever you run `python -m FaceMap`


## [*HOW TO GUI* (MATLAB)](docs/svd_matlab_tutorial.md)

To start the GUI, run the command `MovieGUI` in this folder. The following window should appear. After you click an ROI button and draw an area, you have to **double-click** inside the drawn box to confirm it. To compute the SVD across multiple simultaneously acquired videos you need to use the "multivideo SVD" options to draw ROI's on each video one at a time.

<div align="center">
<img src="figs/GUIscreenshot.png" width="80%" alt="gui screenshot" >
</div>


