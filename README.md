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
  
## [GUI Instructions]()
For pose tracking, load video and check `keypoints` then click `process` button. A dialog box will appear for selecting a bounding box for the face. The keypoints will be tracked in the selected bounding box. Please ensure that the bouding box is focused on the face where all the keypoints shown above will be visible. See example frames [here](figs/mouse_views.png). 

Use the file menu to set path of output folder. The processed keypoints file will be saved in the output folder with an extension of `.h5` and corresponding metadata file with extension `.pkl`.

## [CLI Instructions]()

For more examples, please see [tutorial notebooks](https://github.com/MouseLand/facemap/tree/dev/notebooks).

## User contributions :video_camera:
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

## *HOW TO GUI* (Python)

([video](https://www.youtube.com/watch?v=Rq8fEQ-DOm4) with old install instructions)

<img src="figs/face_fast.gif" width="100%" alt="face gif">

Run the following command in a terminal
```
python -m facemap
```
Default starting folder is set to wherever you run `python -m FaceMap`

The following window should appear. The upper left "file" button loads single files, the upper middle "folder" button loads whole folders (from which you can select movies), and the upper right "folder" button loads processed files ("_proc.npy" files). Load a video or a group of videos (see below for file formats for simultaneous videos). The video(s) will pop up in the left side of the GUI. You can zoom in and out with the mouse wheel, and you can drag by holding down the mouse. Double-click to return to the original, full view.

Choose a type of ROI to add and then click "add ROI" to add it to the view. The pixels in the ROI will show up in the right window (with different processing depending on the ROI type - see below). You can move it and resize the ROI anytime. You can delete the ROI with "right-click" and selecting "remove". You can change the saturation of the ROI with the upper right saturation bar. You can also just click on the ROI at any time to see what it looks like in the right view.

By default, the "Compute multivideo SVD" box is unchecked. If you check it, then the motion SVD is computed across ALL videos - all videos are concatenated at each timepoint, and the SVD of this matrix of ALL_PIXELS x timepoints is computed. If you have just one video acquired at a time, then it is the SVD of this video.

<div align="center">
<img src="figs/multivideo_fast.gif" width="100%" alt="multivideo gif" >
</div>

Once processing starts, the interface will no longer be clickable and all information about processing will be in the terminal in which you opened FaceMap:
<div align="center">
<img src="figs/terminal.png" width="50%" alt="terminal" >
</div>

If you want to open the GUI with a movie file specified and/or save path specified, the following command will allow this:
~~~
python -m facemap --movie '/home/carsen/movie.avi' --savedir '/media/carsen/SSD/'
~~~
Note this will only work if you only have one file that you need to load (can't have multiple in series / multiple views).

#### Processing movies captured simultaneously (multiple camera setups)

Both GUIs will then ask *"are you processing multiple videos taken simultaneously?"*. If you say yes, then the script will look if across movies the **FIRST FOUR** letters of the filename vary. If the first four letters of two movies are the same, then the GUI assumed that they were acquired *sequentially* not *simultaneously*.

Example file list:
+ cam1_G7c1_1.avi
+ cam1_G7c1_2.avi
+ cam2_G7c1_1.avi
+ cam2_G7c1_2.avi
+ cam3_G7c1_1.avi
+ cam3_G7c1_2.avi

*"are you processing multiple videos taken simultaneously?"* ANSWER: Yes

Then the GUIs assume {cam1_G7c1_1.avi, cam2_G7c1_1.avi, cam3_G7c1_1.avi} were acquired simultaneously and {cam1_G7c1_2.avi, cam2_G7c1_2.avi, cam3_G7c1_2.avi} were acquired simultaneously. They will be processed in alphabetical order (1 before 2) and the results from the videos will be concatenated in time. If one of these files was missing, then the GUI will error and you will have to choose file folders again. Also you will get errors if the files acquired at the same time aren't the same frame length (e.g. {cam1_G7c1_1.avi, cam2_G7c1_1.avi, cam3_G7c1_1.avi} should all have the same number of frames).

Note: if you have many simultaneous videos / overall pixels (e.g. 2000 x 2000) you will need around 32GB of RAM to compute the full SVD motion masks.

You will be able to see all the videos that were simultaneously collected at once. However, you can only draw ROIs that are within ONE video. Only the "multivideo SVD" is computed over all videos.


##### Batch processing (python only)

Load a video or a set of videos and draw your ROIs and choose your processing settings. Then click **save ROIs**. This will save a *_proc.npy file in the folder in the specified **save folder**. The name of this proc file will be listed below **process batch** (this button will also activate). You can then repeat this process: load the video(s), draw ROIs, choose settings, and click **save ROIs**. Then to process all the listed *_proc.npy files click **process batch**.

#### Multivideo SVD ROIs

Check box "Compute multivideo SVD" to compute the SVD of all pixels in all videos.

The GUIs create one file for all videos (saved in current folder), the processed .npy file has name "_proc.npy" which contains:

**PYTHON**:
- **filenames**: list of lists of video filenames - each list are the videos taken simultaneously
- **Ly**, **Lx**: list of number of pixels in Y (Ly) and X (Lx) for each video taken simultaneously
- **sbin**: spatial bin size for motion SVDs
- **Lybin**, **Lxbin**: list of number of pixels binned by sbin in Y (Ly) and X (Lx) for each video taken simultaneously
- **sybin**, **sxbin**: coordinates of multivideo (for plotting/reshaping ONLY)
- **LYbin**, **LXbin**: full-size of all videos embedded in rectangle (binned)
- **fullSVD**: whether or not "multivideo SVD" is computed
- **save_mat**: whether or not to save proc as *.mat file
- **avgframe**: list of average frames for each video from a subset of frames (binned by sbin)
- **avgframe_reshape**: average frame reshaped to be y-pixels x x-pixels
- **avgmotion**: list of average motions for each video from a subset of frames (binned by sbin)
- **avgmotion_reshape**: average motion reshaped to be y-pixels x x-pixels
- **iframes**: array containing number of frames in each consecutive video
- **motion**: list of absolute motion energies across time - first is "multivideo" motion energy (empty if not computed)
- **motSVD**: list of motion SVDs - first is "multivideo SVD" (empty if not computed) - each is nframes x components
- **motMask**: list of motion masks for each motion SVD - each motMask is pixels x components
- **motMask_reshape**: motion masks reshaped to be y-pixels x x-pixels x components
- **pupil**: list of pupil ROI outputs - each is a dict with 'area', 'area_smooth', and 'com' (center-of-mass)
- **blink**: list of blink ROI outputs - each is nframes, the blink area on each frame
- **running**: list of running ROI outputs - each is nframes x 2, for X and Y motion on each frame
- **rois**: ROIs that were drawn and computed
    - *rind*: type of ROI in number
    - *rtype*: what type of ROI ('motion SVD', 'pupil', 'blink', 'running')
    - *ivid*: in which video is the ROI
    - *color*: color of ROI
    - *yrange*: y indices of ROI
    - *xrange*: x indices of ROI
    - *saturation*: saturation of ROI (0-255)
    - *pupil_sigma*: number of stddevs used to compute pupil radius (for pupil ROIs)
    - *yrange_bin*: binned indices in y (if motion SVD)
    - *xrange_bin*: binned indices in x (if motion SVD)

Note this is a dict, so say *.item() after loading:
```
import numpy as np
proc = np.load('cam1_proc.npy').item()
```

These *_proc.npy* files can be loaded into the GUI (and will automatically be loaded after processing). The checkboxes in the lower left allow you to view different traces from the processing.

## [*HOW TO GUI* (MATLAB)](https://github.com/MouseLand/facemap/blob/dev/docs/svd_tutorial_matlab.md)

To start the GUI, run the command `MovieGUI` in this folder. The following window should appear. After you click an ROI button and draw an area, you have to **double-click** inside the drawn box to confirm it. To compute the SVD across multiple simultaneously acquired videos you need to use the "multivideo SVD" options to draw ROI's on each video one at a time.

<div align="center">
<img src="figs/GUIscreenshot.png" width="80%" alt="gui screenshot" >
</div>


