Outputs
=======================

ROI and SVD processing 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Proccessed output
~~~~~~~~~~~~~~~~~

The GUIs create one file for all videos (saved in current folder), the
npy file has name “videofile_proc.npy” and the mat file has name
“videofile_proc.mat”.

- **filenames**: list of lists of video filenames - each list are the videos taken simultaneously 

- **Ly**, **Lx**: list of number of pixels in Y (Ly) and X (Lx) for each video taken simultaneously 

- **sbin**: spatial bin size for motion SVDs 

- **Lybin**, **Lxbin**: list of number of pixels binned by sbin in Y (Ly) and X (Lx) for each video taken simultaneously 

- **sybin**, **sxbin**: coordinates of multivideo (for plotting/reshaping ONLY) 

- **LYbin**, **LXbin**: full-size of all videos embedded in rectangle (binned) 

- **fullSVD**: whether or not “multivideo SVD” is computed 

- **save_mat**: whether or not to save proc as `\*.mat` file 

- **avgframe**: list of average frames for each video from a subset of frames (binned by sbin)

- **avgframe_reshape**: average frame reshaped to be y-pixels x x-pixels 

- **avgmotion**: list of average motions for each video from a subset of frames (binned by sbin) 

- **avgmotion_reshape**: average motion reshaped to be y-pixels x x-pixels 

- **iframes**: array containing number of frames in each consecutive video

- **motion**: list of absolute motion energies across time - first is “multivideo” motion energy (empty if not computed) 

- **motSVD**: list of motion SVDs - first is “multivideo SVD” (empty if not computed) - each is nframes x components 

- **motMask**: list of motion masks for each motion SVD - each motMask is pixels x components

- **motMask_reshape**: motion masks reshaped to be y-pixels x x-pixels x components 

- **motSv**: array containing singular values for motSVD

- **movSv**: array containing singular values for movSVD

- **pupil**: list of pupil ROI outputs - each is a dict with ‘area’, ‘area_smooth’, and ‘com’ (center-of-mass)

- **blink**: list of blink ROI outputs - each is nframes, the blink area on each frame 

- **running**: list of running ROI outputs - each is nframes x 2, for X and Y motion on each frame 

- **rois**: ROIs that were drawn and computed:

    - rind: type of ROI in number

    - rtype: what type of ROI (‘motion SVD’, ‘pupil’, ‘blink’, ‘running’) 

    - ivid: in which video is the ROI 

    - color: color of ROI 

    - yrange: y indices of ROI 

    - xrange: x indices of ROI

    - saturation: saturation of ROI (0-255) 

    - pupil_sigma: number of stddevs used to compute pupil radius (for pupil ROIs)

    - yrange_bin: binned indices in y (if motion SVD) 

    - xrange_bin: binned indices in x (if motion SVD)

Loading outputs
''''''''''''''''''''

Note this is a dict, e.g. to load in python:

::

   import numpy as np
   proc = np.load('cam1_proc.npy', allow_pickle=True).item()
   print(proc.keys())
   motion = proc['motion']

These \*_proc.npy\* files can be loaded into the GUI (and will
automatically be loaded after processing). The checkboxes in the lower
left allow you to view different traces from the processing.
