Outputs
--------

ROI and SVD processing 
~~~~~~~~~~~~~~~~~~~~~~~
SVD processing saves two outputs: a \*.npy file and a \*.mat file. The output file contains the following variables:

- **filenames**: A 2D list of video filenames - a list within the 2D list consists of videos recorded simultaneously whereas sequential videos are stored as a separate list

- **Ly**, **Lx**: list of frame length in y-dim (Ly) and x-dim (Lx) for each video taken simultaneously 

- **sbin**: spatial bin size for SVDs 

- **Lybin**, **Lxbin**: list of number of pixels binned by sbin in Y (Ly) and X (Lx) for each video taken simultaneously 

- **sybin**, **sxbin**: coordinates of multivideo (for plotting/reshaping ONLY) 

- **LYbin**, **LXbin**: full-size of all videos embedded in rectangle (binned) 

- **fullSVD**: bool flag indicating whether “multivideo SVD” is computed 

- **save_mat**: bool flag indicating whether to save proc as `\*.mat` file 

- **avgframe**: list of average frames for each video from a subset of frames (binned by sbin)

- **avgframe_reshape**: average frame reshaped to size y-pixels by x-pixels 

- **avgmotion**: list of average motion computed for each video from a subset of frames (binned by sbin) 

- **avgmotion_reshape**: average motion reshaped to size y-pixels by x-pixels 

- **iframes**: an array containing the number of frames in each consecutive video

- **motion**: list of absolute motion energies across time - first is “multivideo” motion energy (empty if not computed) 

- **motSVD**: list of motion SVDs - first is “multivideo SVD” (empty if not computed) - each is of size number of frames by number of components (500)

- **motMask**: list of motion masks for each motion SVD - each motMask is pixels x components

- **motMask_reshape**: motion masks reshaped to: y-pixels x x-pixels x components 

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

The \*.npy saved is a dict which can be loaded in python as follows:

::

   import numpy as np
   proc = np.load('filename_proc.npy', allow_pickle=True).item()
   print(proc.keys())
   motion = proc['motion']

These \*_proc.npy\* files can be loaded in the GUI (and is
automatically loaded after processing). The checkboxes on the lower
left panel of the GUI can be used to toggle display of different traces/variables.

Keypoints processing 
~~~~~~~~~~~~~~~~~~~~

Keypoints processing saves two outputs: a \*.h5 and a \*metadata.pkl file. 
   - \*.h5 file contains: Keypoints stored as a 3D array of shape (3, number of bodyparts, number of frames). The first dimension of size 3 is in the order: (x, y, likelihood). For more details on using/loading the \*.h5 file in python see this `tutorial <https://github.com/MouseLand/facemap/blob/main/docs/notebooks/load_visualize_keypoints.ipynb>`__.
   - \*metadata.pkl file: contains a dictionary consisting of the following variables:
        -  batch_size: batch size used for inference
        -  image_size: frame size
        -  bbox: bounding box for cropping the video [x1, x2, y1, y2]
        -  total_frames: number of frames
        -  bodyparts: names of bodyparts 
        -  inference_speed: processing speed
       To load the pkl file in python, use the following code:
        
        ::

            import pickle
            with open('filename_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
            print(metadata.keys())
            print(metadata['bodyparts'])


Neural activity prediction output 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The output of neural activity prediction is saved in \*.npy file and optionally in \*.mat file. The output contains a dictionary with the following keys:

- predictions: a 2D array containing the predicted neural activity of shape (number of features x time)
- test_indices: a list of indices indicating sections of data used as test data for computing variance explained by the model
- variance_explained: variance explained by the model for test data
- plot_extent: extent of the plot used for plotting the predicted neural activity in the order [x1, y1, x2, y2]


