# Pose tracking **(GUI)** :mouse:

<img src="../figs/tracker.gif" width="100%" height="500" title="Tracker" alt="tracker" algin="middle" vspace = "10">

The latest python version is integrated with Facemap network for tracking 14 distinct keypoints on mouse face and an additional point for tracking paw. The keypoints can be tracked from different camera views (some examples shown below). 

<p float="middle">
<img src="../figs/mouse_face1_keypoints.png"  width="310" height="290" title="View 1" alt="view1" align="left" vspace = "10" hspace="30" style="border: 0.5px solid white"  />
<img src="../figs/mouse_face0_keypoints.png" width="310" height="290" title="View 2" alt="view2" algin="right" vspace = "10" style="border: 0.5px solid white">
</p>

## Generate keypoints

Follow the steps below to generate keypoints for your videos:

1. Load video 
    - Select `File` from the menu bar
    - For processing single video, select `Load video`. Alternatively, for processing multiple videos, select `Load multiple videos` to select the folder containing the videos. (Note: Pose estimation for multipl videos is only supported for videos recorded simultaneously i.e. have the same time duration and frame rate).

    (Optional) Set output folder
    - Use the file menu to `Set output folder`. 
    - The processed keypoints (`*.h5`) and metadata (`*.pkl`) will be saved in the selected output folder or folder containing the video (default).
2. Process video(s)
    - Check `Keypoints` for pose tracking.
    - Click `process`.
3. Set ROI/bounding box for face region
    - A dialog box for selecting a bounding box for the face will appear. Drag the red rectangle to select region of interest on the frame where the keypoints will be tracked. Please ensure that the bouding box is focused on the face where all the keypoints will be visible. See example frames [here](figs/mouse_views.png). If a 'Face (pose)' ROI has already been added then this step will be skipped.
    - Click `Done` to process video. Alternatively, click `Skip` to use the entire frame region. Monitor progress bar at the bottom of the window for updates.
4. View keypoints
    - Keypoints will be automatically loaded after processing.
    - Processed keypoints file will be saved as `[videoname]_FacemapPose.h5` in the selected output folder. 

## Visualize pose estimates 

For plotting pose estimates, generated using Facemap or other software that save output (*.h5) in the same format as DeepLabCut, follow the following steps:

1. Load video: Select `File` from the menu bar and select `Load single movie file`
2. Load data: From the file menu options, select `Load pose data`
3. Select the `Keypoints` checkbox on GUI to plot the keypoints on the loaded video. Toggle the checkbox to change visibility of the points.

Note: this feature currently only supports single video.

## Finetune model to refine keypoints for a video


