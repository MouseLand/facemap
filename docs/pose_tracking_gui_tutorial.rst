Pose tracking **(GUI)**
========================

The latest python version is integrated with Facemap network for
tracking 14 distinct keypoints on mouse face and an additional point for
tracking paw. The keypoints can be tracked from different camera views (see `examples <https://github.com/MouseLand/facemap/blob/dev/figs/mouse_views.png>`__).


Generate keypoints
------------------

Follow the steps below to generate keypoints for your videos:

1. Load video

   -  Select ``File`` from the menu bar
   -  For processing single video, select ``Load video``. Alternatively,
      for processing multiple videos, select ``Load multiple videos`` to
      select the folder containing the videos. (Note: Pose estimation
      for multipl videos is only supported for videos recorded
      simultaneously i.e. have the same time duration and frame rate).

   (Optional) Set output folder

   -  Use the file menu to ``Set output folder``.
   -  The processed keypoints (``*.h5``) and metadata (``*.pkl``) will
      be saved in the selected output folder or folder containing the
      video (by default).

2. Process video(s)

   -  Check ``Keypoints`` for pose tracking.
   -  Click ``process``.
   -  Note: The first time facemap runs for processing keypoints it downloads the latest available trained model weights from our website.

3. Set ROI/bounding box for face region

   -  A dialog box for selecting a bounding box for the face will
      appear. Drag the red rectangle to select region of interest on the
      frame where the keypoints will be tracked. Please ensure that the
      bouding box is focused on the face where all the keypoints will be
      visible. See example frames `here <https://github.com/MouseLand/facemap/blob/main/figs/mouse_views.png>`__. If a
      ‘Face (pose)’ ROI has already been added then this step will be
      skipped.
   -  Click ``Done`` to process video. Alternatively, click ``Skip`` to
      use the entire frame region. Monitor progress bar at the bottom of
      the window for updates.

4. View keypoints

   -  Keypoints will be automatically loaded after processing.
   -  Processed keypoints file will be saved as
      ``[videoname]_FacemapPose.h5`` in the selected output folder.

Visualize keypoints
-------------------

To load keypoints (\*.h5) for a video generated using Facemap or other
software in the same format (such as DeepLabCut and SLEAP), follow the
steps below:

1. Load video

   -  Select ``File`` from the menu bar
   -  Select ``Load video``

2. Load keypoints

   -  Select ``Pose`` from the menu bar
   -  Select ``Load keypoints``
   -  Select the keypoints (\*.h5) file

3. View keypoints

   -  Use the “Keypoints” checkbox to toggle the visibility of
      keypoints.
   -  Change value of “Threshold (%)” under pose settings to filter
      keypoints with lower confidence estimates. Higher threshold will
      show keypoints with higher confidence estimates.

Finetune model to refine keypoints for a video
----------------------------------------------

To improve keypoints predictions for a video, follow the steps below:

1. Load video

   -  Select ``File`` from the menu bar
   -  Select ``Load video``

2. Set finetuned model’s output folder

   -  Select ``Pose`` from the menu bar
   -  Select ``Finetune model``
   -  Set output folder path for finetuned model

3. Select training data and set training parameters

   -  Set ``Initial model`` to use for training. By default, Facemap’s
      base model trained on our dataset will be used for fine-tuning.
      Alternatively, you can select a model previously finetuned on your
      own dataset.
   -  Set ``Output model name`` for the finetuned model.
   -  Choose ``Yes/No`` to refine keypoints prediction for the video
      loaded and set ``# Frames`` to use for training. You can also
      choose proportion of random vs. outlier frames to use for
      training. The outlier frames are selected using the
      ``Difficulty threshold (percentile)``, which determines the
      percentile of confidence scores to use as the threshold for
      selecting frames with the highest error.
   -  Choose ``Yes/No`` to add previously refined keypoints to the
      training set.
   -  ``Set training parameters`` or use default values.
   -  Click ``Next``

4. Refine keypoints

   -  If a ROI/bounding box was not added, then a dialog box for
      selecting a bounding box for the face will appear. Drag the red
      rectangle to select region of interest on the frame where the
      keypoints will be tracked.
   -  Click ``Done`` to process video. Alternatively, click ``Skip`` to
      use the entire frame region. Monitor progress bar at the bottom of
      the window for updates.
   -  Drag keypoints to refine predictions. Use ``Shift+D`` to delete a
      keypoint. Right click to add a deleted keypoint. Use ``Previous``
      and ``Next`` buttons to change frame. Click ``Help`` for more
      details.
   -  Click ``Train model`` to start training. A progress bar will
      appear for training updates.

5. Evaluate training

   -  View predicted keypoints for test frames from the video loaded.
      For further refinement, Click ``Continue training`` that will
      repeat steps 3-5.
   -  Click ``Save model`` to save the finetuned model. The finetuned
      model will be saved as ``*.pt`` in the selected output folder.

6. Generate keypoints using the finetuned model

   -  Use the ``Pose model`` dropdown menu to set the finetuned model to
      use for generating keypoints predictions.
   -  (Optional) Change “Batch size” under pose settings.
   -  Click ``Process`` to generate keypoints predictions. See `Generate
      keypoints <#generate-keypoints>`__ for more details.
