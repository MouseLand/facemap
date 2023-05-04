Neural activity prediction
==========================

This tutorial shows how to use the deep neural network encoding model
for neural prediction using mouse orofacial behavior.

To process neural activity prediction using pose estimates extracted
using the tracker:

1. Load or process keypoints (`see pose tracking
   tutorial <https://github.com/MouseLand/facemap/blob/main/docs/pose_tracking_gui_tutorial.md>`__).
2. Select ``Neural activity`` from file menu to ``Load neural data``.
3. Load neural activity data (2D-array stored in *.npy) and (optionally)
   timestamps for neural and behavioral data (1D-array stored in*.npy)
   then click ``Done``.
4. Select ``Run neural prediction`` from the ``Neural activity`` file
   menu.
5. Select ``Keypoints`` as input data and set whether the output of the
   model’s prediction to be ``neural PCs`` or neural activity. Use help
   button to set training parameters for the model.
6. The predicted neural activity \*.npy file will be saved in the
   selected output folder.

To process neural activity prediction using pose estimates extracted
using the tracker:

1. Load or process SVDs for the video. (`see SVD
   tutorial <https://github.com/MouseLand/facemap/blob/main/docs/svd_python_tutorial.md>`__).
2. Follow steps 2-5 above.

Note: a linear model is used for prediction using SVDs.

Predicted neural activity will be plotted in the bottom-right window of
the GUI. You can highlight test data by selecting
``Highlight test data`` from the ``Neural activity`` file menu. Further
information about neural prediction, including variance explained can be
found in the saved neural prediction file.
