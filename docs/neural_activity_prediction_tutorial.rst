Neural activity prediction
==========================

This tutorial shows how to use the deep neural network encoding model to
predict neural activity based on mouse orofacial behavior.

To process neural activity prediction using pose estimates extracted
using the keypoint tracker:

1. Load or process keypoints (`see pose tracking
   tutorial <https://github.com/MouseLand/facemap/blob/main/docs/pose_tracking_gui_tutorial.md>`__).
2. Select ``Neural activity`` from file menu.
3. Click on ``Launch neural activity window``.
4. Select ``Load neural activity`` (2D-array stored in \*.npy) and
   (optionally) timestamps for neural and behavioral data (1D-arrays
   stored in*.npy) then click ``Done``.
5. Once the neural data is loaded, click on ``Run neural predictions``.
6. Select ``Keypoints`` as input data and select one of the options for
   output of the model’s prediction, which can be ``Neural PCs`` or
   neural activity. CLick on ``Help`` button for more information.
7. The predicted neural activity (\*.npy) file will be saved in the
   selected output folder.

To predict neural activity using SVDs from Facemap:

1. Load or process SVDs for the video. (`see SVD
   tutorial <https://github.com/MouseLand/facemap/blob/main/docs/svd_python_tutorial.md>`__).
2. Follow steps 2-5 above.

Note: a linear model is used for prediction using SVDs.

Predicted neural activity will be plotted in the neural activity window.
Toggle ``Highlight test data`` to highlight time segments not used for
training i.e. test data. Further information about neural prediction,
including variance explained can be found in the saved neural prediction
file (\*.npy).