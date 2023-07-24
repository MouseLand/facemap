GUI
-----

Starting the GUI 
~~~~~~~~~~~~~~~~~~~~~~~

The quickest way to start is to open the GUI from a command line terminal:
::

    python -m facemap


Command line usage
~~~~~~~~~~~~~~~~~~~

Run ``python -m facemap --help`` to see usage of the following optional parameters:

    -h, --help              show this help message and exit
    --ops                   options
    --movie                 Absolute path to video(s)
    --keypoints             Absolute path to keypoints file (\*.h5)
    --proc_npy              Absolute path to proc file (\*_proc.npy)
    --neural_activity       Absolute path to neural activity file (\*.npy)
    --neural_prediction     Absolute path to neural prediction file (\*.npy)
    --tneural               Absolute path to neural timestamps file (\*.npy)
    --tbehavior             Absolute path to behavior timestamps file (\*.npy)
    --savedir               save directory
    --autoload_keypoints    Automatically load keypoints in the same directory as the movie
    --autoload_proc         Automatically load \*_proc.npy in the same directory as the movie


Using the GUI
~~~~~~~~~~~~~~
The GUI can be used for the processing keypoints and SVD of mouse behavioral videos. The GUI can also be used for predicting neural activity using the behavioral data. For more details on each feature, see the following tutorials:

.. toctree:: 
    :maxdepth: 3

    pose_tracking_gui_tutorial
    roi_proc
    neural_activity_prediction_tutorial
