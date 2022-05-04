import numpy as np


class NeuralActivity:
    """
    Neural activity class for neural displaying neural data as a heatmap in the GUI.
    """

    def __init__(
        self,
        parent=None,
        data=None,
        data_type=None,
        data_viz_method=None,
        neural_timestamps=None,
        neural_tstart=None,
        neural_tstop=None,
        behavior_timestamps=None,
        behavior_tstart=None,
        behavior_tstop=None,
    ):
        """
        Initialize the neural activity class.
        """
        self.parent = parent
        self.data = data
        self.data_type = data_type
        self.data_viz_method = data_viz_method
        if self.data is not None:
            self.num_neurons = self.data.shape[0]
        self.neural_timestamps = neural_timestamps
        self.neural_tstart = neural_tstart
        self.neural_tstop = neural_tstop
        self.behavior_timestamps = behavior_timestamps
        self.behavior_tstart = behavior_tstart
        self.behavior_tstop = behavior_tstop

    def get_data(self):
        """
        Return the data.
        """

        return self.data

    def set_data(
        self,
        neural_data_filepath,
        neural_data_type,
        data_viz_type,
        neural_timestamps_filepath,
        neural_tstart,
        neural_tend,
        behav_data_timestamps_filepath,
        behav_tstart,
        behav_tend,
    ):
        """
        Set the data.
        """

        self.load_neural_data(neural_data_filepath)
        self.data_type = neural_data_type
        self.data_viz_method = data_viz_type
        print("neural timestamp file: ", neural_timestamps_filepath)
        # CHeck if string is not empty
        if neural_timestamps_filepath != "":
            self.load_neural_timestamps(neural_timestamps_filepath)
        self.neural_tstart = neural_tstart
        self.neural_tstop = neural_tend
        if behav_data_timestamps_filepath != "":
            self.load_behavior_timestamps(behav_data_timestamps_filepath)
        self.behavior_tstart = behav_tstart
        self.behavior_tstop = behav_tend
        self.num_neurons = self.data.shape[0]
        self.parent.plot_neural_data()

    def load_neural_data(self, file_name):
        """
        Load the neural data from a file.
        """
        if file_name.endswith(".npy") or file_name.endswith(".npz"):
            self.data = np.load(file_name)
        else:
            raise ValueError("File type not recognized.")
        # Check if timestamps file is not loaded then create timestamp array

    def load_neural_timestamps(self, file_name):
        """
        Load the timestamps from a file.
        """
        if file_name.endswith(".npy") or file_name.endswith(".npz"):
            self.neural_timestamps = np.load(file_name)
        else:
            raise ValueError("File type not recognized.")

    def load_behavior_timestamps(self, file_name):
        """
        Load the timestamps from a file.
        """
        if file_name.endswith(".npy") or file_name.endswith(".npz"):
            self.behavior_timestamps = np.load(file_name)
        else:
            raise ValueError("File type not recognized.")
