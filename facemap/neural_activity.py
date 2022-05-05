import numpy as np


class NeuralActivity:
    """
    Neural activity class for storing and visualizing neural activity data.
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
        self.set_data(
            data,
            data_type,
            data_viz_method,
            neural_timestamps,
            neural_tstart,
            neural_tstop,
            behavior_timestamps,
            behavior_tstart,
            behavior_tstop,
        )

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
        # Check if neural_data_filepath is not None and is string
        if isinstance(neural_data_filepath, str) and neural_data_filepath != "":
            self.load_neural_data(neural_data_filepath)
        else:
            self.data = neural_data_filepath  # filepath not passed so use data passed
        self.data_type = neural_data_type
        self.data_viz_method = data_viz_type
        # Check if string is not empty
        if (
            isinstance(neural_timestamps_filepath, str)
            and neural_timestamps_filepath != ""
        ):
            self.load_neural_timestamps(neural_timestamps_filepath)
        else:
            self.neural_timestamps = (
                neural_timestamps_filepath  # filepath not passed so use data passed
            )
        self.neural_tstart = neural_tstart
        self.neural_tstop = neural_tend
        if (
            isinstance(behav_data_timestamps_filepath, str)
            and behav_data_timestamps_filepath != ""
        ):
            self.load_behavior_timestamps(behav_data_timestamps_filepath)
        else:
            self.behavior_timestamps = (
                behav_data_timestamps_filepath  # filepath not passed so use data passed
            )
        self.behavior_tstart = behav_tstart
        self.behavior_tstop = behav_tend
        if self.data is not None:
            self.num_neurons = self.data.shape[0]
        if self.parent.neural_data_loaded:
            self.parent.plot_neural_data()

    def load_neural_data(self, file_name):
        """
        Load the neural data from a file.
        """
        if file_name.endswith(".npy") or file_name.endswith(".npz"):
            self.data = np.load(file_name)
            self.parent.neural_data_loaded = True
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
