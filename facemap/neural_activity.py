import numpy as np

from facemap import utils


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
        self.set_neural_data(neural_data_filepath, neural_data_type, data_viz_type)
        self.set_neural_timestamps(
            neural_timestamps_filepath, neural_tstart, neural_tend
        )
        self.set_behavior_timestamps(
            behav_data_timestamps_filepath, behav_tstart, behav_tend
        )

        if self.parent.neural_data_loaded:
            self.parent.plot_neural_data()

        if self.behavior_timestamps is not None and self.neural_timestamps is not None:
            self.behavior_timestamps_resampled = self.resample_behavior_to_neural()
            self.neural_timestamps_resampled = self.resample_neural_to_behavior()
            print("behav_resampled shape: ", self.behavior_timestamps_resampled.shape)
            print("behav resampled: ", self.behavior_timestamps_resampled)
            print("neural_resampled shape: ", self.neural_timestamps_resampled.shape)
            print("neural resampled: ", self.neural_timestamps_resampled)

    def set_neural_data(self, neural_data_filepath, neural_data_type, data_viz_type):
        """
        Set the neural data.
        """
        if isinstance(neural_data_filepath, str) and neural_data_filepath != "":
            self.load_neural_data(neural_data_filepath)
        elif isinstance(neural_data_filepath, np.ndarray):
            self.data = neural_data_filepath  # filepath not passed so use data passed
        else:
            self.data = None
        self.data_type = neural_data_type
        self.data_viz_method = data_viz_type
        if self.data is not None:
            self.num_neurons = self.data.shape[0]

    def set_neural_timestamps(
        self, neural_timestamps_filepath, neural_tstart, neural_tend
    ):
        """
        Set the neural timestamps.
        """
        if (
            isinstance(neural_timestamps_filepath, str)
            and neural_timestamps_filepath != ""
        ):
            self.load_neural_timestamps(neural_timestamps_filepath)
        elif isinstance(neural_timestamps_filepath, np.ndarray):
            self.neural_timestamps = (
                neural_timestamps_filepath  # filepath not passed so use data passed
            )
        else:
            self.neural_timestamps = None
        self.neural_tstart = neural_tstart
        self.neural_tstop = neural_tend

    def set_behavior_timestamps(self, behavior_timestamps, behav_tstart, behav_tend):
        """
        Set the behavior timestamps.
        """
        if isinstance(behavior_timestamps, str) and behavior_timestamps != "":
            self.load_behavior_timestamps(behavior_timestamps)
        elif isinstance(behavior_timestamps, np.ndarray):
            self.behavior_timestamps = (
                behavior_timestamps  # filepath not passed so use data passed
            )
        else:
            self.behavior_timestamps = None
        self.behavior_tstart = behav_tstart
        self.behavior_tstop = behav_tend

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

    def resample_behavior_to_neural(self):
        """
        Resample the behavior timestamps to the neural timestamps.
        Returns
        -------
        resampled_behavior_timestamps : 1D-array
            The resampled behavior timestamps that can be used to get indices of frames in behavioral data that correspond to neural data timestamps.
        """
        return utils.resample_timestamps(
            self.behavior_timestamps, self.neural_timestamps
        )

    def resample_neural_to_behavior(self):
        """
        Resample the neural timestamps to the behavior timestamps.
        """
        return utils.resample_timestamps(
            self.neural_timestamps, self.behavior_timestamps
        )
