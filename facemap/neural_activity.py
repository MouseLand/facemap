import numpy as np
import matplotlib.pyplot as plt


class NeuralActivity:
    """
    Neural activity class for neural displaying neural data as a heatmap in the GUI.
    """

    def __init__(self, data, time, label):
        """
        Initialize the neural activity class.
        """
        self.data = data
        self.num_neurons = data.shape[0]
        self.neuron_id = np.arange(self.num_neurons)

    def get_data(self):
        """
        Return the data.
        """

        return self.data

    def set_data(self, data):
        """
        Set the data.
        """

        self.data = data
        self.set_num_neurons(data.shape[0])

    def get_num_neurons(self):
        """
        Return the number of neurons.
        """

        return self.num_neurons

    def set_num_neurons(self, num_neurons):
        """
        Set the number of neurons.
        """

        self.num_neurons = num_neurons
        self.neuron_id = np.arange(self.num_neurons)

    def get_neuron_id(self):
        """
        Return the neuron id.
        """

        return self.neuron_id

    def load_neural_data(self, file_name):
        """
        Load the neural data from a file.
        """

        self.data = np.load(file_name)
