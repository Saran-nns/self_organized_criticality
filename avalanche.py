import numpy as np
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter
from utils import compute_spike_count


class IsingModel(object):

    """Ising model of the reservoir pool acitivity at various time scales"""

    def __init__(self):
        pass

    @property
    def running_mean(self, X: np.array, window_size: int):

        """Measure the moving average of neural firing rate given the time window

        Args:
            X (np.array): Neural activity
            
            window_size (int): Time window size

        Returns:
            list: Moving average with the size equals len(X)//window_size
        """

        cumsum = np.cumsum(np.insert(X, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    def compute_spin(self, X: np.array, window_sizes: list = [1]):

        """Compute Sigma {-1,1}^N of neural population activity vector, given window sizes for the mean

        Args:
            X (np.array): Neural activity
            
            window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [1].

        Returns:
            dict : Neural spins at different time scales
        """

        # Placeholder for sigma of neural population
        sigma = {}
        sigma.fromkeys(window_sizes)
        for window in window_sizes:
            sigma_ = []
            for neuron in list(range(X.shape[1])):
                mean = self.running_mean(X[:, neuron], window)
                mean[mean > 0] = 1
                mean[mean <= 0] = -1
                sigma_.append(list(mean))
            sigma["%s" % window] = np.asarray(
                sigma_
            ).transpose()  # Rows= time dim, col= neurons
        return sigma

    def mean_spiking_activity(self, X: np.array, window_sizes: list = [1]):

        """Compute mean spiking probability of each neuron given their spins at different time scales

        Args:
            X (np.array): Neural activity
            
            window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [1].

        Returns:
            dict : Mean spiking probability of each neuron given their spins at different time scales
        """

        # Compute spins with respect to window size
        sigma = self.compute_spin(X, window_sizes=window_sizes)
        # Compute mean of each neuron's spin variance
        m = {}
        m.fromkeys(window_sizes)
        for window in window_sizes:
            m["%s" % window] = np.mean(sigma["%s" % window], axis=1)
        return m

    def compute_Q(self, X: np.array, window_sizes: list = [1]):

        """Compute two point function between pairs of neurons in the network

        Args:
            X (np.array): Neural activity
            
            window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [1].

        Returns:
            [type]: [description]
        """
        num_neurons = X.shape[1]
        # Compute two-point function of spins with respect to window size
        sigma = self.compute_spin(X, window_sizes=window_sizes)

        # Two point function between neurons given their spin
        Q = {}
        Q.fromkeys(window_sizes)
        q = np.zeros((num_neurons, num_neurons))
        for window in window_sizes:
            for i in list(range(1, num_neurons)):
                for j in range(i):
                    q[i][j] = np.mean(
                        sigma["%s" % window][:, i] * sigma["%s" % window][:, j]
                    )
            q_ = q + q.T  # Is symmetric
            np.fill_diagonal(q_, 1)
            Q["%s" % window] = q_
        return Q, sigma

    def plotQ(self, Q: dict):

        """Plot Q, the two point function values between pairs of neurons in the network
        Args:
            Q (dict): Two point function between pairs of neurons in the network
        
        Returns:
            plot
        """

        # Create subplots
        ax_size = len(Q.keys())
        assert ax_size != 0, "Missing time window"

        if ax_size != 1:
            # Check is ax_size even or odd
            if ax_size % 2 != 0:
                ax_size += 1
            rows, cols = ax_size // 2, ax_size // 2

            fig, axs = plt.subplots(rows, cols)
            fig.suptitle("Horizontally stacked subplots")
            for i in range(len(axs)):
                axs[i].imshow(Q.values[i])
                axs[i].title(f"Q at time scale {Q.keys()[i]}")
                axs[i].tight_layout()
                axs[i].colorbar(fraction=0.046, pad=0.04)

        else:

            plt.imshow(Q.values()[0])
            plt.title(f"Q at time scale {Q.keys()[0]}")
            plt.tight_layout()
            plt.colorbar(fraction=0.046, pad=0.04)

        return plt.show()

    def avalanche_observables(self, X: np.array, activity_threshold: int = 1):
        """Avalanche sizes, durations and interval sizes

        - Set the neural activity =0 if < activity_threshold % of size of the network
        - Slice the array by non-zero value indices
        - Count the number of items in each slices: Duration of avalanches
        - Sum the items in each slices: Size of avalanches
        - Slice the array into zero value indices
        - Count number of items in each slices: Duration of inter avalanche intervals
        
        Args:
            X (np.array): Neural activity
            activity_threshold (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """

        spike_count = np.asarray(compute_spike_count(X))
        threshold = activity_threshold * X.shape[1] / 100
        spike_count[spike_count < threshold] = 0

        # Avalanche size and duration

        # Get the non zero indices
        aval_idx = np.nonzero(spike_count)[0]

        # Group indices by a consecutiveness
        aval_indices = []
        for k, g in itertools.groupby(enumerate(aval_idx), lambda ix: ix[0] - ix[1]):
            aval_indices.append(list(map(itemgetter(1), g)))

        # Using group indices, pick the correpondning items in the spike_count list
        avalanches = []
        for val in aval_indices:
            avalanches.append(list(spike_count[val]))

        # Avalanche sizes
        avalanche_sizes = [sum(avalanche) for avalanche in avalanches]
        # Avalanche duration
        avalanche_durations = [len(avalanche) for avalanche in avalanches]

        # Inter avalanche intervals

        # Get the indices where spike count =0
        silent_idx = np.where(spike_count == 0)[0]

        silent_indices = []
        # Group indices by consecutiveness
        for k, g in itertools.groupby(enumerate(silent_idx), lambda ix: ix[0] - ix[1]):
            silent_indices.append(list(map(itemgetter(1), g)))
        iai_ = []
        for val in silent_indices:
            iai_.append(list(spike_count[val]))
        # Duration of inter-avalanche intervals
        iai = [len(intervals) for intervals in iai_]

        return spike_count, avalanche_durations, avalanche_sizes, iai


def test_avalanche(X, window_sizes=[1, 2, 5, 10, 20]):
    """Test avalanche module

    Args:
        X (np.array): Neural activity
        window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [1].
    """
    # Compute moving average and spin
    sigma = IsingModel().compute_spin(X, window_sizes=window_sizes)
    m = IsingModel().mean_spiking_activity(X, window_sizes=window_sizes)
    Q, sigma = IsingModel().compute_Q(X, window_sizes=window_sizes)

