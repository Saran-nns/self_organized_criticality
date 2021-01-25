import unittest
import pickle
import numpy as np
from compnet import (
    avalanche,
    branching_factor,
    exponent_relationships,
    ising,
    power_law,
    scaling_functions,
)
from compnet import utils

# Getting back the pickled matrices:
with open("sample_matrices.pkl", "rb") as f:
    (
        matrices_dict,
        Exc_activity,
        Inh_activity,
        Rec_activity,
        num_active_connections,
    ) = pickle.load(f)

# Default Test inputs
simulation_inputs = np.random.rand(10, 2)
gym_input = np.random.rand(10, 1)
sequence_input = np.random.rand(10, 1)

# Overriding defaults: Sample input
num_features = 4
time_steps = 1000
inputs = np.random.rand(num_features, time_steps)


class TestCompNet(unittest.TestCase):
    def test_avalanche(self, X: np.array, activity_threshold: int = 1):

        self.assertRaises(
            Exception,
            avalanche.Avalanches().avalanche_observables(
                X, activity_threshold=activity_threshold
            ),
        )
        (
            avalanches,
            avalanche_durations,
            avalanche_sizes,
            iai,
        ) = avalanche.Avalanches().avalanche_observables(
            X, activity_threshold=activity_threshold
        )

        self.assertRaises(Exception, avalanche.Avalanches().plot(avalanches))
        self.assertRaises(
            Exception,
            avalanche.Avalanches().durations_histogram(avalanche_durations, plot=True),
        )
        self.assertRaises(
            Exception, avalanche.Avalanches().size_histogram(avalanche_sizes, plot=True)
        )
        self.assertRaises(
            Exception, avalanche.Avalanches().iai_histogram(iai, plot=True)
        )

    def test_branching_factor(self):
        pass

    def test_exponential_relationships(self):
        pass

    def test_ising(self, X, window_sizes=[2, 5]):
        """Test avalanche module

        Args:
            X (np.array): Neural activity
            
            window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [2, 5].
        """
        self.assertRaises(
            Exception, ising.IsingModel().compute_spin(X, window_sizes=window_sizes)
        )
        self.assertRaises(
            Exception,
            ising.IsingModel().mean_spiking_activity(X, window_sizes=window_sizes),
        )
        self.assertRaises(
            Exception, ising.IsingModel().compute_Q(X, window_sizes=window_sizes)
        )

        Q, _ = ising.IsingModel().compute_Q(X, window_sizes=window_sizes)

        self.assertRaises(
            Exception, ising.IsingModel().compute_Q(X, window_sizes=window_sizes)
        )
        self.assertRaises(Exception, ising.IsingModel().plotQ(Q))

    def test_powerlaw(self, X: np.array, activity_threshold: float = 1.0):

        (_, _, avalanche_sizes, _,) = avalanche.Avalanches().avalanche_observables(
            X, activity_threshold=activity_threshold
        )

        avalanche_size_hist = avalanche.Avalanches().size_histogram(avalanche_sizes)

        xmin = 1
        # Avalanche histogram
        x = avalanche_size_hist[1]
        a, b = 1.01, 10

        # Exponent estimate
        self.assertRaises(Exception, power_law.PowerLaw().exponent(x, xmin, a, b))

        alpha_hat = power_law.PowerLaw().exponent(x, xmin, a, b)

        # Confidence interval of the above estimate, assuming the network shows at n, the avalanche behavoir kicks in
        self.assertRaises(
            Exception,
            power_law.PowerLaw().confidence_interval(n=1.0, alpha_hat=alpha_hat),
        )

    def test_scaling_functions(self):
        pass

    def test_utils(self,):

        self.assertRaises(Exception, utils.compute_spike_count(spike_train))


if __name__ == "__main__":
    unittest.main()

