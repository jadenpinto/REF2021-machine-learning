import pytest
import collections
import random
import numpy as np

from machine_learning import size_constrained_clustering

class Test_Deterministic_Annealing:

    def test_size_constrained_clustering_input_validation(self):
        # Throw error if distribution does not sum to 1 (100%)
        with pytest.raises(AssertionError):
            size_constrained_clustering.DeterministicAnnealing(n_clusters=2, distribution=[0.8, 0.3])

        # Throw error if more distributions are specified than the number of clusters
        with pytest.raises(AssertionError):
            size_constrained_clustering.DeterministicAnnealing(n_clusters=1, distribution=[0.5, 0.5])

        # Throw error if the number of clusters exceeds specified distributed
        with pytest.raises(AssertionError):
            size_constrained_clustering.DeterministicAnnealing(n_clusters=4, distribution=[0.33, 0.33, 0.33])

        # If the temperature is specified, assert an error is throw if it is not a list
        with pytest.raises(AssertionError):
            size_constrained_clustering.DeterministicAnnealing(n_clusters=2, distribution=[0.4, 0.6], T=0.75)

        # Assert an error is thrown if less than oen cluster is specified
        with pytest.raises(AssertionError):
            size_constrained_clustering.DeterministicAnnealing(n_clusters=0, distribution=[])

    def test_size_constrained_clustering_results(self):
        # Specify the number of data points
        n_samples = 500
        # Define a random state for Reproducibility
        random_state = 10

        random.seed(random_state)
        np.random.seed(random_state)

        # Create random data points for the training data used to fit the model
        X = np.random.rand(n_samples, 2)

        # Five clusters with equal distribution of data points
        n_clusters = 5
        distribution = [0.20] * n_clusters

        # Training the model using data points and specified cluster configuration
        model = size_constrained_clustering.DeterministicAnnealing(n_clusters, distribution)
        model.fit(X)

        # Obtain predictions
        labels = model.labels_
        label_counter = collections.Counter(labels) # How many data points are assigned to each cluster
        label_dist = list(label_counter.values())
        label_dist = [d / np.sum(label_dist) for d in label_dist] # Obtain the proportion of data points across clusters

        # Assert the specified distribution of cluster assignments matches the actual distribution of data points
        # Include a very small error margin - 1e-6
        assert np.sum(np.array(label_dist) - np.array(distribution)) <= 1e-6
