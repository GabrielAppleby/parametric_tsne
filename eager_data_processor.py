from typing import Tuple

import numpy as np
from sklearn.datasets import fetch_openml

DATA_NAME: str = "mnist.npz"


def fetch_mnist() -> Tuple[np.array, np.array]:
    """
    Fetches the mnist dataset from another source.
    :return: The features and labels matrices of the dataset.
    """
    features, labels = fetch_openml("mnist_784", version=1, return_X_y=True)
    return features, labels


def standardize_datatypes(features: np.array, labels: np.array) -> Tuple[np.array, np.array]:
    """
    Make sure all datasets end up with the same datatypes.
    :param features: The features of the dataset.
    :param labels: The labels of the dataset. Can be none.
    :return: The features, labels tuple of the dataset.
    """
    features = features.astype(np.float32)
    if labels is not None:
        labels = labels.astype(np.float32)
    return features, labels


def normalize_data(data: np.array) -> np.array:
    """
    Normalizes the data between 0 and 1.
    :param data: The data to normalize.
    :return: The normalized data
    """
    min = np.min(data)
    data = data - min
    max = np.max(data)
    data = data / max
    return data


def sample_data(features: np.array, labels: np.array, num_instances: int) -> \
        Tuple[np.array, np.array]:
    """
    Grabs a sample from some features and labels.
    :param features: The features matrix to sample from.
    :param labels: The labels matrix to sample from.
    :param num_instances: The number of instances to sample.
    :return: The sample of features and labels.
    """
    idxs = np.random.choice(features.shape[0], num_instances, replace=False)
    features = features[idxs]
    labels = labels[idxs]
    return features, labels


def main():
    np.random.seed(42)
    features, labels = fetch_mnist()
    features, labels = sample_data(features, labels, num_instances=20000)
    features = normalize_data(features)
    features, labels = standardize_datatypes(features, labels)
    np.savez(DATA_NAME,
             features=features,
             labels=labels)


if __name__ == '__main__':
    main()
