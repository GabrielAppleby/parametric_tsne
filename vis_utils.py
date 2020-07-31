import matplotlib.pyplot as plt
import numpy as np


def show_scatter(projections: np.array, labels, name: str) -> None:
    """
    Plots a projection scatter plot.
    :param projections: The projections to plot.
    :param labels: The labels to color by.
    :param name: The name to save it under.
    :return: None, the images are saved as a side effect.
    """
    plt.scatter(x=projections[:, 0], y=projections[:, 1], c=labels)
    plt.savefig(name)
    plt.clf()


def show_images(images: np.array, name: str) -> None:
    """
    Plots an MNIST image.
    :param images: An MNIST image.
    :param name: The name to save it under.
    :return: None, the images are saved as a side effect.
    """
    plt.gray()
    fig = plt.figure(figsize=(16, 7))
    for i in range(0, 15):
        ax = fig.add_subplot(3, 5, i + 1)
        ax.matshow(images[i].reshape((28, 28)).astype(float))
    plt.savefig(name)
    plt.clf()
