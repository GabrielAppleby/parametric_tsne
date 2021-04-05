from typing import Tuple, Union

import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.model_selection import train_test_split

from eager_data_processor import DATA_NAME
from restricted_boltzmann_machine import RBM
from tsne_utils import estimate_low_d_joint_of_neighbors, estimate_high_d_conditional_of_neighbors, \
    high_d_conditional_to_joint
from vis_utils import show_scatter, show_images

MODEL_NAME: str = "ptsne"


def load_data() -> Tuple[np.array, np.array]:
    """
    Load the data from disk.
    :return: The features, and labels of the data.
    """
    data = np.load(DATA_NAME)
    return data['features'][0:600], data['labels'][0:600]


def train_val_test_split(features: np.array, labels: Union[np.array, None]) \
        -> Tuple:
    """
    Gets a training set, validation set, and test set given features, embeddings, and labels
    matrices.
    :param features: The features matrix.
    :param embeddings: The embeddings matrix.
    :param labels: The labels matrix. Can be None.
    :return: The train, val, and test splits of the three matrices.
    """
    if labels is None:
        labels_train, labels_val, labels_test = None, None, None
        features_train, features_both, = \
            train_test_split(features, test_size=0.5, shuffle=True, random_state=42)
        features_val, features_test = \
            train_test_split(features_both,
                             test_size=0.5,
                             shuffle=True,
                             random_state=42)
    else:
        features_train, features_both, labels_train, labels_both = \
            train_test_split(features, labels, test_size=0.5, shuffle=True,
                             random_state=42)
        features_val, features_test, labels_val, labels_test = \
            train_test_split(features_both,
                             labels_both,
                             test_size=0.5,
                             shuffle=True,
                             random_state=42)

    return features_train, \
           features_val, \
           features_test, \
           labels_train, \
           labels_val, \
           labels_test


def train_and_save_model():
    """
    Trains the model and saves it to disk.
    :return: None.
    """
    # features, labels = load_data()
    # _, instance_shp = features.shape
    #
    # features_train, \
    # features_val, \
    # _, \
    # labels_train, \
    # labels_val, \
    # _ = train_val_test_split(features, labels)

    # tf.keras.backend.set_floatx('float64')
    train = sio.loadmat('mnist_train.mat')
    features_train = train['train_X'].astype(np.float32)
    labels_train = train['train_labels'].astype(np.float32)
    test = sio.loadmat('mnist_test.mat')
    features_test = test['test_X'].astype(np.float32)
    labels_test = test['test_labels'].astype(np.float32)


    n = features_train.shape[0]
    model_sizes = [(784, 500), (500, 500), (500, 2000), (2000, 2)]
    rbms = []
    trainable_weights = []
    rbm_train = features_train
    for i, units in enumerate(model_sizes):
        print("Training model: " + str(units))
        visible_units, hidden_units = units
        linear = False
        if i == len(model_sizes) - 1:
            linear = True
        rbm = RBM(visible_units, hidden_units, linear=linear)
        rbm = rbm.fit(rbm_train)
        rbm_train = rbm.transform(rbm_train)
        rbms.append(rbm)
        trainable_weights.append(rbm.get_w())
        trainable_weights.append(rbm.get_hidden_bias())

    images = []
    for example in rbm_train[0:15]:
        next = tf.reshape(example, (1, 2))
        for rbm in reversed(rbms):
            next = rbm.transform_backwards(next)
        images.append(next)
    np.set_printoptions(precision=6, suppress=True)
    high_d_joints = []
    training_data = []
    inds = np.random.permutation(n)
    for start_idx in range(0, n, 5000):
        print("Getting h d for: " + str(start_idx))
        end_idx = start_idx + 5000
        if end_idx > n:
            print("gotcha")
            exit()
        batch = features_train[inds[start_idx:end_idx]]
        high_d_conditional_of_neighbors = estimate_high_d_conditional_of_neighbors(batch)
        high_d_joint_of_neighbors = high_d_conditional_to_joint(high_d_conditional_of_neighbors)
        high_d_joints.append(high_d_joint_of_neighbors)
        training_data.append(batch)
    optimizer = tf.keras.optimizers.Adam()
    kld_loss = tf.keras.losses.KLDivergence()

    def train():
        for epoch in range(2000):
            for start_idx in range(0, n, 5000):
                end_idx = start_idx + 5000
                if end_idx > n:
                    print("gotcha")
                    exit()
                batch = training_data[start_idx // 5000]
                with tf.GradientTape(persistent=True) as tape:
                    o1 = rbms[0].transform(batch)
                    o2 = rbms[1].transform(o1)
                    o3 = rbms[2].transform(o2)
                    low_d_representation = rbms[3].transform(o3)
                    low_d_joint_of_neighbors = estimate_low_d_joint_of_neighbors(low_d_representation)
                    tsne_loss = kld_loss(high_d_joints[start_idx // 5000], low_d_joint_of_neighbors)
                grads = tape.gradient([tsne_loss], trainable_weights)
                optimizer.apply_gradients(zip(grads, trainable_weights))
            if epoch % 100 == 0:
                print(epoch)
                print(tsne_loss)

    train()

    low_d_reps = []
    for start_idx in range(0, features_test.shape[0], 5000):
        end_idx = start_idx + 5000
        if end_idx > n:
            print("gotcha")
            exit()
        batch = features_test[start_idx:end_idx]
        o1 = rbms[0].transform(batch)
        o2 = rbms[1].transform(o1)
        o3 = rbms[2].transform(o2)
        low_d_representation = rbms[3].transform(o3)
        low_d_reps.append(low_d_representation)

    scat = np.concatenate(low_d_reps, axis=0)
    show_scatter(scat, labels_test, "tsne_scatter_real")


def main():
    train_and_save_model()


if __name__ == '__main__':
    main()
