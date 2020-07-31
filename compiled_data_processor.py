from typing import Tuple

import tensorflow as tf
import tensorflow_datasets as tfds

from tsne_utils import estimate_high_d_conditional_of_neighbors, high_d_conditional_to_joint


def load_data() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """
    Loads the mnist dataset.
    :return: The mnist dataset. Split into train, val, and test. Along with metadata.
    """
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train[:75%]', 'train[:25%]', 'test'],
        as_supervised=True,
        with_info=True
    )
    return ds_train, ds_val, ds_test, ds_info


def process_data(
        ds_train: tf.data.Dataset,
        ds_val: tf.data.Dataset,
        ds_test: tf.data.Dataset,
        ds_info,
        pretraining: bool,
        batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Processes all portions of the dataset.
    :param ds_train: The training portion of the dataset.
    :param ds_val: The validation portion of the dataset.
    :param ds_test: The test portion of the dataset.
    :param pretraining: If true, processes for autoencoder style training, if false processes for
    tsne training.
    :param batch_size: The batch size to use.
    :return: The processed datasets.
    """
    transformed_datasets = []
    map_fnc = process_examples_for_tsne_training
    if pretraining:
        map_fnc = process_examples_for_pretraining

    for idx, ds in enumerate((ds_train, ds_val, ds_test)):
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(
            map_fnc, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        if idx == 0:
            ds = ds.shuffle(ds_info.splits['train[:75%]'].num_examples)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        transformed_datasets.append(ds)
    return transformed_datasets[0], transformed_datasets[1], transformed_datasets[2]


def process_examples_for_pretraining(image, _):
    """
    Processes examples from a dataset by replacing the label with the features (since we are
    training autoencoder style models). Also normalizes and reshapes the images into the shape we
    expect.
    :param image: The image(s) to process.
    :param _: The labels, which we don't care about.
    :return: Processed image(s), processed image(s)
    """
    image = tf.reshape(image, (tf.shape(image)[0], 784))
    image = tf.cast(image, tf.float32) / 255.
    return image, image


def process_examples_for_tsne_training(image, _):
    """
    Processes examples from a dataset by replacing the label with the tsne joint (since we are
    training tsne). Also normalizes and reshapes the images into the shape we expect.
    :param image: The image(s) to process.
    :param _: The labels, which we don't care about.
    :return: Processed image(s), tsne joint(s)
    """
    image = tf.reshape(image, (tf.shape(image)[0], 784))
    image = tf.cast(image, tf.float32) / 255.
    high_d_conditional = estimate_high_d_conditional_of_neighbors(image)
    return image, high_d_conditional_to_joint(high_d_conditional)


if __name__ == '__main__':
    load_data()
