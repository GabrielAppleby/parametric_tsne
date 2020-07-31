import tensorflow as tf
import tensorflow_datasets as tfds

from compiled_data_processor import load_data, process_data
from parametric_tsne import ParametricTSNE
from vis_utils import show_images, show_scatter


def train_and_save_model():
    """
    Trains the model and saves it to disk.
    :return: None.
    """
    ds_train, ds_val, ds_test, ds_info = load_data()
    ds_train_pretrain, ds_val_pretrain, ds_test_pretrain = process_data(
        ds_train, ds_val, ds_test, ds_info, pretraining=True)

    ptsne = ParametricTSNE()

    pretrainer = ptsne.pretrainer_model()

    pretrainer.fit(ds_train_pretrain,
                   epochs=30,
                   validation_data=ds_val_pretrain,
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       monitor='val_loss',
                       patience=5,
                       restore_best_weights=True)])

    images = pretrainer.predict(ds_test_pretrain)


    ds_train_tsne, ds_val_tsne, ds_test_tsne = process_data(
        ds_train, ds_val, ds_test, ds_info, pretraining=False)

    actual_model = ptsne.tsne_model()

    labels = list(map(lambda x: x[1], tfds.as_numpy(ds_test)))[0:300]
    tsne_coords = actual_model.predict(ds_test_tsne)
    show_scatter(tsne_coords[0:300], labels, "tsne_scatter_no_fine")

    actual_model.fit(ds_train_tsne,
                     epochs=30,
                     validation_data=ds_val_tsne,
                     callbacks=[tf.keras.callbacks.EarlyStopping(
                         monitor='val_loss',
                         patience=5,
                         restore_best_weights=True)])

    tsne_coords = actual_model.predict(ds_test_tsne)

    show_scatter(tsne_coords[0:300], labels, "tsne_scatter")
    show_images(images[0:15], "blah")


def main():
    train_and_save_model()


if __name__ == '__main__':
    main()
