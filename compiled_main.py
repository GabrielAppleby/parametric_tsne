import tensorflow as tf

from compiled_data_processor import load_data, process_data
from parametric_tsne import ParametricTSNE


def train_and_save_model():
    """
    Trains the model and saves it to disk.
    :return: None.
    """
    ds_train, ds_val, ds_test = load_data()
    ds_train_pretrain, ds_val_pretrain, _ = process_data(
        ds_train, ds_val, ds_test, pretraining=True)

    ptsne = ParametricTSNE()

    pretrainer = ptsne.pretrainer_model()

    pretrainer.fit(ds_train_pretrain,
                   epochs=300,
                   validation_data=ds_val_pretrain,
                   callbacks=[tf.keras.callbacks.EarlyStopping(
                       monitor='val_loss',
                       patience=5,
                       restore_best_weights=True)])

    ds_train_tsne, ds_val_tsne, _ = process_data(
        ds_train, ds_val, ds_test, pretraining=False)

    actual_model = ptsne.tsne_model()

    actual_model.fit(ds_train_tsne,
                     epochs=300,
                     validation_data=ds_val_tsne,
                     callbacks=[tf.keras.callbacks.EarlyStopping(
                         monitor='val_loss',
                         patience=10,
                         restore_best_weights=True)])


def main():
    train_and_save_model()


if __name__ == '__main__':
    main()
