import tensorflow as tf
from tensorflow.keras import layers, models


def create_model(input_shape=(50, 4)):
    model = models.Sequential()

    # Bidirectional LSTM
    # return_sequences=True because we want a prediction for every note in the sequence
    model.add(layers.Bidirectional(layers.LSTM(
        64, return_sequences=True), input_shape=input_shape))
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))

    # Dense layer for classification
    # TimeDistributed applies the Dense layer to every temporal slice of the input
    model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))

    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()
