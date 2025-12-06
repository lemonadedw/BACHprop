import tensorflow as tf
from tensorflow.keras import layers, models

def create_bidirectional_model(input_shape=(50, 4)):
    model = models.Sequential()

    # Bidirectional LSTM - Layer 1 (High Capacity)
    # return_sequences=True because we want a prediction for every note in the sequence
    model.add(layers.Bidirectional(layers.LSTM(
        128, return_sequences=True), input_shape=input_shape))
    model.add(layers.Dropout(0.3))  # Regularization

    # Bidirectional LSTM - Layer 2 (Compression)
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.Dropout(0.3))  # Regularization

    # Dense layer for classification
    # TimeDistributed applies the Dense layer to every temporal slice of the input
    model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))

    return model


def create_unidirectional_model(input_shape=(50, 4)):
    """
    Simple LSTM model for left/right hand prediction.
    Uses unidirectional LSTM (not bidirectional) to avoid looking at future notes.
    """
    model = models.Sequential()

    # Unidirectional LSTM layers
    # return_sequences=True because we want a prediction for every note in the sequence
    model.add(layers.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.3))

    # Dense layer for left/right hand classification
    # TimeDistributed applies the Dense layer to every temporal slice of the input
    # Output: 0 = right hand, 1 = left hand
    model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))

    return model

def create_model(input_shape=(50, 4), bidirectional=False):
    """
    Create a hand classification model.
    
    Args:
        input_shape: Shape of input sequences (seq_length, features)
        bidirectional: If True, uses bidirectional LSTM. If False, uses unidirectional LSTM.
    
    Returns:
        Keras model for hand classification
    """
    if bidirectional:
        return create_bidirectional_model(input_shape)
    else:
        return create_unidirectional_model(input_shape)

if __name__ == "__main__":
    print("Unidirectional LSTM model:")
    print("="*60)
    model_uni = create_model(bidirectional=False)
    model_uni.summary()
    
    print("\n" + "="*60)
    print("Bidirectional LSTM model:")
    print("="*60)
    model_bi = create_model(bidirectional=True)
    model_bi.summary()
