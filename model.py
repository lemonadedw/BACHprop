import tensorflow as tf
from tensorflow.keras import layers, models

def create_bidirectional_model(input_shape=(50, 4)):
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

def create_unidirectional_prediction_model(input_shape=(50, 4)):
    """
    Multi-task LSTM model that predicts:
    1. Left/Right hand for current note (output[0])
    2. Next note's pitch, 0-1 normalized (output[1])
    
    Uses Sequential with Dense(2) output layer.
    """
    model = models.Sequential()
    
    # LSTM backbone
    model.add(layers.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.3))
    
    # Output layer with 2 units:
    # output[:, :, 0] = hand classification (0=right, 1=left)
    # output[:, :, 1] = next pitch prediction (0-1 normalized)
    model.add(layers.TimeDistributed(layers.Dense(2, activation='sigmoid')))
    
    return model


def create_model(input_shape=(50, 4)):
    """
    Create a model based on the task type.
    
    Args:
        input_shape: Shape of input sequences (seq_length, features)
        multi_task: If True, returns multi-task model. If False, returns simple hand classification model.
    
    Returns:
        Keras model for the specified task
    """
    return create_bidirectional_model(input_shape)

if __name__ == "__main__":
    model = create_model()["model"]
    model.summary()
