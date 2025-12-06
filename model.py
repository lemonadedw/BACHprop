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

def create_unidirectional_prediction_model(input_shape=(50, 4)):
    """
    Multi-task LSTM model that predicts:
    1. Left/Right hand for current note (output[0])
    2. Next note's pitch as one-hot over 88 piano keys (output[1:89])
    
    Uses Sequential with Dense(89) output layer.
    Output shape: (batch, seq, 89)
    - output[:, :, 0] = hand classification (sigmoid: 0=right, 1=left)
    - output[:, :, 1:89] = next pitch one-hot (softmax over 88 keys)
    """
    model = models.Sequential()
    
    # LSTM backbone
    model.add(layers.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.3))
    
    # Output layer with 89 units:
    # output[:, :, 0] = hand classification (sigmoid)
    # output[:, :, 1:89] = next pitch one-hot (will apply softmax separately)
    # Note: Using linear activation here, will apply sigmoid/softmax in custom activation
    model.add(layers.TimeDistributed(layers.Dense(89)))
    
    return model

def create_transformer_prediction_model(input_shape=(50, 4), 
                                       num_heads=4, 
                                       ff_dim=128, 
                                       num_transformer_blocks=2,
                                       dropout_rate=0.3):
    """
    Transformer-based multi-task model that predicts:
    1. Left/Right hand for current note (output[0])
    2. Next note's pitch as one-hot over 88 piano keys (output[1:89])
    
    Uses multi-head self-attention instead of LSTM.
    
    Args:
        input_shape: Shape of input sequences (seq_length, features)
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        num_transformer_blocks: Number of transformer blocks to stack
        dropout_rate: Dropout rate for regularization
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Initial projection to embedding dimension
    embedding_dim = 128
    x = layers.Dense(embedding_dim)(x)
    
    # Positional encoding (simple learned encoding)
    seq_len = input_shape[0]
    pos_encoding = layers.Embedding(input_dim=seq_len, output_dim=embedding_dim)(
        tf.range(start=0, limit=seq_len, delta=1)
    )
    x = x + pos_encoding
    
    # Stack transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head self-attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate
        )(x, x)
        attn_output = layers.Dropout(dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward network
        ffn = layers.Dense(ff_dim, activation='relu')(x)
        ffn = layers.Dropout(dropout_rate)(ffn)
        ffn = layers.Dense(embedding_dim)(ffn)
        ffn = layers.Dropout(dropout_rate)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    
    # Output layer with 89 units:
    # output[:, :, 0] = hand classification (sigmoid)
    # output[:, :, 1:89] = next pitch one-hot (softmax)
    # Using linear activation, will apply sigmoid/softmax separately
    outputs = layers.Dense(89)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def create_model(input_shape=(50, 4), multitask=False, use_transformer=False):
    """
    Create a model based on the task type.
    
    Args:
        input_shape: Shape of input sequences (seq_length, features)
        multitask: If True, returns multi-task model. If False, returns simple hand classification model.
        use_transformer: If True, uses transformer architecture instead of LSTM.
    
    Returns:
        Keras model for the specified task
    """
    if multitask:
        if use_transformer:
            return create_transformer_prediction_model(input_shape)
        else:
            return create_unidirectional_prediction_model(input_shape)
    else:
        return create_unidirectional_model(input_shape)

if __name__ == "__main__":
    print("Simple hand classification model (LSTM):")
    model = create_model(multitask=False)
    model.summary()
    
    print("\n" + "="*60)
    print("Multi-task model - LSTM (hand + next pitch):")
    print("="*60)
    model_mt = create_model(multitask=True, use_transformer=False)
    model_mt.summary()
    
    print("\n" + "="*60)
    print("Multi-task model - Transformer (hand + next pitch):")
    print("="*60)
    model_transformer = create_model(multitask=True, use_transformer=True)
    model_transformer.summary()
