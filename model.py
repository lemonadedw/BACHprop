import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable(package="model", name="PositionalEncoding")
class PositionalEncoding(layers.Layer):
    """
    Custom layer for positional encoding using learned embeddings.
    This avoids Lambda layers with Python lambdas for safe model serialization.
    """
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.pos_emb = layers.Embedding(seq_len, embed_dim)
        
    def call(self, inputs):
        # Create positions [0, 1, 2, ..., seq_len-1]
        positions = tf.range(self.seq_len, dtype=tf.int32)
        # Expand to [1, seq_len] for broadcasting
        positions = tf.expand_dims(positions, 0)
        # Get positional embeddings: [1, seq_len, embed_dim]
        pos_encoding = self.pos_emb(positions)
        # Add positional encoding to inputs (broadcasts across batch dimension)
        return inputs + pos_encoding
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'embed_dim': self.embed_dim
        })
        return config


def create_bilstm_model(input_shape=(50, 4)):
    model = models.Sequential()

    # Bidirectional LSTM - Layer 1 (High Capacity)
    # return_sequences=True because we want a prediction for every note in the sequence
    model.add(layers.Bidirectional(layers.LSTM(
        128, return_sequences=True), input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))  # Regularization

    # Bidirectional LSTM - Layer 2 (Compression)
    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))  # Regularization

    # Dense layer for classification
    # TimeDistributed applies the Dense layer to every temporal slice of the input
    model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))

    return model

def create_lstm_model(input_shape=(50, 4)):
    model = models.Sequential()
    model.add(layers.LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def create_attention_model(input_shape=(50, 4)):
    """
    Create a pure attention-based model for piano hand prediction.
    Uses Transformer encoder blocks without LSTM layers.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Project input to higher dimension for attention
    x = layers.Dense(128)(inputs)
    
    # Positional encoding (learned embeddings)
    seq_len = input_shape[0]
    x = PositionalEncoding(seq_len, 128)(x)
    
    # Transformer encoder blocks (attention + feed-forward)
    num_blocks = 3
    
    for _ in range(num_blocks):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.2
        )(x, x)  # Self-attention: query=key=value=x
        
        # Residual connection and layer normalization
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed-forward network
        ff = layers.Dense(256, activation='relu')(x)
        ff = layers.Dropout(0.3)(ff)
        ff = layers.Dense(128)(ff)
        ff = layers.Dropout(0.3)(ff)
        
        # Residual connection and layer normalization
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)
    
    # Final classification layer
    output = layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))(x)
    
    model = models.Model(inputs=inputs, outputs=output)
    return model

def create_mlp_model(input_shape=(50, 4)):
    model = models.Sequential()
    model.add(layers.Dense(128, input_shape=input_shape, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def create_cnn_model(input_shape=(50, 4)):
    model = models.Sequential()
    # Use padding='same' to preserve sequence length
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.1))
    # Use TimeDistributed to apply Dense layer to each timestep
    model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))
    return model

def create_model(input_shape=(50, 4)):
    return create_bilstm_model(input_shape)

if __name__ == "__main__":
    model = create_model()
    model.summary()
