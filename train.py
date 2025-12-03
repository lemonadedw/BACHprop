import tensorflow as tf
from data_loader import PIGDataLoader
from model import create_model
import os

def train():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    SEQUENCE_LENGTH = 50
    DATA_DIR = "PianoFingeringDataset_v1.2/FingeringFiles"
    
    # Load Data
    loader = PIGDataLoader(DATA_DIR, sequence_length=SEQUENCE_LENGTH)
    X, y = loader.get_data()
    
    # Split into train and val
    # We can use sklearn or just manual split. Let's do manual for simplicity to avoid extra dependency if possible, 
    # but sklearn is standard. Let's just slice.
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape: {X_val.shape}, {y_val.shape}")
    
    # Create Model
    model = create_model(input_shape=(SEQUENCE_LENGTH, 4))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor='val_accuracy'
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb]
    )
    
    print("Training complete.")

if __name__ == "__main__":
    train()
