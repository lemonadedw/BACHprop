import tensorflow as tf
from data_loader import PIGDataLoader
from model import create_model
from data_augmentation import apply_augmentations
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Set seeds for reproducibility
SEED = 190273
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def train():
    # Hyperparameters
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 16
    SEQUENCE_LENGTH = 50
    DATA_DIRS = [
        "PianoFingeringDataset_v1.2/FingeringFiles",
        "Musescore/FingeringFiles"
    ]
    AUGMENTATION = True
    if AUGMENTATION:
        print("Data augmentation enabled.")
    
    # Load Data from all directories
    all_X = []
    all_y = []
    
    for data_dir in DATA_DIRS:
        if os.path.exists(data_dir):
            print(f"Loading data from {data_dir}...")
            loader = PIGDataLoader(data_dir, sequence_length=SEQUENCE_LENGTH)
            X, y = loader.get_data()
            all_X.append(X)
            all_y.append(y)
        else:
            print(f"Warning: {data_dir} does not exist, skipping...")
    
    # Combine data from all directories
    if not all_X:
        raise ValueError("No data directories found!")
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"Total combined data: {X.shape}, {y.shape}")
    
    # Split into train and val
    # We can use sklearn or just manual split. Let's do manual for simplicity to avoid extra dependency if possible, 
    # but sklearn is standard. Let's just slice.
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape: {X_val.shape}, {y_val.shape}")
    
    # Apply data augmentation to training data only
    if AUGMENTATION:
        print("\n" + "="*50)
        X_train, y_train = apply_augmentations(X_train, y_train)
        print("="*50 + "\n")
        print(f"Train shape after augmentation: {X_train.shape}, {y_train.shape}")
    
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
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('train_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    plt.title('Average Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('train_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training complete.")
    print("Plots saved: train_loss.png, train_accuracy.png")

if __name__ == "__main__":
    train()
