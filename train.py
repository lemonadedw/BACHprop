import tensorflow as tf
from data_loader import PIGDataLoader
from model import create_model
import numpy as np
import os

def train(bidirectional=False):
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    SEQUENCE_LENGTH = 50
    
    # Data directories
    PIANO_FINGERING_DIR = "data/PianoFingeringDataset_v1.2/FingeringFiles"
    POP909_DIR = "data/POP909-Dataset/FingeringFiles"
    
    # Load data from both datasets
    print("=" * 60)
    print("Loading PianoFingeringDataset...")
    print("=" * 60)
    loader1 = PIGDataLoader(PIANO_FINGERING_DIR, sequence_length=SEQUENCE_LENGTH)
    X1, y1, file_indices1, num_files1 = loader1.get_data_with_file_indices()
    
    print("\n" + "=" * 60)
    print("Loading POP909 Dataset...")
    print("=" * 60)
    loader2 = PIGDataLoader(POP909_DIR, sequence_length=SEQUENCE_LENGTH)
    X2, y2, file_indices2, num_files2 = loader2.get_data_with_file_indices()
    
    # Combine datasets
    print("\n" + "=" * 60)
    print("Combining datasets...")
    print("=" * 60)
    
    # Adjust file indices for second dataset to be unique
    file_indices2_adjusted = file_indices2 + num_files1
    
    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    file_indices = np.concatenate([file_indices1, file_indices2_adjusted], axis=0)
    num_files = num_files1 + num_files2
    
    print(f"Combined {num_files1} + {num_files2} = {num_files} total files")
    print(f"Total sequences: {len(X)}")
    print(f"Combined data shape: {X.shape}")
    print(f"Combined labels shape: {y.shape}")
    
    # Split by FILE to prevent data leakage (sequences from same file stay together)
    train_file_count = int(0.8 * num_files)
    train_mask = file_indices < train_file_count
    val_mask = file_indices >= train_file_count
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    
    print(f"\nSplit: {train_file_count} training files, {num_files - train_file_count} validation files")
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape: {X_val.shape}, {y_val.shape}")
    
    # Create Model
    model_type = "Bidirectional" if bidirectional else "Unidirectional"
    print("\n" + "=" * 60)
    print(f"Creating {model_type} LSTM model...")
    print("=" * 60)
    model = create_model(input_shape=(SEQUENCE_LENGTH, 4), bidirectional=bidirectional)
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Callbacks
    model_filename = "best_model_bidirectional.keras" if bidirectional else "best_model.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_filename, save_best_only=True, monitor='val_accuracy'
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    history = model.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb]
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {model_filename}")

if __name__ == "__main__":
    import sys
    
    bidirectional = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--bidirectional" or sys.argv[1] == "--bi":
            bidirectional = True
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python train.py                    # Train unidirectional LSTM model")
            print("  python train.py --bidirectional   # Train bidirectional LSTM model")
            print("  python train.py --help            # Show this help message")
            sys.exit(0)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage info.")
            sys.exit(1)
    
    train(bidirectional=bidirectional)
