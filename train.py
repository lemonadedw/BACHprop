import tensorflow as tf
from data_loader import PIGDataLoader
from model import create_model
import numpy as np
import os

def multitask_loss(y_true, y_pred):
    """
    Custom loss for multi-task learning:
    - Binary cross-entropy for hand classification (output[:, :, 0])
    - Categorical cross-entropy for next pitch prediction (output[:, :, 1:89])
    
    y_true shape: (batch, seq, 89)
    y_pred shape: (batch, seq, 89)
    """
    # Hand classification loss (binary cross-entropy)
    # Apply sigmoid to hand predictions
    hand_pred_sigmoid = tf.nn.sigmoid(y_pred[:, :, 0])
    hand_loss = tf.keras.backend.binary_crossentropy(y_true[:, :, 0], hand_pred_sigmoid)
    
    # Pitch prediction loss (categorical cross-entropy)
    # Apply softmax to pitch predictions (indices 1:89)
    pitch_pred_logits = y_pred[:, :, 1:]  # Shape: (batch, seq, 88)
    pitch_true_onehot = y_true[:, :, 1:]  # Shape: (batch, seq, 88)
    
    # Categorical cross-entropy with logits
    pitch_pred_softmax = tf.nn.softmax(pitch_pred_logits, axis=-1)
    # Clip to avoid log(0)
    pitch_pred_softmax = tf.clip_by_value(pitch_pred_softmax, 1e-7, 1.0)
    pitch_loss = -tf.reduce_sum(pitch_true_onehot * tf.math.log(pitch_pred_softmax), axis=-1)
    
    # Combined loss with weights
    # You can adjust these weights to prioritize one task over the other
    hand_weight = 1.0
    pitch_weight = 1.0
    
    total_loss = hand_weight * tf.keras.backend.mean(hand_loss) + pitch_weight * tf.keras.backend.mean(pitch_loss)
    return total_loss

def hand_accuracy(y_true, y_pred):
    """Metric for hand classification accuracy"""
    hand_pred_sigmoid = tf.nn.sigmoid(y_pred[:, :, 0])
    hand_pred_binary = tf.keras.backend.cast(hand_pred_sigmoid > 0.5, tf.float32)
    hand_true = y_true[:, :, 0]
    return tf.keras.backend.mean(tf.keras.backend.cast(tf.keras.backend.equal(hand_true, hand_pred_binary), tf.float32))

def pitch_accuracy(y_true, y_pred):
    """Metric for pitch prediction accuracy (top-1 accuracy)"""
    # Get predicted pitch class (argmax of softmax over 88 keys)
    pitch_pred_logits = y_pred[:, :, 1:]  # Shape: (batch, seq, 88)
    pitch_pred_class = tf.argmax(pitch_pred_logits, axis=-1)  # Shape: (batch, seq)
    
    # Get true pitch class (argmax of one-hot)
    pitch_true_onehot = y_true[:, :, 1:]  # Shape: (batch, seq, 88)
    pitch_true_class = tf.argmax(pitch_true_onehot, axis=-1)  # Shape: (batch, seq)
    
    # Calculate accuracy
    correct = tf.keras.backend.cast(tf.keras.backend.equal(pitch_true_class, pitch_pred_class), tf.float32)
    return tf.keras.backend.mean(correct)

def train():
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
    model = create_model(input_shape=(SEQUENCE_LENGTH, 4), multitask=False)
    
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

def train_multitask(use_transformer=False):
    """
    Train the multi-task model that predicts both:
    1. Hand classification (left/right)
    2. Next note's pitch
    
    Args:
        use_transformer: If True, uses transformer architecture instead of LSTM
    """
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
    y1_multitask = loader1.get_multitask_labels()
    
    print("\n" + "=" * 60)
    print("Loading POP909 Dataset...")
    print("=" * 60)
    loader2 = PIGDataLoader(POP909_DIR, sequence_length=SEQUENCE_LENGTH)
    X2, y2, file_indices2, num_files2 = loader2.get_data_with_file_indices()
    y2_multitask = loader2.get_multitask_labels()
    
    # Combine datasets
    print("\n" + "=" * 60)
    print("Combining datasets...")
    print("=" * 60)
    
    # Adjust file indices for second dataset to be unique
    file_indices2_adjusted = file_indices2 + num_files1
    
    X = np.concatenate([X1, X2], axis=0)
    y_multitask = np.concatenate([y1_multitask, y2_multitask], axis=0)
    file_indices = np.concatenate([file_indices1, file_indices2_adjusted], axis=0)
    num_files = num_files1 + num_files2
    
    print(f"Combined {num_files1} + {num_files2} = {num_files} total files")
    print(f"Total sequences: {len(X)}")
    print(f"Combined data shape: {X.shape}")
    print(f"Combined multitask labels shape: {y_multitask.shape}")
    print(f"  - labels[:, :, 0] = hand classification")
    print(f"  - labels[:, :, 1:89] = next pitch one-hot (88 piano keys)")
    
    # Split by FILE to prevent data leakage (sequences from same file stay together)
    train_file_count = int(0.8 * num_files)
    train_mask = file_indices < train_file_count
    val_mask = file_indices >= train_file_count
    
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y_multitask[train_mask], y_multitask[val_mask]
    
    print(f"\nSplit: {train_file_count} training files, {num_files - train_file_count} validation files")
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Val shape: {X_val.shape}, {y_val.shape}")
    
    # Create Multi-task Model
    print("\n" + "=" * 60)
    model_type = "Transformer" if use_transformer else "LSTM"
    print(f"Creating multi-task {model_type} model...")
    print("=" * 60)
    model = create_model(input_shape=(SEQUENCE_LENGTH, 4), multitask=True, use_transformer=use_transformer)
    model.summary()
    
    # Compile with custom multi-task loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=multitask_loss,
        metrics=[hand_accuracy, pitch_accuracy]
    )
    
    # Callbacks
    model_filename = "best_model_multitask_transformer.keras" if use_transformer else "best_model_multitask.keras"
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_filename, 
        save_best_only=True, 
        monitor='val_hand_accuracy',
        mode='max'  # Maximize accuracy
    )
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_hand_accuracy',
        patience=5,
        restore_best_weights=True,
        mode='max'  # Maximize accuracy
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Training multi-task model...")
    print("=" * 60)
    history = model.fit(
        X_train, y_train,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb, early_stopping_cb]
    )
    
    print("\n" + "=" * 60)
    print("Multi-task training complete!")
    print("=" * 60)
    print(f"Model saved to: {model_filename}")
    
    return history

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--multitask":
            use_transformer = "--transformer" in sys.argv
            train_multitask(use_transformer=use_transformer)
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python train.py                          # Train simple hand classification model")
            print("  python train.py --multitask              # Train multi-task LSTM model")
            print("  python train.py --multitask --transformer # Train multi-task Transformer model")
        else:
            print("Unknown argument. Use --help for usage info.")
    else:
        print("Usage:")
        print("  python train.py                          # Train simple hand classification model")
        print("  python train.py --multitask              # Train multi-task LSTM model")
        print("  python train.py --multitask --transformer # Train multi-task Transformer model")
        print("\nDefaulting to simple model...")
        train()
