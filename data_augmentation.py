import numpy as np


def augment_pitch_shift(X, y, semitones_range=(-3, 3), repeat=1):
    """Shift pitches up or down by a randomly selected semitone while keeping within valid MIDI range.
    
    Args:
        repeat: Number of augmented copies to create, each with different random shifts.
    """
    possible_shifts = [s for s in range(semitones_range[0], semitones_range[1] + 1) if s != 0]
    
    augmented_X = []
    augmented_y = []
    
    for _ in range(repeat):
        # Randomly select a different semitone shift for each sample (excluding 0)
        semitones = np.random.choice(possible_shifts, size=(X.shape[0], 1))  # shape: (num_samples, 1)
        
        X_shifted = X.copy()
        # Convert normalized pitch back to MIDI, shift, then normalize
        pitches = X_shifted[:, :, 0] * 128.0
        pitches_shifted = np.clip(pitches + semitones, 0, 127)  # broadcasts across sequence length
        X_shifted[:, :, 0] = pitches_shifted / 128.0
        
        augmented_X.append(X_shifted)
        augmented_y.append(y)
    
    return augmented_X, augmented_y


def augment_time_stretch(X, y, stretch_factors=[0.9, 1.1]):
    """Stretch or compress time (duration and dt) by a factor."""
    augmented_X = []
    augmented_y = []
    
    for factor in stretch_factors:
        X_stretched = X.copy()
        # Scale duration (feature 1) and dt (feature 2)
        X_stretched[:, :, 1] *= factor  # duration
        X_stretched[:, :, 2] *= factor  # dt
        
        augmented_X.append(X_stretched)
        augmented_y.append(y)
    
    return augmented_X, augmented_y


def augment_noise(X, y, noise_std=0.02):
    """Add small random noise to features."""
    X_noisy = X.copy()
    noise = np.random.normal(0, noise_std, X.shape)
    X_noisy = X_noisy + noise
    
    # Clip normalized features to valid ranges
    X_noisy[:, :, 0] = np.clip(X_noisy[:, :, 0], 0, 1)  # pitch
    X_noisy[:, :, 1] = np.maximum(X_noisy[:, :, 1], 0)  # duration (non-negative)
    X_noisy[:, :, 2] = np.maximum(X_noisy[:, :, 2], 0)  # dt (non-negative)
    X_noisy[:, :, 3] = np.clip(X_noisy[:, :, 3], 0, 1)  # velocity
    
    return [X_noisy], [y]


def augment_velocity_scale(X, y, scale_factors=[0.8, 1.2]):
    """Scale velocity values."""
    augmented_X = []
    augmented_y = []
    
    for factor in scale_factors:
        X_scaled = X.copy()
        X_scaled[:, :, 3] = np.clip(X_scaled[:, :, 3] * factor, 0, 1)  # velocity
        
        augmented_X.append(X_scaled)
        augmented_y.append(y)
    
    return augmented_X, augmented_y


def apply_augmentations(X, y):
    """Apply all augmentation techniques to the dataset."""
    print("Applying data augmentation...")
    augmented_X_list = [X]
    augmented_y_list = [y]
    
    # 1. Pitch shift
    X_ps, y_ps = augment_pitch_shift(X, y, semitones_range=(-24, 24), repeat=1) # 1 - 2 octave shift
    augmented_X_list.extend(X_ps)
    augmented_y_list.extend(y_ps)
    print(f"  Pitch shift: Added {len(X_ps)} variations")
    
    # 2. Time stretch
    X_ts, y_ts = augment_time_stretch(X, y, stretch_factors=[0.95, 1.05])
    augmented_X_list.extend(X_ts)
    augmented_y_list.extend(y_ts)
    print(f"  Time stretch: Added {len(X_ts)} variations")
    
    # 3. Noise
    X_noise, y_noise = augment_noise(X, y, noise_std=0.015)
    augmented_X_list.extend(X_noise)
    augmented_y_list.extend(y_noise)
    print(f"  Noise: Added {len(X_noise)} variations")
    
    # 4. Velocity scale
    X_vs, y_vs = augment_velocity_scale(X, y, scale_factors=[0.85, 1.15])
    augmented_X_list.extend(X_vs)
    augmented_y_list.extend(y_vs)
    print(f"  Velocity scale: Added {len(X_vs)} variations")
    
    # Combine all augmented data
    X_augmented = np.concatenate(augmented_X_list, axis=0)
    y_augmented = np.concatenate(augmented_y_list, axis=0)
    
    print(f"Augmentation complete: {len(X)} -> {len(X_augmented)} samples ({len(X_augmented)/len(X):.1f}x)")
    
    return X_augmented, y_augmented

