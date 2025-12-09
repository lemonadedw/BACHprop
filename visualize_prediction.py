#!/usr/bin/env python3
"""
Visualize model predictions on a fingering file.
Shows correct predictions in green, incorrect in red.

Usage:
    python visualize_prediction.py fingering_file.txt
    python visualize_prediction.py fingering_file.txt --model best_model.keras
"""

import sys
import argparse
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("Error: tensorflow is required. Install with: pip install tensorflow")
    sys.exit(1)

try:
    import pretty_midi
except ImportError:
    print("Error: pretty_midi is required. Install with: pip install pretty_midi")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

from html_visualization import create_html_visualization
# Import custom layer so it can be deserialized when loading saved models
from model import PositionalEncoding


def parse_fingering_file(file_path):
    """
    Parse a fingering file and extract notes with ground truth hand labels.
    
    Returns:
        List of dicts with keys: onset, offset, pitch, pitch_name, velocity, hand
    """
    notes = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header if present
    start_idx = 0
    if lines[0].startswith('//'):
        start_idx = 1
    
    for line in lines[start_idx:]:
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
        
        try:
            onset = float(parts[1])
            offset = float(parts[2])
            pitch_name = parts[3]
            velocity = int(parts[4])
            hand = int(parts[6])  # 0 = Right, 1 = Left
            
            pitch = pretty_midi.note_name_to_number(pitch_name)
            
            notes.append({
                'onset': onset,
                'offset': offset,
                'pitch': pitch,
                'pitch_name': pitch_name,
                'velocity': velocity,
                'hand': hand  # Ground truth
            })
        except (ValueError, IndexError):
            continue
    
    return notes


def extract_features(notes):
    """
    Extract features from notes for model input.
    
    Returns:
        numpy array of shape (num_notes, 4)
    """
    features = []
    prev_onset = 0.0
    
    for n in notes:
        duration = n['offset'] - n['onset']
        dt = n['onset'] - prev_onset
        prev_onset = n['onset']
        
        features.append([
            n['pitch'] / 128.0,
            duration,
            dt,
            n['velocity'] / 127.0
        ])
    
    return np.array(features, dtype=np.float32)


def get_predictions(model, features, sequence_length=50):
    """
    Get predictions from model, handling variable length sequences.
    """
    num_notes = len(features)
    predictions = np.zeros(num_notes, dtype=int)
    
    # Process in chunks of sequence_length
    for i in range(0, num_notes, sequence_length):
        end_idx = min(i + sequence_length, num_notes)
        chunk = features[i:end_idx]
        
        # Pad if necessary
        if len(chunk) < sequence_length:
            padded = np.zeros((sequence_length, 4), dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
        
        # Predict
        x = np.expand_dims(chunk, axis=0)  # (1, seq_len, 4)
        output = model.predict(x, verbose=0)  # (1, seq_len, 1)
        preds = (output[0, :, 0] > 0.5).astype(int)
        
        # Store predictions (only for actual notes, not padding)
        actual_len = end_idx - i
        predictions[i:end_idx] = preds[:actual_len]
    
    return predictions


def pitch_to_name(pitch):
    """Convert MIDI pitch to note name."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = pitch // 12 - 1
    note = note_names[pitch % 12]
    return f"{note}{octave}"


def create_piano_roll(notes, predictions, ground_truth, max_notes=200, time_range=None):
    """
    Create a piano roll visualization with colored notes.
    Green = correct prediction, Red = incorrect prediction
    """
    # Limit notes for visualization
    if len(notes) > max_notes:
        print(f"Limiting visualization to first {max_notes} notes...")
        notes = notes[:max_notes]
        predictions = predictions[:max_notes]
        ground_truth = ground_truth[:max_notes]
    
    # Calculate statistics
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    incorrect = len(predictions) - correct
    accuracy = correct / len(predictions) * 100 if predictions.size > 0 else 0
    
    # Create figure with two subplots (right hand on top, left hand on bottom)
    fig, (ax_right, ax_left) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Separate notes by hand
    right_hand_notes = []
    left_hand_notes = []
    
    for i, n in enumerate(notes):
        note_data = {
            'onset': n['onset'],
            'offset': n['offset'],
            'pitch': n['pitch'],
            'correct': predictions[i] == ground_truth[i]
        }
        
        if ground_truth[i] == 0:  # Right hand
            right_hand_notes.append(note_data)
        else:  # Left hand
            left_hand_notes.append(note_data)
    
    # Plot right hand
    for n in right_hand_notes:
        color = '#2ecc71' if n['correct'] else '#e74c3c'  # Green or Red
        rect = mpatches.Rectangle(
            (n['onset'], n['pitch'] - 0.4),
            n['offset'] - n['onset'],
            0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.8
        )
        ax_right.add_patch(rect)
    
    # Plot left hand
    for n in left_hand_notes:
        color = '#2ecc71' if n['correct'] else '#e74c3c'  # Green or Red
        rect = mpatches.Rectangle(
            (n['onset'], n['pitch'] - 0.4),
            n['offset'] - n['onset'],
            0.8,
            linewidth=1,
            edgecolor='black',
            facecolor=color,
            alpha=0.8
        )
        ax_left.add_patch(rect)
    
    # Configure right hand axis
    if right_hand_notes:
        right_pitches = [n['pitch'] for n in right_hand_notes]
        ax_right.set_ylim(min(right_pitches) - 2, max(right_pitches) + 2)
    else:
        ax_right.set_ylim(60, 96)  # Default treble range
    
    # Configure left hand axis
    if left_hand_notes:
        left_pitches = [n['pitch'] for n in left_hand_notes]
        ax_left.set_ylim(min(left_pitches) - 2, max(left_pitches) + 2)
    else:
        ax_left.set_ylim(36, 60)  # Default bass range
    
    # Set x-axis limits
    if notes:
        max_time = max(n['offset'] for n in notes)
        min_time = min(n['onset'] for n in notes)
        ax_right.set_xlim(min_time - 0.5, max_time + 0.5)
        ax_left.set_xlim(min_time - 0.5, max_time + 0.5)
    
    # Add pitch labels (show some key notes)
    for ax in [ax_right, ax_left]:
        y_min, y_max = ax.get_ylim()
        yticks = list(range(int(y_min), int(y_max) + 1, 2))
        ax.set_yticks(yticks)
        ax.set_yticklabels([pitch_to_name(p) for p in yticks])
        ax.grid(True, alpha=0.3)
    
    # Labels and title
    ax_right.set_ylabel('Right Hand (Treble)', fontsize=12)
    ax_left.set_ylabel('Left Hand (Bass)', fontsize=12)
    ax_left.set_xlabel('Time (seconds)', fontsize=12)
    
    # Create legend
    correct_patch = mpatches.Patch(color='#2ecc71', label=f'Correct ({correct})')
    incorrect_patch = mpatches.Patch(color='#e74c3c', label=f'Incorrect ({incorrect})')
    fig.legend(handles=[correct_patch, incorrect_patch], loc='upper right', fontsize=11)
    
    # Title with accuracy
    fig.suptitle(f'Piano Hand Prediction Results - Accuracy: {accuracy:.1f}%', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig, accuracy, correct, incorrect


def visualize_predictions(fingering_path, model_path='best_model.keras', max_notes=200, output_html=None):
    """
    Main function to visualize predictions on a fingering file.
    """
    fingering_path = Path(fingering_path)
    
    if not fingering_path.exists():
        print(f"Error: File not found: {fingering_path}")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained a model first (run train.py)")
        sys.exit(1)
    
    # Parse fingering file
    print(f"Parsing {fingering_path}...")
    notes = parse_fingering_file(fingering_path)
    
    if not notes:
        print("Error: No notes found in fingering file")
        sys.exit(1)
    
    print(f"Found {len(notes)} notes")
    
    # Extract features and ground truth
    features = extract_features(notes)
    ground_truth = np.array([n['hand'] for n in notes], dtype=int)
    
    # Get model predictions
    print("Getting model predictions...")
    predictions = get_predictions(model, features)
    
    # Generate HTML visualization
    if output_html is None:
        output_html = fingering_path.stem + '_visualization.html'
    
    print(f"Creating HTML visualization: {output_html}...")
    accuracy, correct, incorrect = create_html_visualization(notes, predictions, ground_truth, output_html)
    
    print(f"\nPrediction Results:")
    print(f"  Correct:   {correct} (green)")
    print(f"  Incorrect: {incorrect} (red)")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"\nHTML visualization saved to: {output_html}")
    print(f"Open {output_html} in your browser to view the results.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions with color coding"
    )
    parser.add_argument(
        "fingering_file",
        type=str,
        help="Path to the fingering file to visualize"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="best_model.keras",
        help="Path to the trained model (default: best_model.keras)"
    )
    parser.add_argument(
        "-n", "--max-notes",
        type=int,
        default=200,
        help="Maximum number of notes to display (default: 200) - DEPRECATED: HTML shows all notes"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: <input_filename>_visualization.html)"
    )
    
    args = parser.parse_args()
    visualize_predictions(args.fingering_file, args.model, args.max_notes, args.output)


if __name__ == "__main__":
    main()
