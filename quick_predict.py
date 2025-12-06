#!/usr/bin/env python3
"""
Quick prediction script - predict hand assignments for MIDI file
Usage: python quick_predict.py <midi_file> [--bidirectional]
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pretty_midi
from mido import MidiFile, MidiTrack, Message

def load_midi_notes(midi_path):
    """Load all notes from a MIDI file"""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'velocity': note.velocity
                })
    
    notes = sorted(notes, key=lambda x: x['start'])
    
    features = []
    prev_onset = 0.0
    
    for note in notes:
        features.append([
            note['pitch'] / 128.0,
            note['end'] - note['start'],
            note['start'] - prev_onset,
            note['velocity'] / 127.0
        ])
        prev_onset = note['start']
    
    return np.array(features, dtype=np.float32), notes

def predict_hands(model, notes_features, sequence_length=50):
    """Predict hand assignments for all notes"""
    all_hands = []
    
    # Process in chunks of sequence_length
    for i in range(0, len(notes_features), sequence_length):
        chunk = notes_features[i:i+sequence_length]
        
        # Pad if necessary
        if len(chunk) < sequence_length:
            padding = np.zeros((sequence_length - len(chunk), 4), dtype=np.float32)
            chunk = np.vstack([padding, chunk])
        
        # Predict
        input_seq = np.array([chunk])
        prediction = model.predict(input_seq, verbose=0)
        
        # Extract hand predictions (sigmoid already applied in model)
        chunk_hands = prediction[0, :, 0]
        
        # Trim padding if needed
        if len(notes_features[i:i+sequence_length]) < sequence_length:
            chunk_hands = chunk_hands[-(len(notes_features[i:i+sequence_length])):]
        
        all_hands.extend(chunk_hands)
    
    return np.array(all_hands)

def visualize_and_save(notes_features, notes_data, hands, midi_path):
    """Create visualization and save MIDI"""
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    times = []
    current_time = 0
    for i, note_feat in enumerate(notes_features):
        pitch = note_feat[0] * 128.0
        duration = note_feat[1]
        dt = note_feat[2]
        current_time += dt
        times.append(current_time)
        
        hand = hands[i]
        color = 'blue' if hand < 0.5 else 'red'
        
        rect = patches.Rectangle(
            (current_time, pitch - 0.5), duration, 1.0,
            linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)
    
    ax.set_xlim(0, max(times) + 2 if times else 2)
    ax.set_ylim(30, 90)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pitch (MIDI Note)')
    ax.set_title('Piano Roll: Blue=Right Hand, Red=Left Hand')
    ax.grid(True, alpha=0.3)
    
    base_name = os.path.splitext(os.path.basename(midi_path))[0]
    viz_path = f"{base_name}_hand_prediction_viz.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization: {viz_path}")
    plt.close()
    
    # Save MIDI
    mid = MidiFile()
    right_track = MidiTrack()
    left_track = MidiTrack()
    mid.tracks.extend([right_track, left_track])
    
    right_track.append(Message('track_name', name='Right Hand', time=0))
    left_track.append(Message('track_name', name='Left Hand', time=0))
    
    prev_time = 0
    for i, note in enumerate(notes_data):
        pitch = note['pitch']
        start_time = note['start']
        end_time = note['end']
        velocity = note['velocity']
        hand = hands[i]
        
        track = right_track if hand < 0.5 else left_track
        
        # Calculate delta time in ticks
        dt_ticks = int((start_time - prev_time) * mid.ticks_per_beat * 2)
        duration_ticks = int((end_time - start_time) * mid.ticks_per_beat * 2)
        
        track.append(Message('note_on', note=pitch, velocity=velocity, time=dt_ticks))
        track.append(Message('note_off', note=pitch, velocity=0, time=duration_ticks))
        
        prev_time = start_time
    
    midi_out_path = f"{base_name}_hand_prediction.mid"
    mid.save(midi_out_path)
    print(f"✅ MIDI file: {midi_out_path}")
    
    # Print statistics
    right_count = np.sum(hands < 0.5)
    left_count = np.sum(hands >= 0.5)
    print(f"\n📊 Statistics:")
    print(f"   Right hand notes: {right_count} ({100*right_count/len(hands):.1f}%)")
    print(f"   Left hand notes: {left_count} ({100*left_count/len(hands):.1f}%)")

def quick_predict(midi_path, model_path=None):
    """Quick hand prediction and visualization"""
    if model_path is None:
        # Auto-select: try unidirectional first, then bidirectional
        if os.path.exists("best_model.keras"):
            model_path = "best_model.keras"
        elif os.path.exists("best_model_bidirectional.keras"):
            model_path = "best_model_bidirectional.keras"
        else:
            model_path = "best_model.keras"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train a model first:")
        print("  python train.py                    # Unidirectional model")
        print("  python train.py --bidirectional   # Bidirectional model")
        return
    
    print(f"Using model: {model_path}")
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading MIDI file: {midi_path}...")
    notes_features, notes_data = load_midi_notes(midi_path)
    print(f"Found {len(notes_features)} notes")
    
    print("Predicting hand assignments...")
    hands = predict_hands(model, notes_features)
    
    print("\nGenerating visualization and saving MIDI...")
    visualize_and_save(notes_features, notes_data, hands, midi_path)
    
    print("\nDone! 🎹")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_predict.py <midi_file> [--bidirectional]")
        print("\nExample:")
        print("  python quick_predict.py static/bach.mid")
        print("  python quick_predict.py static/mozart.mid --bidirectional")
        sys.exit(1)
    
    # Parse arguments
    midi_path = sys.argv[1]
    model_path = None
    
    # Parse remaining arguments
    for arg in sys.argv[2:]:
        if arg == "--bidirectional" or arg == "--bi":
            model_path = "best_model_bidirectional.keras"
        elif arg == "--help":
            print("Usage: python quick_predict.py <midi_file> [--bidirectional]")
            print("\nOptions:")
            print("  --bidirectional   Use bidirectional model (default: unidirectional)")
            print("  --help           Show this help message")
            sys.exit(0)
    
    quick_predict(midi_path, model_path)

