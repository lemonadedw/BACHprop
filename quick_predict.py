#!/usr/bin/env python3
"""
Quick prediction script - play 20 notes and predict the rest!
Usage: python quick_predict.py <midi_file> [num_initial_notes] [num_predictions]
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pretty_midi
from mido import MidiFile, MidiTrack, Message

def load_midi_notes(midi_path, num_notes=20):
    """Load initial notes from a MIDI file"""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity
            })
    
    notes = sorted(notes, key=lambda x: x['start'])[:num_notes]
    
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
    
    return np.array(features, dtype=np.float32)

def predict_sequence(model, initial_notes, num_predictions=30, sequence_length=50):
    """Predict future notes"""
    num_initial = len(initial_notes)
    
    if num_initial < sequence_length:
        padding = np.zeros((sequence_length - num_initial, 4), dtype=np.float32)
        sequence = np.vstack([padding, initial_notes])
    else:
        sequence = initial_notes[-sequence_length:]
    
    all_notes = list(sequence)
    predicted_pitches = []
    
    for i in range(num_predictions):
        input_seq = np.array([sequence])
        prediction = model.predict(input_seq, verbose=0)
        last_pred = prediction[0, -1, :]
        next_pitch_normalized = last_pred[1]
        
        predicted_pitches.append(next_pitch_normalized * 128.0)
        
        next_note = np.array([next_pitch_normalized, 0.5, 0.25, 0.7], dtype=np.float32)
        all_notes.append(next_note)
        sequence = np.vstack([sequence[1:], next_note])
    
    return {
        'predicted_pitches': np.array(predicted_pitches),
        'all_notes': np.array(all_notes),
        'num_initial': num_initial
    }

def visualize_and_save(result, midi_path, model):
    """Create visualization and save MIDI"""
    all_notes = result['all_notes']
    num_initial = result['num_initial']
    
    # Get hand predictions
    hands = []
    for i in range(0, len(all_notes), 50):
        chunk = all_notes[i:i+50]
        if len(chunk) < 50:
            padding = np.zeros((50 - len(chunk), 4), dtype=np.float32)
            chunk = np.vstack([padding, chunk])
        pred = model.predict(np.array([chunk]), verbose=0)
        chunk_hands = pred[0, :, 0]
        if len(all_notes[i:i+50]) < 50:
            chunk_hands = chunk_hands[-(len(all_notes[i:i+50])):]
        hands.extend(chunk_hands)
    hands = np.array(hands)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 6))
    
    times = []
    current_time = 0
    for i, note in enumerate(all_notes):
        pitch = note[0] * 128.0
        duration = note[1]
        dt = note[2]
        current_time += dt
        times.append(current_time)
        
        hand = hands[i]
        color = 'blue' if hand < 0.5 else 'red'
        alpha = 0.3 if i < num_initial else 0.7
        
        rect = patches.Rectangle(
            (current_time, pitch - 0.5), duration, 1.0,
            linewidth=1, edgecolor='black', facecolor=color, alpha=alpha
        )
        ax.add_patch(rect)
    
    ax.set_xlim(0, max(times) + 2)
    ax.set_ylim(30, 90)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Pitch (MIDI Note)')
    ax.set_title('Piano Roll: Blue=Right, Red=Left (Light=Input, Dark=Predicted)')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=times[num_initial-1] if num_initial > 0 else 0, 
               color='green', linestyle='--', linewidth=2)
    
    base_name = os.path.splitext(os.path.basename(midi_path))[0]
    viz_path = f"{base_name}_prediction_viz.png"
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
    
    for i, note in enumerate(all_notes):
        pitch = int(note[0] * 128.0)
        duration = note[1]
        dt = note[2]
        velocity = int(note[3] * 127.0)
        hand = hands[i]
        
        track = right_track if hand < 0.5 else left_track
        dt_ticks = int(dt * mid.ticks_per_beat * 2)
        duration_ticks = int(duration * mid.ticks_per_beat * 2)
        
        track.append(Message('note_on', note=pitch, velocity=velocity, time=dt_ticks))
        track.append(Message('note_off', note=pitch, velocity=0, time=duration_ticks))
    
    midi_out_path = f"{base_name}_predicted.mid"
    mid.save(midi_out_path)
    print(f"✅ MIDI file: {midi_out_path}")

def quick_predict(midi_path, num_initial=20, num_predictions=30, model_path=None):
    """Quick prediction and visualization"""
    if model_path is None:
        # Auto-select: try LSTM first, then transformer
        if os.path.exists("best_model_multitask.keras"):
            model_path = "best_model_multitask.keras"
        elif os.path.exists("best_model_multitask_transformer.keras"):
            model_path = "best_model_multitask_transformer.keras"
        else:
            model_path = "best_model_multitask.keras"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Train a model first:")
        print("  python train.py --multitask")
        print("  python train.py --multitask --transformer")
        return
    
    print(f"Using model: {model_path}")
    
    print(f"Loading model...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'multitask_loss': lambda y_true, y_pred: tf.reduce_mean(y_pred),
            'hand_accuracy': lambda y_true, y_pred: tf.reduce_mean(y_pred),
            'pitch_mae': lambda y_true, y_pred: tf.reduce_mean(y_pred)
        }
    )
    
    print(f"Loading {num_initial} notes from {midi_path}...")
    initial_notes = load_midi_notes(midi_path, num_notes=num_initial)
    
    print(f"Predicting {num_predictions} notes...")
    result = predict_sequence(model, initial_notes, num_predictions)
    
    print(f"\nPredicted pitches: {[int(p) for p in result['predicted_pitches'][:10]]}...")
    
    print("\nGenerating visualization...")
    visualize_and_save(result, midi_path, model)
    
    print("\nDone! 🎹")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_predict.py <midi_file> [num_initial] [num_predictions] [--lstm|--transformer]")
        print("\nExample:")
        print("  python quick_predict.py static/bach.mid")
        print("  python quick_predict.py static/mozart.mid 15 40")
        print("  python quick_predict.py static/bach.mid 20 30 --transformer")
        sys.exit(1)
    
    # Parse arguments
    midi_path = sys.argv[1]
    num_initial = 20
    num_predictions = 30
    model_path = None
    
    # Parse remaining arguments
    for arg in sys.argv[2:]:
        if arg == "--lstm":
            model_path = "best_model_multitask.keras"
        elif arg == "--transformer":
            model_path = "best_model_multitask_transformer.keras"
        elif arg.isdigit() and num_initial == 20:
            num_initial = int(arg)
        elif arg.isdigit():
            num_predictions = int(arg)
    
    quick_predict(midi_path, num_initial, num_predictions, model_path)

