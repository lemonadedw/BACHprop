import tensorflow as tf
import pretty_midi
import numpy as np
import argparse

def predict(midi_path, output_path, model_path='best_model.keras'):
    # Load Model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load MIDI
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file: {e}")
        return

    # Collect all notes
    all_notes = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)
            
    # Sort notes
    all_notes.sort(key=lambda x: x.start)
    
    if not all_notes:
        print("No notes found in MIDI file.")
        return
        
    # Extract features
    features = []
    prev_onset = 0.0
    
    for note in all_notes:
        pitch = note.pitch
        duration = note.end - note.start
        dt = note.start - prev_onset
        velocity = note.velocity
        prev_onset = note.start
        
        features.append([
            pitch / 128.0,
            duration,
            dt,
            velocity / 127.0
        ])
        
    # Create input tensor
    # Model expects (batch, seq_len, 4)
    # We have one sequence.
    x = np.array([features], dtype=np.float32)
    
    # Inference
    outputs = model.predict(x) # (1, seq_len, 1)
    predictions = (outputs[0, :, 0] > 0.5).astype(int)
    
    # Create new MIDI
    new_pm = pretty_midi.PrettyMIDI()
    right_hand = pretty_midi.Instrument(program=0, name="Right Hand")
    left_hand = pretty_midi.Instrument(program=0, name="Left Hand")
    
    for i, note in enumerate(all_notes):
        new_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start,
            end=note.end
        )
        
        if predictions[i] == 0:
            right_hand.notes.append(new_note)
        else:
            left_hand.notes.append(new_note)
            
    new_pm.instruments.append(right_hand)
    new_pm.instruments.append(left_hand)
    
    new_pm.write(output_path)
    print(f"Saved split MIDI to {output_path}")
    print(f"Right hand notes: {len(right_hand.notes)}")
    print(f"Left hand notes: {len(left_hand.notes)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("midi_path", help="Path to input MIDI file")
    parser.add_argument("--output", default="output.mid", help="Path to output MIDI file")
    args = parser.parse_args()
    
    predict(args.midi_path, args.output)
