#!/usr/bin/env python3
"""
Preprocessing script to convert POP909 MIDI files to fingering file format.
"""

import os
import mido
from pathlib import Path


def midi_note_to_name(note_number):
    """Convert MIDI note number to note name (e.g., 60 -> C4)."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (note_number // 12) - 1
    note_name = note_names[note_number % 12]
    return f"{note_name}{octave}"


def process_midi_track(track, ticks_per_beat, tempo):
    """
    Process a MIDI track and extract note events.
    
    Returns a list of note events with timing information.
    Each event is a dict with: start_time, end_time, note, velocity, channel
    """
    # Convert tempo to seconds per tick
    microseconds_per_beat = tempo
    seconds_per_beat = microseconds_per_beat / 1_000_000
    seconds_per_tick = seconds_per_beat / ticks_per_beat
    
    current_time_ticks = 0
    active_notes = {}  # (note, channel) -> (start_time, velocity)
    note_events = []
    
    for msg in track:
        current_time_ticks += msg.time
        current_time_seconds = current_time_ticks * seconds_per_tick
        
        if msg.type == 'note_on':
            key = (msg.note, msg.channel)
            
            if msg.velocity > 0:
                # Note on
                active_notes[key] = (current_time_seconds, msg.velocity)
            else:
                # Note off (velocity = 0)
                if key in active_notes:
                    start_time, velocity = active_notes.pop(key)
                    note_events.append({
                        'start_time': start_time,
                        'end_time': current_time_seconds,
                        'note': msg.note,
                        'velocity': velocity,
                        'channel': msg.channel
                    })
        elif msg.type == 'note_off':
            key = (msg.note, msg.channel)
            if key in active_notes:
                start_time, velocity = active_notes.pop(key)
                note_events.append({
                    'start_time': start_time,
                    'end_time': current_time_seconds,
                    'note': msg.note,
                    'velocity': velocity,
                    'channel': msg.channel
                })
    
    return note_events


def merge_melody_and_bridge(midi_file):
    """
    Merge MELODY and BRIDGE tracks from a MIDI file (excluding PIANO track).
    
    Returns a list of all note events sorted by start time.
    """
    # Get tempo and ticks per beat from the MIDI file
    ticks_per_beat = midi_file.ticks_per_beat
    tempo = 500000  # Default tempo (120 BPM)
    
    # Extract tempo from the first track (usually contains meta messages)
    for msg in midi_file.tracks[0]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break
    
    all_events = []
    
    for track in midi_file.tracks:
        # Only process MELODY and BRIDGE tracks
        if track.name in ['MELODY', 'BRIDGE']:
            # Process the track
            events = process_midi_track(track, ticks_per_beat, tempo)
            all_events.extend(events)
    
    # Sort events by start time
    all_events.sort(key=lambda x: x['start_time'])
    
    return all_events


def assign_hand_and_finger(events):
    """
    Assign hand (0 for right, 1 for left) and finger values.
    
    Simple heuristic: notes >= C4 (60) are right hand, notes < C4 are left hand.
    Finger values are assigned based on relative pitch within each hand.
    Right hand: 1-5 (thumb to pinky)
    Left hand: -1 to -5 (thumb to pinky)
    """
    # Separate events by hand
    right_hand_events = []
    left_hand_events = []
    
    for event in events:
        # Simple hand assignment based on pitch
        if event['note'] >= 60:  # C4 and above
            event['hand'] = 0  # Right hand
            right_hand_events.append(event)
        else:
            event['hand'] = 1  # Left hand
            left_hand_events.append(event)
    
    # Assign fingers based on pitch patterns
    # For right hand: lower notes use lower fingers (1=thumb), higher notes use higher fingers (5=pinky)
    # For left hand: higher notes use lower fingers (-1=thumb), lower notes use higher fingers (-5=pinky)
    
    for event in right_hand_events:
        # Simple cyclic assignment based on note modulo
        # This is a placeholder; real fingering would consider hand position and movement
        relative_pitch = event['note'] - 60  # Relative to C4
        finger = (relative_pitch % 5) + 1  # Cycle through fingers 1-5
        event['finger'] = finger
    
    for event in left_hand_events:
        # For left hand, use negative fingers
        relative_pitch = 60 - event['note']  # Distance below C4
        finger = (relative_pitch % 5) + 1
        event['finger'] = -finger  # Negative for left hand
    
    return events


def write_fingering_file(events, output_path):
    """
    Write events to a fingering file in the required format.
    
    Format:
    index  start_time  end_time  note_name  ??  velocity  hand  finger
    """
    with open(output_path, 'w') as f:
        # Write version header
        f.write("//Version: PianoFingering_v170101\n")
        
        # Write each note event
        for idx, event in enumerate(events):
            note_name = midi_note_to_name(event['note'])
            
            # Format: index, start_time, end_time, note_name, 64 (unknown), velocity, hand, finger
            # The "64" appears to be a constant in the original files
            line = f"{idx}\t{event['start_time']:.6f}\t{event['end_time']:.6f}\t{note_name}\t64\t{event['velocity']}\t{event['hand']}\t{event['finger']}\n"
            f.write(line)


def preprocess_pop909(pop909_dir="data/POP909-Dataset/POP909", output_dir="data/POP909-Dataset/FingeringFiles"):
    """
    Main preprocessing function to convert POP909 MIDI files to fingering format.
    
    Args:
        pop909_dir: Path to POP909 dataset directory
        output_dir: Path to output directory for fingering files
    """
    pop909_path = Path(pop909_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing POP909 dataset from: {pop909_path}")
    print(f"Output directory: {output_path}")
    
    # Get all subdirectories (each song has its own directory)
    song_dirs = sorted([d for d in pop909_path.iterdir() if d.is_dir()])
    
    processed_count = 0
    error_count = 0
    
    for song_dir in song_dirs:
        # Look for the main MIDI file (e.g., 001.mid, not version files)
        midi_files = list(song_dir.glob("*.mid"))
        
        # Filter to get only the main file (not in versions subdirectory)
        main_midi = None
        for midi_file in midi_files:
            if midi_file.parent == song_dir:
                main_midi = midi_file
                break
        
        if not main_midi:
            print(f"Warning: No main MIDI file found in {song_dir}")
            error_count += 1
            continue
        
        try:
            # Load MIDI file
            midi = mido.MidiFile(main_midi)
            
            # Process: merge MELODY and BRIDGE tracks only
            events = merge_melody_and_bridge(midi)
            
            # Assign hand and finger
            events = assign_hand_and_finger(events)
            
            # Generate output filename
            song_name = song_dir.name
            output_file = output_path / f"{song_name}_fingering.txt"
            
            # Write fingering file
            write_fingering_file(events, output_file)
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"Processed {processed_count} files...")
        
        except Exception as e:
            print(f"Error processing {main_midi}: {e}")
            error_count += 1
            continue
    
    print(f"\nPreprocessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    preprocess_pop909()

