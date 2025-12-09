#!/usr/bin/env python3
"""
Visualize MIDI files as piano sheet music with treble and bass clefs.

Usage:
    python visualize_midi.py file.mid
"""

import sys
import argparse
from pathlib import Path

try:
    from music21 import converter, instrument, stream, environment
except ImportError:
    print("Error: music21 is required. Install with: pip install music21")
    sys.exit(1)

# Configure MuseScore 4 path for macOS
def configure_musescore():
    """Configure music21 to use MuseScore 4."""
    env = environment.Environment()
    
    # MuseScore 4 path on macOS
    musescore_path = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
    
    if Path(musescore_path).exists():
        env['musicxmlPath'] = musescore_path
        env['musescoreDirectPNGPath'] = musescore_path
        print(f"Configured MuseScore at: {musescore_path}")
        return True
    
    print("Warning: Could not find MuseScore 4. Trying default viewer...")
    return False

configure_musescore()


def visualize_midi(midi_path: str, show_measures: int = None):
    """
    Load a MIDI file and display it as piano sheet music.
    
    Args:
        midi_path: Path to the MIDI file
        show_measures: Optional, limit display to first N measures
    """
    midi_path = Path(midi_path)
    
    if not midi_path.exists():
        print(f"Error: File not found: {midi_path}")
        sys.exit(1)
    
    print(f"Loading {midi_path}...")
    
    # Parse the MIDI file
    score = converter.parse(str(midi_path))
    
    # Create a piano score with two staves (right hand / left hand)
    piano_score = stream.Score()
    
    # Get all parts from the MIDI
    parts = score.parts
    
    if len(parts) == 0:
        print("Error: No parts found in MIDI file")
        sys.exit(1)
    
    print(f"Found {len(parts)} parts in MIDI file")
    
    # If there are exactly 2 parts, assume they are right hand and left hand
    if len(parts) >= 2:
        # First part -> treble clef (right hand)
        right_hand = parts[0]
        right_hand.partName = "Right Hand"
        right_hand.insert(0, instrument.Piano())
        
        # Second part -> bass clef (left hand)
        left_hand = parts[1]
        left_hand.partName = "Left Hand"
        left_hand.insert(0, instrument.Piano())
        
        piano_score.insert(0, right_hand)
        piano_score.insert(0, left_hand)
    else:
        # Single part - try to split by pitch (middle C = 60)
        print("Single part detected - splitting by pitch...")
        
        part = parts[0]
        right_hand = stream.Part()
        left_hand = stream.Part()
        
        right_hand.partName = "Right Hand"
        left_hand.partName = "Left Hand"
        right_hand.insert(0, instrument.Piano())
        left_hand.insert(0, instrument.Piano())
        
        for element in part.flatten().notes:
            # Check if it's a chord or single note
            if hasattr(element, 'pitches'):
                # It's a chord - check average pitch
                avg_pitch = sum(p.midi for p in element.pitches) / len(element.pitches)
                if avg_pitch >= 60:
                    right_hand.append(element)
                else:
                    left_hand.append(element)
            else:
                # Single note
                if element.pitch.midi >= 60:
                    right_hand.append(element)
                else:
                    left_hand.append(element)
        
        piano_score.insert(0, right_hand)
        piano_score.insert(0, left_hand)
    
    # Optionally limit to first N measures
    if show_measures:
        piano_score = piano_score.measures(1, show_measures)
    
    print("Opening score viewer...")
    print("(This may open MuseScore, Finale, or your default music notation app)")
    
    # Show the score
    piano_score.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MIDI files as piano sheet music"
    )
    parser.add_argument(
        "midi_file",
        type=str,
        help="Path to the MIDI file to visualize"
    )
    parser.add_argument(
        "-m", "--measures",
        type=int,
        default=None,
        help="Limit display to first N measures"
    )
    
    args = parser.parse_args()
    visualize_midi(args.midi_file, args.measures)


if __name__ == "__main__":
    main()

