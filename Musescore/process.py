import os
import glob
import pretty_midi
from pathlib import Path

def process_midi_to_fingering(midi_path, output_dir):
    """
    Convert a MIDI file to fingering format.
    
    Args:
        midi_path: Path to input MIDI file
        output_dir: Directory to save output fingering file
    """
    try:
        # Load MIDI file
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        # Collect all notes with hand information
        all_notes = []
        
        # Process each track (assuming first track is right hand, second is left hand)
        for track_idx, instrument in enumerate(pm.instruments):
            if instrument.is_drum:
                continue
                
            # Determine hand: 0 = right hand (first track), 1 = left hand (second track)
            hand = 0 if track_idx == 0 else 1
            
            for note in instrument.notes:
                # Convert MIDI note number to note name (e.g., 60 -> 'C4')
                note_name = pretty_midi.note_number_to_name(note.pitch)
                
                all_notes.append({
                    'onset': note.start,
                    'offset': note.end,
                    'note_name': note_name,
                    'velocity': note.velocity,
                    'hand': hand
                })
        
        # Sort all notes by onset time
        all_notes.sort(key=lambda x: (x['onset'], x['hand']))
        
        # Generate output filename
        midi_filename = Path(midi_path).stem
        output_filename = f"{midi_filename}_fingering.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write fingering file
        with open(output_path, 'w') as f:
            # Write header
            f.write("//Version: PianoFingering_v170101\n")
            
            # Write each note
            for idx, note in enumerate(all_notes):
                # Format: index\tonset\toffset\tnote_name\tvelocity\t80\thand\tfingering
                # Using -1 for fingering since we don't have actual fingering data
                line = f"{idx}\t{note['onset']:.5f}\t{note['offset']:.5f}\t{note['note_name']}\t{note['velocity']}\t80\t{note['hand']}\t-1\n"
                f.write(line)
        
        print(f"Processed: {midi_path} -> {output_path} ({len(all_notes)} notes)")
        return True
        
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return False


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    raw_data_dir = script_dir / "raw_data"
    output_dir = script_dir / "FingeringFiles"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all MIDI files
    midi_files = glob.glob(str(raw_data_dir / "*.mid"))
    
    if not midi_files:
        print(f"No MIDI files found in {raw_data_dir}")
        return
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Process each MIDI file
    success_count = 0
    for midi_path in sorted(midi_files):
        if process_midi_to_fingering(midi_path, output_dir):
            success_count += 1
    
    print(f"\nProcessed {success_count}/{len(midi_files)} files successfully")


if __name__ == "__main__":
    main()

