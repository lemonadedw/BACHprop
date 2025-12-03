import pretty_midi

def create_test_midi(filename="test_input.mid"):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    
    # Add some notes
    # C4 (Middle C) - likely Left Hand or Right Hand depending on context, but let's put some low and high notes
    
    # Low notes (Left Hand likely)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=48, start=0, end=0.5)) # C3
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=52, start=0.5, end=1.0)) # E3
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=55, start=1.0, end=1.5)) # G3
    
    # High notes (Right Hand likely)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=72, start=0, end=0.5)) # C5
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=76, start=0.5, end=1.0)) # E5
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=79, start=1.0, end=1.5)) # G5
    
    pm.instruments.append(inst)
    pm.write(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_test_midi()
