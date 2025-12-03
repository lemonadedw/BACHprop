import os
import glob
import pretty_midi
import numpy as np

class PIGDataLoader:
    def __init__(self, data_dir, sequence_length=50):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*_fingering.txt")))
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        print(f"Loading data from {len(self.file_paths)} files...")
        for file_path in self.file_paths:
            try:
                notes, hands = self._parse_file(file_path)
                if len(notes) > self.sequence_length:
                    # Create sequences
                    for i in range(0, len(notes) - self.sequence_length, self.sequence_length):
                        seq_notes = notes[i:i+self.sequence_length]
                        seq_hands = hands[i:i+self.sequence_length]
                        self.data.append(seq_notes)
                        self.labels.append(seq_hands)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32) # Keras likes float labels for BCE usually, or int
        # Expand dims for labels to be (batch, seq, 1) if needed, or just (batch, seq)
        # For TimeDistributed(Dense(1)), we usually want (batch, seq, 1)
        self.labels = np.expand_dims(self.labels, axis=-1)
        
        print(f"Created {len(self.data)} sequences.")
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")

    def _parse_file(self, file_path):
        notes = []
        hands = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header if present
        start_idx = 0
        if lines[0].startswith('//'):
            start_idx = 1
            
        prev_onset = 0.0
        
        for line in lines[start_idx:]:
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
                
            try:
                onset = float(parts[1])
                offset = float(parts[2])
                pitch_name = parts[3]
                velocity = int(parts[4])
                hand_channel = int(parts[6]) # 0 = Right, 1 = Left
                
                pitch = pretty_midi.note_name_to_number(pitch_name)
                duration = offset - onset
                dt = onset - prev_onset
                prev_onset = onset
                
                # Features: [pitch, duration, dt, velocity]
                features = [
                    pitch / 128.0,
                    duration,
                    dt,
                    velocity / 127.0
                ]
                
                notes.append(features)
                hands.append(hand_channel)
                
            except ValueError:
                continue
                
        return np.array(notes, dtype=np.float32), np.array(hands, dtype=np.int64)

    def get_data(self):
        return self.data, self.labels

if __name__ == "__main__":
    # Test the loader
    loader = PIGDataLoader("PianoFingeringDataset_v1.2/FingeringFiles", sequence_length=50)
    X, y = loader.get_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
