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
        self.file_indices = []  # Track which file each sequence came from
        self._load_data()

    def _load_data(self):
        print(f"Loading data from {len(self.file_paths)} files...")
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                notes, hands = self._parse_file(file_path)
                if len(notes) > self.sequence_length:
                    # Create sequences
                    for i in range(0, len(notes) - self.sequence_length, self.sequence_length):
                        seq_notes = notes[i:i+self.sequence_length]
                        seq_hands = hands[i:i+self.sequence_length]
                        self.data.append(seq_notes)
                        self.labels.append(seq_hands)
                        self.file_indices.append(file_idx)  # Track which file this came from
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.file_indices = np.array(self.file_indices, dtype=np.int32)
        
        # Expand dims for labels to be (batch, seq, 1) if needed, or just (batch, seq)
        # For TimeDistributed(Dense(1)), we usually want (batch, seq, 1)
        self.labels = np.expand_dims(self.labels, axis=-1)
        
        print(f"Created {len(self.data)} sequences from {len(self.file_paths)} files.")
        print(f"Data shape: {self.data.shape}")
        print(f"Labels shape: {self.labels.shape}")
    
    def get_multitask_labels(self):
        """
        Create multi-task labels: [hand, next_pitch]
        Returns labels of shape (batch, seq, 2) where:
        - labels[:, :, 0] = hand classification (0=right, 1=left)
        - labels[:, :, 1] = next note's pitch (0-1 normalized)
        """
        batch_size, seq_len, _ = self.labels.shape
        multitask_labels = np.zeros((batch_size, seq_len, 2), dtype=np.float32)
        
        # Hand labels (already have these)
        multitask_labels[:, :, 0] = self.labels[:, :, 0]
        
        # Next pitch labels - extract from the data (pitch is first feature)
        # For each note at position i, the target is the pitch of note at i+1
        for i in range(batch_size):
            for j in range(seq_len - 1):
                # Next pitch is already normalized in data (pitch / 128.0)
                multitask_labels[i, j, 1] = self.data[i, j+1, 0]
            # For last note in sequence, use same pitch as target (or could use 0)
            multitask_labels[i, seq_len-1, 1] = self.data[i, seq_len-1, 0]
        
        return multitask_labels

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
    
    def get_data_with_file_indices(self):
        """Return data with file indices to enable file-level splitting"""
        return self.data, self.labels, self.file_indices, len(self.file_paths)

if __name__ == "__main__":
    # Test the loader
    loader = PIGDataLoader("PianoFingeringDataset_v1.2/FingeringFiles", sequence_length=50)
    X, y = loader.get_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
