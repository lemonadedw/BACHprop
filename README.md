# BACHprop

## Data and Preprocessing

### Datasets

**PianoFingeringDataset_v1.2**
- Reference dataset with expert-annotated piano fingering
- Location: `data/PianoFingeringDataset_v1.2/FingeringFiles/`
- Contains fingering files in tab-delimited format with timing, pitch, velocity, hand, and finger annotations

**POP909-Dataset**
- Collection of 909 popular piano songs
- Location: `data/POP909-Dataset/POP909/`
- Each song contains:
  - Main MIDI file (e.g., `001/001.mid`)
  - Multiple tracks: MELODY, BRIDGE, and PIANO
  - Beat and chord annotations in text files

### Preprocessing Pipeline

Convert POP909 MIDI files to fingering format:

```bash
python3 preprocess.py
```

**Processing Steps:**

1. **Track Selection**: Extracts and merges MELODY and BRIDGE tracks only (PIANO track is excluded)
2. **Timing Extraction**: Converts MIDI ticks to absolute timestamps in seconds
3. **Note Events**: Extracts note-on/note-off events with pitch, velocity, and duration
4. **Hand Assignment**: 
   - Right hand (0): Notes >= C4 (MIDI note 60)
   - Left hand (1): Notes < C4
5. **Finger Assignment**:
   - Right hand: Fingers 1-5 (thumb to pinky) based on relative pitch
   - Left hand: Fingers -1 to -5 (negative notation)
6. **Output Format**: Generates tab-delimited fingering files matching PianoFingeringDataset format

**Output:**
- Directory: `data/POP909-Dataset/FingeringFiles/`
- Files: `{song_id}_fingering.txt` (e.g., `001_fingering.txt`)
- Format: `index  start_time  end_time  note_name  64  velocity  hand  finger`

### Training

Train the model using both datasets:

**Simple Hand Classification Model:**
```bash
python3 train.py
```

**Multi-Task Model (Hand + Next Pitch Prediction):**
```bash
python3 train.py --multitask              # LSTM model
python3 train.py --multitask --transformer # Transformer model
```

**Note:** Run preprocessing first to generate POP909 fingering files before training.

The training script:
- Loads data from both PianoFingeringDataset and POP909 datasets
- Combines ~150 expert-annotated files + 909 POP909 files = ~1059 total files
- Splits data 80/20 for training and validation (by file to prevent leakage)
- Simple model: Predicts hand assignment (left/right) for piano notes
- Multi-task model: Predicts both hand assignment AND next note's pitch

### Interactive Keyboard

Play piano on your keyboard and let AI predict what comes next:

```bash
# Interactive keyboard (auto-selects available model)
python3 interactive_keyboard.py

# Choose specific model
python3 interactive_keyboard.py --lstm
python3 interactive_keyboard.py --transformer

# MIDI file prediction
python3 quick_predict.py static/bach.mid
```

All controls and instructions are shown in the GUI.