# Piano Hand Detection Model Implementation Plan

## Goal Description
Build a machine learning model to detect which hand (left or right) plays which notes in a piano solo MIDI file. The model will take a MIDI file as input and output hand assignments for each note.

## User Review Required
> [!IMPORTANT]
> **Dataset**: Real-world datasets like PIG require registration. I will provide a `dummy_data_generator.py` that creates synthetic data (splitting based on pitch) to demonstrate the pipeline. You will need to replace this with real labeled MIDI data for a production-quality model.

## Proposed Changes

### Data Processing
#### [MODIFY] [data_loader.py](file:///Users/henrywang/Library/CloudStorage/GoogleDrive-zixuan_wang5@brown.edu/My%20Drive/Courses/S3/CSCI%201470/BACHprop/data_loader.py)
- Functions to parse PIG dataset `.txt` files.
- Convert Pitch Names (e.g., "C4") to MIDI note numbers.
- Extract features: Pitch, Duration, Start Time, Velocity, Interval from previous note.
- **Change**: Return Numpy arrays or a `tf.data.Dataset` instead of a PyTorch Dataset.

#### [DELETE] dummy_data_generator.py
- No longer needed as we have the PIG dataset.

### Model Architecture
#### [MODIFY] [model.py](file:///Users/henrywang/Library/CloudStorage/GoogleDrive-zixuan_wang5@brown.edu/My%20Drive/Courses/S3/CSCI%201470/BACHprop/model.py)
- **Architecture**: Bidirectional LSTM (Bi-LSTM) using `tensorflow.keras`.
- **Input**: Sequence of notes (features: pitch, duration, dt, velocity).
- **Output**: Binary classification (0=Right, 1=Left) for each note.

### Training & Inference
#### [MODIFY] [train.py](file:///Users/henrywang/Library/CloudStorage/GoogleDrive-zixuan_wang5@brown.edu/My%20Drive/Courses/S3/CSCI%201470/BACHprop/train.py)
- Training loop using `model.fit()` from Keras.
- Loss function: BinaryCrossentropy.
- Saves the best model checkpoint (e.g., `.keras` or `.h5`).

#### [MODIFY] [predict.py](file:///Users/henrywang/Library/CloudStorage/GoogleDrive-zixuan_wang5@brown.edu/My%20Drive/Courses/S3/CSCI%201470/BACHprop/predict.py)
- Loads the trained Keras model.
- Processes a new MIDI file using `pretty_midi`.
- Outputs a new MIDI file with separated hands.

## Verification Plan
### Automated Tests
- Run `train.py` on the PIG dataset using TensorFlow.
- Run `predict.py` on a sample MIDI file.

### Manual Verification
- Inspect the output MIDI from `predict.py` to ensure hands are split correctly.
