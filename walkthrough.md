# Piano Hand Detection Walkthrough (TensorFlow Version)

I have successfully built and trained a model to detect which hand plays which notes in a piano MIDI file, using **TensorFlow/Keras**.

## Accomplishments
- **Data Loading**: Implemented `data_loader.py` to parse the PIG dataset and return Numpy arrays compatible with Keras.
- **Model**: Built a Bidirectional LSTM model in `model.py` using `tensorflow.keras`.
- **Training**: Trained the model for 20 epochs in `train.py`, achieving **~95% validation accuracy**.
- **Inference**: Created `predict.py` to take any MIDI file and output a new MIDI file with notes separated into "Right Hand" and "Left Hand" tracks.

## Verification Results
I verified the pipeline by:
1. **Training**: The model converged well.
   ```
   Epoch 20/20
   46/46 ━━━━━━━━━━━━━━━━━━━━ 1s 17ms/step - accuracy: 0.9831 - loss: 0.0535 - val_accuracy: 0.9495 - val_loss: 0.1513
   ```
2. **Prediction**: I generated a simple test MIDI (`test_input.mid`) and ran prediction.
   - Running `predict.py` successfully split the notes.
   - Output saved to `test_output_tf.mid`.

## How to Use
1. **Activate Environment** (Python 3.13):
   ```bash
   source .venv/bin/activate
   ```
2. **Train Model** (Optional, pretrained model `best_model.keras` is already saved):
   ```bash
   python train.py
   ```
3. **Run Prediction**:
   ```bash
   python predict.py path/to/your_song.mid --output split_song.mid
   ```
