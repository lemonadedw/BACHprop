#!/usr/bin/env python3
"""
Interactive Piano Keyboard - Play notes and predict what comes next!
Uses pygame for keyboard interface and audio playback.
"""

import pygame
import numpy as np
import tensorflow as tf
import os
import sys
from collections import deque
import time

# Piano synthesizer using pygame
class PianoSynthesizer:
    def __init__(self, sample_rate=22050):
        pygame.mixer.init(frequency=sample_rate, channels=1)
        self.sample_rate = sample_rate
        self.note_cache = {}
    
    def midi_to_freq(self, midi_note):
        """Convert MIDI note number to frequency in Hz"""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    def generate_note(self, midi_note, duration=0.5, volume=0.3):
        """Generate a piano-like sound for a MIDI note"""
        if midi_note in self.note_cache:
            return self.note_cache[midi_note]
        
        freq = self.midi_to_freq(midi_note)
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Piano-like sound using multiple harmonics with ADSR envelope
        wave = np.zeros(num_samples)
        
        # Fundamental and harmonics
        harmonics = [1.0, 0.5, 0.25, 0.125, 0.0625]
        for i, amp in enumerate(harmonics):
            wave += amp * np.sin(2 * np.pi * freq * (i + 1) * t)
        
        # ADSR envelope
        attack = int(0.01 * self.sample_rate)
        decay = int(0.1 * self.sample_rate)
        release = int(0.2 * self.sample_rate)
        
        envelope = np.ones(num_samples)
        # Attack
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        # Decay
        if decay > 0:
            envelope[attack:attack+decay] = np.linspace(1, 0.7, decay)
        # Sustain (stays at 0.7)
        # Release
        envelope[-release:] = np.linspace(0.7, 0, release)
        
        wave = wave * envelope * volume
        
        # Convert to 16-bit integers
        wave = np.int16(wave * 32767)
        
        # Create stereo
        stereo_wave = np.column_stack((wave, wave))
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        self.note_cache[midi_note] = sound
        return sound
    
    def play_note(self, midi_note, duration=0.5, volume=0.3):
        """Play a note immediately"""
        sound = self.generate_note(midi_note, duration, volume)
        sound.play()
    
    def play_sequence(self, notes, dt=0.3):
        """Play a sequence of notes with timing"""
        for note_info in notes:
            if isinstance(note_info, (int, float)):
                midi_note = int(note_info)
                duration = dt
            else:
                midi_note = int(note_info[0])
                duration = note_info[1] if len(note_info) > 1 else dt
            
            self.play_note(midi_note, duration)
            time.sleep(duration * 0.8)  # Slight overlap


class InteractivePianoApp:
    def __init__(self, model_path="best_model_multitask.keras"):
        # Initialize pygame
        pygame.init()
        
        # Screen settings
        self.width = 1600
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Interactive Piano Predictor 🎹")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.DARK_GRAY = (100, 100, 100)
        self.BLUE = (100, 150, 255)
        self.RED = (255, 100, 100)
        self.GREEN = (100, 255, 100)
        self.YELLOW = (255, 255, 100)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Piano keyboard settings (5 octaves: C3 to C8)
        self.start_note = 48  # C3
        self.num_octaves = 5
        self.num_white_keys = self.num_octaves * 7 + 1  # 36 white keys (5 octaves + C)
        self.white_key_width = 40
        self.white_key_height = 200
        self.black_key_width = 25
        self.black_key_height = 120
        
        # Keyboard offset
        self.keyboard_x = 50
        self.keyboard_y = 450
        
        # Key rectangles for mouse detection
        self.key_rects = {}  # midi_note -> pygame.Rect
        
        # Note patterns (which notes are black keys)
        self.black_key_pattern = [1, 3, 6, 8, 10]  # C#, D#, F#, G#, A#
        
        # Computer keyboard mapping to piano keys
        self.key_mapping = {
            pygame.K_a: 60,  # C4
            pygame.K_w: 61,  # C#4
            pygame.K_s: 62,  # D4
            pygame.K_e: 63,  # D#4
            pygame.K_d: 64,  # E4
            pygame.K_f: 65,  # F4
            pygame.K_t: 66,  # F#4
            pygame.K_g: 67,  # G4
            pygame.K_y: 68,  # G#4
            pygame.K_h: 69,  # A4
            pygame.K_u: 70,  # A#4
            pygame.K_j: 71,  # B4
            pygame.K_k: 72,  # C5
            pygame.K_o: 73,  # C#5
            pygame.K_l: 74,  # D5
        }
        
        # State
        self.recorded_notes = []
        self.predicted_notes = []
        self.num_predictions = 20
        self.is_recording = False
        self.is_playing = False
        
        # Synth
        self.synth = PianoSynthesizer()
        
        # Model
        self.model = None
        self.sequence_length = 50
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'multitask_loss': lambda y_true, y_pred: tf.reduce_mean(y_pred),
                    'hand_accuracy': lambda y_true, y_pred: tf.reduce_mean(y_pred),
                    'pitch_accuracy': lambda y_true, y_pred: tf.reduce_mean(y_pred)
                }
            )
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model not found at {model_path}")
            print("Predictions will not be available.")
    
    def midi_to_key_position(self, midi_note):
        """Get the visual position of a piano key"""
        relative_note = midi_note - self.start_note
        octave = relative_note // 12
        note_in_octave = relative_note % 12
        
        # Calculate white key position
        white_key_positions = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]
        white_pos = octave * 7 + white_key_positions[note_in_octave]
        
        is_black = note_in_octave in self.black_key_pattern
        
        return white_pos, is_black
    
    def draw_piano_key(self, midi_note, pressed=False, color=None):
        """Draw a single piano key"""
        white_pos, is_black = self.midi_to_key_position(midi_note)
        
        if is_black:
            x = self.keyboard_x + white_pos * self.white_key_width + self.white_key_width - self.black_key_width // 2
            y = self.keyboard_y
            width = self.black_key_width
            height = self.black_key_height
            key_color = self.DARK_GRAY if pressed else self.BLACK
            if color:
                key_color = color
        else:
            x = self.keyboard_x + white_pos * self.white_key_width
            y = self.keyboard_y
            width = self.white_key_width
            height = self.white_key_height
            key_color = self.GRAY if pressed else self.WHITE
            if color:
                key_color = color
        
        pygame.draw.rect(self.screen, key_color, (x, y, width, height))
        pygame.draw.rect(self.screen, self.BLACK, (x, y, width, height), 2)
        
        return pygame.Rect(x, y, width, height)
    
    def draw_piano_keyboard(self, pressed_notes=set()):
        """Draw the entire piano keyboard and store key rectangles for mouse detection"""
        self.key_rects = {}  # Clear previous rectangles
        
        # Draw white keys first
        max_note = self.start_note + self.num_octaves * 12
        for i in range(max_note - self.start_note + 1):
            midi_note = self.start_note + i
            if midi_note % 12 not in self.black_key_pattern:
                is_pressed = midi_note in pressed_notes
                rect = self.draw_piano_key(midi_note, pressed=is_pressed)
                self.key_rects[midi_note] = rect
        
        # Draw black keys on top
        for i in range(max_note - self.start_note + 1):
            midi_note = self.start_note + i
            if midi_note % 12 in self.black_key_pattern:
                is_pressed = midi_note in pressed_notes
                rect = self.draw_piano_key(midi_note, pressed=is_pressed)
                self.key_rects[midi_note] = rect
    
    def draw_note_display(self):
        """Draw the recorded and predicted notes"""
        y = 50
        
        # Title
        title = self.font_large.render("🎹 Interactive Piano Predictor", True, self.BLACK)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, y))
        y += 60
        
        # Status
        if self.is_recording:
            status = self.font_medium.render("🔴 RECORDING", True, self.RED)
        elif self.is_playing:
            status = self.font_medium.render("▶️  PLAYING", True, self.GREEN)
        else:
            status = self.font_medium.render("⏸️  READY", True, self.BLUE)
        self.screen.blit(status, (50, y))
        y += 40
        
        # Recorded notes
        recorded_text = f"Recorded: {len(self.recorded_notes)} notes"
        text = self.font_small.render(recorded_text, True, self.BLACK)
        self.screen.blit(text, (50, y))
        y += 30
        
        # Show last few recorded notes
        if self.recorded_notes:
            notes_str = ", ".join([str(int(n[0] * 128)) for n in self.recorded_notes[-10:]])
            if len(self.recorded_notes) > 10:
                notes_str = "..." + notes_str
            text = self.font_small.render(f"  {notes_str}", True, self.DARK_GRAY)
            self.screen.blit(text, (50, y))
        y += 40
        
        # Predicted notes
        if self.predicted_notes:
            predicted_text = f"Predicted: {len(self.predicted_notes)} notes"
            text = self.font_small.render(predicted_text, True, self.BLUE)
            self.screen.blit(text, (50, y))
            y += 30
            
            notes_str = ", ".join([str(int(n)) for n in self.predicted_notes[:10]])
            if len(self.predicted_notes) > 10:
                notes_str += "..."
            text = self.font_small.render(f"  {notes_str}", True, self.DARK_GRAY)
            self.screen.blit(text, (50, y))
    
    def draw_controls(self):
        """Draw control instructions"""
        y = self.keyboard_y + self.white_key_height + 30
        
        # Title
        title = self.font_medium.render("HOW TO USE:", True, self.BLACK)
        self.screen.blit(title, (50, y))
        y += 35
        
        # Instructions
        instructions = [
            "1. Press SPACE to start recording",
            "2. Play notes with A-L keys OR click piano keys with mouse",
            "3. Press SPACE to stop recording",
            "4. Press ENTER to predict next notes",
            "5. Press P to play back everything!",
        ]
        
        for instruction in instructions:
            text = self.font_small.render(instruction, True, self.DARK_GRAY)
            self.screen.blit(text, (50, y))
            y += 25
        
        y += 15
        
        # Controls reference
        controls = [
            f"+/- = Adjust predictions (current: {self.num_predictions})",
            "C = Clear all  |  ESC = Quit"
        ]
        
        for control in controls:
            text = self.font_small.render(control, True, self.DARK_GRAY)
            self.screen.blit(text, (50, y))
            y += 25
    
    def play_note(self, midi_note):
        """Play a note and add to recording if active"""
        self.synth.play_note(midi_note, duration=0.5)
        
        if self.is_recording:
            # Create feature vector [pitch, duration, dt, velocity]
            note_features = np.array([
                midi_note / 128.0,  # Normalized pitch
                0.5,  # Default duration
                0.3,  # Default dt
                0.7   # Default velocity
            ], dtype=np.float32)
            self.recorded_notes.append(note_features)
    
    def predict_next_notes(self):
        """Use the model to predict the next N notes"""
        if self.model is None:
            print("No model loaded!")
            return
        
        if len(self.recorded_notes) == 0:
            print("No notes recorded!")
            return
        
        print(f"\nPredicting {self.num_predictions} notes...")
        
        # Prepare sequence
        notes_array = np.array(self.recorded_notes)
        
        if len(notes_array) < self.sequence_length:
            # Pad with zeros
            padding = np.zeros((self.sequence_length - len(notes_array), 4), dtype=np.float32)
            sequence = np.vstack([padding, notes_array])
        else:
            # Take last sequence_length notes
            sequence = notes_array[-self.sequence_length:]
        
        predicted_pitches = []
        
        # Predict iteratively
        for i in range(self.num_predictions):
            input_seq = np.array([sequence])
            prediction = self.model.predict(input_seq, verbose=0)
            
            # Get prediction for last timestep
            # prediction shape: (1, seq_len, 89)
            # [:, :, 0] = hand classification
            # [:, :, 1:89] = pitch one-hot (88 classes)
            last_pred = prediction[0, -1, :]
            
            # Get pitch prediction (indices 1:89 are the one-hot encoding)
            pitch_logits = last_pred[1:]  # 88 values
            predicted_pitch_index = np.argmax(pitch_logits)  # 0-87
            
            # Convert to MIDI note (piano range: 21-108)
            predicted_midi = predicted_pitch_index + 21
            predicted_pitches.append(predicted_midi)
            
            # Normalize pitch for next iteration
            next_pitch_normalized = predicted_midi / 128.0
            
            # Update sequence for next prediction
            next_note = np.array([
                next_pitch_normalized,
                0.5, 0.3, 0.7
            ], dtype=np.float32)
            sequence = np.vstack([sequence[1:], next_note])
        
        self.predicted_notes = predicted_pitches
        print(f"Predicted: {predicted_pitches}")
    
    def playback(self):
        """Play back recorded notes + predictions"""
        if len(self.recorded_notes) == 0 and len(self.predicted_notes) == 0:
            print("Nothing to play!")
            return
        
        self.is_playing = True
        print("\nPlaying back...")
        
        # Play recorded notes
        for note in self.recorded_notes:
            if not self.is_playing:
                break
            midi_note = int(note[0] * 128)
            self.synth.play_note(midi_note, duration=0.4)
            
            # Update display
            self.screen.fill(self.WHITE)
            self.draw_piano_keyboard(pressed_notes={midi_note})
            self.draw_note_display()
            self.draw_controls()
            pygame.display.flip()
            
            time.sleep(0.3)
        
        # Visual separator
        time.sleep(0.5)
        
        # Play predicted notes
        for midi_note in self.predicted_notes:
            if not self.is_playing:
                break
            self.synth.play_note(midi_note, duration=0.4)
            
            # Update display with different color
            self.screen.fill(self.WHITE)
            self.draw_piano_keyboard(pressed_notes={midi_note})
            self.draw_note_display()
            self.draw_controls()
            
            # Draw "PREDICTION" indicator
            pred_text = self.font_large.render("🔮 PREDICTION", True, self.BLUE)
            self.screen.blit(pred_text, (self.width // 2 - pred_text.get_width() // 2, 150))
            
            pygame.display.flip()
            time.sleep(0.3)
        
        self.is_playing = False
        print("Playback complete!")
    
    def get_clicked_key(self, mouse_pos):
        """Get the MIDI note of the key clicked at mouse position"""
        # Check black keys first (they're on top)
        for midi_note, rect in self.key_rects.items():
            if midi_note % 12 in self.black_key_pattern:
                if rect.collidepoint(mouse_pos):
                    return midi_note
        
        # Then check white keys
        for midi_note, rect in self.key_rects.items():
            if midi_note % 12 not in self.black_key_pattern:
                if rect.collidepoint(mouse_pos):
                    return midi_note
        
        return None
    
    def run(self):
        """Main application loop"""
        clock = pygame.time.Clock()
        running = True
        pressed_keys = set()
        mouse_pressed_key = None
        
        print("\n🎹 Interactive Piano Predictor")
        print("All instructions are shown in the GUI window!")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    # Piano keys
                    if event.key in self.key_mapping and not self.is_playing:
                        midi_note = self.key_mapping[event.key]
                        pressed_keys.add(midi_note)
                        self.play_note(midi_note)
                    
                    # Control keys
                    elif event.key == pygame.K_SPACE:
                        self.is_recording = not self.is_recording
                        print(f"Recording: {'ON' if self.is_recording else 'OFF'}")
                    
                    elif event.key == pygame.K_RETURN:
                        if not self.is_playing:
                            self.predict_next_notes()
                    
                    elif event.key == pygame.K_p:
                        if not self.is_playing:
                            self.playback()
                    
                    elif event.key == pygame.K_c:
                        self.recorded_notes = []
                        self.predicted_notes = []
                        print("Cleared all notes")
                    
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.num_predictions = min(50, self.num_predictions + 5)
                        print(f"Predictions: {self.num_predictions}")
                    
                    elif event.key == pygame.K_MINUS:
                        self.num_predictions = max(5, self.num_predictions - 5)
                        print(f"Predictions: {self.num_predictions}")
                    
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                
                elif event.type == pygame.KEYUP:
                    if event.key in self.key_mapping:
                        midi_note = self.key_mapping[event.key]
                        pressed_keys.discard(midi_note)
                
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.is_playing:
                    # Mouse click on piano key
                    mouse_pos = pygame.mouse.get_pos()
                    clicked_key = self.get_clicked_key(mouse_pos)
                    if clicked_key is not None:
                        mouse_pressed_key = clicked_key
                        pressed_keys.add(clicked_key)
                        self.play_note(clicked_key)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    # Release mouse key
                    if mouse_pressed_key is not None:
                        pressed_keys.discard(mouse_pressed_key)
                        mouse_pressed_key = None
            
            # Draw everything
            self.screen.fill(self.WHITE)
            self.draw_piano_keyboard(pressed_notes=pressed_keys)
            self.draw_note_display()
            self.draw_controls()
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS
        
        pygame.quit()
        print("\nGoodbye! 👋")


def main():
    import sys
    
    # Check command line arguments
    model_path = None
    if len(sys.argv) > 1:
        if sys.argv[1] == "--transformer":
            model_path = "best_model_multitask_transformer.keras"
        elif sys.argv[1] == "--lstm":
            model_path = "best_model_multitask.keras"
        elif sys.argv[1] == "--help":
            print("Interactive Piano Keyboard")
            print("\nUsage:")
            print("  python interactive_keyboard.py [--lstm|--transformer]")
            print("\nOptions:")
            print("  --lstm        Use LSTM model (default)")
            print("  --transformer Use Transformer model")
            print("  --help        Show this help message")
            return
        else:
            model_path = sys.argv[1]  # Custom path
    else:
        # Default: try LSTM first, then transformer
        if os.path.exists("best_model_multitask.keras"):
            model_path = "best_model_multitask.keras"
        elif os.path.exists("best_model_multitask_transformer.keras"):
            model_path = "best_model_multitask_transformer.keras"
        else:
            model_path = "best_model_multitask.keras"  # Default even if doesn't exist
    
    if not os.path.exists(model_path):
        print("\n⚠️  WARNING: Model not found!")
        print(f"Model: {model_path}")
        print("\nTrain a model first:")
        print("  python train.py --multitask              # LSTM model")
        print("  python train.py --multitask --transformer # Transformer model")
        print("\nContinuing without predictions (keyboard only)...\n")
    else:
        print(f"\n✓ Using model: {model_path}\n")
    
    app = InteractivePianoApp(model_path)
    app.run()


if __name__ == "__main__":
    main()

