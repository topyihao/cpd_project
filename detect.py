import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

# Constants
SAMPLE_RATE = 22050

class PiczakCNN(nn.Module):
    def __init__(self):
        super(PiczakCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 80, kernel_size=(57, 6), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 3), stride=(1, 3))
        self.conv2 = nn.Conv2d(80, 80, kernel_size=(1, 3), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.fc1 = nn.Linear(80 * 3 * 1, 5000)  # Adjusted to match output size from conv layers
        self.fc2 = nn.Linear(5000, 5000)
        self.fc3 = nn.Linear(5000, 10)  # Change 10 to the number of classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def load_and_process_audio(file_path, sr=SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def extract_log_mel_spectrogram(y, sr, n_mels=60, n_fft=1024, hop_length=512):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

def split_spectrogram(spectrogram, segment_length, overlap):
    segments = []
    step = int(segment_length * (1 - overlap))
    for start in range(0, spectrogram.shape[1] - segment_length + 1, step):
        segment = spectrogram[:, start:start + segment_length]
        delta = librosa.feature.delta(segment)
        combined_segment = np.stack((segment, delta), axis=-1)
        segments.append(combined_segment)
    return np.array(segments)

def augment_audio(y, sr, num_augmentations=10):
    augmented_ys = [y]
    for _ in range(num_augmentations):
        shift = np.random.randint(sr * 0.5)  # Shift up to half a second
        augmented_y = np.roll(y, shift)
        augmented_ys.append(augmented_y)
        stretch = np.random.uniform(0.8, 1.2)
        pitch_shift = np.random.randint(-2, 2)
        augmented_y = librosa.effects.time_stretch(y=y, rate=stretch)
        augmented_y = librosa.effects.pitch_shift(y=augmented_y, sr=sr, n_steps=pitch_shift)
        augmented_ys.append(augmented_y)
    return augmented_ys

def prepare_data(directory, sr=SAMPLE_RATE, n_mels=60, n_fft=1024, hop_length=512, short_segment=41, long_segment=101, short_overlap=0.5, long_overlap=0.9):
    X_short = []
    X_long = [] 
    y_, sr = load_and_process_audio(directory, sr)
    augmentations = augment_audio(y_, sr)
    for aug_y in augmentations:
        log_mel_spectrogram = extract_log_mel_spectrogram(aug_y, sr, n_mels, n_fft, hop_length)
        short_segments = split_spectrogram(log_mel_spectrogram, short_segment, short_overlap)
        long_segments = split_spectrogram(log_mel_spectrogram, long_segment, long_overlap)
        X_short.extend(short_segments)
        X_long.extend(long_segments)
    X_short = np.array(X_short)
    X_long = np.array(X_long)
    return X_short, X_long

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X

def detect_sound(model_path, audio_path, label_to_int, int_to_label):
    # Load the model
    model = PiczakCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process the input audio file
    X_short, X_long = prepare_data(directory=audio_path)

    # Prepare the data for prediction
    segments = torch.tensor(X_long.transpose(0, 3, 1, 2), dtype=torch.float32).to(device)  # Shape: (batch_size, channels, height, width)

    # Perform prediction
    with torch.no_grad():
        outputs = model(segments)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_probabilities = probabilities.mean(dim=0)
        predicted_class = torch.argmax(predicted_probabilities).item()

    # Decode the prediction
    predicted_label = int_to_label[predicted_class]
    print(f"Predicted sound: {predicted_label}")

# Define label mapping
label_to_int = {
    'dog': 0,
    'rain': 1,
    'sea_waves': 2,
    'baby_cry': 3,
    'clock_tick': 4,
    'person_sneeze': 5,
    'helocopter': 6,
    'chainsaw': 7,
    'rooster': 8,
    'fire_crackling': 9
}
int_to_label = {v: k for k, v in label_to_int.items()}


model_path = 'model_fold_1.pth'  
audio_path = 'ESC-10/1/3-180256-A.ogg'  
detect_sound(model_path, audio_path, label_to_int, int_to_label) 
