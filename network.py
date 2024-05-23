import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
SAMPLE_RATE = 22050

# Dataset class
class SoundDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Ensure the data is in the correct shape (channels, height, width)
        sample = self.data[idx].transpose(2, 0, 1)  # from (height, width, channels) to (channels, height, width)
        return torch.tensor(sample, dtype=torch.float32), self.labels[idx]

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
    y = []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                y_, sr = load_and_process_audio(file_path, sr)
                augmentations = augment_audio(y_, sr)
                for aug_y in augmentations:
                    log_mel_spectrogram = extract_log_mel_spectrogram(aug_y, sr, n_mels, n_fft, hop_length)
                    short_segments = split_spectrogram(log_mel_spectrogram, short_segment, short_overlap)
                    long_segments = split_spectrogram(log_mel_spectrogram, long_segment, long_overlap)
                    X_short.extend(short_segments)
                    X_long.extend(long_segments)
                    y.extend([label] * len(short_segments))
    X_short = np.array(X_short)
    X_long = np.array(X_long)
    y = np.array(y)
    return X_short, X_long, y

def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X

# Define the CNN model
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
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=300):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    return model

# Save the model
def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def cross_validate(model_class, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    accuracies = []
    confusion_matrices = []
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_loader = create_dataloaders(X_train, y_train)
        val_loader = create_dataloaders(X_val, y_val)

        model = model_class().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9, weight_decay=0.001, nesterov=True)
        model = train_model(model, train_loader, criterion, optimizer, num_epochs=300)

        preds, true_labels = evaluate_model(model, val_loader)
        accuracy = accuracy_score(true_labels, preds)
        cm = confusion_matrix(true_labels, preds)
        accuracies.append(accuracy)
        confusion_matrices.append(cm)

        # Save the model for each fold
        save_model(model, f'model_fold_{fold+1}.pth')

    return accuracies, confusion_matrices

# Function to create dataloaders
def create_dataloaders(X, y, batch_size=1000, num_workers=2):  # Batch size adjusted to 1000
    dataset = SoundDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
directory = "./ESC-10"
X_short, X_long, y = prepare_data(directory)
X_short = normalize_data(X_short)
X_long = normalize_data(X_long)

# Convert labels to integers
label_to_int = {label: idx for idx, label in enumerate(np.unique(y))}
y = np.array([label_to_int[label] for label in y])

# Perform 5-fold cross-validation
accuracies, confusion_matrices = cross_validate(PiczakCNN, X_short, y, n_splits=5)

# Print accuracies
print(f"Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies)}")

# Plot confusion matrices
class_names = list(label_to_int.keys())
for cm in confusion_matrices:
    plot_confusion_matrix(cm, class_names)