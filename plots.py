import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


# Function to load and return waveform, spectrogram, mel spectrogram, and Fourier Transform data
def analyze_audio(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Waveform data
    waveform_data = (np.arange(len(y)) / sr, y)

    # Spectrogram data
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    spectrogram_data = (D, sr)

    # Mel spectrogram data
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    mel_spectrogram_data = (S_dB, sr)

    # Fourier Transform data
    n = len(y)
    T = 1.0 / sr
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(n, T)[: n // 2]
    fourier_transform_data = (xf, 2.0 / n * np.abs(yf[: n // 2]))

    return waveform_data, spectrogram_data, mel_spectrogram_data, fourier_transform_data


sound_files = os.listdir("./ESC-10")
all_data = []

# Collect data for all files
for folder in sound_files:
    f = os.listdir("./ESC-10/" + folder)
    for file in f:
        all_data.append(
            (folder, file, analyze_audio("./ESC-10/" + folder + "/" + file))
        )

# Determine the layout of the subplots
num_files = len(all_data)
cols = 4
rows = (num_files + cols - 1) // cols  # Calculate number of rows needed

fig, axes = plt.subplots(rows, cols, figsize=(16, 9 * rows))

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, (folder, file, data) in enumerate(all_data):
    waveform_data, spectrogram_data, mel_spectrogram_data, fourier_transform_data = data

    # Plot waveform
    axes[4 * i].plot(waveform_data[0], waveform_data[1])
    axes[4 * i].set_title(f"Waveform: {folder}/{file}")
    axes[4 * i].set_xlabel("Time (s)")
    axes[4 * i].set_ylabel("Amplitude")

    # Plot spectrogram
    librosa.display.specshow(
        spectrogram_data[0],
        sr=spectrogram_data[1],
        x_axis="time",
        y_axis="log",
        ax=axes[4 * i + 1],
    )
    axes[4 * i + 1].set_title(f"Spectrogram: {folder}/{file}")
    fig.colorbar(axes[4 * i + 1].images[0], ax=axes[4 * i + 1], format="%+2.0f dB")

    # Plot mel spectrogram
    librosa.display.specshow(
        mel_spectrogram_data[0],
        sr=mel_spectrogram_data[1],
        x_axis="time",
        y_axis="mel",
        ax=axes[4 * i + 2],
    )
    axes[4 * i + 2].set_title(f"Mel Spectrogram: {folder}/{file}")
    fig.colorbar(axes[4 * i + 2].images[0], ax=axes[4 * i + 2], format="%+2.0f dB")

    # Plot Fourier Transform
    axes[4 * i + 3].plot(fourier_transform_data[0], fourier_transform_data[1])
    axes[4 * i + 3].set_title(f"Fourier Transform: {folder}/{file}")
    axes[4 * i + 3].set_xlabel("Frequency (Hz)")
    axes[4 * i + 3].set_ylabel("Amplitude")
    axes[4 * i + 3].grid()

# Hide any unused subplots
for j in range(4 * num_files, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()
