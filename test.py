import librosa
import matplotlib.pyplot as plt
import numpy as np


# Function to load and display waveform and spectrogram
def analyze_audio(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Display waveform
    axs[0].plot(np.arange(len(y)) / sr, y)
    axs[0].set_title(f"Waveform: {file_path}")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    # Display spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=axs[1])
    axs[1].set_title(f"Spectrogram: {file_path}")
    fig.colorbar(img, ax=axs[1], format="%+2.0f dB")

    # Adjust layout
    plt.tight_layout()
    plt.show()


# Example usage
analyze_audio("dog.ogg")
