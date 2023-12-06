import os
import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, noise_dir, clear_dir, target_length=3, sampling_rate=22050):
        """
        Initialize the dataset with paths and configurations.

        Args:
        - noise_dir (str): Path to the directory with noisy audio files.
        - clear_dir (str): Path to the directory with clean audio files.
        - target_length (int): Target length of audio clips in seconds.
        - sampling_rate (int): Sampling rate of the audio files.
        """
        self.noise_dir = noise_dir
        self.clear_dir = clear_dir
        self.target_length = target_length
        self.sampling_rate = sampling_rate
        self.max_length = self.target_length * self.sampling_rate

        self.noise_files = os.listdir(noise_dir)
        self.clear_files = os.listdir(clear_dir)

    def __len__(self):
        return min(len(self.noise_files), len(self.clear_files))

    def __getitem__(self, idx):
        noise_file_path = os.path.join(self.noise_dir, self.noise_files[idx])
        clear_file_path = os.path.join(self.clear_dir, self.clear_files[idx])

        noise_audio, _ = librosa.load(noise_file_path, sr=self.sampling_rate)
        clear_audio, _ = librosa.load(clear_file_path, sr=self.sampling_rate)

        padded_noise_audio = self.pad_or_truncate_audio(noise_audio)
        padded_clear_audio = self.pad_or_truncate_audio(clear_audio)

        # Convert audio waveforms to spectrograms
        noise_spectrogram = self.audio_to_spectrogram(padded_noise_audio)
        clear_spectrogram = self.audio_to_spectrogram(padded_clear_audio)

        return noise_spectrogram, clear_spectrogram

    def audio_to_spectrogram(self, audio):
        # Convert audio to a spectrogram
        stft = librosa.stft(audio, n_fft=2048, hop_length=512, win_length=2048)
        spectrogram = np.abs(stft)
        return librosa.amplitude_to_db(spectrogram).astype(np.float32)

    def pad_or_truncate_audio(self, audio):
        """
        Pad or truncate the audio to the max_length.

        Args:
        - audio (numpy.ndarray): Audio time-series.

        Returns:
        - numpy.ndarray: Padded or truncated audio time-series.
        """
        if len(audio) > self.max_length:
            return audio[:self.max_length]
        elif len(audio) < self.max_length:
            padding = self.max_length - len(audio)
            return np.pad(audio, (0, padding), mode='constant', constant_values=(0, 0))
        return audio


# Usage
noise_dir = './data/noisy_train'
clear_dir = './data/clean_train'
dataset = AudioDataset(noise_dir, clear_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Example of iterating over the dataloader
for noise, clear in dataloader:
    print(noise.shape, clear.shape)  # Example processing
