import torch
import librosa
import numpy as np
import os
import soundfile as sf
from model_CNN import AudioDenoisingCNN  # Import your model definition

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioDenoisingCNN().to(device)
model.load_state_dict(torch.load('audio_denoising_model.pth', map_location=device))
model.eval()


def preprocess_audio(file_path, sr=22050, n_fft=2048, hop_length=512):
    # Load and convert the audio to a spectrogram
    audio, _ = librosa.load(file_path, sr=sr)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrogram = np.abs(stft)
    log_spectrogram = librosa.amplitude_to_db(spectrogram)
    return torch.tensor(log_spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)


def postprocess_audio(spectrogram, hop_length=512):
    # Convert the spectrogram back to an audio waveform
    spectrogram = librosa.db_to_amplitude(spectrogram.squeeze().cpu().numpy())
    return librosa.istft(spectrogram, hop_length=hop_length)


# Directory containing noisy test audio files
noisy_test_dir = './data/noisy_test'

# Process each file in the directory
for filename in os.listdir(noisy_test_dir):
    if filename.endswith('.wav'):  # or .mp3 or whatever format your files are in
        file_path = os.path.join(noisy_test_dir, filename)

        # Preprocess the audio
        preprocessed_audio = preprocess_audio(file_path)

        # Run the model
        with torch.no_grad():
            denoised_spectrogram = model(preprocessed_audio)

        # Postprocess the output
        denoised_audio = postprocess_audio(denoised_spectrogram)

        # Save the denoised audio
        output_path = os.path.join('./data/clean_test', 'denoised_' + filename)
        sf.write(output_path, denoised_audio, samplerate=22050)
