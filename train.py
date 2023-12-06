import torch
import torch.nn as nn
import torch.optim as optim
from model_CNN import AudioDenoisingCNN
# from model_RNNs import AudioDenoisingRNN
from dataloader import AudioDataset  # Assuming you have a dataset class defined
from torch.utils.data import DataLoader
from utils import save_model, load_model  # Assuming these functions are defined in utils.py
import torch.nn.functional as F
from tqdm import tqdm

# Parameters
learning_rate = 0.001
batch_size = 16
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
noise_dir = 'data/noisy_train'
clear_dir = 'data/clean_train'
train_dataset = AudioDataset(noise_dir, clear_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = AudioDenoisingCNN().to(device)

# Loss and optimizer
criterion = nn.MSELoss()  # Mean Squared Error is common for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop with tqdm progress bar
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for i, (noisy, clear) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch+1}")

            noisy, clear = noisy.unsqueeze(1).to(device), clear.unsqueeze(1).to(device) #CNN

            # Forward pass
            outputs = model(noisy)

            # Resize output to match target tensor size
            outputs_resized = F.interpolate(outputs, size=(1025, 130), mode='bilinear', align_corners=False) #CNN


            # Calculate loss
            loss = criterion(outputs_resized, clear)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tepoch.set_postfix(loss=epoch_loss / (i + 1))

# Save the model after training
save_model(model, 'audio_denoising_model.pth')

