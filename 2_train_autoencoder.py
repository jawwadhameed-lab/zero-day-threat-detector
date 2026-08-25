import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import os

# 1. Load Preprocessed Safe Data
print("--- Loading Normal Training Data ---")
X_train_normal = np.load('data/X_train_normal.npy')

# PyTorch Tensors & DataLoader setup
train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
dataset = TensorDataset(train_tensor)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

input_dim = X_train_normal.shape[1]
print(f"Number of Input Features: {input_dim}")

# 2. Define Autoencoder Architecture
class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAutoencoder, self).__init__()
        # Compression (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU()
        )
        # Reconstruction (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

model = AnomalyAutoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Model Training Loop
epochs = 25
print(f"\n--- Training Autoencoder on Normal Patterns ({epochs} Epochs) ---")
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for data in train_loader:
        inputs = data[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss (Reconstruction Error): {epoch_loss:.6f}")

# 4. Calculate Dynamic Anomaly Threshold
print("\n--- Calculating Anomaly Threshold ---")
model.eval()
with torch.no_grad():
    train_reconstructed = model(train_tensor)
    # Calculate Reconstruction Loss per sample
    losses = torch.mean((train_tensor - train_reconstructed) ** 2, dim=1).numpy()

# Set threshold to 95th percentile of normal losses
threshold = float(np.percentile(losses, 95))
print(f"Calculated Anomaly Threshold (95th Percentile): {threshold:.6f}")

# 5. Save Trained Model and Metadata
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/autoencoder.pth')

metadata = {
    'input_dim': input_dim,
    'threshold': threshold
}
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f)

print("\nSuccess! Model saved to 'models/autoencoder.pth' & metadata to 'models/metadata.json'.")