import numpy as np
import torch
import torch.nn as nn
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

print("--- Loading Model & Saved Metadata ---")
with open('models/metadata.json', 'r') as f:
    metadata = json.load(f)

input_dim = metadata['input_dim']

class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = AnomalyAutoencoder(input_dim)
model.load_state_dict(torch.load('models/autoencoder.pth'))
model.eval()

# Load Test Data
X_test = np.load('data/X_test.npy')
y_test = np.load('data/y_test.npy')
test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Calculate Reconstruction Error
with torch.no_grad():
    reconstructed = model(test_tensor)
    errors = torch.mean((test_tensor - reconstructed) ** 2, dim=1).numpy()

# Calculate Youden's J Index (Optimal Balance: True Positive Rate - False Positive Rate)
fpr, tpr, thresholds = roc_curve(y_test, errors)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]

y_pred = (errors > best_thresh).astype(int)

print("\n================ BALANCED ZERO-DAY DETECTION RESULTS ================")
print(f"Optimal Balanced Threshold (Youden's J): {best_thresh:.6f}")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc_score(y_test, errors):.4f}\n")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"True Normal Traffic (Correctly Safe): {cm[0][0]}")
print(f"False Alarm (Safe marked as Threat): {cm[0][1]}")
print(f"Missed Attacks (False Negative):    {cm[1][0]}")
print(f"Detected Zero-Day Threats (True Positive): {cm[1][1]}\n")

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal Traffic', 'Zero-Day Threat']))

# Update metadata with balanced threshold
metadata['threshold'] = float(best_thresh)
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f)
print("\nSaved balanced threshold to 'models/metadata.json'.")