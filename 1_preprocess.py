import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os

print("--- Step 1: Loading Data ---")
train_df = pd.read_parquet('data/UNSW_NB15_training-set.parquet')
test_df = pd.read_parquet('data/UNSW_NB15_testing-set.parquet')

# Drop unnecessary identifier columns if present
cols_to_drop = ['id'] if 'id' in train_df.columns else []
train_df = train_df.drop(columns=cols_to_drop)
test_df = test_df.drop(columns=cols_to_drop)

print(f"Train Raw Shape: {train_df.shape}")
print(f"Test Raw Shape: {test_df.shape}")

print("\n--- Step 2: Categorical Feature Encoding ---")
# Select ALL non-numeric columns regardless of parquet string type
cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove target category column from feature encoder if present
if 'attack_cat' in cat_cols:
    cat_cols.remove('attack_cat')

print(f"Categorical Columns to Encode: {cat_cols}")

# Convert text features to numeric integers
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    encoders[col] = le

print("\n--- Step 3: Zero-Day Strategy Filtering ---")
# Training Autoencoder strictly on SAFE / NORMAL data (label == 0)
train_normal = train_df[train_df['label'] == 0].copy()

X_train = train_normal.drop(columns=['label', 'attack_cat'], errors='ignore')
X_test = test_df.drop(columns=['label', 'attack_cat'], errors='ignore')
y_test = test_df['label']

print(f"Normal Safe Traffic Samples for Training: {X_train.shape[0]}")
print(f"Mixed (Safe + Attacks) Samples for Testing: {X_test.shape[0]}")

print("\n--- Step 4: MinMax Feature Scaling ---")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed arrays and scaler for PyTorch Training & Dashboard
os.makedirs('models', exist_ok=True)
np.save('data/X_train_normal.npy', X_train_scaled)
np.save('data/X_test.npy', X_test_scaled)
np.save('data/y_test.npy', y_test.to_numpy())
joblib.dump(scaler, 'models/scaler.pkl')

print("\nSuccess! Processed data saved to 'data/' folder and scaler to 'models/scaler.pkl'.")