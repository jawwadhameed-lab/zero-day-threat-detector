import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Zero-Day Threat Detector", layout="wide", page_icon="🛡️")

# Custom Title Header
st.title("🛡️ AI-Based Zero-Day Threat Detection System")
st.markdown("Automated Behavioral Anomaly Detection using Deep Autoencoder")

# Load Model Metadata & Architecture
@st.cache_resource
def load_system():
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
    return model, metadata['threshold']

model, threshold = load_system()

# Sidebar Configuration
st.sidebar.title("📊 System Configuration")
st.sidebar.metric("Active Anomaly Threshold", f"{threshold:.6f}")
st.sidebar.info("Traffic packets with Reconstruction Loss > Threshold are flagged as Zero-Day Anomalies.")

# Dashboard Tabs Navigation
tab1, tab2 = st.tabs(["⚡ Live Packet Inspector", "📈 Batch Analytics & Loss Graph"])

# TAB 1: Single Packet Inspection
with tab1:
    st.subheader("Real-Time Traffic Inspection")
    if st.button("Run Random Traffic Test Sample", type="primary"):
        X_test = np.load('data/X_test.npy')
        y_test = np.load('data/y_test.npy')
        
        idx = np.random.randint(0, len(X_test))
        sample = X_test[idx : idx + 1]
        actual_label = "Attack / Anomaly" if y_test[idx] == 1 else "Normal Safe Traffic"
        
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = model(sample_tensor)
            error = torch.mean((sample_tensor - reconstructed) ** 2).item()
        
        is_anomaly = error > threshold
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Reconstruction Error Loss", f"{error:.6f}")
        col2.metric("AI Verdict", "🚨 ZERO-DAY THREAT" if is_anomaly else "✅ SAFE TRAFFIC")
        col3.metric("Ground Truth Label", actual_label)
        
        if is_anomaly:
            st.error("⚠️ ALERT: High Reconstruction Loss detected! Behavior deviates from normal baseline profiles.")
        else:
            st.success("✅ SAFE: Traffic pattern closely matches trusted network baseline.")

# TAB 2: Batch Analysis & Visual Chart
with tab2:
    st.subheader("Batch Sample Analytics (1,000 Traffic Packets)")
    if st.button("Run Batch Evaluation Analysis"):
        X_test = np.load('data/X_test.npy')[:1000]
        y_test = np.load('data/y_test.npy')[:1000]
        
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            reconstructed = model(test_tensor)
            errors = torch.mean((test_tensor - reconstructed) ** 2, dim=1).numpy()
        
        preds = (errors > threshold).astype(int)
        
        c1, c2 = st.columns(2)
        c1.metric("Flagged Zero-Day Threats", int(sum(preds)))
        c2.metric("Verified Safe Traffic Packets", int(len(preds) - sum(preds)))
        
        # Loss Distribution Chart
        fig, ax = plt.subplots(figsize=(10, 3.5))
        sns.histplot(errors, bins=50, kde=True, ax=ax, color='purple')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.5f})')
        ax.set_title("Reconstruction Error Distribution vs Anomaly Threshold")
        ax.set_xlabel("Reconstruction Loss")
        ax.legend()
        st.pyplot(fig)
        
        # Data Table
        df_res = pd.DataFrame({
            'Packet Index': range(1, 1001),
            'Reconstruction Loss': errors,
            'Prediction': ["Threat (Anomaly)" if p == 1 else "Safe Traffic" for p in preds],
            'Actual Label': ["Attack" if y == 1 else "Normal" for y in y_test]
        })
        st.dataframe(df_res, use_container_width=True)