import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# Page Layout
st.set_page_config(page_title="AI Zero-Day Attack Detection", layout="wide")
st.title("🛡️ AI-Based Zero-Day Threat Detection")

# --- IPS SESSION STATE INITIALIZATION ---
if 'blacklist' not in st.session_state:
    st.session_state.blacklist = []
if 'total_blocked' not in st.session_state:
    st.session_state.total_blocked = 0

# Helper Function: Mock IP Generator for Demo
def generate_mock_ip(idx):
    np.random.seed(int(idx))
    return f"192.168.{np.random.randint(1, 10)}.{np.random.randint(2, 254)}"

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    scaler = joblib.load("models/scaler.pkl")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    
    # Autoencoder Architecture
    class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
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

    model = Autoencoder(X_test.shape[1])
    model.load_state_dict(torch.load("models/autoencoder.pth", weights_only=True))
    model.eval()
    return model, scaler, X_test, y_test

model, scaler, X_test, y_test = load_assets()
OPTIMAL_THRESHOLD = 0.001652

# --- SIDEBAR: IPS STATUS & METRICS ---
st.sidebar.header("🛡️ IPS Control Panel")
metric1_holder = st.sidebar.empty()
metric2_holder = st.sidebar.empty()

if st.sidebar.button("Clear Blacklist / Reset IPS"):
    st.session_state.blacklist = []
    st.session_state.total_blocked = 0
    st.sidebar.success("IPS Rules Reset Successfully!")

# --- MAIN DASHBOARD TABS ---
tab1, tab2 = st.tabs(["🚀 Live Traffic Inspection & IPS", "📋 Active Blacklist & Firewall Rules"])

with tab1:
    st.subheader("Real-Time Traffic Packet Inspection")
    
    if st.button("Run Random Traffic Test Sample"):
        sample_idx = np.random.randint(0, len(X_test))
        sample_data = X_test[sample_idx:sample_idx+1]
        actual_label = y_test[sample_idx]
        mock_ip = generate_mock_ip(sample_idx)

        # Model Prediction
        with torch.no_grad():
            sample_tensor = torch.FloatTensor(sample_data)
            reconstruction = model(sample_tensor)
            loss = torch.mean((sample_tensor - reconstruction) ** 2, dim=1).item()

        is_anomaly = loss > OPTIMAL_THRESHOLD

        # Display Traffic Info
        col1, col2, col3 = st.columns(3)
        col1.metric("Source IP Address", mock_ip)
        col2.metric("Reconstruction Loss", f"{loss:.6f}")
        col3.metric("Ground Truth Label", "Attack" if actual_label == 1 else "Normal")

        st.markdown("---")

        # --- IPS PREVENTION LOGIC ---
        if is_anomaly:
            if mock_ip not in st.session_state.blacklist:
                st.session_state.blacklist.append(mock_ip)
                st.session_state.total_blocked += 1

            st.error(f"🛑 **[IPS PREVENTION ACTION ACTIVATED]**")
            st.warning(f"Zero-Day Anomaly Detected from Source IP: **{mock_ip}** (Loss > {OPTIMAL_THRESHOLD:.6f})")
            st.info("⚡ **Automated Mitigation:** Packet status set to **DROPPED & BLOCKED**.")
            
            st.markdown("**Generated System Command (Firewall Rule):**")
            st.code(f"sudo iptables -A INPUT -s {mock_ip} -j DROP", language="bash")

        else:
            if mock_ip in st.session_state.blacklist:
                st.error(f"🛑 Traffic from IP **{mock_ip}** was rejected because IP is currently BLACKLISTED.")
            else:
                st.success(f"✅ **[IPS ACTION ALLOWED]** Traffic packet from IP **{mock_ip}** is safe. Packet forwarded.")

with tab2:
    st.subheader("📋 Active Blacklisted IPs (Firewall Engine)")
    
    if len(st.session_state.blacklist) > 0:
        df_bl = pd.DataFrame({
            "Blocked Source IP": st.session_state.blacklist,
            "Action Taken": ["PACKET_DROP"] * len(st.session_state.blacklist),
            "Firewall Protocol": ["IPTABLES_DROP"] * len(st.session_state.blacklist)
        })
        st.table(df_bl)
    else:
        st.info("No IPs are currently blacklisted. All traffic operating normally.")
        
metric1_holder.metric("Active Blocked IPs", len(st.session_state.blacklist))
metric2_holder.metric("Total Packets Dropped", st.session_state.total_blocked)        