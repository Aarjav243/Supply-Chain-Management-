import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# GNN Architecture (Redefined for loading)
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return F.relu(output)

class SupplyChainGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SupplyChainGNN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        return x
    
    def predict_edge(self, embeds, src, dst):
        src_embed = embeds[src]
        dst_embed = embeds[dst]
        edge_repr = torch.cat([src_embed, dst_embed], dim=0)
        logit = self.edge_predictor(edge_repr)
        return torch.sigmoid(logit.squeeze())

# Cache loading to improve performance
@st.cache_resource
def load_all():
    if not os.path.exists('model.pth'):
        return None, None, None, None
        
    # Initialize model
    model = SupplyChainGNN(input_dim=16, hidden_dim=32, output_dim=8)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # Load features and adjacency matrix
    features = torch.load('features.pt', map_location=torch.device('cpu'))
    adj_norm = torch.load('adj_norm.pt', map_location=torch.device('cpu'))
    
    # Load mappings
    with open('mappings.json', 'r') as f:
        mappings = json.load(f)
        
    # Pre-compute final embeddings
    with torch.no_grad():
        final_embeds = model(features, adj_norm)
        
    return model, final_embeds, mappings['city_to_idx'], mappings

# Page Setup
st.set_page_config(page_title="Supply Chain Risk Predictor", layout="wide")

# Sidebar
st.sidebar.title("🚚 SC Risk GNN")
st.sidebar.info("This model uses Graph Neural Networks to predict delivery delays based on origin and destination cities.")

# Main Title
st.title("🚚 Supply Chain Risk Prediction")

# Load data
model, final_embeds, city_to_idx, full_mappings = load_all()

if model is None:
    st.error("Model artifacts not found. Please run the training script first.")
    st.info("Run: `python train_and_save.py` to generate the necessary files.")
    st.stop()

# Tabs
tab1, tab2 = st.tabs(["🔍 Check Route Risk", "📊 View High-Risk Analysis"])

with tab1:
    st.header("📍 Select Route")
    
    # Dropdowns for cities
    cities = sorted(list(city_to_idx.keys()))
    col1, col2 = st.columns(2)
    
    with col1:
        origin_city = st.selectbox("Origin City", cities, index=cities.index("Caguas") if "Caguas" in cities else 0)
    with col2:
        dest_city = st.selectbox("Destination City", cities, index=1 if len(cities) > 1 else 0)
    
    if st.button("🔍 PREDICT RISK", use_container_width=True):
        src_idx = city_to_idx[origin_city]
        dst_idx = city_to_idx[dest_city]
        
        with torch.no_grad():
            risk_score = model.predict_edge(final_embeds, src_idx, dst_idx).item()
        
        st.divider()
        st.subheader("📊 RESULT:")
        
        # Risk Score with Progress Bar
        st.write(f"**Risk Score:** {risk_score:.4f}")
        color = "red" if risk_score > 0.6 else "orange" if risk_score > 0.4 else "green"
        st.progress(risk_score, text=f"{risk_score:.1%}")
        
        # Prediction Label
        if risk_score > 0.6:
            st.error(f"Prediction: 🔴 HIGH RISK - Likely Delayed")
        elif risk_score > 0.4:
            st.warning(f"Prediction: 🟡 MEDIUM RISK - Possibility of Delay")
        else:
            st.success(f"Prediction: 🟢 LOW RISK - Likely On-Time")
            
        st.info(f"Analyzing route: {origin_city} \u2192 {dest_city}")

with tab2:
    st.header("📊 Supply Chain Risk Analysis")
    
    # Model Performance
    st.subheader("🎯 MODEL PERFORMANCE")
    metrics = full_mappings['metrics']
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy']}%")
    m2.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
    m3.metric("Routes Analyzed", metrics['total_routes'])
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🏙️ TOP 10 HIGH-RISK CITIES")
        top_cities_df = pd.DataFrame(full_mappings['top_cities'])
        top_cities_df.index = top_cities_df.index + 1
        st.table(top_cities_df)
        st.caption("Risk score based on node embedding magnitude.")
        
    with col_right:
        st.subheader("🚨 TOP 10 HIGH-RISK ROUTES")
        top_routes_df = pd.DataFrame(full_mappings['top_routes'])
        # Rename for display
        display_routes = top_routes_df[['route', 'risk', 'actual', 'correct']].copy()
        display_routes['correct'] = display_routes['correct'].apply(lambda x: "✓" if x else "✗")
        st.table(display_routes)

    st.divider()
    st.info("💡 These insights come from analyzing the supply chain network structure using a Graph Convolutional Network (GCN).")

# Footer
st.markdown("---")
st.markdown("*Built with PyTorch, NetworkX, and Streamlit*")
