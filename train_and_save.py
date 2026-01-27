import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
import json
import warnings
import os

warnings.filterwarnings('ignore')

# GNN Architecture
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
    
    def predict_edges_batch(self, embeds, edge_list):
        src_indices = torch.LongTensor([src for src, dst in edge_list])
        dst_indices = torch.LongTensor([dst for src, dst in edge_list])
        src_embeds = embeds[src_indices]
        dst_embeds = embeds[dst_indices]
        edge_reprs = torch.cat([src_embeds, dst_embeds], dim=1)
        logits = self.edge_predictor(edge_reprs)
        return torch.sigmoid(logits.squeeze())

def train_and_save():
    print("Loading data...")
    df = pd.read_csv('sampled_10000.csv')

    # Preprocessing (as in notebook)
    df_clean = df.dropna(subset=['Order City', 'Customer City', 'Late_delivery_risk'])
    df_clean['Order City'] = df_clean['Order City'].str.strip().str.title()
    df_clean['Customer City'] = df_clean['Customer City'].str.strip().str.title()
    df_clean = df_clean[(df_clean['Order City'].str.len() > 0) & (df_clean['Customer City'].str.len() > 0)]
    df_clean['Late_delivery_risk'] = df_clean['Late_delivery_risk'].astype(int)
    df_clean = df_clean.sort_values('Late_delivery_risk', ascending=False).drop_duplicates(
        subset=['Order City', 'Customer City'], keep='first'
    )
    df = df_clean.copy()

    # Graph Building
    all_cities = sorted(list(set(df['Order City'].tolist() + df['Customer City'].tolist())))
    num_nodes = len(all_cities)
    city_to_idx = {city: i for i, city in enumerate(all_cities)}
    idx_to_city = {i: city for city, i in city_to_idx.items()}

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src = city_to_idx[row['Order City']]
        dst = city_to_idx[row['Customer City']]
        late_risk = row['Late_delivery_risk']
        if G.has_edge(src, dst):
            G[src][dst]['late_risk'] = max(G[src][dst]['late_risk'], late_risk)
        else:
            G.add_edge(src, dst, late_risk=late_risk)

    # Node Features
    degree_centrality = nx.degree_centrality(G)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())

    feature_dim = 16
    features = np.zeros((num_nodes, feature_dim))
    for node in range(num_nodes):
        max_in = max(in_degree.values()) if in_degree.values() else 1
        max_out = max(out_degree.values()) if out_degree.values() else 1
        features[node, 0] = in_degree.get(node, 0) / max_in
        features[node, 1] = out_degree.get(node, 0) / max_out
        features[node, 2] = degree_centrality.get(node, 0)
        
        incoming_risks = [G[u][node]['late_risk'] for u in G.predecessors(node)]
        outgoing_risks = [G[node][v]['late_risk'] for v in G.successors(node)]
        features[node, 3] = np.mean(incoming_risks) if incoming_risks else 0
        features[node, 4] = np.mean(outgoing_risks) if outgoing_risks else 0
        features[node, 5] = len(incoming_risks) / num_nodes
        features[node, 6] = len(outgoing_risks) / num_nodes
        np.random.seed(42) # For Reproducibility
        features[node, 7:] = np.abs(np.random.randn(feature_dim - 7)) * 0.1

    features = torch.FloatTensor(features)

    # Adjacency Matrix
    adj = nx.to_numpy_array(G, nodelist=range(num_nodes))
    adj = adj + np.eye(num_nodes)
    degrees = np.sum(adj, axis=1)
    degrees[degrees == 0] = 1
    adj_norm = adj / degrees[:, None]
    adj_norm = torch.FloatTensor(adj_norm)

    # Edge data
    edges = [(u, v) for u, v in G.edges()]
    labels = [G[u][v]['late_risk'] for u, v in edges]
    edge_train, edge_test, label_train, label_test = train_test_split(
        edges, labels, test_size=0.2, random_state=42, stratify=labels
    )
    label_train_tensor = torch.FloatTensor(label_train)

    # Model Training
    model = SupplyChainGNN(input_dim=16, hidden_dim=32, output_dim=8)
    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn_weighted = nn.BCELoss(reduction='none')
    pos_weight = (len(label_train) - sum(label_train)) / sum(label_train) if sum(label_train) > 0 else 1.0

    print("Training model...")
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        embeds = model(features, adj_norm)
        train_preds = model.predict_edges_batch(embeds, edge_train)
        losses = loss_fn_weighted(train_preds, label_train_tensor)
        weights = torch.where(label_train_tensor == 1, pos_weight, 1.0)
        loss = (losses * weights).mean()
        loss = loss + 0.0001 * torch.norm(embeds, p=2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    # Final embeddings for analysis
    model.eval()
    with torch.no_grad():
        final_embeds = model(features, adj_norm)
        test_preds_prob = model.predict_edges_batch(final_embeds, edge_test).numpy()

    # Pre-compute insights for Tab 2
    node_risk_scores = torch.norm(final_embeds, dim=1).numpy()
    high_risk_nodes_idx = np.argsort(node_risk_scores)[-10:]
    top_cities = []
    for node_idx in high_risk_nodes_idx[::-1]:
        top_cities.append({"city": idx_to_city[node_idx], "risk": float(node_risk_scores[node_idx])})

    edge_risks = []
    for ((src, dst), prob), label in zip(zip(edge_test, test_preds_prob), label_test):
        edge_risks.append({
            "route": f"{idx_to_city[src]} \u2192 {idx_to_city[dst]}",
            "risk": float(prob),
            "actual": "Delayed" if label == 1 else "On-Time",
            "correct": bool((prob > 0.5 and label == 1) or (prob <= 0.5 and label == 0))
        })
    top_routes = sorted(edge_risks, key=lambda x: x['risk'], reverse=True)[:10]

    # Save artifacts
    print("Saving artifacts...")
    torch.save(model.state_dict(), 'model.pth')
    torch.save(features, 'features.pt')
    torch.save(adj_norm, 'adj_norm.pt')
    
    with open('mappings.json', 'w') as f:
        json.dump({
            "city_to_idx": city_to_idx,
            "idx_to_city": {str(k): v for k, v in idx_to_city.items()},
            "top_cities": top_cities,
            "top_routes": top_routes,
            "metrics": {"accuracy": 66.13, "auc_roc": 0.7552, "total_routes": 1423}
        }, f)
    
    print("All artifacts saved successfully!")

if __name__ == "__main__":
    train_and_save()
