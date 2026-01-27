# Supply Chain Risk Predictor Using Graph Neural Networks

### Predicting Delivery Delays Through Network Analysis and Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

---

## Overview

This project implements a Graph Convolutional Network (GCN) to predict supply chain delivery risks by modeling shipping routes as an interconnected network. Unlike traditional time-series forecasting methods that treat suppliers as isolated entities, this approach leverages graph neural networks to capture structural dependencies and propagate risk signals through the supply chain network.

**Key Achievement:** 66% prediction accuracy with 0.75 AUC-ROC score on real commercial supply chain data.

---

## Problem Statement

### The Business Challenge

Modern supply chains are complex networks where a delay at one node can trigger cascading failures downstream—a phenomenon known as the "bullwhip effect." Traditional forecasting methods (ARIMA, LSTMs) fail to capture these network externalities because they analyze suppliers independently.

**Real-World Impact:**
- A tier-2 supplier delay in China cascades to tier-1 manufacturers in Vietnam
- Distribution centers in the USA face stockouts
- End customers experience delivery failures
- Companies face revenue loss, customer dissatisfaction, and contract penalties



## Solution Approach

### Graph-Based Supply Chain Modeling

Instead of treating suppliers as isolated entities, we model the supply chain as a directed graph:

**Graph Structure:**
- **Nodes (Vertices):** Cities, warehouses, distribution centers
- **Edges:** Shipping routes between locations
- **Node Features:** Degree centrality, historical risk statistics, network position
- **Edge Labels:** Binary classification (0 = on-time, 1 = delayed)

**Mathematical Formulation:**
```
G = (V, E) where:
V = {v₁, v₂, ..., vₙ} (set of nodes/cities)
E = {(vᵢ, vⱼ) | route exists from vᵢ to vⱼ} (set of edges/routes)

Node features: X ∈ ℝⁿˣᵈ (n nodes, d features per node)
Adjacency matrix: A ∈ ℝⁿˣⁿ (network structure)
```

### Graph Convolutional Network Architecture

**Core Innovation:** Using GCN layers to aggregate information from neighboring nodes, capturing both local features and network topology.

**Architecture Components:**

1. **Input Layer:**
   - Node feature matrix: 16 dimensions per node
   - Normalized adjacency matrix (D⁻¹ × A where D is degree matrix)

2. **GCN Layer 1:** 
   - Input: 16 dimensions
   - Output: 32 dimensions
   - Activation: ReLU
   - Dropout: 0.3

3. **GCN Layer 2:**
   - Input: 32 dimensions
   - Output: 8 dimensions (node embeddings)
   - Activation: ReLU
   - Dropout: 0.3

4. **Edge Prediction Module:**
   - Multi-Layer Perceptron (MLP) architecture
   - Input: Concatenated source and destination embeddings (16 dimensions total)
   - Hidden layers: 32 → 16 dimensions
   - Output: Single probability score via sigmoid activation

**GCN Layer Mathematical Operation:**
```
H^(l+1) = σ(Â × H^(l) × W^(l))

where:
Â = D⁻¹ × A (normalized adjacency matrix)
H^(l) = node features at layer l
W^(l) = learnable weight matrix at layer l
σ = ReLU activation function
```

### Why Graph Neural Networks?

**Comparison with Traditional Methods:**

| Approach | Captures Network Structure | Models Contagion | Scalability |
|----------|---------------------------|------------------|-------------|
| ARIMA/Time-Series | No | No | High |
| LSTM/RNN | No | No | Medium |
| Random Forest/XGBoost | No (requires manual features) | No | High |
| **Graph Neural Networks** | **Yes (automatic)** | **Yes (via aggregation)** | **Medium-High** |

**Key Advantages of GCN:**

1. **Network Externalities:** GCNs naturally model how one agent's outcome depends on neighbors' outcomes through message passing
2. **Shock Propagation:** Multi-layer aggregation captures cascading effects across the network
3. **Structural Learning:** Embeddings encode both node features AND topological position
4. **Inductive Learning:** Can make predictions on unseen nodes/edges

**Economic Intuition:**
- Traditional ML: "This warehouse has 100 delays/year → high risk"
- GCN: "This warehouse has 100 delays/year AND is connected to 3 high-risk suppliers AND is a central hub → extremely high systemic risk"

---

## Technical Implementation

### Data Pipeline

**Step 1: Graph Construction**
```python
# Build directed graph from shipping data
G = networkx.DiGraph()
for each order:
    source_city = city_to_index[order['Origin']]
    dest_city = city_to_index[order['Destination']]
    risk_label = order['Late_delivery_risk']
    G.add_edge(source_city, dest_city, late_risk=risk_label)
```

**Step 2: Feature Engineering**
```python
# Node features (16 dimensions per node):
features = [
    normalized_in_degree,
    normalized_out_degree,
    degree_centrality,
    avg_incoming_risk,
    avg_outgoing_risk,
    normalized_edge_count_in,
    normalized_edge_count_out,
    random_embeddings  # 9 additional dimensions
]
```

**Step 3: Adjacency Matrix Normalization**
```python
# Normalize by degree: D^-1 × A
A_norm = adjacency_matrix / degree_matrix
# Add self-loops for stability
A_norm = A_norm + I
```

**Step 4: Model Training**
```python
# Forward pass
embeddings = GCN(node_features, adjacency_matrix)

# Edge prediction
edge_embedding = concat(embeddings[source], embeddings[destination])
risk_probability = MLP(edge_embedding)

# Loss calculation with class weighting
loss = BCELoss(predictions, labels, class_weights)
```

### Key Technical Innovations

**Problem:** Original dot-product edge prediction caused saturation (all predictions = 0.5)

**Solution:** Replaced with MLP edge predictor that learns complex non-linear relationships

**Before (Saturated):**
```python
risk = sigmoid(dot_product(normalize(src_embed), normalize(dst_embed)))
# Result: All predictions = 0.5, std = 0.0
```

**After (Fixed):**
```python
edge_repr = concatenate(src_embed, dst_embed)
risk = MLP(edge_repr)  # 3-layer neural network
# Result: Predictions range [0.0, 1.0], std = 0.21
```

---

## Model Performance

### Quantitative Results

**Primary Metrics:**
- **Accuracy:** 66.13%
- **AUC-ROC:** 0.7552
- **Test Set Size:** 1,423 shipping routes

**Prediction Distribution:**
- Minimum Prediction: 0.0044
- Maximum Prediction: 0.9592
- Mean Prediction: 0.5021
- Standard Deviation: 0.2080
- Median Prediction: 0.4957

**Calibration Quality:**
- Well-calibrated predictions (0.05-0.95): 94.2%
- Over-confident high (>0.95): 1.0%
- Over-confident low (<0.05): 4.8%

### Classification Performance

**Confusion Matrix:**
```
                    Predicted
                No Risk    Late Risk
Actual  No Risk    420        165      (585 total)
        Late Risk  317        521      (838 total)
```

**Interpretation:**
- True Negatives (420): Correctly identified safe routes
- True Positives (521): Correctly flagged risky routes
- False Positives (165): Over-cautious predictions (conservative approach)
- False Negatives (317): Missed risks (area for improvement)

**Class-Specific Metrics:**
- Precision (No Risk): 0.57
- Recall (No Risk): 0.72
- F1-Score (No Risk): 0.64

### Business Impact Quantification

**Hypothetical ROI Calculation:**

If a company ships 10,000 orders per month with:
- 10% baseline delay rate (1,000 delays)
- $100 average penalty per delay
- Model reduces delays by 40% (400 fewer delays)

**Monthly Savings:** $40,000
**Annual Savings:** $480,000

**Additional Benefits:**
- Improved customer satisfaction
- Enhanced brand reputation
- Reduced operational firefighting
- More efficient resource allocation

---

## Dataset

### DataCo Smart Supply Chain Dataset

**Source:** Kaggle - Real commercial supply chain data
**Link:** https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis


## Limitations and Constraints

### 1. Data Quality Dependencies
**Limitation:** Model accuracy depends on completeness of historical data

**Impact:** Missing values in critical columns require row removal, reducing training data

**Mitigation:** Implement robust imputation strategies; invest in real-time data collection infrastructure

### 2. Static Graph Assumption
**Limitation:** Current implementation treats network structure as fixed

**Impact:** Cannot model seasonal patterns or evolving supplier relationships

**Mitigation:** Extend to Temporal Graph Networks or Dynamic GNN architectures

### 3. Limited Temporal Modeling
**Limitation:** No explicit time-series features (no LSTM integration)

**Impact:** Cannot capture long-term trends or cyclic patterns (e.g., holiday seasons)

**Mitigation:** Develop hybrid GNN-LSTM architecture for sequential graph data

### 4. Binary Classification Only
**Limitation:** Predicts delayed vs. on-time but not delay severity

**Impact:** Cannot differentiate between 2-day delay vs. 2-week delay

**Mitigation:** Extend to multi-class classification or regression for delay duration

### 5. Class Imbalance Sensitivity
**Limitation:** Performance degrades with extreme imbalance (<5% or >95%)

**Impact:** May underpredict rare catastrophic events

**Mitigation:** Apply SMOTE oversampling, focal loss, or cost-sensitive learning

### 6. Interpretability Challenges
**Limitation:** GNN embeddings are black-box representations

**Impact:** Difficult to explain predictions to non-technical stakeholders

**Mitigation:** Implement GNNExplainer or Graph Attention Networks (GAT) for attention visualization

### 7. Scalability Constraints
**Limitation:** Full-batch training on 10,000+ nodes requires significant memory

**Impact:** Cannot easily scale to global supply chains (100,000+ nodes)

**Mitigation:** Implement mini-batch training with GraphSAINT or neighbor sampling

### 8. Cold Start Problem
**Limitation:** New nodes (suppliers) have no historical data

**Impact:** Cannot make accurate predictions for newly onboarded vendors

**Mitigation:** Use meta-learning or few-shot learning; leverage network position for initial estimates

### 9. External Factors Not Captured
**Limitation:** Model doesn't account for exogenous shocks

**Impact:** Cannot predict impacts of weather events, geopolitical crises, or pandemics

**Mitigation:** Integrate external data sources (weather APIs, news sentiment, economic indicators)

### 10. Correlation vs. Causation
**Limitation:** GCN learns correlations, not causal relationships

**Impact:** Cannot answer counterfactual questions ("What if we remove Supplier X?")

**Mitigation:** Explore Causal GNN architectures or intervention-based simulations

---

## Installation and Usage

### Prerequisites

**Required Libraries:**
```
Python 3.8+
PyTorch 2.0+
NetworkX 3.0+
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
```

```

---

**Last Updated:** January 2026  
**Version:** 1.0  
**Status:** Production Ready
