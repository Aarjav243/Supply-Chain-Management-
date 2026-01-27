# 🚚 Supply Chain Risk Prediction (GNN)

AI-powered supply chain risk predictor using Graph Neural Networks (GNN). Treats shipping routes as interconnected networks to model contagion effects. Identifies vulnerable cities & high-risk routes with 66% accuracy.

## 🚀 Deployment Instructions for Render

To deploy this model to Render, follow these steps:

### 1. Locally Prepare Artifacts (Optional but Recommended)
Run the training script to generate the model and data artifacts:
```bash
python train_and_save.py
```
This will create:
- `model.pth` (Trained GNN state)
- `features.pt` (Node features)
- `adj_norm.pt` (Normalized adjacency matrix)
- `mappings.json` (City mappings and pre-computed insights)

### 2. Push to GitHub
Make sure all files are in your repository:
```bash
git add .
git commit -m "Add deployment files"
git push origin main
```

### 3. Deploy to Render
1. Go to [Render](https://render.com) and log in.
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository `Supply-Chain`.
4. Configure the service:
   - **Name:** `supply-chain-risk-predictor`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python train_and_save.py` (The second part ensures artifacts are generated on the server)
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Deploy Web Service**.

## 🛠 Project Structure
- `app.py`: Streamlit application interface.
- `train_and_save.py`: Script to train the GNN and save results.
- `requirements.txt`: Python dependencies.
- `Procfile`: Render deployment configuration.
- `sampled_10000.csv`: Dataset used for training.
- `Supply Chain.ipynb`: Original research notebook.

## 📊 Features
- **Tab 1: Route Risk Checker**: Select origin and destination cities to get a real-time risk prediction.
- **Tab 2: High-Risk Analysis**: View overall model performance and top 10 high-risk cities/routes.
