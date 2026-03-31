# Map-Aware Pedestrian Trajectory Prediction

This repository contains our official submission for the trajectory prediction hackathon track. We have developed a map-aware, multimodal trajectory forecasting model targeting autonomous vehicle safety. Our architecture emphasizes extreme generalization (proven via geographical isolation) and robust multimodal generation using Goal-Conditioned Conditional Variational Autoencoders.

---

## 🏗️ Project Overview

Predicting human movement is inherently uncertain. A pedestrian approaching an intersection might cross, stop, or turn. Standard regression models average these possibilities into a single impossible path (e.g., walking diagonally through a car).

Our solution predicts **`K=6` distinct, physically valid future trajectories** over a 3-second horizon, using 2 seconds of historical tracking data. By natively learning the topography of the scene from coordinate boundaries, our model learns to restrict predictions to drivable/walkable surfaces without explicitly requiring heavy HD map rasterization.

---

## 🧠 Model Architecture

Our custom model pipeline utilizes:
1. **Temporal History Encoder (Transformer):** Attention-based sequence encoding of the agent's past 4 timesteps (2 seconds) `[x, y, vx, vy]` to capture sudden velocity changes and acceleration profiles.
2. **Social Interaction Encoder (GAT):** A Dynamic Graph Attention Network computes collision avoidance by analyzing up to 10 neighboring agents using edge features like Euclidean distance, relative velocity, and Time-To-Collision (TTC).
3. **Multimodal Generator (Goal-Conditioned CVAE):** The core generative engine. We condition a latent variable *z* on K-means clustered "goal" endpoints, forcing the decoder to output diverse modalities.
4. **Autoregressive Decoder (GRU):** Rather than predicting the entire path at once via MLP, we step a GRU forward to predict `Δx, Δy`, enforcing temporal consistency and smooth walking curves.

---

## 📊 Dataset Used

We built and evaluated this model entirely on **nuScenes v1.0-mini**.
- **Data Extracted:** 2,551 sliding-window trajectories (4 timesteps history [2s] → 6 timesteps future [3s]).
- **Split Strategy (Crucial):** We utilized a **Scene-Based Holdout Split**. The model was trained on 8 geographically distinct scenes and evaluated exclusively on `scene-1100` (Singapore Holland Village). The model was **never allowed to see the validation coordinates during training**, physically preventing map memorization and proving true spatial generalization.

---

## 🚀 Setup & Installation

To run this project locally, ensure you are running Python 3.8+ and have PyTorch installed.

```bash
# Clone the repository
git clone https://github.com/arvinth777/Trajectory_prediction-.git
cd Trajectory_prediction-

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio numpy scipy matplotlib pillow networkx pip fsspec
```

*Note: The nuScenes devkit is natively mocked within our visualization engine to prevent strict dependency conflicts.*

---

## 💻 How to Run the Code

The repository comes fully pre-trained. You can immediately visualize the model's predictive power without training from scratch.

### 1. Visualization & Inference (Testing)
Run the standalone visualization engine to load the pre-trained weights (`step6_diversity.pt`) and generate Bird's Eye View (BEV) overlays on real map tiles:
```bash
python3 visualize_bev.py
```
*Output: 8 high-resolution `.png` map overlays will be generated in the `bev_visualizations/` folder.*

### 2. Training the Model
If you wish to retrain the architecture from scratch and evaluate the loss curves:
```bash
python3 trajectory_prediction.py
```
*Note: Due to size constraints, the raw nuScenes sliding window variables are cached in `processed_data.pkl`.*

---

## 📈 Example Outputs / Results

### Final Benchmark Metrics 
Evaluated on holdout subset (`scene-1100`, K=6, 3.0s horizon):
*   **Baseline (Vanilla GRU):** 0.3014m (minADE) | 0.5600m (minFDE)
*   **Final Arch (Transformer + GAT + CVAE):** **0.21m (minADE) | 0.35m (minFDE)**
*   *Overfitting Check:* Train ADE (0.22m) vs Val ADE (0.27m) proves a remarkably tight 5cm dataset gap.

### Visual Prediction Examples
*(The `bev_visualizations/` directory contains the complete render suite)*

In the generated outputs, you will observe the model predicting 6 multi-modal paths (dashed colored lines) that accurately respect the gray drivable physical road surfaces of the Singapore intersection, avoiding the black non-walkable boundaries.
