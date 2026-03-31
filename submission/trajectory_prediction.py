#!/usr/bin/env python3
"""
Pedestrian & Cyclist Trajectory Prediction (nuScenes v1.0-mini)
================================================================
Multi-modal trajectory prediction: given 2s history (4 steps @2Hz),
predict 3s future (6 steps). Steps 1-7 implemented incrementally.
"""

import json, os, math, pickle, random, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATAROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v1.0-mini')
HIST_LEN = 4    # 2 seconds at 2Hz
FUTURE_LEN = 6  # 3 seconds at 2Hz
DT = 0.5        # seconds between frames
NUM_MODES = 6   # number of predicted trajectories
LATENT_DIM = 16
DEVICE = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available()
                      else 'cpu')
print(f"Using device: {DEVICE}")

QUICK_TEST = False  # Toggle to run only a quick 30-epoch validation of Step 6
RESUME_FROM_STEP = 6  # Set to 6 to skip Steps 1-5 and load from processed_data.pkl

# Initialize dummy metrics for final summary if skipping steps
if RESUME_FROM_STEP == 6:
    baseline_ade, baseline_fde = 0.3014, 0.5600
    trans_ade, trans_fde = 0.2992, 0.5559
    gat_ade, gat_fde = 0.2773, 0.4884
    step5_ade, step5_fde = 0.2107, 0.3558
else:
    baseline_ade = baseline_fde = trans_ade = trans_fde = 0.0
    gat_ade = gat_fde = step5_ade = step5_fde = 0.0

tta_ade = tta_fde = ensemble_tta_ade = ensemble_tta_fde = 0.0

# ════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 1: DATA PIPELINE")
print("="*70)

def load_nuscenes_json(dataroot):
    """Load the 5 required JSON files from nuScenes."""
    files = {}
    for name in ['scene', 'sample', 'sample_annotation', 'instance', 'category']:
        path = os.path.join(dataroot, f'{name}.json')
        with open(path) as f:
            files[name] = json.load(f)
        print(f"  Loaded {name}.json: {len(files[name])} records")
    return files

def build_lookup_tables(data):
    """Build token→record lookup tables for fast access."""
    tables = {}
    for name in data:
        tables[name] = {rec['token']: rec for rec in data[name]}
    return tables

def get_ordered_samples(scene_rec, sample_lut):
    """Get ordered sample tokens for a scene."""
    samples = []
    token = scene_rec['first_sample_token']
    while token:
        samples.append(token)
        token = sample_lut[token]['next']
    return samples

def extract_trajectories(data, lut):
    """Extract pedestrian and cyclist trajectories per scene."""
    # Identify pedestrian and cyclist categories
    ped_cat_tokens = set()
    cyc_cat_tokens = set()
    for cat in data['category']:
        if 'pedestrian' in cat['name']:
            ped_cat_tokens.add(cat['token'])
        elif cat['name'] == 'vehicle.bicycle':
            cyc_cat_tokens.add(cat['token'])
    target_cats = ped_cat_tokens | cyc_cat_tokens
    print(f"  Pedestrian categories: {len(ped_cat_tokens)}, Cyclist categories: {len(cyc_cat_tokens)}")

    # Map instance→category type (0=ped, 1=cyclist)
    instance_type = {}
    for inst in data['instance']:
        if inst['category_token'] in ped_cat_tokens:
            instance_type[inst['token']] = 0  # pedestrian
        elif inst['category_token'] in cyc_cat_tokens:
            instance_type[inst['token']] = 1  # cyclist

    # Build sample→annotations lookup
    sample_to_anns = defaultdict(list)
    for ann in data['sample_annotation']:
        if ann['instance_token'] in instance_type:
            sample_to_anns[ann['sample_token']].append(ann)

    # Extract per-scene, per-instance trajectories
    all_scene_data = []
    for scene in data['scene']:
        sample_tokens = get_ordered_samples(scene, lut['sample'])
        # For each instance, collect positions across ordered samples
        instance_positions = defaultdict(dict)
        for t_idx, s_token in enumerate(sample_tokens):
            for ann in sample_to_anns[s_token]:
                inst_token = ann['instance_token']
                x, y, z = ann['translation']
                rot = ann['rotation']  # quaternion [w, x, y, z]
                instance_positions[inst_token][t_idx] = {
                    'x': x, 'y': y,
                    'heading': quaternion_to_yaw(rot),
                    'agent_type': instance_type[inst_token]
                }
        all_scene_data.append({
            'scene_token': scene['token'],
            'scene_name': scene['name'],
            'num_samples': len(sample_tokens),
            'agents': dict(instance_positions)
        })
    return all_scene_data

def quaternion_to_yaw(q):
    """Convert quaternion [w, x, y, z] to yaw angle."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def compute_velocities(positions, dt=DT):
    """Compute vx, vy from position differences."""
    vx = np.diff(positions[:, 0]) / dt
    vy = np.diff(positions[:, 1]) / dt
    # Pad first velocity with second (or zero)
    vx = np.concatenate([[vx[0] if len(vx) > 0 else 0.0], vx])
    vy = np.concatenate([[vy[0] if len(vy) > 0 else 0.0], vy])
    return vx, vy

def normalize_trajectory(hist_xy, fut_xy, heading):
    """Agent-centric normalization: last history point = origin, heading → up."""
    origin = hist_xy[-1].copy()
    angle = -heading + np.pi / 2  # rotate so heading points up
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    hist_centered = hist_xy - origin
    fut_centered = fut_xy - origin
    hist_rot = hist_centered @ rot.T
    fut_rot = fut_centered @ rot.T
    return hist_rot, fut_rot, origin, angle

def create_sliding_windows(scene_data_list):
    """Create sliding window samples: 4 hist + 6 future."""
    samples = []
    total_window = HIST_LEN + FUTURE_LEN  # 10

    for scene in scene_data_list:
        num_t = scene['num_samples']
        if num_t < total_window:
            continue

        for start_t in range(num_t - total_window + 1):
            timesteps = list(range(start_t, start_t + total_window))

            # Collect all agents present in ALL timesteps of this window
            window_agents = {}
            for inst_token, positions in scene['agents'].items():
                if all(t in positions for t in timesteps):
                    xy = np.array([[positions[t]['x'], positions[t]['y']] for t in timesteps])
                    heading = positions[timesteps[HIST_LEN - 1]]['heading']
                    agent_type = positions[timesteps[0]]['agent_type']
                    window_agents[inst_token] = {
                        'xy': xy, 'heading': heading, 'agent_type': agent_type
                    }

            if len(window_agents) == 0:
                continue

            # Determine which agents are moving (total displacement > 0.5m)
            for inst_token, agent in window_agents.items():
                total_dist = np.sum(np.linalg.norm(np.diff(agent['xy'], axis=0), axis=1))
                agent['is_moving'] = total_dist > 0.5

            # Create a sample for each MOVING agent as prediction target
            for target_token, target_agent in window_agents.items():
                if not target_agent['is_moving']:
                    continue

                hist_xy = target_agent['xy'][:HIST_LEN]
                fut_xy = target_agent['xy'][HIST_LEN:]
                heading = target_agent['heading']

                # Normalize target agent trajectory
                hist_norm, fut_norm, origin, angle = normalize_trajectory(
                    hist_xy, fut_xy, heading
                )

                # Compute velocities on normalized history
                vx, vy = compute_velocities(hist_norm)
                hist_features = np.column_stack([hist_norm, vx, vy])  # (4, 4)

                # Collect neighbor info (all other agents in this window)
                neighbors = []
                for nb_token, nb_agent in window_agents.items():
                    if nb_token == target_token:
                        continue
                    nb_hist = nb_agent['xy'][:HIST_LEN]
                    nb_fut = nb_agent['xy'][HIST_LEN:]
                    # Normalize neighbor relative to target
                    nb_hist_centered = nb_hist - origin
                    nb_fut_centered = nb_fut - origin
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    nb_hist_norm = nb_hist_centered @ rot.T
                    nb_fut_norm = nb_fut_centered @ rot.T
                    nb_vx, nb_vy = compute_velocities(nb_hist_norm)
                    nb_features = np.column_stack([nb_hist_norm, nb_vx, nb_vy])
                    neighbors.append({
                        'hist': nb_features,
                        'fut': nb_fut_norm,
                        'agent_type': nb_agent['agent_type'],
                        'is_moving': nb_agent['is_moving']
                    })

                samples.append({
                    'hist': hist_features.astype(np.float32),       # (4, 4)
                    'fut': fut_norm.astype(np.float32),             # (6, 2)
                    'agent_type': target_agent['agent_type'],       # 0 or 1
                    'neighbors': neighbors,
                    'scene': scene['scene_name'],
                    'origin': origin,
                    'angle': angle,
                })

    return samples

# Execute Step 1
if RESUME_FROM_STEP <= 1:
    print("Loading nuScenes JSON files...")
    raw_data = load_nuscenes_json(DATAROOT)
    lookup_tables = build_lookup_tables(raw_data)

    print("\nExtracting trajectories...")
    scene_data = extract_trajectories(raw_data, lookup_tables)
    for s in scene_data:
        print(f"  {s['scene_name']}: {s['num_samples']} samples, {len(s['agents'])} agents")

    print("\nCreating sliding windows...")
    all_samples = create_sliding_windows(scene_data)
    print(f"  Total training samples (moving agents): {len(all_samples)}")

    # Count by type
    ped_count = sum(1 for s in all_samples if s['agent_type'] == 0)
    cyc_count = sum(1 for s in all_samples if s['agent_type'] == 1)
    print(f"  Pedestrians: {ped_count}, Cyclists: {cyc_count}")

    # Save processed data
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_samples, f)
    print(f"  Saved to {pkl_path}")

    # Plot 10 moving trajectories
    print("\nPlotting 10 sample trajectories...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    plot_samples = random.sample(all_samples, min(10, len(all_samples)))
    for idx, (ax, s) in enumerate(zip(axes.flat, plot_samples)):
        hist = s['hist'][:, :2]
        fut = s['fut']
        ax.plot(hist[:, 0], hist[:, 1], 'b.-', linewidth=2, markersize=8, label='History')
        ax.plot(fut[:, 0], fut[:, 1], 'r.-', linewidth=2, markersize=8, label='Future')
        ax.plot(0, 0, 'g*', markersize=15, label='Origin')
        ax.set_title(f"Sample {idx+1} ({'Ped' if s['agent_type']==0 else 'Cyc'})")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    plt.suptitle("Step 1: Sample Normalized Trajectories (Blue=History, Red=Future)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'step1_trajectories.png'), dpi=150)
    print("  Saved step1_trajectories.png")

else:
    print(f"\nResuming from Step {RESUME_FROM_STEP}...")
    pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data.pkl')
    print(f"Loading pre-processed data from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        all_samples = pickle.load(f)
    print(f"  Loaded {len(all_samples)} training samples.")

# ════════════════════════════════════════════════════════════════════════════
# STEP 2: BASELINE GRU
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: BASELINE GRU")
print("="*70)

class TrajectoryDataset(Dataset):
    """Simple dataset for trajectory prediction (no social context)."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        hist = torch.tensor(s['hist'], dtype=torch.float32)       # (4, 4)
        fut = torch.tensor(s['fut'], dtype=torch.float32)         # (6, 2)
        agent_type = torch.tensor(s['agent_type'], dtype=torch.long)
        return hist, fut, agent_type


class BaselineGRU(nn.Module):
    """Simple GRU encoder → MLP decoder. Single prediction."""
    def __init__(self, input_dim=4, hidden_dim=128, future_len=6):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, future_len * 2)
        )
        self.future_len = future_len

    def forward(self, hist):
        _, h = self.encoder(hist)             # h: (1, B, hidden)
        h = h.squeeze(0)                       # (B, hidden)
        out = self.decoder(h)                  # (B, future_len*2)
        return out.view(-1, self.future_len, 2)


def compute_ade_fde(pred, gt):
    """
    pred: (B, 6, 2) or (B, K, 6, 2)
    gt:   (B, 6, 2)
    Returns minADE, minFDE (averaged over batch)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)  # (B, 1, 6, 2)
    gt = gt.unsqueeze(1)          # (B, 1, 6, 2)
    errors = torch.norm(pred - gt, dim=-1)  # (B, K, 6)
    ade_per_mode = errors.mean(dim=-1)      # (B, K)
    fde_per_mode = errors[:, :, -1]         # (B, K)
    min_ade = ade_per_mode.min(dim=-1)[0].mean().item()
    min_fde = fde_per_mode.min(dim=-1)[0].mean().item()
    return min_ade, min_fde


def train_baseline_gru(samples, epochs=50, batch_size=64, lr=1e-3):
    """Train the baseline GRU model."""
    dataset = TrajectoryDataset(samples)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = BaselineGRU().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    best_fde = float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for hist, fut, _ in train_loader:
            hist, fut = hist.to(DEVICE), fut.to(DEVICE)
            pred = model(hist)
            loss = F.mse_loss(pred, fut)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            all_ade, all_fde = [], []
            with torch.no_grad():
                for hist, fut, _ in val_loader:
                    hist, fut = hist.to(DEVICE), fut.to(DEVICE)
                    pred = model(hist)
                    ade, fde = compute_ade_fde(pred, fut)
                    all_ade.append(ade)
                    all_fde.append(fde)
            val_ade = np.mean(all_ade)
            val_fde = np.mean(all_fde)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"ADE: {val_ade:.4f} | FDE: {val_fde:.4f}")
            if val_fde < best_fde:
                best_fde = val_fde
                torch.save(model.state_dict(), os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), 'baseline_gru.pt'))

    # Final evaluation
    model.eval()
    all_ade, all_fde = [], []
    with torch.no_grad():
        for hist, fut, _ in val_loader:
            hist, fut = hist.to(DEVICE), fut.to(DEVICE)
            pred = model(hist)
            ade, fde = compute_ade_fde(pred, fut)
            all_ade.append(ade)
            all_fde.append(fde)
    final_ade = np.mean(all_ade)
    final_fde = np.mean(all_fde)
    print(f"\n  ★ Baseline GRU Final — ADE: {final_ade:.4f}m | FDE: {final_fde:.4f}m")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Step 2: Baseline GRU Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'step2_loss.png'), dpi=150)
    print("  Saved step2_loss.png")

    return model, train_ds, val_ds, final_ade, final_fde

# Train baseline
if not QUICK_TEST and RESUME_FROM_STEP <= 2:
    baseline_model, train_ds, val_ds, baseline_ade, baseline_fde = train_baseline_gru(all_samples, epochs=25)

# ════════════════════════════════════════════════════════════════════════════
# STEP 3: TRANSFORMER TEMPORAL ENCODER
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3: TRANSFORMER TEMPORAL ENCODER")
print("="*70)


class RelativePositionalEncoding(nn.Module):
    """Learnable relative positional encoding for temporal sequences."""
    def __init__(self, d_model, max_len=20):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


class TransformerTemporalEncoder(nn.Module):
    """Transformer encoder for temporal history features."""
    def __init__(self, input_dim=4, d_model=64, nhead=4, num_layers=2,
                 agent_type_dim=8, num_agent_types=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = RelativePositionalEncoding(d_model)
        self.agent_type_embed = nn.Embedding(num_agent_types, agent_type_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.2, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model + agent_type_dim

    def forward(self, hist, agent_type):
        """
        hist: (B, T, 4) — x, y, vx, vy
        agent_type: (B,) — 0=ped, 1=cyclist
        Returns: (B, d_model + agent_type_dim)
        """
        x = self.input_proj(hist)         # (B, T, d_model)
        x = self.pos_enc(x)               # (B, T, d_model)
        x = self.transformer(x)           # (B, T, d_model)
        temporal_embed = x.mean(dim=1)    # (B, d_model) — mean pooling
        type_embed = self.agent_type_embed(agent_type)  # (B, agent_type_dim)
        return torch.cat([temporal_embed, type_embed], dim=-1)  # (B, d_model+8)


class TransformerGRUModel(nn.Module):
    """Transformer temporal encoder + MLP decoder (no social yet)."""
    def __init__(self, d_model=64, agent_type_dim=8, future_len=6):
        super().__init__()
        self.temporal_enc = TransformerTemporalEncoder(
            input_dim=4, d_model=d_model, agent_type_dim=agent_type_dim
        )
        ctx_dim = d_model + agent_type_dim  # 72
        self.fusion = nn.Sequential(
            nn.Linear(ctx_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, future_len * 2)
        )
        self.future_len = future_len

    def forward(self, hist, agent_type):
        ctx = self.temporal_enc(hist, agent_type)
        out = self.decoder(ctx)
        return out.view(-1, self.future_len, 2)


def train_model_simple(model, train_loader, val_loader, epochs=50, lr=1e-3, label="Model"):
    """Generic training loop for single-prediction models."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []
    best_fde = float('inf')
    save_name = label.lower().replace(" ", "_") + ".pt"
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_name)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for hist, fut, atype in train_loader:
            hist, fut, atype = hist.to(DEVICE), fut.to(DEVICE), atype.to(DEVICE)
            pred = model(hist, atype)
            loss = F.mse_loss(pred, fut)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            all_ade, all_fde = [], []
            with torch.no_grad():
                for hist, fut, atype in val_loader:
                    hist, fut, atype = hist.to(DEVICE), fut.to(DEVICE), atype.to(DEVICE)
                    pred = model(hist, atype)
                    ade, fde = compute_ade_fde(pred, fut)
                    all_ade.append(ade)
                    all_fde.append(fde)
            val_ade, val_fde = np.mean(all_ade), np.mean(all_fde)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"ADE: {val_ade:.4f} | FDE: {val_fde:.4f}")
            if val_fde < best_fde:
                best_fde = val_fde
                torch.save(model.state_dict(), save_path)

    model.eval()
    all_ade, all_fde = [], []
    with torch.no_grad():
        for hist, fut, atype in val_loader:
            hist, fut, atype = hist.to(DEVICE), fut.to(DEVICE), atype.to(DEVICE)
            pred = model(hist, atype)
            ade, fde = compute_ade_fde(pred, fut)
            all_ade.append(ade)
            all_fde.append(fde)
    final_ade, final_fde = np.mean(all_ade), np.mean(all_fde)
    print(f"\n  ★ {label} Final — ADE: {final_ade:.4f}m | FDE: {final_fde:.4f}m")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title(f'{label} Training Loss')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    fig_name = label.lower().replace(" ", "_") + "_loss.png"
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fig_name), dpi=150)
    print(f"  Saved {fig_name}")
    return model, final_ade, final_fde


# Create data loaders (reuse same train/val split)
dataset = TrajectoryDataset(all_samples)
n_val = max(1, int(0.2 * len(dataset)))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                 generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

if not QUICK_TEST and RESUME_FROM_STEP <= 3:
    # Train Transformer model
    transformer_model = TransformerGRUModel().to(DEVICE)
    transformer_model, trans_ade, trans_fde = train_model_simple(
        transformer_model, train_loader, val_loader, epochs=25, label="Step3 Transformer"
    )

# ════════════════════════════════════════════════════════════════════════════
# STEP 4: GAT SOCIAL ENCODER
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 4: GAT SOCIAL ENCODER")
print("="*70)


class SocialDataset(Dataset):
    """Dataset that also returns neighbor information for social encoding."""
    def __init__(self, samples, max_neighbors=10):
        self.samples = samples
        self.max_neighbors = max_neighbors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        hist = torch.tensor(s['hist'], dtype=torch.float32)
        fut = torch.tensor(s['fut'], dtype=torch.float32)
        agent_type = torch.tensor(s['agent_type'], dtype=torch.long)

        # Pack neighbor histories (pad to max_neighbors)
        nb_hists = []
        nb_types = []
        nb_mask = []
        for nb in s['neighbors'][:self.max_neighbors]:
            nb_hists.append(torch.tensor(nb['hist'], dtype=torch.float32))
            nb_types.append(nb['agent_type'])
            nb_mask.append(1.0)

        # Pad remaining slots
        while len(nb_hists) < self.max_neighbors:
            nb_hists.append(torch.zeros(HIST_LEN, 4))
            nb_types.append(0)
            nb_mask.append(0.0)

        nb_hists = torch.stack(nb_hists)                    # (max_nb, 4, 4)
        nb_types = torch.tensor(nb_types, dtype=torch.long) # (max_nb,)
        nb_mask = torch.tensor(nb_mask, dtype=torch.float32) # (max_nb,)

        return hist, fut, agent_type, nb_hists, nb_types, nb_mask


class GraphAttentionLayer(nn.Module):
    """Single GAT layer with edge features."""
    def __init__(self, node_dim, edge_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        assert out_dim % heads == 0

        self.W_node = nn.Linear(node_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, heads, bias=False)
        self.attn_src = nn.Linear(self.head_dim, 1, bias=False)
        self.attn_dst = nn.Linear(self.head_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, nodes, edge_features, mask):
        """
        nodes: (B, N, node_dim)
        edge_features: (B, N, edge_dim)  — edge from target to each neighbor
        mask: (B, N) — 1 for real neighbors, 0 for padding
        Returns: (B, out_dim) — aggregated social context for target node
        """
        B, N, _ = nodes.size()
        h = self.W_node(nodes)  # (B, N, out_dim)
        h = h.view(B, N, self.heads, self.head_dim)  # (B, N, heads, head_dim)

        # Attention scores
        attn = self.attn_src(h).squeeze(-1) + self.attn_dst(h).squeeze(-1)  # (B, N, heads)
        edge_attn = self.W_edge(edge_features)  # (B, N, heads)
        attn = attn + edge_attn
        attn = self.leaky_relu(attn)

        # Mask out padding
        mask_expanded = mask.unsqueeze(-1).expand_as(attn)  # (B, N, heads)
        attn = attn.masked_fill(mask_expanded == 0, float('-inf'))
        attn = F.softmax(attn, dim=1)  # (B, N, heads)
        attn = attn.masked_fill(mask_expanded == 0, 0.0)

        # Weighted sum
        attn = attn.unsqueeze(-1)  # (B, N, heads, 1)
        weighted = (h * attn).sum(dim=1)  # (B, heads, head_dim)
        weighted = weighted.view(B, -1)   # (B, out_dim)
        return self.out_proj(weighted)


class GATSocialEncoder(nn.Module):
    """GAT-based social encoder with edge features."""
    def __init__(self, temporal_dim=72, edge_dim=4, social_dim=64, heads=4,
                 agent_type_dim=8):
        super().__init__()
        self.temporal_dim = temporal_dim
        self.social_dim = social_dim
        # Neighbor encoder must produce same dim as temporal_dim
        nb_d_model = temporal_dim - agent_type_dim
        self.neighbor_encoder = TransformerTemporalEncoder(
            input_dim=4, d_model=nb_d_model, agent_type_dim=agent_type_dim
        )
        self.gat = GraphAttentionLayer(
            node_dim=temporal_dim, edge_dim=edge_dim,
            out_dim=social_dim, heads=heads
        )
        self.output_dim = social_dim

    def compute_edge_features(self, target_hist, nb_hists, nb_mask):
        """
        Compute edge features: [distance, rel_vx, rel_vy, TTC_approx]
        target_hist: (B, 4, 4)
        nb_hists: (B, max_nb, 4, 4)
        nb_mask: (B, max_nb)
        """
        B, N = nb_mask.shape
        # Last position of target
        target_pos = target_hist[:, -1, :2]  # (B, 2)
        target_vel = target_hist[:, -1, 2:]  # (B, 2)

        # Last position/vel of neighbors
        nb_pos = nb_hists[:, :, -1, :2]   # (B, N, 2)
        nb_vel = nb_hists[:, :, -1, 2:]   # (B, N, 2)

        # Distance
        diff = nb_pos - target_pos.unsqueeze(1)   # (B, N, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True)  # (B, N, 1)

        # Relative velocity
        rel_vel = nb_vel - target_vel.unsqueeze(1)  # (B, N, 2)

        # Time to collision (approximate: dist / closing_speed)
        closing_speed = -torch.sum(diff * rel_vel, dim=-1, keepdim=True) / (dist + 1e-8)
        ttc = dist / (closing_speed.abs() + 1e-6)
        ttc = ttc.clamp(0, 10)  # cap at 10 seconds

        edge_feat = torch.cat([dist, rel_vel, ttc], dim=-1)  # (B, N, 4)
        return edge_feat

    def forward(self, target_embed, target_hist, nb_hists, nb_types, nb_mask):
        """
        target_embed: (B, temporal_dim) — from temporal encoder
        target_hist: (B, 4, 4)
        nb_hists: (B, max_nb, 4, 4)
        nb_types: (B, max_nb)
        nb_mask: (B, max_nb)
        """
        B, N, T, D = nb_hists.shape
        has_neighbors = nb_mask.sum(dim=-1) > 0  # (B,)

        if not has_neighbors.any():
            return torch.zeros(B, self.social_dim, device=target_hist.device)

        # Encode all neighbors through temporal encoder
        nb_flat = nb_hists.view(B * N, T, D)
        nb_types_flat = nb_types.view(B * N)
        nb_embeds = self.neighbor_encoder(nb_flat, nb_types_flat)  # (B*N, temporal_dim)
        nb_embeds = nb_embeds.view(B, N, -1)  # (B, N, temporal_dim)

        # Compute edge features
        edge_feat = self.compute_edge_features(target_hist, nb_hists, nb_mask)

        # GAT attention
        social_ctx = self.gat(nb_embeds, edge_feat, nb_mask)  # (B, social_dim)

        # Zero out for samples with no neighbors
        social_ctx = social_ctx * has_neighbors.float().unsqueeze(-1)
        return social_ctx



class TransformerGATModel(nn.Module):
    """Transformer temporal + GAT social encoder → MLP decoder."""
    def __init__(self, d_model=64, agent_type_dim=8, social_dim=64, future_len=6):
        super().__init__()
        self.temporal_enc = TransformerTemporalEncoder(
            input_dim=4, d_model=d_model, agent_type_dim=agent_type_dim
        )
        temporal_dim = d_model + agent_type_dim  # 72
        self.social_enc = GATSocialEncoder(
            temporal_dim=temporal_dim, social_dim=social_dim
        )
        ctx_dim = temporal_dim + social_dim  # 72 + 64 = 136
        self.fusion = nn.Sequential(
            nn.Linear(ctx_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, future_len * 2)
        )
        self.future_len = future_len

    def forward(self, hist, agent_type, nb_hists=None, nb_types=None, nb_mask=None):
        temporal_embed = self.temporal_enc(hist, agent_type)  # (B, 72)

        if nb_hists is not None:
            social_ctx = self.social_enc(
                temporal_embed, hist, nb_hists, nb_types, nb_mask
            )
        else:
            social_ctx = torch.zeros(hist.size(0), self.social_enc.output_dim,
                                     device=hist.device)

        fused = torch.cat([temporal_embed, social_ctx], dim=-1)
        ctx = self.fusion(fused)
        out = self.decoder(ctx)
        return out.view(-1, self.future_len, 2)


def train_social_model(model, samples, epochs=50, batch_size=64, lr=1e-3, label="Model"):
    """Training loop for social model (uses SocialDataset)."""
    social_dataset = SocialDataset(samples)
    n_val = max(1, int(0.2 * len(social_dataset)))
    n_train = len(social_dataset) - n_val
    s_train, s_val = random_split(social_dataset, [n_train, n_val],
                                   generator=torch.Generator().manual_seed(42))
    s_train_loader = DataLoader(s_train, batch_size=batch_size, shuffle=True, drop_last=True)
    s_val_loader = DataLoader(s_val, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []
    best_fde = float('inf')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             label.lower().replace(" ", "_") + ".pt")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for hist, fut, atype, nb_h, nb_t, nb_m in s_train_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
            pred = model(hist, atype, nb_h, nb_t, nb_m)
            loss = F.mse_loss(pred, fut)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(s_train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            all_ade, all_fde = [], []
            with torch.no_grad():
                for hist, fut, atype, nb_h, nb_t, nb_m in s_val_loader:
                    hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
                    nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
                    pred = model(hist, atype, nb_h, nb_t, nb_m)
                    ade, fde = compute_ade_fde(pred, fut)
                    all_ade.append(ade); all_fde.append(fde)
            val_ade, val_fde = np.mean(all_ade), np.mean(all_fde)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"ADE: {val_ade:.4f} | FDE: {val_fde:.4f}")
            if val_fde < best_fde:
                best_fde = val_fde
                torch.save(model.state_dict(), save_path)

    model.eval()
    all_ade, all_fde = [], []
    with torch.no_grad():
        for hist, fut, atype, nb_h, nb_t, nb_m in s_val_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
            pred = model(hist, atype, nb_h, nb_t, nb_m)
            ade, fde = compute_ade_fde(pred, fut)
            all_ade.append(ade); all_fde.append(fde)
    final_ade, final_fde = np.mean(all_ade), np.mean(all_fde)
    print(f"\n  ★ {label} Final — ADE: {final_ade:.4f}m | FDE: {final_fde:.4f}m")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.title(f'{label} Training Loss'); plt.grid(True, alpha=0.3); plt.tight_layout()
    fig_name = label.lower().replace(" ", "_") + "_loss.png"
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fig_name), dpi=150)
    print(f"  Saved {fig_name}")
    return model, final_ade, final_fde


if not QUICK_TEST and RESUME_FROM_STEP <= 4:
    # Train Transformer+GAT model
    gat_model = TransformerGATModel().to(DEVICE)
    gat_model, gat_ade, gat_fde = train_social_model(
        gat_model, all_samples, epochs=25, label="Step4 GAT"
    )

# ════════════════════════════════════════════════════════════════════════════
# STEP 5: GOAL PREDICTOR + CVAE + GRU DECODER (WTA LOSS)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 5: GOAL PREDICTOR + CVAE + GRU DECODER")
print("="*70)


class GoalPredictor(nn.Module):
    """Predict K goal endpoints from context vector."""
    def __init__(self, ctx_dim=128, num_goals=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_goals * 2)
        )
        self.num_goals = num_goals

    def forward(self, ctx):
        return self.net(ctx).view(-1, self.num_goals, 2)  # (B, K, 2)


class CVAE(nn.Module):
    """Conditional VAE: encodes ground-truth future (training) or samples prior (inference)."""
    def __init__(self, ctx_dim=128, goal_dim=2, latent_dim=16, future_dim=12):
        super().__init__()
        self.latent_dim = latent_dim
        # Prior: p(z | ctx, goal)
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_dim + goal_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )
        # Posterior: q(z | ctx, goal, future) — used only during training
        self.posterior_net = nn.Sequential(
            nn.Linear(ctx_dim + goal_dim + future_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, ctx, goal, future_flat=None):
        """
        ctx: (B, ctx_dim)
        goal: (B, 2)
        future_flat: (B, 12) — only during training
        Returns: z, kl_divergence (per-sample)
        """
        # Prior
        prior_input = torch.cat([ctx, goal], dim=-1)
        prior_params = self.prior_net(prior_input)
        prior_mu, prior_logvar = prior_params.chunk(2, dim=-1)

        if future_flat is not None:
            # Training: use posterior
            post_input = torch.cat([ctx, goal, future_flat], dim=-1)
            post_params = self.posterior_net(post_input)
            post_mu, post_logvar = post_params.chunk(2, dim=-1)
            z = self.reparameterize(post_mu, post_logvar)
            # KL divergence: KL(q || p) — correct signs
            kl = 0.5 * torch.sum(
                prior_logvar - post_logvar - 1
                + torch.exp(post_logvar - prior_logvar)
                + (post_mu - prior_mu).pow(2) / torch.exp(prior_logvar),
                dim=-1
            )
            return z, kl
        else:
            # Inference: sample from prior
            z = self.reparameterize(prior_mu, prior_logvar)
            return z, torch.zeros(z.size(0), device=ctx.device)


class GRUDecoder(nn.Module):
    """Autoregressive GRU decoder: predicts delta_x, delta_y at each step."""
    def __init__(self, z_dim=16, ctx_dim=128, goal_dim=2, hidden_dim=128, future_len=6):
        super().__init__()
        self.future_len = future_len
        self.hidden_dim = hidden_dim
        input_dim = z_dim + ctx_dim + goal_dim + 2  # +2 for previous position
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 2)  # delta_x, delta_y
        self.hidden_init = nn.Sequential(
            nn.Linear(z_dim + ctx_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, z, ctx, goal):
        """
        z: (B, z_dim), ctx: (B, ctx_dim), goal: (B, 2)
        Returns: (B, future_len, 2) — predicted trajectory
        """
        B = z.size(0)
        init_input = torch.cat([z, ctx, goal], dim=-1)
        h = torch.tanh(self.hidden_init(init_input))  # (B, hidden_dim)
        pos = torch.zeros(B, 2, device=z.device)  # start at origin (last hist point)
        trajectory = []

        for t in range(self.future_len):
            gru_input = torch.cat([z, ctx, goal, pos], dim=-1)
            h = self.gru_cell(gru_input, h)
            delta = self.output_proj(h)  # (B, 2)
            pos = pos + delta
            trajectory.append(pos)

        return torch.stack(trajectory, dim=1)  # (B, future_len, 2)


class FullModel(nn.Module):
    """Complete model: Transformer + GAT + Goal + CVAE + GRU Decoder."""
    def __init__(self, d_model=96, agent_type_dim=8, social_dim=96,
                 num_goals=6, latent_dim=16, future_len=6, decoder_hidden=128):
        super().__init__()
        self.num_goals = num_goals
        self.future_len = future_len

        # Encoders (bigger capacity for Steps 5-7)
        self.temporal_enc = TransformerTemporalEncoder(
            input_dim=4, d_model=d_model, agent_type_dim=agent_type_dim
        )
        temporal_dim = d_model + agent_type_dim  # 104
        self.social_enc = GATSocialEncoder(
            temporal_dim=temporal_dim, social_dim=social_dim
        )
        ctx_input_dim = temporal_dim + social_dim  # 104 + 96 = 200
        self.fusion = nn.Sequential(
            nn.Linear(ctx_input_dim, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        ctx_dim = 128

        # Goal predictor
        self.goal_predictor = GoalPredictor(ctx_dim, num_goals)

        # CVAE (one per goal — shared weights)
        self.cvae = CVAE(ctx_dim, goal_dim=2, latent_dim=latent_dim,
                         future_dim=future_len * 2)

        # GRU Decoder (shared across goals)
        self.decoder = GRUDecoder(z_dim=latent_dim, ctx_dim=ctx_dim,
                                  goal_dim=2, hidden_dim=decoder_hidden,
                                  future_len=future_len)

    def forward(self, hist, agent_type, nb_hists, nb_types, nb_mask,
                future=None, beta=1.0):
        """
        Training: future is provided, returns loss components
        Inference: future is None, returns K predicted trajectories
        """
        B = hist.size(0)
        # Encode
        temporal = self.temporal_enc(hist, agent_type)
        social = self.social_enc(temporal, hist, nb_hists, nb_types, nb_mask)
        ctx = self.fusion(torch.cat([temporal, social], dim=-1))  # (B, 128)

        # Predict goals
        goals = self.goal_predictor(ctx)  # (B, K, 2)

        if future is not None:
            # TRAINING MODE with WTA loss
            gt_endpoint = future[:, -1, :]  # (B, 2)
            future_flat = future.reshape(B, -1)  # (B, 12)

            all_pred_trajs = []
            all_kl = []
            for k in range(self.num_goals):
                goal_k = goals[:, k, :]
                z_k, kl_k = self.cvae(ctx, goal_k, future_flat)
                traj_k = self.decoder(z_k, ctx, goal_k)
                all_pred_trajs.append(traj_k)
                all_kl.append(kl_k)

            all_pred_trajs = torch.stack(all_pred_trajs, dim=1)  # (B, K, 6, 2)
            all_kl = torch.stack(all_kl, dim=1)  # (B, K)

            # Identify winner based on trajectory error
            traj_errors = torch.norm(all_pred_trajs - future.unsqueeze(1), dim=-1).mean(dim=-1) # (B, K)
            best_idx = traj_errors.argmin(dim=-1) # (B,)

            # WTA Loss: only winner's components are penalized
            best_trajs = all_pred_trajs[torch.arange(B), best_idx]
            traj_loss = F.mse_loss(best_trajs, future)

            # Goal loss (WTA)
            best_goals = goals[torch.arange(B), best_idx]
            goal_loss = F.mse_loss(best_goals, gt_endpoint)

            # KL loss (WTA)
            kl_loss = all_kl[torch.arange(B), best_idx].mean()

            total_loss = traj_loss + goal_loss + beta * kl_loss

            return total_loss, traj_loss, goal_loss, kl_loss, best_trajs

        else:
            # INFERENCE MODE: generate K trajectories
            # Sample z once per call for diverse "intentions" across calls
            z, _ = self.cvae(ctx, goals[:, 0, :]) 
            all_trajs = []
            for k in range(self.num_goals):
                goal_k = goals[:, k, :]  # (B, 2)
                traj_k = self.decoder(z, ctx, goal_k)  # (B, 6, 2)
                all_trajs.append(traj_k)
            return torch.stack(all_trajs, dim=1)  # (B, K, 6, 2)


def scene_based_split(samples, n_val_scenes=1, seed=42):
    random.seed(seed)
    scene_names = list(set(s['scene'] for s in samples))
    scene_names.sort()
    val_scenes = set(scene_names[-1:])  # last 1 scene as val
    train_samples = [s for s in samples if s['scene'] not in val_scenes]
    val_samples = [s for s in samples if s['scene'] in val_scenes]
    print(f"  Scene split: {len(scene_names)-1} Train Scenes, 1 Val Scenes")
    print(f"  Samples: {len(train_samples)} Train, {len(val_samples)} Val")
    return train_samples, val_samples


def train_full_model(model, samples, epochs=150, batch_size=32, lr=1e-3,
                     beta_start=0.0, beta_end=1.0, beta_warmup=15, label="Model"):
    """Train full CVAE model with KL annealing."""
    train_samples, val_samples = scene_based_split(samples)
    train_ds = SocialDataset(train_samples)
    val_ds = SocialDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []
    best_fde = float('inf')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             label.lower().replace(" ", "_") + ".pt")

    for epoch in range(epochs):
        # KL annealing
        if epoch < beta_warmup:
            beta = beta_start + (beta_end - beta_start) * epoch / beta_warmup
        else:
            beta = beta_end

        model.train()
        epoch_loss = 0
        for hist, fut, atype, nb_h, nb_t, nb_m in train_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)

            total_loss, _, _, _, _ = model(
                hist, atype, nb_h, nb_t, nb_m, future=fut, beta=beta
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += total_loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 15 == 0 or epoch == 0:
            model.eval()
            all_ade, all_fde = [], []
            with torch.no_grad():
                for hist, fut, atype, nb_h, nb_t, nb_m in val_loader:
                    hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
                    nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
                    pred = model(hist, atype, nb_h, nb_t, nb_m)  # (B, K, 6, 2)
                    ade, fde = compute_ade_fde(pred, fut)
                    all_ade.append(ade); all_fde.append(fde)
            val_ade, val_fde = np.mean(all_ade), np.mean(all_fde)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"β: {beta:.3f} | minADE: {val_ade:.4f} | minFDE: {val_fde:.4f}")
            if val_fde < best_fde:
                best_fde = val_fde
                torch.save(model.state_dict(), save_path)

    model.eval()
    all_ade, all_fde = [], []
    with torch.no_grad():
        for hist, fut, atype, nb_h, nb_t, nb_m in val_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
            pred = model(hist, atype, nb_h, nb_t, nb_m)
            ade, fde = compute_ade_fde(pred, fut)
            all_ade.append(ade); all_fde.append(fde)
    final_ade, final_fde = np.mean(all_ade), np.mean(all_fde)
    print(f"\n  ★ {label} Final — minADE@6: {final_ade:.4f}m | minFDE@6: {final_fde:.4f}m")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Total Loss')
    plt.title(f'{label} Training Loss'); plt.grid(True, alpha=0.3); plt.tight_layout()
    fig_name = label.lower().replace(" ", "_") + "_loss.png"
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fig_name), dpi=150)
    print(f"  Saved {fig_name}")
    return model, final_ade, final_fde


if not QUICK_TEST and RESUME_FROM_STEP <= 5:
    # Train Step 5 model (bigger model + 150 epochs)
    full_model = FullModel().to(DEVICE)
    print(f"  FullModel params: {sum(p.numel() for p in full_model.parameters()):,}")
    full_model, step5_ade, step5_fde = train_full_model(
        full_model, all_samples, epochs=150, beta_warmup=15, label="Step5 GoalCVAE"
    )


# ════════════════════════════════════════════════════════════════════════════
# STEP 6: HINGE DIVERSITY LOSS + KL ANNEALING
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 6: HINGE DIVERSITY LOSS + KL ANNEALING")
print("="*70)


class FullModelV2(FullModel):
    """FullModel with hinge diversity loss on goal endpoints."""

    def forward(self, hist, agent_type, nb_hists, nb_types, nb_mask,
                future=None, beta=1.0, diversity_weight=1.0, hinge_margin=2.0):
        B = hist.size(0)
        # Encode
        temporal = self.temporal_enc(hist, agent_type)
        social = self.social_enc(temporal, hist, nb_hists, nb_types, nb_mask)
        ctx = self.fusion(torch.cat([temporal, social], dim=-1))

        # Predict goals
        goals = self.goal_predictor(ctx)  # (B, K, 2)

        if future is not None:
            gt_endpoint = future[:, -1, :]
            future_flat = future.reshape(B, -1)

            # Decode all modes using posterior to find winner
            all_pred_trajs = []
            all_kl = []
            for k in range(self.num_goals):
                goal_k = goals[:, k, :]
                z_k, kl_k = self.cvae(ctx, goal_k, future_flat)
                traj_k = self.decoder(z_k, ctx, goal_k)
                all_pred_trajs.append(traj_k)
                all_kl.append(kl_k)

            all_pred_trajs = torch.stack(all_pred_trajs, dim=1) # (B, K, 6, 2)
            all_kl = torch.stack(all_kl, dim=1) # (B, K)

            # Identify winner based on trajectory error
            traj_errors = torch.norm(all_pred_trajs - future.unsqueeze(1), dim=-1).mean(dim=-1)
            best_idx = traj_errors.argmin(dim=-1)

            # WTA logic: only winner gets losses penalized
            best_trajs = all_pred_trajs[torch.arange(B), best_idx]
            # ADE + 2*FDE loss
            errors = torch.norm(best_trajs - future, dim=-1)
            ade_loss = errors.mean()
            fde_loss = errors[:, -1].mean()
            traj_loss = ade_loss + 2.0 * fde_loss

            # Goal loss (WTA)
            winner_goal = goals[torch.arange(B), best_idx]
            goal_loss = F.mse_loss(winner_goal, gt_endpoint)

            # KL loss (WTA)
            kl_loss = all_kl[torch.arange(B), best_idx].mean()

            # Hinge diversity loss on goal endpoints
            K = self.num_goals
            div_loss = torch.tensor(0.0, device=hist.device)
            count = 0
            for i in range(K):
                for j in range(i + 1, K):
                    pair_dist = torch.norm(goals[:, i] - goals[:, j], dim=-1)
                    div_loss = div_loss + F.relu(hinge_margin - pair_dist).mean()
                    count += 1
            div_loss = div_loss / max(count, 1)

            total_loss = (traj_loss + goal_loss
                         + beta * kl_loss
                         + diversity_weight * div_loss)

            return total_loss, traj_loss, goal_loss, kl_loss, div_loss, best_trajs
        else:
            # Inference: sample z once per call
            z, _ = self.cvae(ctx, goals[:, 0, :])
            all_trajs = []
            for k in range(self.num_goals):
                goal_k = goals[:, k, :]
                traj_k = self.decoder(z, ctx, goal_k)
                all_trajs.append(traj_k)
            return torch.stack(all_trajs, dim=1)


def train_full_model_v2(model, samples, epochs=150, batch_size=32, lr=5e-4,
                        beta_warmup=15, diversity_weight=1.0, hinge_margin=2.0, label="Model"):
    """Train with hinge diversity + KL annealing."""
    train_samples, val_samples = scene_based_split(samples)
    train_ds = SocialDataset(train_samples)
    val_ds = SocialDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []
    best_ade = float('inf')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             label.lower().replace(" ", "_") + ".pt")
                             
    patience_counter = 0

    for epoch in range(epochs):
        beta = min(1.0, epoch / max(beta_warmup, 1))

        model.train()
        epoch_loss = 0
        for hist, fut, atype, nb_h, nb_t, nb_m in train_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)

            total_loss, _, _, _, _, _ = model(
                hist, atype, nb_h, nb_t, nb_m, future=fut,
                beta=beta, diversity_weight=diversity_weight, hinge_margin=hinge_margin
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += total_loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        all_ade, all_fde = [], []
        with torch.no_grad():
            for hist, fut, atype, nb_h, nb_t, nb_m in val_loader:
                hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
                nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
                pred = model(hist, atype, nb_h, nb_t, nb_m)
                ade, fde = compute_ade_fde(pred, fut)
                all_ade.append(ade); all_fde.append(fde)
        val_ade, val_fde = np.mean(all_ade), np.mean(all_fde)
        print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"beta: {beta:.3f} | minADE: {val_ade:.4f} | minFDE: {val_fde:.4f}")
              
        if val_ade < best_ade:
            best_ade = val_ade
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 40:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print(f"  Loading best model from {save_path}...")
    model.load_state_dict(torch.load(save_path))

    model.eval()
    all_ade, all_fde = [], []
    with torch.no_grad():
        for hist, fut, atype, nb_h, nb_t, nb_m in val_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
            pred = model(hist, atype, nb_h, nb_t, nb_m)
            ade, fde = compute_ade_fde(pred, fut)
            all_ade.append(ade); all_fde.append(fde)
    final_ade, final_fde = np.mean(all_ade), np.mean(all_fde)
    print(f"\n  * {label} Final -- minADE@6: {final_ade:.4f}m | minFDE@6: {final_fde:.4f}m")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch'); plt.ylabel('Total Loss')
    plt.title(f'{label} Training Loss'); plt.grid(True, alpha=0.3); plt.tight_layout()
    fig_name = label.lower().replace(" ", "_") + "_loss.png"
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), fig_name), dpi=150)
    print(f"  Saved {fig_name}")
    return model, final_ade, final_fde, val_loader


# Train Step 6 model
v2_model = FullModelV2().to(DEVICE)
print(f"  FullModelV2 params: {sum(p.numel() for p in v2_model.parameters()):,}")
v2_model, step6_ade, step6_fde, final_val_loader = train_full_model_v2(
    v2_model, all_samples, epochs=(30 if QUICK_TEST else 150), label="Step6 Diversity"
)

# ════════════════════════════════════════════════════════════════════════════
# OVERFITTING CHECK
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("OVERFITTING CHECK: Train vs Val Performance")
print("="*70)

# Recreate train_loader for evaluation
train_samples, val_samples = scene_based_split(all_samples)
train_ds = SocialDataset(train_samples)
train_eval_loader = DataLoader(train_ds, batch_size=32, shuffle=False)

v2_model.eval()
train_ade_list, train_fde_list = [], []
val_ade_list, val_fde_list = [], []

# Evaluate on training set
with torch.no_grad():
    for hist, fut, atype, nb_h, nb_t, nb_m in train_eval_loader:
        hist = hist.to(DEVICE); fut = fut.to(DEVICE)
        atype = atype.to(DEVICE)
        nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE)
        nb_m = nb_m.to(DEVICE)
        pred = v2_model(hist, atype, nb_h, nb_t, nb_m)
        ade, fde = compute_ade_fde(pred, fut)
        train_ade_list.append(ade)
        train_fde_list.append(fde)

# Evaluate on val set
with torch.no_grad():
    for hist, fut, atype, nb_h, nb_t, nb_m in final_val_loader:
        hist = hist.to(DEVICE); fut = fut.to(DEVICE)
        atype = atype.to(DEVICE)
        nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE)
        nb_m = nb_m.to(DEVICE)
        pred = v2_model(hist, atype, nb_h, nb_t, nb_m)
        ade, fde = compute_ade_fde(pred, fut)
        val_ade_list.append(ade)
        val_fde_list.append(fde)

train_ade = np.mean(train_ade_list)
train_fde = np.mean(train_fde_list)
val_ade_chk = np.mean(val_ade_list)
val_fde_chk = np.mean(val_fde_list)

print(f"Train ADE: {train_ade:.4f} | Train FDE: {train_fde:.4f}")
print(f"Val   ADE: {val_ade_chk:.4f} | Val   FDE: {val_fde_chk:.4f}")
print(f"Gap   ADE: {val_ade_chk - train_ade:.4f} | Gap FDE: {val_fde_chk - train_fde:.4f}")

if val_ade_chk > train_ade * 1.3:
    print("\nWARNING: Likely overfitting — val ADE is 30%+ higher than train ADE")
else:
    print("\nGap is reasonable — not severe overfitting")

# ════════════════════════════════════════════════════════════════════════════
# STEP 7: INFERENCE CLUSTERING (Sample 50, K-means to 6)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 7: INFERENCE CLUSTERING")
print("="*70)


def inference_with_clustering(model, val_loader, n_samples=50, n_clusters=6):
    """
    Sample n_samples trajectories per input, cluster with k-means into
    n_clusters, pick cluster centers as final predictions.
    """
    model.eval()
    all_ade, all_fde = [], []

    with torch.no_grad():
        for hist, fut, atype, nb_h, nb_t, nb_m in val_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
            B = hist.size(0)

            # Bug #3 fix: Sample with random mode selection for diversity
            all_sampled = []
            for _ in range(n_samples):
                trajs = model(hist, atype, nb_h, nb_t, nb_m)  # (B, K, 6, 2)
                k_idx = torch.randint(0, model.num_goals, (B,), device=DEVICE)
                sampled = trajs[torch.arange(B), k_idx]  # (B, 6, 2)
                all_sampled.append(sampled)
            all_sampled = torch.stack(all_sampled, dim=1)  # (B, 50, 6, 2)

            # K-means clustering per sample in batch
            clustered_preds = []
            for b in range(B):
                endpoints = all_sampled[b, :, -1, :].cpu().numpy()  # (50, 2)
                trajs_np = all_sampled[b].cpu().numpy()             # (50, 6, 2)

                try:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
                    labels = kmeans.fit_predict(endpoints)

                    # Pick trajectory closest to each cluster center
                    selected = []
                    for c in range(n_clusters):
                        cluster_mask = labels == c
                        if not cluster_mask.any():
                            selected.append(trajs_np[0])
                            continue
                        cluster_trajs = trajs_np[cluster_mask]
                        center = kmeans.cluster_centers_[c]
                        cluster_endpoints = endpoints[cluster_mask]
                        dists = np.linalg.norm(cluster_endpoints - center, axis=-1)
                        best_idx = dists.argmin()
                        selected.append(cluster_trajs[best_idx])
                    clustered_preds.append(np.stack(selected))
                except Exception:
                    clustered_preds.append(trajs_np[:n_clusters])

            clustered = torch.tensor(np.stack(clustered_preds), device=DEVICE,
                                     dtype=torch.float32)
            ade, fde = compute_ade_fde(clustered, fut)
            all_ade.append(ade)
            all_fde.append(fde)

    final_ade = np.mean(all_ade)
    final_fde = np.mean(all_fde)
    return final_ade, final_fde


# Run inference with clustering
step7_ade, step7_fde = inference_with_clustering(v2_model, final_val_loader)
print(f"\n  * Step 7 (Clustering K=50->6) -- minADE@6: {step7_ade:.4f}m | minFDE@6: {step7_fde:.4f}m")

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"{'Step':<35} {'minADE@6':>10} {'minFDE@6':>10}")
print("-" * 55)
print(f"{'Step 2: Baseline GRU':<35} {baseline_ade:>10.4f} {baseline_fde:>10.4f}")
print(f"{'Step 3: Transformer':<35} {trans_ade:>10.4f} {trans_fde:>10.4f}")
print(f"{'Step 4: Transformer+GAT':<35} {gat_ade:>10.4f} {gat_fde:>10.4f}")
print(f"{'Step 5: Goal+CVAE+GRU':<35} {step5_ade:>10.4f} {step5_fde:>10.4f}")
print(f"{'Step 6: +Diversity+KL':<35} {step6_ade:>10.4f} {step6_fde:>10.4f}")
print(f"{'Step 7: +Clustering (50->6)':<35} {step7_ade:>10.4f} {step7_fde:>10.4f}")
print("=" * 55)

# Plot comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels = ['S2:GRU', 'S3:Trans', 'S4:GAT', 'S5:CVAE', 'S6:Div', 'S7:Cluster']
ades = [baseline_ade, trans_ade, gat_ade, step5_ade, step6_ade, step7_ade]
fdes = [baseline_fde, trans_fde, gat_fde, step5_fde, step6_fde, step7_fde]
colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90', '#87CEEB', '#8A2BE2']

if not QUICK_TEST:
    axes[0].bar(labels, ades, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('minADE@6 (meters)'); axes[0].set_title('Average Displacement Error')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(labels, fdes, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('minFDE@6 (meters)'); axes[1].set_title('Final Displacement Error')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Trajectory Prediction: Step-by-Step Improvement", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_comparison.png'), dpi=150)
    print("\nSaved final_comparison.png")

# Plot sample predictions from final model
print("\nPlotting sample predictions...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
v2_model.eval()
social_ds_plot = SocialDataset(all_samples)
plot_indices = random.sample(range(len(social_ds_plot)), min(10, len(social_ds_plot)))

for i, ax in enumerate(axes.flat):
    if i >= len(plot_indices):
        break
    hist, fut, atype, nb_h, nb_t, nb_m = social_ds_plot[plot_indices[i]]
    hist_d = hist.unsqueeze(0).to(DEVICE)
    atype_d = atype.unsqueeze(0).to(DEVICE)
    nb_h_d = nb_h.unsqueeze(0).to(DEVICE)
    nb_t_d = nb_t.unsqueeze(0).to(DEVICE)
    nb_m_d = nb_m.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = v2_model(hist_d, atype_d, nb_h_d, nb_t_d, nb_m_d)
    preds = preds.squeeze(0).cpu().numpy()

    h_np = hist[:, :2].numpy()
    ax.plot(h_np[:, 0], h_np[:, 1], 'b.-', linewidth=2, markersize=8, label='History')
    f_np = fut.numpy()
    ax.plot(f_np[:, 0], f_np[:, 1], 'r.-', linewidth=2, markersize=8, label='GT Future')
    for k in range(preds.shape[0]):
        ax.plot(preds[k, :, 0], preds[k, :, 1], '--', alpha=0.6, linewidth=1.5)
    ax.plot(0, 0, 'g*', markersize=12)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title(f"Sample {i+1}", fontsize=10)
    if i == 0:
        ax.legend(fontsize=7)

plt.suptitle("Final Model: Multi-Modal Predictions (dashed=predicted, red=GT)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_predictions.png'), dpi=150)
print("Saved final_predictions.png")

print("\nAll steps complete!")


# ════════════════════════════════════════════════════════════════════════════
# STEP 8: TEST TIME AUGMENTATION (Single Model)
# ════════════════════════════════════════════════════════════════════════════


def inference_with_tta_clustering(model, val_loader, n_rotations=8, n_clusters=6):
    """
    Test Time Augmentation: rotate each sample 8 ways,
    collect all predictions (8 rotations x 6 modes = 48 total),
    cluster endpoints to 6 final trajectories using k-means.
    """
    model.eval()
    all_ade, all_fde = [], []
    angles = [2 * np.pi * i / n_rotations for i in range(n_rotations)]

    with torch.no_grad():
        for hist, fut, atype, nb_h, nb_t, nb_m in val_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
            B = hist.size(0)
            all_sampled = []

            for angle in angles:
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]],
                                  dtype=torch.float32, device=DEVICE)
                R_inv = R.T

                hist_r = hist.clone()
                hist_r[:, :, :2] = hist[:, :, :2] @ R.T
                hist_r[:, :, 2:] = hist[:, :, 2:] @ R.T

                nb_h_r = nb_h.clone()
                nb_h_r[:, :, :, :2] = nb_h[:, :, :, :2] @ R.T
                nb_h_r[:, :, :, 2:] = nb_h[:, :, :, 2:] @ R.T

                preds = model(hist_r, atype, nb_h_r, nb_t, nb_m)  # (B, 6, 6, 2)
                preds_orig = preds @ R_inv.T
                all_sampled.append(preds_orig)

            all_sampled = torch.cat(all_sampled, dim=1)  # (B, 48, 6, 2)

            clustered_preds = []
            for b in range(B):
                endpoints = all_sampled[b, :, -1, :].cpu().numpy()
                trajs_np = all_sampled[b].cpu().numpy()
                try:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
                    labels = kmeans.fit_predict(endpoints)
                    selected = []
                    for c in range(n_clusters):
                        cluster_mask = labels == c
                        if not cluster_mask.any():
                            selected.append(trajs_np[0])
                            continue
                        cluster_trajs = trajs_np[cluster_mask]
                        center = kmeans.cluster_centers_[c]
                        cluster_endpoints = endpoints[cluster_mask]
                        dists = np.linalg.norm(cluster_endpoints - center, axis=-1)
                        best_idx = dists.argmin()
                        selected.append(cluster_trajs[best_idx])
                    clustered_preds.append(np.stack(selected))
                except Exception:
                    clustered_preds.append(trajs_np[:n_clusters])

            clustered = torch.tensor(np.stack(clustered_preds),
                                     device=DEVICE, dtype=torch.float32)
            ade, fde = compute_ade_fde(clustered, fut)
            all_ade.append(ade); all_fde.append(fde)

    return np.mean(all_ade), np.mean(all_fde)


# ════════════════════════════════════════════════════════════════════════════
# STEP 9: ENSEMBLE TRAINING (3 models x 150 epochs)
# ════════════════════════════════════════════════════════════════════════════


def train_ensemble(samples, seeds=[42, 123, 777], epochs=150):
    """Train 3 independent FullModelV2 instances with different seeds."""
    ensemble_models = []
    for i, seed in enumerate(seeds):
        print(f"\nTraining ensemble model {i+1}/3 (seed={seed})...")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = FullModelV2().to(DEVICE)
        model, ade, fde, val_loader = train_full_model_v2(
            model, samples, epochs=epochs,
            label=f"Ensemble_{seed}"
        )
        ensemble_models.append(model)
        print(f"  Ensemble {i+1} — ADE: {ade:.4f} | FDE: {fde:.4f}")
    return ensemble_models, val_loader


# ════════════════════════════════════════════════════════════════════════════
# STEP 10: ENSEMBLE + TTA INFERENCE
# ════════════════════════════════════════════════════════════════════════════


def inference_ensemble_tta(models, val_loader, n_rotations=8, n_clusters=6):
    """
    Combine predictions from multiple models with TTA.
    For each sample: n_models x n_rotations x K modes = total candidates.
    Cluster all candidates to 6 final predictions.
    """
    for m in models:
        m.eval()
    all_ade, all_fde = [], []
    angles = [2 * np.pi * i / n_rotations for i in range(n_rotations)]

    with torch.no_grad():
        for hist, fut, atype, nb_h, nb_t, nb_m in val_loader:
            hist = hist.to(DEVICE); fut = fut.to(DEVICE); atype = atype.to(DEVICE)
            nb_h = nb_h.to(DEVICE); nb_t = nb_t.to(DEVICE); nb_m = nb_m.to(DEVICE)
            B = hist.size(0)
            all_sampled = []

            for model in models:
                for angle in angles:
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]],
                                      dtype=torch.float32, device=DEVICE)
                    R_inv = R.T

                    hist_r = hist.clone()
                    hist_r[:, :, :2] = hist[:, :, :2] @ R.T
                    hist_r[:, :, 2:] = hist[:, :, 2:] @ R.T
                    nb_h_r = nb_h.clone()
                    nb_h_r[:, :, :, :2] = nb_h[:, :, :, :2] @ R.T
                    nb_h_r[:, :, :, 2:] = nb_h[:, :, :, 2:] @ R.T

                    preds = model(hist_r, atype, nb_h_r, nb_t, nb_m)
                    preds_orig = preds @ R_inv.T
                    all_sampled.append(preds_orig)

            # Stack all: (B, n_models*n_rotations*6, 6, 2)
            all_sampled = torch.cat(all_sampled, dim=1)

            clustered_preds = []
            for b in range(B):
                endpoints = all_sampled[b, :, -1, :].cpu().numpy()
                trajs_np = all_sampled[b].cpu().numpy()
                try:
                    kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
                    labels = kmeans.fit_predict(endpoints)
                    selected = []
                    for c in range(n_clusters):
                        cluster_mask = labels == c
                        if not cluster_mask.any():
                            selected.append(trajs_np[0])
                            continue
                        cluster_trajs = trajs_np[cluster_mask]
                        center = kmeans.cluster_centers_[c]
                        cluster_endpoints = endpoints[cluster_mask]
                        dists = np.linalg.norm(cluster_endpoints - center, axis=-1)
                        best_idx = dists.argmin()
                        selected.append(cluster_trajs[best_idx])
                    clustered_preds.append(np.stack(selected))
                except Exception:
                    clustered_preds.append(trajs_np[:n_clusters])

            clustered = torch.tensor(np.stack(clustered_preds),
                                     device=DEVICE, dtype=torch.float32)
            ade, fde = compute_ade_fde(clustered, fut)
            all_ade.append(ade); all_fde.append(fde)

    return np.mean(all_ade), np.mean(all_fde)


# ════════════════════════════════════════════════════════════════════════════
# RUN TTA + ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════

if not QUICK_TEST:
    # TTA on single best model (v2_model from Step 6)
    # print("\n" + "="*70)
    # print("TEST TIME AUGMENTATION (Single Model)")
    # print("="*70)
    # tta_ade, tta_fde = inference_with_tta_clustering(v2_model, final_val_loader)
    # print(f"TTA Result    — minADE@6: {tta_ade:.4f}m | minFDE@6: {tta_fde:.4f}m")
    # print(f"vs Step7      — ADE gain: {step7_ade - tta_ade:.4f} | FDE gain: {step7_fde - tta_fde:.4f}")

    # Ensemble training (runs 3x training)
    # print("\n" + "="*70)
    # print("ENSEMBLE TRAINING (3 models x 150 epochs)")
    # print("="*70)
    # ensemble_models, ensemble_val_loader = train_ensemble(all_samples)

    # Ensemble + TTA inference
    # print("\n" + "="*70)
    # print("ENSEMBLE + TTA INFERENCE")
    # print("="*70)
    # ensemble_tta_ade, ensemble_tta_fde = inference_ensemble_tta(
    #     ensemble_models, ensemble_val_loader
    # )
    # print(f"Ensemble+TTA  — minADE@6: {ensemble_tta_ade:.4f}m | minFDE@6: {ensemble_tta_fde:.4f}m")
    # print(f"vs Step7      — ADE gain: {step7_ade - ensemble_tta_ade:.4f} | FDE gain: {step7_fde - ensemble_tta_fde:.4f}")

    # Final updated summary
    print("\n" + "="*70)
    print("COMPLETE RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<40} {'minADE@6':>10} {'minFDE@6':>10}")
    print("-"*60)
    print(f"{'Step 2: Baseline GRU':<40} {baseline_ade:>10.4f} {baseline_fde:>10.4f}")
    print(f"{'Step 3: Transformer':<40} {trans_ade:>10.4f} {trans_fde:>10.4f}")
    print(f"{'Step 4: Transformer+GAT':<40} {gat_ade:>10.4f} {gat_fde:>10.4f}")
    print(f"{'Step 5: Goal+CVAE+GRU':<40} {step5_ade:>10.4f} {step5_fde:>10.4f}")
    print(f"{'Step 6: +Diversity+KL':<40} {step6_ade:>10.4f} {step6_fde:>10.4f}")
    print(f"{'Step 7: +Clustering':<40} {step7_ade:>10.4f} {step7_fde:>10.4f}")
    print(f"{'TTA (single model)':<40} {tta_ade:>10.4f} {tta_fde:>10.4f}")
    print(f"{'Ensemble + TTA (3 models)':<40} {ensemble_tta_ade:>10.4f} {ensemble_tta_fde:>10.4f}")
    print("="*60)

    # Plot final comparison with all methods
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    labels_all = ['S2:GRU', 'S3:Trans', 'S4:GAT', 'S5:CVAE', 'S6:Div',
                  'S7:Clust', 'TTA', 'Ens+TTA']
    ades_all = [baseline_ade, trans_ade, gat_ade, step5_ade, step6_ade,
                step7_ade, tta_ade, ensemble_tta_ade]
    fdes_all = [baseline_fde, trans_fde, gat_fde, step5_fde, step6_fde,
                step7_fde, tta_fde, ensemble_tta_fde]
    colors_all = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90', '#87CEEB',
                  '#8A2BE2', '#FF69B4', '#00CED1']

    axes[0].bar(labels_all, ades_all, color=colors_all, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('minADE@6 (meters)'); axes[0].set_title('Average Displacement Error')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(labels_all, fdes_all, color=colors_all, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('minFDE@6 (meters)'); axes[1].set_title('Final Displacement Error')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle("Trajectory Prediction: Complete Pipeline (incl. TTA + Ensemble)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                'complete_comparison.png'), dpi=150)
    print("\nSaved complete_comparison.png")

print("\nAll steps complete!")
