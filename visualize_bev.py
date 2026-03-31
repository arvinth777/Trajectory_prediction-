#!/usr/bin/env python3
"""
visualize_bev.py

Bird's Eye View (BEV) trajectory overlay on nuScenes map tiles.
Uses the trained FullModelV2 from step6_diversity.pt to generate
6 multi-modal predictions and overlays them on the real map PNG.

Run with:
    source venv_vis/bin/activate
    python3 visualize_bev.py
"""

import os
import sys
import json
import math
import pickle
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None  # Maps are huge — disable decompression bomb check
from PIL import Image
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
DATAROOT        = './v1.0-mini'
MAPS_DIR        = './maps'
MODEL_PATH      = 'step6_diversity.pt'
PKL_PATH        = 'processed_data.pkl'
OUTPUT_DIR      = 'bev_visualizations'
N_SAMPLES       = 8          # How many BEV images to generate
CONTEXT_RADIUS  = 15         # Meters around agent to crop from map
MAP_RESOLUTION  = 0.1        # nuScenes standard: 0.1 m/pixel
DEVICE          = 'cpu'      # Visualization doesn't need GPU
HIST_LEN        = 4
FUT_LEN         = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# MAP METADATA: scene -> png file
# Hardcoded from nuScenes v1.0-mini log.json / map.json analysis
# ─────────────────────────────────────────────────────────
SCENE_TO_MAP = {
    'scene-0061': '53992ee3023e5494b90c316c183be829.png',  # singapore-onenorth
    'scene-0103': '36092f0b03a857c6a3403e25b4b7aab3.png',  # boston-seaport
    'scene-0553': '36092f0b03a857c6a3403e25b4b7aab3.png',
    'scene-0655': '36092f0b03a857c6a3403e25b4b7aab3.png',
    'scene-0757': '36092f0b03a857c6a3403e25b4b7aab3.png',
    'scene-0796': '93406b464a165eaba6d9de76ca09f5da.png',  # singapore-queenstown
    'scene-0916': '93406b464a165eaba6d9de76ca09f5da.png',
    'scene-1077': '37819e65e09e5547b8a3ceaefba56bb2.png',  # singapore-hollandvillage
    'scene-1094': '37819e65e09e5547b8a3ceaefba56bb2.png',
    'scene-1100': '37819e65e09e5547b8a3ceaefba56bb2.png',  # ← our val scene
}

# Map image origins (top-left corner in real-world meters).
# nuScenes map coordinate system: origin is top-left, y increases downward.
# These offsets align the PNG pixel grid with real-world UTM coordinates.
MAP_ORIGINS = {
    '53992ee3023e5494b90c316c183be829.png': (0.0, 0.0),       # singapore-onenorth
    '36092f0b03a857c6a3403e25b4b7aab3.png': (0.0, 0.0),       # boston-seaport
    '93406b464a165eaba6d9de76ca09f5da.png': (0.0, 0.0),       # singapore-queenstown
    '37819e65e09e5547b8a3ceaefba56bb2.png': (0.0, 0.0),       # singapore-hollandvillage
}

# ─────────────────────────────────────────────────────────
# PASTE IN THE MODEL ARCHITECTURE
# (Copy of FullModelV2 from trajectory_prediction.py — keeps this script standalone)
# ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────
# CONSTANTS (copied from trajectory_prediction.py)
# ─────────────────────────────────────────────────────────
HIST_LEN = 4
FUT_LEN  = 6
DT       = 0.5   # seconds per step (nuScenes 2Hz)
MAX_NB   = 10

# ─────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (copied from trajectory_prediction.py)
# ─────────────────────────────────────────────────────────
import torch.nn.functional as F

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    def forward(self, x):
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        return x + self.pe(positions).unsqueeze(0)

class TransformerTemporalEncoder(nn.Module):
    def __init__(self, input_dim=4, d_model=96, nhead=4, num_layers=2,
                 agent_type_dim=8, num_agent_types=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = RelativePositionalEncoding(d_model)
        self.agent_type_embed = nn.Embedding(num_agent_types, agent_type_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.2, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model + agent_type_dim
    def forward(self, hist, agent_type):
        x = self.input_proj(hist)
        x = self.pos_enc(x)
        x = self.transformer(x)
        temporal_embed = x.mean(dim=1)
        type_embed = self.agent_type_embed(agent_type)
        return torch.cat([temporal_embed, type_embed], dim=-1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        self.W_node   = nn.Linear(node_dim, out_dim, bias=False)
        self.W_edge   = nn.Linear(edge_dim, heads, bias=False)
        self.attn_src = nn.Linear(self.head_dim, 1, bias=False)
        self.attn_dst = nn.Linear(self.head_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(out_dim, out_dim)
    def forward(self, nodes, edge_features, mask):
        B, N, _ = nodes.size()
        h = self.W_node(nodes).view(B, N, self.heads, self.head_dim)
        attn = self.attn_src(h).squeeze(-1) + self.attn_dst(h).squeeze(-1)
        edge_attn = self.W_edge(edge_features)
        attn = self.leaky_relu(attn + edge_attn)
        mask_exp = mask.unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(mask_exp == 0, float('-inf'))
        attn = F.softmax(attn, dim=1).masked_fill(mask_exp == 0, 0.0)
        weighted = (h * attn.unsqueeze(-1)).sum(dim=1).view(B, -1)
        return self.out_proj(weighted)

class GATSocialEncoder(nn.Module):
    def __init__(self, temporal_dim=104, edge_dim=4, social_dim=96, heads=4, agent_type_dim=8):
        super().__init__()
        nb_d_model = temporal_dim - agent_type_dim
        self.neighbor_encoder = TransformerTemporalEncoder(
            input_dim=4, d_model=nb_d_model, agent_type_dim=agent_type_dim)
        self.gat = GraphAttentionLayer(temporal_dim, edge_dim, social_dim, heads)
        self.output_dim = temporal_dim + social_dim
    def compute_edge_features(self, target_hist, nb_hists, nb_mask):
        B, N = nb_mask.shape
        target_pos = target_hist[:, -1, :2]; target_vel = target_hist[:, -1, 2:]
        nb_pos = nb_hists[:, :, -1, :2]; nb_vel = nb_hists[:, :, -1, 2:]
        diff = nb_pos - target_pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        rel_vel = nb_vel - target_vel.unsqueeze(1)
        closing_speed = -torch.sum(diff * rel_vel, dim=-1, keepdim=True) / (dist + 1e-8)
        ttc = (dist / (closing_speed.abs() + 1e-6)).clamp(0, 10)
        return torch.cat([dist, rel_vel, ttc], dim=-1)
    def forward(self, target_embed, target_hist, nb_hists, nb_types, nb_mask):
        B, N, T, D = nb_hists.shape
        has_neighbors = nb_mask.sum(dim=-1) > 0
        if not has_neighbors.any():
            return torch.zeros(B, self.output_dim, device=target_hist.device)
        nb_embeds = self.neighbor_encoder(
            nb_hists.view(B*N, T, D), nb_types.view(B*N)).view(B, N, -1)
        edge_feat = self.compute_edge_features(target_hist, nb_hists, nb_mask)
        social_ctx = self.gat(nb_embeds, edge_feat, nb_mask)
        social_ctx = social_ctx * has_neighbors.float().unsqueeze(-1)
        return torch.cat([target_embed, social_ctx], dim=-1)

class GoalPredictor(nn.Module):
    def __init__(self, ctx_dim=128, num_goals=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),      nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_goals * 2))
        self.num_goals = num_goals
    def forward(self, ctx):
        return self.net(ctx).view(-1, self.num_goals, 2)

class CVAE(nn.Module):
    def __init__(self, ctx_dim=128, goal_dim=2, latent_dim=16, future_dim=12):
        super().__init__()
        self.latent_dim = latent_dim
        self.prior_net = nn.Sequential(
            nn.Linear(ctx_dim + goal_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim * 2))
        self.posterior_net = nn.Sequential(
            nn.Linear(ctx_dim + goal_dim + future_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim * 2))
    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    def forward(self, ctx, goal, future_flat=None):
        prior_params = self.prior_net(torch.cat([ctx, goal], dim=-1))
        prior_mu, prior_logvar = prior_params.chunk(2, dim=-1)
        if future_flat is not None:
            post_params = self.posterior_net(torch.cat([ctx, goal, future_flat], dim=-1))
            post_mu, post_logvar = post_params.chunk(2, dim=-1)
            z = self.reparameterize(post_mu, post_logvar)
            kl = 0.5 * torch.sum(prior_logvar - post_logvar - 1
                + post_logvar.exp() / prior_logvar.exp()
                + (post_mu - prior_mu).pow(2) / prior_logvar.exp(), dim=-1)
            return z, kl.mean()
        z = self.reparameterize(prior_mu, prior_logvar)
        return z, torch.tensor(0.0, device=ctx.device)

class GRUDecoder(nn.Module):
    def __init__(self, z_dim=16, ctx_dim=128, goal_dim=2, hidden_dim=128, future_len=6):
        super().__init__()
        self.future_len = future_len
        self.gru_cell = nn.GRUCell(z_dim + ctx_dim + goal_dim + 2, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 2)
        self.hidden_init = nn.Sequential(
            nn.Linear(z_dim + ctx_dim + goal_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2))
    def forward(self, z, ctx, goal):
        B = z.size(0)
        h = torch.tanh(self.hidden_init(torch.cat([z, ctx, goal], dim=-1)))
        pos = torch.zeros(B, 2, device=z.device)
        traj = []
        for _ in range(self.future_len):
            h = self.gru_cell(torch.cat([z, ctx, goal, pos], dim=-1), h)
            pos = pos + self.output_proj(h)
            traj.append(pos)
        return torch.stack(traj, dim=1)

class FullModelV2(nn.Module):
    def __init__(self, d_model=96, agent_type_dim=8, social_dim=96,
                 num_goals=6, latent_dim=16, future_len=6, decoder_hidden=128):
        super().__init__()
        self.num_goals = num_goals
        self.future_len = future_len
        self.temporal_enc = TransformerTemporalEncoder(
            input_dim=4, d_model=d_model, agent_type_dim=agent_type_dim)
        temporal_dim = d_model + agent_type_dim
        self.social_enc = GATSocialEncoder(temporal_dim=temporal_dim, social_dim=social_dim)
        ctx_input_dim = temporal_dim + self.social_enc.output_dim
        self.fusion = nn.Sequential(nn.Linear(ctx_input_dim, 128), nn.ReLU(), nn.Dropout(0.2))
        ctx_dim = 128
        self.goal_predictor = GoalPredictor(ctx_dim, num_goals)
        self.cvae = CVAE(ctx_dim, goal_dim=2, latent_dim=latent_dim, future_dim=future_len * 2)
        self.decoder = GRUDecoder(z_dim=latent_dim, ctx_dim=ctx_dim,
                                  goal_dim=2, hidden_dim=decoder_hidden, future_len=future_len)
    def forward(self, hist, agent_type, nb_hists, nb_types, nb_mask):
        temporal = self.temporal_enc(hist, agent_type)
        social   = self.social_enc(temporal, hist, nb_hists, nb_types, nb_mask)
        ctx      = self.fusion(torch.cat([temporal, social], dim=-1))
        goals    = self.goal_predictor(ctx)
        all_trajs = []
        for k in range(self.num_goals):
            goal_k = goals[:, k, :]
            z_k, _ = self.cvae(ctx, goal_k)
            all_trajs.append(self.decoder(z_k, ctx, goal_k))
        return torch.stack(all_trajs, dim=1)  # (B, K, T, 2)

def scene_based_split(samples, n_val_scenes=1, seed=42):
    random.seed(seed)
    scene_names = sorted(set(s['scene'] for s in samples))
    val_scenes = set(scene_names[-1:])
    train_s = [s for s in samples if s['scene'] not in val_scenes]
    val_s   = [s for s in samples if s['scene'] in val_scenes]
    print(f"  Scene split: {len(scene_names)-1} Train Scenes, 1 Val Scenes")
    print(f"  Samples: {len(train_s)} Train, {len(val_s)} Val")
    return train_s, val_s

print("✓ Model classes defined inline")


# ─────────────────────────────────────────────────────────
# INVERSE NORMALIZATION
# ─────────────────────────────────────────────────────────
def denormalize(local_traj, origin, angle):
    """
    Convert agent-centric normalized coords back to global map coords.

    Forward transform was:
        angle = -heading + pi/2
        rot = [[cos_a, -sin_a], [sin_a, cos_a]]
        local = (global - origin) @ rot.T

    Inverse:
        global = local @ rot + origin
        which is: local @ R_inv.T + origin where R_inv = rot.T
    """
    # The forward rotation matrix
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a],
                    [sin_a,  cos_a]])
    # Inverse = transpose of rotation matrix
    rot_inv = rot.T
    return local_traj @ rot_inv + origin


# ─────────────────────────────────────────────────────────
# GLOBAL COORD → MAP PIXEL
# ─────────────────────────────────────────────────────────
def global_to_pixel(xy_global, map_img_size, map_origin=(0.0, 0.0)):
    """
    Convert global (x, y) meters to (col, row) pixel indices.

    nuScenes map convention:
      - x increases → East  (right in image)
      - y increases → North (UP in image, but image row 0 = top)
      So pixel_row = img_height - y/resolution
    """
    img_w, img_h = map_img_size
    ox, oy = map_origin

    px = (xy_global[:, 0] - ox) / MAP_RESOLUTION
    # y increases north but image rows increase downward
    py = img_h - (xy_global[:, 1] - oy) / MAP_RESOLUTION
    return px, py


# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
print(f"\nLoading model from {MODEL_PATH}...")
model = FullModelV2().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print("✓ Model loaded")


# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
print(f"Loading data from {PKL_PATH}...")
with open(PKL_PATH, 'rb') as f:
    all_samples = pickle.load(f)
print(f"✓ Loaded {len(all_samples)} total samples")

# Use the same scene split as training — pick validation samples
train_samples, val_samples = scene_based_split(all_samples)
print(f"  Using {len(val_samples)} val samples for visualization")

# Pick a diverse set: samples with real movement (large future displacement)
def trajectory_length(s):
    fut = s['fut']
    return np.sum(np.linalg.norm(np.diff(fut, axis=0), axis=1))

# Sort by movement, pick most interesting ones spread across the range
val_samples_sorted = sorted(val_samples, key=trajectory_length, reverse=True)
# Pick every Nth for diversity
step = max(1, len(val_samples_sorted) // N_SAMPLES)
selected = val_samples_sorted[::step][:N_SAMPLES]
print(f"  Selected {len(selected)} samples for visualization")


# ─────────────────────────────────────────────────────────
# LOAD MAP IMAGES (lazy cache — only load each once)
# ─────────────────────────────────────────────────────────
map_cache = {}

def get_map_image(scene_name):
    map_file = SCENE_TO_MAP.get(scene_name)
    if not map_file:
        return None, None
    if map_file not in map_cache:
        path = os.path.join(MAPS_DIR, map_file)
        if not os.path.exists(path):
            print(f"  ⚠ Map file not found: {path}")
            return None, None
        print(f"  Loading map: {map_file[:16]}... (this may take a moment for large maps)")
        map_cache[map_file] = np.array(Image.open(path))
        print(f"  ✓ Map loaded: {map_cache[map_file].shape}")
    return map_cache[map_file], map_file


# ─────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────
def run_inference(sample):
    """Run the full model and return 6 predicted trajectories in local coords."""
    max_nb = 10
    hist   = torch.tensor(sample['hist'], dtype=torch.float32).unsqueeze(0)
    atype  = torch.tensor(sample['agent_type'], dtype=torch.long).unsqueeze(0)

    # Build neighbor tensors (same logic as SocialDataset)
    nb_hists, nb_types_list, nb_mask_list = [], [], []
    for nb in sample['neighbors'][:max_nb]:
        nb_hists.append(torch.tensor(nb['hist'], dtype=torch.float32))
        nb_types_list.append(nb['agent_type'])
        nb_mask_list.append(1.0)
    while len(nb_hists) < max_nb:
        nb_hists.append(torch.zeros(HIST_LEN, 4))
        nb_types_list.append(0)
        nb_mask_list.append(0.0)

    nb_h = torch.stack(nb_hists).unsqueeze(0)
    nb_t = torch.tensor(nb_types_list, dtype=torch.long).unsqueeze(0)
    nb_m = torch.tensor(nb_mask_list, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(hist, atype, nb_h, nb_t, nb_m)  # (1, K, 6, 2)
    return pred.squeeze(0).cpu().numpy()  # (K, 6, 2)


# ─────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────
MODE_COLORS = ['#FF4B4B', '#FF8C00', '#FFD700', '#7CFC00', '#00CED1', '#9370DB']

def visualize_sample(sample, idx):
    scene_name = sample.get('scene', 'unknown')
    origin     = sample['origin']   # global (x,y) of agent at last history step
    angle      = sample['angle']    # rotation applied during normalization

    # ── 1. Run model ──────────────────────────────────────
    pred_local = run_inference(sample)  # (K, 6, 2)
    K = pred_local.shape[0]

    hist_local = sample['hist'][:, :2]  # (4, 2) — position only
    fut_local  = sample['fut']          # (6, 2)

    # ── 2. Convert to global coords ───────────────────────
    hist_global = denormalize(hist_local, origin, angle)  # (4, 2)
    fut_global  = denormalize(fut_local,  origin, angle)  # (6, 2)
    pred_global = [denormalize(pred_local[k], origin, angle) for k in range(K)]  # list of (6,2)

    # ── 3. Compute crop bounds ────────────────────────────
    all_pts = np.vstack([hist_global, fut_global] + pred_global)
    cx, cy  = origin  # crop center = agent's last known position
    margin  = CONTEXT_RADIUS

    x_min, x_max = cx - margin, cx + margin
    y_min, y_max = cy - margin, cy + margin

    # ── 4. Load and crop map ──────────────────────────────
    map_arr, map_file = get_map_image(scene_name)
    use_map = map_arr is not None

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0d0d0d')
    ax.set_facecolor('#0d0d0d')

    if use_map:
        img_h, img_w = map_arr.shape[:2]
        origin_offset = MAP_ORIGINS.get(map_file, (0.0, 0.0))

        # Convert world bounds to pixel bounds
        px_min = int((x_min - origin_offset[0]) / MAP_RESOLUTION)
        px_max = int((x_max - origin_offset[0]) / MAP_RESOLUTION)
        # y is flipped: world y_max → smaller row index
        py_min = int(img_h - (y_max - origin_offset[1]) / MAP_RESOLUTION)
        py_max = int(img_h - (y_min - origin_offset[1]) / MAP_RESOLUTION)

        # Clamp to image bounds
        px_min = max(0, px_min); px_max = min(img_w, px_max)
        py_min = max(0, py_min); py_max = min(img_h, py_max)

        if px_max > px_min and py_max > py_min:
            crop = map_arr[py_min:py_max, px_min:px_max]
            ax.imshow(crop, extent=[x_min, x_max, y_min, y_max],
                      origin='lower', cmap='gray', alpha=0.55, zorder=0)
        else:
            use_map = False

    # ── 5. Plot trajectories ────────────────────────────────
    # History — thick dashed blue
    ax.plot(hist_global[:, 0], hist_global[:, 1],
            color='#4FC3F7', linewidth=3, linestyle='--',
            marker='o', markersize=7, zorder=3, label='History (2s)')

    # Ground truth — solid bright green
    full_gt = np.vstack([hist_global[-1:], fut_global])
    ax.plot(full_gt[:, 0], full_gt[:, 1],
            color='#69FF47', linewidth=3,
            marker='s', markersize=8, zorder=4, label='Ground Truth (3s)')
    ax.plot(fut_global[-1, 0], fut_global[-1, 1],
            color='#69FF47', marker='*', markersize=20, zorder=5)

    # 6 predicted modes — coloured rays
    for k in range(K):
        traj = np.vstack([hist_global[-1:], pred_global[k]])
        color = MODE_COLORS[k % len(MODE_COLORS)]
        ax.plot(traj[:, 0], traj[:, 1],
                color=color, linewidth=2.5, linestyle='--',
                alpha=0.85, zorder=3,
                label=f'Mode {k+1}' if k < 6 else '')
        # Endpoint dot
        ax.scatter(pred_global[k][-1, 0], pred_global[k][-1, 1],
                   color=color, s=120, zorder=4, edgecolors='white', linewidths=1.5)

    # Agent start position
    ax.scatter(cx, cy, color='white', s=200, marker='D',
               zorder=6, edgecolors='black', linewidths=2, label='Agent Now')

    # ── 6. Styling ────────────────────────────────────────
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    ax.tick_params(colors='#888888')
    ax.spines[:].set_edgecolor('#333333')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    ax.set_xlabel('X (m)', color='#aaaaaa', fontsize=11)
    ax.set_ylabel('Y (m)', color='#aaaaaa', fontsize=11)
    ax.tick_params(colors='#aaaaaa')

    map_label = 'singapore-hollandvillage' if '37819e' in (map_file or '') else \
                'boston-seaport'           if '36092f' in (map_file or '') else \
                'singapore-queenstown'     if '93406b' in (map_file or '') else \
                'singapore-onenorth'       if '53992e' in (map_file or '') else 'map'

    ax.set_title(
        f'BEV Trajectory Prediction  ·  {scene_name}  ·  {map_label}\n'
        f'6 Multi-Modal Predictions (minADE@6 ≈ 0.21m)',
        color='white', fontsize=13, fontweight='bold', pad=12
    )

    legend = ax.legend(loc='upper left', framealpha=0.35,
                       facecolor='#1a1a1a', edgecolor='#555555',
                       labelcolor='white', fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f'bev_sample_{idx:03d}.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
print(f"\nGenerating {len(selected)} BEV visualizations...")
saved = []
for i, sample in enumerate(selected):
    print(f"\n[{i+1}/{len(selected)}] Scene: {sample.get('scene','?')}")
    try:
        path = visualize_sample(sample, i + 1)
        saved.append(path)
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback; traceback.print_exc()

print(f"\n{'='*50}")
print(f"Done! Generated {len(saved)}/{len(selected)} images")
print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}/")
for p in saved:
    print(f"  {p}")
