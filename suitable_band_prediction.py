"""train_model.py
Train a simple neural network that selects the optimal satellite band
based on weather‑severity and per‑band interference.  The resulting
PyTorch model is exported to ONNX with a **dynamic batch dimension** so
it can accept any number of rows from JavaScript.  The script keeps the
original 0‑100 value range, so the existing HTML front‑end does **not**
need to change.
"""

from __future__ import annotations

import os
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 1. Reproducibility ---------------------------------------------------------
# ---------------------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 2. Synthetic data generation ----------------------------------------------
# ---------------------------------------------------------------------------
NUM_SAMPLES = 5_000
FEATURES = []  # [weather, UHF_intf, L_intf, ..., V_intf]
LABELS = []    # optimal band index (0‑based)

for _ in range(NUM_SAMPLES):
    # Weather severity 0‑100
    weather = np.random.rand() * 100

    # Interference for nine bands 0‑100
    interference = np.random.rand(9) * 100

    # Determine which bands are allowed under the weather rule‑set
    allowed = [True] * 9
    if weather > 80:      # Severe weather ⇒ disallow X and above
        allowed[4:] = [False] * 5
    elif weather > 60:    # Heavy rain ⇒ disallow Ku and above
        allowed[5:] = [False] * 4
    elif weather > 40:    # Moderate rain ⇒ disallow K and above
        allowed[6:] = [False] * 3
    elif weather > 20:    # Light rain ⇒ disallow Ka and above
        allowed[7:] = [False] * 2

    # Pick the allowed band with minimum interference (ties → higher band)
    allowed_idx = [i for i, ok in enumerate(allowed) if ok]
    best = min(allowed_idx, key=lambda i: (interference[i], -i))

    FEATURES.append([weather, *interference])
    LABELS.append(best)

X = np.array(FEATURES, dtype=np.float32)
y = np.array(LABELS, dtype=np.int64)

# ---------------------------------------------------------------------------
# 3. Train / validation split ------------------------------------------------
# ---------------------------------------------------------------------------
shuffle_idx = np.random.permutation(NUM_SAMPLES)
train_size = int(NUM_SAMPLES * 0.8)
train_idx, val_idx = shuffle_idx[:train_size], shuffle_idx[train_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)

# ---------------------------------------------------------------------------
# 4. Model -------------------------------------------------------------------
# ---------------------------------------------------------------------------
model = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 9)
).to(device)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# ---------------------------------------------------------------------------
# 5. Training loop with early stopping --------------------------------------
# ---------------------------------------------------------------------------
EPOCHS       = 100
PATIENCE     = 10
best_val_acc = 0.0
patience_ctr = 0

for epoch in range(1, EPOCHS + 1):
    # ----- training -----
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = loss_fn(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ----- validation -----
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
        val_acc = correct / total

    print(f"Epoch {epoch:3d} | val‑accuracy: {val_acc:.4f}")

    # early‑stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_ctr = 0
        torch.save(model.state_dict(), "best.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print("Early stopping – no improvement")
            break

# load the best weights before export
model.load_state_dict(torch.load("best.pt"))
model.eval()

# ---------------------------------------------------------------------------
# 6. Export to ONNX (dynamic batch) -----------------------------------------
# ---------------------------------------------------------------------------
onnx_file = "/Users/phani/Desktop/AI/spectra-luma/model/suitable_band_prediction.onnx"

dummy = torch.zeros((1, 10), dtype=torch.float32, device=device)

torch.onnx.export(
    model, dummy, onnx_file,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=14,
)

print(f"✨ Model exported")