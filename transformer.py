import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx as onnx

# Device selection: use CUDA if available, else MPS (for Apple devices), else CPU
device = torch.device("cuda" if torch.cuda.is_available() 
                      else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
                            else "cpu"))
print("Using device:", device)

# Load hyperspectral data and labels (assuming .mat files)
data_mat = sio.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/Indian_pines_corrected.mat')
image_cube = data_mat['indian_pines_corrected']  # shape (145, 145, 200)
gt_mat = sio.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/Indian_pines_gt.mat')
labels_map = gt_mat['indian_pines_gt']  # shape (145, 145)

# Flatten the image cube to [num_pixels, num_bands] and labels to [num_pixels]
H, W, B = image_cube.shape  # B = 200 spectral bands
pixels = image_cube.reshape(-1, B)              # shape (21025, 200)
labels = labels_map.reshape(-1)                 # shape (21025,)
# Filter out unlabeled pixels (assume 0 indicates no class)
mask = labels > 0
pixels = pixels[mask]
labels = labels[mask] - 1  # convert labels to 0-15 range
print("Total labeled samples:", pixels.shape[0])

# Normalization functions
def normalize_data(X, method='band_minmax', params=None):
    """Normalize data X (shape: [samples, bands]) according to the specified method."""
    if method == 'none':
        return X, {}
    if method == 'global_minmax':
        X_min = X.min()
        X_max = X.max()
        X_norm = (X - X_min) / (X_max - X_min)
        return X_norm, {'min': X_min, 'max': X_max}
    if method == 'band_minmax':
        # Compute per-band min and max (across samples)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)
        return X_norm, {'min': X_min, 'max': X_max}
    if method == 'band_zscore':
        # Compute per-band mean and std
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        X_norm = (X - mu) / (sigma + 1e-8)
        return X_norm, {'mean': mu, 'std': sigma}
    if method == 'l2':
        # Normalize each sample to unit length
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_norm = X / norms
        return X_norm, {}
    raise ValueError(f"Unknown method: {method}")

# Choose a normalization method
method = 'band_zscore'
pixels_norm, norm_params = normalize_data(pixels.astype(np.float32), method=method)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    pixels_norm, labels, test_size=0.2, random_state=42, stratify=labels)
print("Training samples:", X_train.shape[0], "Validation samples:", X_val.shape[0])

class SpectralTransformerClassifier(nn.Module):
    def __init__(self, seq_length=200, num_classes=16, d_model=64, n_heads=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(SpectralTransformerClassifier, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        # Learnable positional embeddings for [CLS] + 200 positions
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length+1, d_model))
        # Linear projection for band values to d_model
        self.value_proj = nn.Linear(1, d_model)
        # Learnable [CLS] token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 200) containing spectral data for each sample.
        Returns:
            logits: (batch_size, num_classes)
            attn_weights: Attention weights from the first encoder layer (batch_size, n_heads, query_len, key_len)
        """
        batch_size = x.shape[0]
        # Project each band value to d_model dimension
        band_embeddings = self.value_proj(x.unsqueeze(-1))  # (batch, 200, d_model)
        # Prepend the CLS token embedding to the sequence
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        seq_embeddings = torch.cat([cls_token, band_embeddings], dim=1)  # (batch, 201, d_model)
        # Add positional encoding
        seq_embeddings = seq_embeddings + self.pos_embedding[:, :seq_embeddings.size(1), :]
        
        # Process the first encoder layer manually to capture attention weights
        first_layer = self.encoder.layers[0]
        src = seq_embeddings
        # Call self_attn with need_weights=True; this returns (attn_output, attn_weights)
        src2, attn_weights = first_layer.self_attn(src, src, src, need_weights=True)
        src = src + first_layer.dropout1(src2)
        src = first_layer.norm1(src)
        src2 = first_layer.linear2(first_layer.dropout(first_layer.activation(first_layer.linear1(src))))
        src = src + first_layer.dropout2(src2)
        src = first_layer.norm2(src)
        
        # Process remaining layers (if any)
        for layer in self.encoder.layers[1:]:
            src = layer(src)
        
        # Extract [CLS] token representation and classify
        cls_output = src[:, 0, :]  # (batch, d_model)
        logits = self.fc(cls_output)  # (batch, num_classes)
        return logits, attn_weights

# Initialize and train model (training code remains the same)
model = SpectralTransformerClassifier()
model = model.to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Convert data to PyTorch tensors and move them to the selected device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor   = torch.tensor(y_val, dtype=torch.long).to(device)

# Simple training loop
num_epochs = 2
batch_size = 32
for epoch in range(1, num_epochs+1):
    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0.0
    model.train()
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X = X_train_tensor[indices]
        batch_y = y_train_tensor[indices]
        
        optimizer.zero_grad()
        logits, _ = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= (X_train_tensor.size(0) / batch_size)
    
    # Validation
    model.eval()
    with torch.no_grad():
        logits_val, _ = model(X_val_tensor)
        val_pred = logits_val.argmax(dim=1)
        val_acc = (val_pred == y_val_tensor).float().mean().item()
    print(f"Epoch {epoch:02d}: Train Loss = {epoch_loss:.3f}, Val Accuracy = {val_acc*100:.2f}%")

# Run a sample through the model to check outputs
model.eval()
sample_X = X_val_tensor[0:1]  # single sample from validation
with torch.no_grad():
    logits, attn_weights = model(sample_X)
    predicted_class = logits.argmax(dim=1).item()
    print("Predicted class:", predicted_class)
    # attn_weights shape: (batch=1, n_heads, query_len, key_len)
    # For demonstration, average attention weights over heads for the [CLS] token:
    attn_weights_np = attn_weights.cpu().numpy()
    cls_attn = attn_weights_np[0, 0, :]  # (key_len,)
    # Ignore the CLS-to-CLS weight (first element) and get top 5 important bands:
    top5_idx = cls_attn[1:].argsort()[-5:][::-1]
    print("Top 5 important bands (0-indexed):", top5_idx)
# Export the trained model to ONNX format with two outputs
model_cpu = model.cpu()
dummy_input = torch.randn(1, 200, dtype=torch.float32)
onnx_file = "/Users/phani/Desktop/AI/spectra-luma/model/indian_pines_transformer.onnx"
onnx.export(model_cpu, dummy_input, onnx_file, 
            input_names=["spectral_vector"], 
            output_names=["class_probabilities", "attention_weights"],
            dynamic_axes={"spectral_vector": {0: "batch_size"}, 
                          "class_probabilities": {0: "batch_size"},
                          "attention_weights": {0: "batch_size"}})
print(f"Model exported to {onnx_file}")