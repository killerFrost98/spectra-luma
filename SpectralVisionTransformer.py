import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json
from collections import Counter

# Set device (MPS if available, else CUDA/CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names for Indian Pines dataset (0-based after background removal)
CLASS_NAMES = {
    0: "Alfalfa",
    1: "Corn-notill",
    2: "Corn-mintill",
    3: "Corn",
    4: "Grass-pasture",
    5: "Grass-trees",
    6: "Grass-pasture-mowed",
    7: "Hay-windrowed",
    8: "Oats",
    9: "Soybean-notill",
    10: "Soybean-mintill",
    11: "Soybean-clean",
    12: "Wheat",
    13: "Woods",
    14: "Buildings-Grass-Trees-Drives",
    15: "Stone-Steel-Towers"
}

# Early stopping settings
min_accuracy = 0.50
improvement_threshold = 0.01
patience = 3
epochs_without_improvement = 0
best_accuracy = 0.0
max_epochs = 100

# Load Indian Pines data from .mat files
ip_data = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/Indian_pines_corrected.mat')['indian_pines_corrected']
ip_labels = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/Indian_pines_gt.mat')['indian_pines_gt']

# Load Pavia University data from .mat files
pu_data = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/PaviaU.mat')['paviaU']
pu_labels = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/PaviaU_gt.mat')['paviaU_gt']

# Preprocessing (no normalization here; model will handle it)
def preprocess_hsi(data_cube, label_map):
    H, W, B = data_cube.shape
    data_cube = data_cube.astype(np.float32).reshape(-1, B)
    labels = label_map.reshape(-1)
    mask = labels > 0  # remove background pixels
    X = data_cube[mask]
    y = labels[mask] - 1  # convert to 0-based labels
    return X, y

X_ip, y_ip = preprocess_hsi(ip_data, ip_labels)
X_pu, y_pu = preprocess_hsi(pu_data, pu_labels)

print("Indian Pines:", X_ip.shape, "classes:", np.unique(y_ip))
print("Pavia University:", X_pu.shape, "classes:", np.unique(y_pu))

# Function to print class distributions
def print_class_distribution(y, dataset_name):
    counter = Counter(y)
    total = sum(counter.values())
    print(f"\n{dataset_name} class distribution:")
    for cls, count in sorted(counter.items()):
        class_name = CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"  {class_name}: {count} samples ({count/total:.2%})")

print_class_distribution(y_ip, "Indian Pines (Original)")
print_class_distribution(y_pu, "Pavia University (Original)")

# Split data into training and testing sets
X_ip_train, X_ip_test, y_ip_train, y_ip_test = train_test_split(X_ip, y_ip, test_size=0.3, stratify=y_ip, random_state=42)
X_pu_train, X_pu_test, y_pu_train, y_pu_test = train_test_split(X_pu, y_pu, test_size=0.3, stratify=y_pu, random_state=42)

print_class_distribution(y_ip_train, "Indian Pines (Train)")
print_class_distribution(y_ip_test, "Indian Pines (Test)")
print_class_distribution(y_pu_train, "Pavia University (Train)")
print_class_distribution(y_pu_test, "Pavia University (Test)")

# Compute per-band min and max from the training data for normalization inside the model
band_min = X_ip_train.min(axis=0)
band_max = X_ip_train.max(axis=0)
print("Band Min:", band_min)
print("Band Max:", band_max)

# Compute static sinusoidal positional encoding
def get_static_positional_encoding(n_bands, d_model):
    position = np.arange(n_bands)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((n_bands, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float32)

# Custom transformer encoder layer that returns attention weights
class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super(TransformerEncoderLayerWithAttn, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        # src: (batch, seq_len, d_model)
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

# Modified SpectralTransformer model using custom transformer layers
class SpectralTransformer(nn.Module):
    def __init__(self, num_bands, num_classes, band_min, band_max, d_model=64, nhead=8, num_layers=2, dim_feedforward=128):
        super(SpectralTransformer, self).__init__()
        # Register normalization parameters as buffers
        self.register_buffer("band_min", torch.tensor(band_min, dtype=torch.float32))
        self.register_buffer("band_max", torch.tensor(band_max, dtype=torch.float32))
        self.eps = 1e-8

        # 1. Linear projection: scalar -> d_model
        self.value_embed = nn.Linear(1, d_model)
        
        # 2. Static positional encoding for 200 bands
        static_pe = get_static_positional_encoding(200, d_model)
        if num_bands < 200:
            static_pe = static_pe[:num_bands, :]
        elif num_bands > 200:
            static_pe = static_pe[:num_bands, :]  # simple truncation (interpolation can be used)
        self.register_buffer("pos_embed", static_pe.unsqueeze(0))  # shape: (1, num_bands, d_model)
        
        # 3. Layer normalization for embedded spectral values
        self.input_norm = nn.LayerNorm(d_model)
        
        # 4. Custom transformer encoder layers (returning attention weights)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        
        # 5. Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, return_probs=False, return_attention=False):
        band_min = self.band_min.to(x.device)
        band_max = self.band_max.to(x.device)
        x = (x - band_min) / (band_max - band_min + self.eps)
        batch_size, seq_len = x.shape
        
        # Linear projection and add positional encoding
        x_emb = self.value_embed(x.unsqueeze(-1))  # (batch, seq_len, d_model)
        x_emb = x_emb + self.pos_embed[:, :seq_len, :]
        x_emb = self.input_norm(x_emb)
        
        attention_weights_all = []
        for layer in self.encoder_layers:
            x_emb, attn_weights = layer(x_emb)
            attention_weights_all.append(attn_weights)
        
        # Mean pooling over the sequence dimension
        seq_avg = x_emb.mean(dim=1)
        logits = self.classifier(seq_avg)
        
        if return_attention:
            # Stack attention weights: shape (num_layers, batch, num_heads, seq_len, seq_len)
            attn_stack = torch.stack(attention_weights_all, dim=0)
            if return_probs:
                return torch.softmax(logits, dim=1), attn_stack
            return logits, attn_stack
        
        if return_probs:
            return torch.softmax(logits, dim=1)
        return logits

# Instantiate the model for Indian Pines
num_bands = X_ip_train.shape[1]
num_classes = len(np.unique(y_ip_train))
model = SpectralTransformer(num_bands=num_bands, num_classes=num_classes, band_min=band_min, band_max=band_max,
                            d_model=64, nhead=8, num_layers=2, dim_feedforward=128)
if torch.backends.mps.is_available():
    print("Using MPS for model training.")
    model.to(device)
print(model)

# Prepare datasets and dataloaders
X_train_tensor = torch.from_numpy(X_ip_train).float().to(device)
y_train_tensor = torch.from_numpy(y_ip_train).long().to(device)
X_test_tensor  = torch.from_numpy(X_ip_test).float().to(device)
y_test_tensor  = torch.from_numpy(y_ip_test).long().to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epoch = 0
while epoch < max_epochs:
    epoch += 1
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    # Evaluate on test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            preds = model(batch_X).argmax(dim=1)
            correct_test += (preds == batch_y).sum().item()
            total_test += batch_y.size(0)
    test_accuracy = correct_test / total_test

    # Evaluate on training set
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            preds = model(batch_X).argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)
    train_accuracy = correct_train / total_train
    
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Train Acc = {train_accuracy*100:.2f}%, Test Acc = {test_accuracy*100:.2f}%")
    
    if train_accuracy - test_accuracy > 0.10:
        print("Warning: Overfitting detected!")
    
    if test_accuracy > best_accuracy + improvement_threshold:
        best_accuracy = test_accuracy
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"No significant improvement in test accuracy for {epochs_without_improvement} consecutive epoch(s).")
    
    if test_accuracy > min_accuracy and epochs_without_improvement >= patience:
        print("Stopping training early due to insufficient improvement.")
        break

# Final evaluation on test set
model.eval()
all_preds = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_probs = model(batch_X, return_probs=True)
        all_preds.append(batch_probs)
probs = torch.cat(all_preds, dim=0)
preds = probs.argmax(dim=1)
final_test_accuracy = (preds.cpu() == y_test_tensor.cpu()).float().mean().item()
print(f"Final Test Accuracy: {final_test_accuracy*100:.2f}%")

# Export the model to ONNX with attention outputs.
dummy_input = torch.randn(1, num_bands, requires_grad=True).cpu()
model.cpu()  # Ensure model is on CPU
model.eval() # Set to evaluation mode if not already
torch.onnx.export(
    model,
    (dummy_input, False, True),  # Now the model returns two outputs
    "/Users/phani/Desktop/AI/spectra-luma/model/SpectralTransformer.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['logits', 'attentions'],
    dynamic_axes={'input': {0: 'batch_size'}, 'logits': {0: 'batch_size'}, 'attentions': {1: 'batch_size'}}
)

# Save inference parameters (for restoring normalization and class mapping)
inference_params = {
    'model_state_dict': model.state_dict(),
    'band_min': band_min.tolist(),
    'band_max': band_max.tolist(),
    'CLASS_NAMES': CLASS_NAMES
}
torch.save(inference_params, "/Users/phani/Desktop/AI/spectra-luma/model/inference_params.pth")
params = {"CLASS_NAMES": CLASS_NAMES, "band_min": band_min.tolist(), "band_max": band_max.tolist()}
with open("/Users/phani/Desktop/AI/spectra-luma/model/inference_params.json", "w") as f:
    json.dump(params, f)

print("Model and inference parameters have been successfully saved.")