import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json
from collections import Counter

# Set device (MPS if available)
device = torch.device("mps")  # fallback: torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Load Indian Pines data from .mat files
ip_data = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/Indian_pines_corrected.mat')['indian_pines_corrected']  # shape: (145, 145, n_bands)
ip_labels = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/Indian_pines_gt.mat')['indian_pines_gt']              # shape: (145, 145)

# Load Pavia University data from .mat files
pu_data = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/PaviaU.mat')['paviaU']       # shape: (610, 610, n_bands)
pu_labels = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/PaviaU_gt.mat')['paviaU_gt']  # shape: (610, 610)

# Preprocessing function without normalization (since normalization will be done in the model)
def preprocess_hsi(data_cube, label_map):
    H, W, B = data_cube.shape
    data_cube = data_cube.astype(np.float32)
    data_cube = data_cube.reshape(-1, B)
    labels = label_map.reshape(-1)
    # Remove background pixels (assumed label 0)
    mask = labels > 0
    X = data_cube[mask]
    y = labels[mask] - 1  # Convert to 0-based labels
    return X, y

# Apply preprocessing
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

# Split data into training and testing sets (using stratification)
X_ip_train, X_ip_test, y_ip_train, y_ip_test = train_test_split(
    X_ip, y_ip, test_size=0.3, stratify=y_ip, random_state=42
)
X_pu_train, X_pu_test, y_pu_train, y_pu_test = train_test_split(
    X_pu, y_pu, test_size=0.3, stratify=y_pu, random_state=42
)

print_class_distribution(y_ip_train, "Indian Pines (Train)")
print_class_distribution(y_ip_test, "Indian Pines (Test)")
print_class_distribution(y_pu_train, "Pavia University (Train)")
print_class_distribution(y_pu_test, "Pavia University (Test)")

# Compute per-band min and max from the training data (for normalization)
band_min = X_ip_train.min(axis=0)
band_max = X_ip_train.max(axis=0)

print("Band Min:", band_min)
print("Band Max:", band_max)

# Define the SpectralTransformer model with integrated normalization
# Define the SpectralTransformer model with integrated normalization and log-softmax
class SpectralTransformer(nn.Module):
    def __init__(self, num_bands, num_classes, band_min, band_max, d_model=64, nhead=8, num_layers=2, dim_feedforward=128):
        super(SpectralTransformer, self).__init__()
        # Register the normalization parameters as buffers (not trainable)
        self.register_buffer("band_min", torch.tensor(band_min, dtype=torch.float32))
        self.register_buffer("band_max", torch.tensor(band_max, dtype=torch.float32))
        self.eps = 1e-8  # small epsilon to avoid division by zero
        
        # 1. Linear projection: from a scalar spectral value to d_model
        self.value_embed = nn.Linear(1, d_model)
        # 2. Learnable positional encoding for each spectral band
        self.pos_embed = nn.Parameter(torch.zeros(1, num_bands, d_model))
        # 3. Layer normalization for the embedded spectral values
        self.input_norm = nn.LayerNorm(d_model)
        # 4. Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 5. Classification head mapping aggregated output to class logits
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x, return_probs=False):
        # x: (batch_size, num_bands)
        # Normalize each spectral band on the fly using stored parameters
        x = (x - self.band_min) / (self.band_max - self.band_min + self.eps)
        
        batch_size, seq_len = x.shape
        # Expand to add a singleton dimension for linear projection: (batch, seq_len, 1)
        x_emb = self.value_embed(x.unsqueeze(-1))
        # Add positional encoding
        x_emb = x_emb + self.pos_embed
        # Apply layer normalization
        x_emb = self.input_norm(x_emb)
        # Pass through transformer encoder layers
        x_enc = self.transformer(x_emb)
        # Aggregate sequence outputs (mean pooling)
        seq_avg = x_enc.mean(dim=1)
        # Classification head producing logits
        logits = self.classifier(seq_avg)
        # Convert logits to log probabilities for numerical stability
        log_probs = torch.log_softmax(logits, dim=1)
        
        if return_probs:
            # Return probabilities by exponentiating the log probabilities
            return torch.exp(log_probs)
        else:
            # Return log probabilities (for use with NLLLoss)
            return log_probs

# Instantiate the model for Indian Pines
num_bands = X_ip_train.shape[1]
num_classes = len(np.unique(y_ip_train))
model = SpectralTransformer(num_bands=num_bands, num_classes=num_classes, band_min=band_min, band_max=band_max,
                            d_model=64, nhead=8, num_layers=2, dim_feedforward=128)
if torch.backends.mps.is_available():
    print("Using MPS for model training.")
    model.to(device)

print(model)

# Prepare PyTorch datasets and dataloaders for training and testing (using Indian Pines)
X_train_tensor = torch.from_numpy(X_ip_train).float().to(device)
y_train_tensor = torch.from_numpy(y_ip_train).long().to(device)
X_test_tensor  = torch.from_numpy(X_ip_test).float().to(device)
y_test_tensor  = torch.from_numpy(y_ip_test).long().to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Update loss function: Use NLLLoss which expects log probabilities
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Early stopping settings
min_accuracy = 0.85  # stop only after reaching at least 85% test accuracy
improvement_threshold = 0.01  # require at least 1% improvement in test accuracy
patience = 3                  # stop if no improvement for 3 consecutive epochs
epochs_without_improvement = 0
best_accuracy = 0.0
max_epochs = 100

# Training loop with early stopping and over/underfitting checks
epoch = 0
while epoch < max_epochs:
    epoch += 1
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        # Forward pass returns log probabilities by default
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    # Evaluation on test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            # For prediction, argmax over log probabilities is equivalent to argmax over probabilities
            preds = model(batch_X).argmax(dim=1)
            correct_test += (preds == batch_y).sum().item()
            total_test += batch_y.size(0)
    test_accuracy = correct_test / total_test

    # Evaluation on training set
    correct_train = 0
    total_train = 0
    with torch.no_grad():
        for batch_X, batch_y in train_loader:
            preds = model(batch_X).argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)
    train_accuracy = correct_train / total_train
    
    print(f"Epoch {epoch}: Avg Training Loss = {avg_loss:.4f}, Train Accuracy = {train_accuracy*100:.2f}%, Test Accuracy = {test_accuracy*100:.2f}%")
    
    if train_accuracy - test_accuracy > 0.10:
        print("Warning: Overfitting detected! Training accuracy is significantly higher than test accuracy.")
    
    if test_accuracy > best_accuracy + improvement_threshold:
        best_accuracy = test_accuracy
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        print(f"No significant improvement in test accuracy for {epochs_without_improvement} consecutive epoch(s).")
    
    if test_accuracy > min_accuracy and epochs_without_improvement >= patience:
        print("Stopping training early due to insufficient improvement.")
        break

# Final evaluation on the full test set using batches
model.eval()
all_preds = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        # Here, we return probabilities for clarity
        batch_probs = model(batch_X, return_probs=True)
        all_preds.append(batch_probs)
probs = torch.cat(all_preds, dim=0)
preds = probs.argmax(dim=1)
final_test_accuracy = (preds.cpu() == y_test_tensor.cpu()).float().mean().item()
print(f"Final Test Accuracy: {final_test_accuracy*100:.2f}%")

# Export the model to ONNX format (the exported model outputs log probabilities; apply softmax as needed)
dummy_input = torch.randn(1, num_bands, requires_grad=True).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "/Users/phani/Desktop/AI/spectra-luma/model/SpectralTransformer.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# Save inference parameters including the CLASS_NAMES mapping
inference_params = {
    'model_state_dict': model.state_dict(),
    'band_min': band_min.tolist(),
    'band_max': band_max.tolist(),
    'CLASS_NAMES': CLASS_NAMES
}
torch.save(inference_params, "/Users/phani/Desktop/AI/spectra-luma/model/inference_params.pth")
params = {"CLASS_NAMES": CLASS_NAMES, "band_min": band_min, "band_max": band_max}
with open("/Users/phani/Desktop/AI/spectra-luma/model/inference_params.json", "w") as f:
    json.dump(params, f)

print("Model and inference parameters have been successfully saved.")