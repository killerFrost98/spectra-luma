import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import json

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

# Preprocessing function: normalize and flatten HSI data
def preprocess_hsi(data_cube, label_map):
    H, W, B = data_cube.shape  # Height, Width, Bands
    # Normalize each spectral band to [0, 1]
    data_cube = data_cube.astype(np.float32)
    data_cube = data_cube.reshape(-1, B)  # flatten pixels
    # Min-max normalization per band:
    band_min = data_cube.min(axis=0)
    band_max = data_cube.max(axis=0)
    data_norm = (data_cube - band_min) / (band_max - band_min + 1e-6)
    # Prepare labels correspondingly
    labels = label_map.reshape(-1)
    # Remove background pixels (assumed label 0 for unlabeled)
    mask = labels > 0
    X = data_norm[mask]               # pixel spectral data
    y = labels[mask] - 1             # convert labels to 0-based index
    return X, y

# Apply preprocessing to each dataset
X_ip, y_ip = preprocess_hsi(ip_data, ip_labels)
X_pu, y_pu = preprocess_hsi(pu_data, pu_labels)

# Check shapes and number of classes
print("Indian Pines:", X_ip.shape, "classes:", np.unique(y_ip))
print("Pavia University:", X_pu.shape, "classes:", np.unique(y_pu))

# Split into training and test sets (e.g., 70% train, 30% test)
X_ip_train, X_ip_test, y_ip_train, y_ip_test = train_test_split(X_ip, y_ip, test_size=0.3, stratify=y_ip, random_state=42)
X_pu_train, X_pu_test, y_pu_train, y_pu_test = train_test_split(X_pu, y_pu, test_size=0.3, stratify=y_pu, random_state=42)

class SpectralTransformer(nn.Module):
    def __init__(self, num_bands, num_classes, d_model=64, nhead=8, num_layers=2, dim_feedforward=128):
        super(SpectralTransformer, self).__init__()
        # 1. Linear projection for spectral values to d_model (embedding size)
        self.value_embed = nn.Linear(1, d_model)
        # 2. Positional encoding for each band (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_bands, d_model))
        # 3. Transformer Encoder: stack of self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 4. Classification head: linear layer to map Transformer output to class scores
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        # x is a batch of spectral sequences, shape (batch_size, num_bands)
        batch_size, seq_len = x.shape
        # Project each spectral value to an embedding vector
        # Expand last dimension for linear projection: (batch, seq_len, 1) -> (batch, seq_len, d_model)
        x_emb = self.value_embed(x.unsqueeze(-1))  
        # Add positional encoding to include band index information
        x_emb = x_emb + self.pos_embed  # shape: (batch, seq_len, d_model)
        # Pass through Transformer encoder layers
        x_enc = self.transformer(x_emb)  # shape: (batch, seq_len, d_model)
        # Aggregate sequence output (mean pooling)
        seq_avg = x_enc.mean(dim=1)     # shape: (batch, d_model)
        # Classification layer
        logits = self.classifier(seq_avg)  # shape: (batch, num_classes)
        return logits

# Instantiate model (example for Indian Pines)
num_bands = X_ip_train.shape[1]       # number of spectral bands (e.g., ~200 for Indian Pines)
num_classes = len(np.unique(y_ip_train))  # number of classes (16 for Indian Pines after removing background)
model = SpectralTransformer(num_bands=num_bands, num_classes=num_classes, 
                             d_model=64, nhead=8, num_layers=2, dim_feedforward=128)

print(model)

# Prepare PyTorch datasets and loaders for training and testing
X_train_tensor = torch.from_numpy(X_ip_train).float()
y_train_tensor = torch.from_numpy(y_ip_train).long()
X_test_tensor  = torch.from_numpy(X_ip_test).float()
y_test_tensor  = torch.from_numpy(y_ip_test).long()

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset  = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(1, num_epochs+1):
    model.train()  # set model to training mode
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)               # forward pass
        loss = criterion(outputs, batch_y)     # compute cross-entropy loss
        loss.backward()                        # backpropagation
        optimizer.step()                       # update parameters
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    # Periodically evaluate on test set
    if epoch % 5 == 0 or epoch == num_epochs:
        model.eval()  # evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                preds = model(batch_X).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        accuracy = correct / total
        print(f"Epoch {epoch}: Average Training Loss = {avg_loss:.4f}, Test Accuracy = {accuracy*100:.2f}%")

# Final evaluation on the full test set
model.eval()

with torch.no_grad():
    preds = model(X_test_tensor).argmax(dim=1)
    test_accuracy = (preds == y_test_tensor).float().mean().item()
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")

# After preprocessing your training data, compute normalization parameters
# (Assuming you're using the training data from Indian Pines)
data_cube = ip_data.astype(np.float32).reshape(-1, ip_data.shape[-1])
band_min = data_cube.min(axis=0)
band_max = data_cube.max(axis=0)


# Create a dummy input with the same number of spectral bands as your model expects
dummy_input = torch.randn(1, num_bands, requires_grad=True)

# Save the ONNX model as before (note: ONNX won't store these extra values)
torch.onnx.export(
    model,                      
    dummy_input,                
    "/Users/phani/Desktop/AI/spectra-luma/model/SpectralTransformer.onnx",
    export_params=True,         
    opset_version=16,           # Updated version
    do_constant_folding=True,   
    input_names=['input'],      
    output_names=['output'],    
)

# Save additional inference parameters (including normalization values) in a dictionary.
inference_params = {
    'model_state_dict': model.state_dict(),
    'band_min': band_min,  # numpy array of shape (num_bands,)
    'band_max': band_max,  # numpy array of shape (num_bands,)
    'CLASS_NAMES': CLASS_NAMES
}

# Save the dictionary to a file using torch.save (you could also use pickle or JSON if you prefer)
torch.save(inference_params, "/Users/phani/Desktop/AI/spectra-luma/model/inference_params.pth")

# Load the PyTorch inference parameters
inference_params = torch.load("/Users/phani/Desktop/AI/spectra-luma/model/inference_params.pth", map_location=torch.device('cpu'))

# Convert tensors/arrays to lists if necessary
band_min = inference_params['band_min']
band_max = inference_params['band_max']
if hasattr(band_min, 'tolist'):
    band_min = band_min.tolist()
if hasattr(band_max, 'tolist'):
    band_max = band_max.tolist()

# Save the parameters to a JSON file
params = {"band_min": band_min, "band_max": band_max, "CLASS_NAMES": CLASS_NAMES}
with open("/Users/phani/Desktop/AI/spectra-luma/model/inference_params.json", "w") as f:
    json.dump(params, f)

print("Model and inference parameters have been successfully saved.")