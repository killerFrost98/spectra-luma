import numpy as np
import torch

# --- Load the saved normalization parameters ---
# These should be computed on your training dataset per spectral band.
# For example, during training you might have done:
#   band_min = X_train.min(axis=0)
#   band_max = X_train.max(axis=0)
# and saved them with np.save("band_min.npy", band_min)
# (and similarly for band_max).
band_min = np.load("band_min.npy")  # shape: (num_bands,)
band_max = np.load("band_max.npy")  # shape: (num_bands,)

# --- Define your model architecture (same as during training) ---
import torch.nn as nn

class SpectralTransformer(nn.Module):
    def __init__(self, num_bands, num_classes, d_model=64, nhead=8, num_layers=2, dim_feedforward=128):
        super(SpectralTransformer, self).__init__()
        self.value_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_bands, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        x_emb = self.value_embed(x.unsqueeze(-1))
        x_emb = x_emb + self.pos_embed
        x_enc = self.transformer(x_emb)
        seq_avg = x_enc.mean(dim=1)
        logits = self.classifier(seq_avg)
        return logits

# --- Load your trained model ---
# Ensure that num_bands and num_classes are set to the values used during training.
# For example, if your training data had 200 bands and 16 classes:
num_bands = band_min.shape[0]
num_classes = 16  # adjust if necessary

model = SpectralTransformer(num_bands=num_bands, num_classes=num_classes,
                            d_model=64, nhead=8, num_layers=2, dim_feedforward=128)

# Load the model parameters (if saved using torch.save)
model.load_state_dict(torch.load("SpectralTransformer_model.pth", map_location=torch.device('cpu')))
model.eval()  # set the model to inference mode

# --- Get raw input from the user ---
# The input should be a comma-separated string of spectral values.
input_str = input("Enter raw hyperspectral data (comma-separated values): ")
# Convert input string to a NumPy array of floats.
raw_input = np.array([float(val.strip()) for val in input_str.split(",")])

# --- Normalize the input ---
# Since the raw input is unnormalized, we normalize each band using the training min and max.
normalized_input = (raw_input - band_min) / (band_max - band_min + 1e-6)

# --- Convert to PyTorch tensor ---
# The model expects input shape (batch_size, num_bands)
input_tensor = torch.from_numpy(normalized_input).float().unsqueeze(0)

# --- Run inference ---
with torch.no_grad():
    logits = model(input_tensor)
    predicted_class = torch.argmax(logits, dim=1).item()

print("Predicted class:", predicted_class)