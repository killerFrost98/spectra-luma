import numpy as np
import scipy.io

# Load Indian Pines data and labels from .mat files
ip_data = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/Indian_pines_corrected.mat')['indian_pines_corrected']
ip_labels = scipy.io.loadmat('/Users/phani/Desktop/AI/spectra-luma/dataset/Indian_pines_gt.mat')['indian_pines_gt']

# Define the class names dictionary
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

# Create a mask to select only valid pixels (non-background: label > 0)
mask = ip_labels > 0
valid_indices = np.argwhere(mask)

# Set seed for reproducibility and randomly select 5 valid pixels
np.random.seed(42)
sample_indices = valid_indices[np.random.choice(valid_indices.shape[0], 5, replace=False)]

# Print formatted output for each selected pixel
for idx in sample_indices:
    i, j = idx
    # Extract raw spectral data (e.g., 200 bands)
    pixel_raw = ip_data[i, j, :]
    # Convert ground truth label (1-indexed) to 0-index for lookup in CLASS_NAMES
    pixel_class_idx = int(ip_labels[i, j]) - 1
    pixel_class_name = CLASS_NAMES[pixel_class_idx]
    
    print(f"Pixel position: height = {i}, width = {j}")
    # Format spectral data as comma separated values with 6 decimal places
    spectral_str = ", ".join([f"{val:.6f}" for val in pixel_raw])
    print("Pixel spectral data:", spectral_str)
    print("Pixel class:", pixel_class_name)
    print("-" * 50)