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

# Set seed for reproducibility
np.random.seed(42)

# Loop over each class and select one random valid pixel
for class_idx, class_name in CLASS_NAMES.items():
    # Ground truth labels are 1-indexed; use class_idx+1 for the current class
    class_mask = (ip_labels == (class_idx + 1))
    valid_indices = np.argwhere(class_mask)
    
    if valid_indices.size == 0:
        print(f"No valid pixels found for class: {class_name}")
        continue

    # Randomly select one index for the current class
    random_index = valid_indices[np.random.choice(valid_indices.shape[0], 1, replace=False)][0]
    i, j = random_index

    # Extract raw spectral data (e.g., 200 bands)
    pixel_raw = ip_data[i, j, :]

    print(f"Pixel position: height = {i}, width = {j}")
    spectral_str = ", ".join([f"{val:.6f}" for val in pixel_raw])
    print("Pixel spectral data:", spectral_str)
    print("Pixel class:", class_name)
    print("-" * 50)