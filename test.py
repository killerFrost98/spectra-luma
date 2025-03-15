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

# Loop over each class and select 2 random valid pixels
for class_idx, class_name in CLASS_NAMES.items():
    # Ground truth labels are 1-indexed; use class_idx+1 for the current class
    class_mask = (ip_labels == (class_idx + 1))
    valid_indices = np.argwhere(class_mask)
    
    if valid_indices.size == 0:
        print(f"{class_name}:")
        print("No valid pixels found for this class.\n")
        continue

    # Determine how many items to select (2 if possible, otherwise as many as available)
    n_items = 2 if valid_indices.shape[0] >= 2 else valid_indices.shape[0]
    selected_indices = valid_indices[np.random.choice(valid_indices.shape[0], n_items, replace=False)]
    
    # Print the class header
    print(f"{class_name}:")
    print("")  # Blank line
    
    # Print each example in the desired format
    for idx, (i, j) in enumerate(selected_indices, start=1):
        # Extract raw spectral data (assuming ~200 bands)
        pixel_raw = ip_data[i, j, :]
        spectral_str = ", ".join([f"{val:.6f}" for val in pixel_raw])
        print(f"Example {idx}:")
        print(f"{spectral_str}")
        print("")  # Blank line for separation