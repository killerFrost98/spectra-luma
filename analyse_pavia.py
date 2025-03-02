import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 1) Dictionary for class names
CLASS_NAMES = {
    1: "Asphalt",
    2: "Meadows",
    3: "Gravel",
    4: "Trees",
    5: "Painted metal sheets",
    6: "Bare Soil",
    7: "Bitumen",
    8: "Self-Blocking Bricks",
    9: "Shadows"
}

def print_meta_data(data_cube, gt):
    print("=== Corrected Hyperspectral Data Info ===")
    print(f"Shape (rows, cols, spectral bands): {data_cube.shape}")
    print(f" - Number of rows: {data_cube.shape[0]}")
    print(f" - Number of columns: {data_cube.shape[1]}")
    print(f" - Number of spectral bands: {data_cube.shape[2]}")
    print(f"Data type: {data_cube.dtype}")
    print(f"Min pixel value: {np.min(data_cube)}")
    print(f"Max pixel value: {np.max(data_cube)}")
    print("")
    
    print("=== Ground Truth Data Info ===")
    print(f"Shape (rows, cols): {gt.shape}")
    unique_labels = np.unique(gt)
    print(f"Unique labels: {unique_labels}")
    print(f"Number of unique labels: {len(unique_labels)}")
    print("")

def load_data(corrected_file, gt_file):
    data_corrected = sio.loadmat(corrected_file)
    data_gt = sio.loadmat(gt_file)
    
    print("Keys in corrected data file:", data_corrected.keys())
    print("Keys in ground truth file:", data_gt.keys())
    
    data_cube = None
    for key in data_corrected.keys():
        if not key.startswith('__'):
            data_cube = data_corrected[key]
            print(f"Selected data key: {key} with shape {data_cube.shape}")
            break
            
    gt = None
    for key in data_gt.keys():
        if not key.startswith('__'):
            gt = data_gt[key]
            print(f"Selected ground truth key: {key} with shape {gt.shape}")
            break
            
    return data_cube, gt

def plot_sample_band(data_cube, band_index=50):
    plt.figure(figsize=(6, 6))
    plt.imshow(data_cube[:, :, band_index], cmap='gray')
    plt.title(f"Spectral Band {band_index}")
    plt.colorbar()
    plt.show()

def plot_ground_truth(gt):
    plt.figure(figsize=(6, 6))
    plt.imshow(gt, cmap='jet')
    plt.title("Ground Truth Classification")
    plt.colorbar()
    plt.show()

# 2) Updated distribution plot to use class names
def plot_class_distribution(gt):
    labels = gt.flatten()
    unique, counts = np.unique(labels, return_counts=True)
    
    # Optional: skip background (label=0)
    mask = unique != 0
    unique = unique[mask]
    counts = counts[mask]
    
    plt.figure(figsize=(8, 4))
    plt.bar(unique, counts, color='blue')
    plt.xlabel("Class Label")
    plt.ylabel("Pixel Count")
    plt.title("Class Distribution")
    
    # Replace numeric labels with class names
    x_ticks_labels = [CLASS_NAMES.get(cls, f"Class {cls}") for cls in unique]
    plt.xticks(unique, x_ticks_labels, rotation=90)
    
    plt.tight_layout()
    plt.show()

# 3) Updated spectral signatures plot to use class names
def plot_mean_spectral_signatures(data_cube, gt):
    labels = gt.flatten()
    unique_labels = np.unique(labels)
    classes = unique_labels[unique_labels != 0]  # skip background
    
    plt.figure(figsize=(8, 6))
    for cls in classes:
        indices = np.where(gt == cls)
        spectra = data_cube[indices[0], indices[1], :]
        mean_signature = np.mean(spectra, axis=0)
        
        class_label = CLASS_NAMES.get(cls, f"Class {cls}")
        plt.plot(mean_signature, label=class_label)
    
    plt.xlabel("Spectral Band Index")
    plt.ylabel("Mean Reflectance")
    plt.title("Mean Spectral Signatures per Class")
    plt.legend()
    plt.show()

def main():
    corrected_file = '/Users/phani/Desktop/AI/spectra-luma/dataset/PaviaU.mat'
    gt_file = '/Users/phani/Desktop/AI/spectra-luma/dataset/PaviaU_gt.mat'
    
    # Load the data
    data_cube, gt = load_data(corrected_file, gt_file)
    
    # Print meta-information
    print_meta_data(data_cube, gt)
    
    # Plot a sample band
    plot_sample_band(data_cube, band_index=50)
    
    # Plot the ground truth classification map
    plot_ground_truth(gt)
    
    # Plot the class distribution with actual class names
    plot_class_distribution(gt)
    
    # Plot the mean spectral signatures with actual class names
    plot_mean_spectral_signatures(data_cube, gt)
    
if __name__ == "__main__":
    main()