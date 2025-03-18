import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import json

# Path to the exported ONNX model and inference parameters file
onnx_file = "/Users/phani/Desktop/AI/spectra-luma/model/indian_pines_transformer.onnx"
params_file = "/Users/phani/Desktop/AI/spectra-luma/model/indian_pines_transformer_inference_params.json"

# Load inference parameters (including PCA and normalization parameters)
with open(params_file, "r") as f:
    params = json.load(f)

# Extract PCA and normalization parameters
# Note: JSON keys for CLASS_NAMES may be strings.
pca_components = np.array(params["pca_components"])  # shape: (num_pca_components, original_dim) e.g., (30, 200)
pca_mean = np.array(params["pca_mean"])              # shape: (original_dim,), e.g., (200,)
band_min = np.array(params["band_min"]) if params["band_min"] is not None else None
band_max = np.array(params["band_max"]) if params["band_max"] is not None else None
CLASS_NAMES = params["CLASS_NAMES"]

num_pca_components = pca_components.shape[0]
original_dim = pca_mean.shape[0]  # should be 200

# Create an inference session with the ONNX model
session = ort.InferenceSession(onnx_file)
output_names = [o.name for o in session.get_outputs()]
print("Model outputs:", output_names)

# Provided CSV input string (with 200 comma-separated values: raw spectral data)
input_str = '''3145.000000, 3971.000000, 4024.000000, 4023.000000, 4327.000000, 4527.000000, 4570.000000, 4387.000000, 4361.000000, 4092.000000, 4111.000000, 3935.000000, 3898.000000, 4031.000000, 4075.000000, 3995.000000, 3888.000000, 3785.000000, 3662.000000, 2760.000000, 2745.000000, 2660.000000, 2581.000000, 3262.000000, 3278.000000, 3155.000000, 3036.000000, 2990.000000, 2952.000000, 2808.000000, 3098.000000, 2784.000000, 3114.000000, 2788.000000, 3770.000000, 4011.000000, 4787.000000, 5437.000000, 5458.000000, 3951.000000, 5730.000000, 5784.000000, 5534.000000, 5588.000000, 5188.000000, 4757.000000, 4977.000000, 5431.000000, 5354.000000, 5342.000000, 5316.000000, 5357.000000, 5228.000000, 4334.000000, 4132.000000, 4031.000000, 3639.000000, 2221.000000, 2386.000000, 2534.000000, 3220.000000, 3725.000000, 4151.000000, 4379.000000, 4340.000000, 4373.000000, 4345.000000, 4337.000000, 4340.000000, 4311.000000, 4186.000000, 4120.000000, 4045.000000, 3888.000000, 2653.000000, 2820.000000, 1707.000000, 1576.000000, 1802.000000, 1706.000000, 2103.000000, 2559.000000, 2699.000000, 2633.000000, 2733.000000, 2739.000000, 2807.000000, 2863.000000, 2927.000000, 2870.000000, 2816.000000, 2552.000000, 2629.000000, 2736.000000, 2593.000000, 2712.000000, 2684.000000, 2526.000000, 2262.000000, 2035.000000, 1664.000000, 1456.000000, 1090.000000, 1026.000000, 1042.000000, 1054.000000, 1071.000000, 1119.000000, 1175.000000, 1129.000000, 1192.000000, 1303.000000, 1431.000000, 1537.000000, 1596.000000, 1643.000000, 1662.000000, 1687.000000, 1686.000000, 1623.000000, 1654.000000, 1670.000000, 1663.000000, 1660.000000, 1701.000000, 1691.000000, 1669.000000, 1647.000000, 1662.000000, 1642.000000, 1625.000000, 1595.000000, 1573.000000, 1526.000000, 1511.000000, 1476.000000, 1434.000000, 1418.000000, 1377.000000, 1309.000000, 1232.000000, 1170.000000, 1083.000000, 1034.000000, 1008.000000, 1020.000000, 1038.000000, 1085.000000, 1112.000000, 1100.000000, 1039.000000, 1049.000000, 1093.000000, 1146.000000, 1138.000000, 1117.000000, 1109.000000, 1128.000000, 1148.000000, 1160.000000, 1158.000000, 1159.000000, 1167.000000, 1164.000000, 1166.000000, 1155.000000, 1150.000000, 1160.000000, 1156.000000, 1149.000000, 1145.000000, 1146.000000, 1145.000000, 1141.000000, 1141.000000, 1124.000000, 1117.000000, 1103.000000, 1103.000000, 1089.000000, 1089.000000, 1091.000000, 1087.000000, 1072.000000, 1074.000000, 1061.000000, 1062.000000, 1062.000000, 1048.000000, 1058.000000, 1055.000000, 1046.000000, 1035.000000, 1036.000000, 1030.000000, 1024.000000, 1034.000000, 1025.000000, 1009.000000, 1005.000000'''
# Convert CSV string to numpy array and reshape it to (1,200)
input_vals = np.array([float(x) for x in input_str.split(',')])
raw_input = input_vals.reshape(1, -1).astype(np.float32)

# ---------------------------------
# 1. Apply PCA Transformation
# ---------------------------------
# Subtract the PCA mean and project the raw input using the PCA components.
# (X - mean) dot components_.T gives the PCA-transformed data.
input_pca = (raw_input - pca_mean) @ pca_components.T  # shape: (1, num_pca_components)

# ---------------------------------
# 2. Apply Normalization (band_minmax)
# ---------------------------------
epsilon = 1e-8
if band_min is not None and band_max is not None:
    input_normalized = (input_pca - band_min) / (band_max - band_min + epsilon)
else:
    input_normalized = input_pca

# Convert to float32 to match the model's expected input type.
input_normalized = input_normalized.astype(np.float32)

# Run inference using the normalized PCA-transformed input.
outputs = session.run(None, {"spectral_vector": input_normalized})
class_probs = outputs[0]  # first output: class probabilities

# Determine the predicted class index.
predicted_class = np.argmax(class_probs, axis=1)[0]
# Due to JSON conversion, keys in CLASS_NAMES may be strings.
predicted_class_name = CLASS_NAMES.get(str(predicted_class), "Unknown")
print("\nPredicted class:", predicted_class_name)

# Print all class probabilities.
print("\nClass Probabilities:")
for key, class_name in CLASS_NAMES.items():
    key_int = int(key)
    print(f"{class_name}: {class_probs[0][key_int]:.4f}")

# ---------------------------------
# 3. Plot the Raw Spectral Data
# ---------------------------------
plt.figure(figsize=(10, 6))
plt.plot(np.arange(200), raw_input[0], marker='o', linestyle='-', color='blue')
plt.xlabel("Spectral Band (0-indexed)")
plt.ylabel("Spectral Value")
plt.title("Raw Spectral Data")
plt.ylim(top=10000)
plt.show()

# ---------------------------------
# 4. Plot the Attention Map for All Bands
# ---------------------------------
if len(outputs) > 1:
    attention_weights = outputs[1]
    # Expected shape: (batch, n_heads, query_len, key_len)
    attn = attention_weights[0]
    print("Attention weights shape:", attn.shape)

    # For visualization, focus on the attention from the [CLS] token.
    # The [CLS] token is assumed to be the first element.
    cls_attn = attn[0, :]  # shape: (query_len,), where query_len = num_pca_components + 1
    
    # Exclude the [CLS] token itself to get the attention for each PCA component.
    a_components = cls_attn[1:]  # shape: (num_pca_components,)
    
    # ----------------------------
    # Option A: Plot Attention for PCA Components
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(num_pca_components), a_components, color='skyblue', edgecolor='black')
    plt.xlabel("PCA Component (0-indexed)")
    plt.ylabel("Attention Weight")
    plt.title("Attention Weights for PCA Components")
    plt.show()
    
    # ----------------------------
    # Option B: Map and Plot Attention for Original Bands
    # ----------------------------
    # Here we map the attention on PCA components back to the original 200 bands.
    # We use the absolute values of the PCA components as a simple weight.
    band_attn = np.dot(a_components, np.abs(pca_components))  # shape: (original_dim,)
    
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(original_dim), band_attn, color='skyblue', edgecolor='black')
    plt.xlabel("Original Band (0-indexed)")
    plt.ylabel("Mapped Attention Weight")
    plt.title(f"Attention Weights Mapped to Original Bands (Predicted Class: {predicted_class_name})")
    plt.show()
else:
    print("Attention weights are not available from the ONNX model export.")