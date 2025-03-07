import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# Path to the exported ONNX model
onnx_file = "/Users/phani/Desktop/AI/spectra-luma/model/indian_pines_transformer.onnx"

# Create an inference session with the ONNX model
session = ort.InferenceSession(onnx_file)
output_names = [o.name for o in session.get_outputs()]
print("Model outputs:", output_names)

# Provided CSV input string (ensure there are 200 comma-separated values)
input_str = '''4020.000000, 4656.000000, 5169.000000, 5181.000000, 5762.000000, 6167.000000, 6358.000000, 6255.000000, 6423.000000, 6105.000000, 6121.000000, 6136.000000, 5968.000000, 6167.000000, 6221.000000, 6225.000000, 6139.000000, 5942.000000, 5919.000000, 5824.000000, 5794.000000, 5805.000000, 5788.000000, 5700.000000, 5695.000000, 5649.000000, 5494.000000, 5626.000000, 5599.000000, 5151.000000, 5238.000000, 5054.000000, 5188.000000, 5413.000000, 5088.000000, 4844.000000, 5291.000000, 5578.000000, 5375.000000, 3690.000000, 5431.000000, 5482.000000, 5297.000000, 5116.000000, 4841.000000, 4438.000000, 4608.000000, 5007.000000, 4904.000000, 4874.000000, 4814.000000, 4770.000000, 4711.000000, 3946.000000, 3799.000000, 3664.000000, 3310.000000, 2125.000000, 2284.000000, 2444.000000, 3086.000000, 2803.000000, 3954.000000, 4181.000000, 4060.000000, 4090.000000, 4008.000000, 3936.000000, 3908.000000, 3850.000000, 3758.000000, 3666.000000, 3603.000000, 2689.000000, 3083.000000, 2566.000000, 1641.000000, 1516.000000, 1764.000000, 1699.000000, 2103.000000, 2620.000000, 2767.000000, 2724.000000, 2793.000000, 2736.000000, 2828.000000, 2865.000000, 2966.000000, 2944.000000, 2856.000000, 2565.000000, 2655.000000, 2758.000000, 2540.000000, 2682.000000, 2681.000000, 2539.000000, 2284.000000, 2126.000000, 1779.000000, 1519.000000, 1101.000000, 1041.000000, 1060.000000, 1099.000000, 1136.000000, 1257.000000, 1344.000000, 1274.000000, 1372.000000, 1629.000000, 1858.000000, 2000.000000, 2103.000000, 2146.000000, 2192.000000, 2174.000000, 2155.000000, 2050.000000, 2056.000000, 2071.000000, 2036.000000, 2010.000000, 2059.000000, 2046.000000, 1993.000000, 1976.000000, 1961.000000, 1947.000000, 1915.000000, 1890.000000, 1852.000000, 1814.000000, 1780.000000, 1719.000000, 1674.000000, 1646.000000, 1599.000000, 1504.000000, 1385.000000, 1269.000000, 1124.000000, 1040.000000, 1026.000000, 1041.000000, 1093.000000, 1194.000000, 1279.000000, 1222.000000, 1098.000000, 1093.000000, 1204.000000, 1333.000000, 1322.000000, 1251.000000, 1244.000000, 1298.000000, 1349.000000, 1368.000000, 1376.000000, 1367.000000, 1363.000000, 1367.000000, 1356.000000, 1322.000000, 1313.000000, 1315.000000, 1307.000000, 1308.000000, 1286.000000, 1304.000000, 1289.000000, 1277.000000, 1259.000000, 1238.000000, 1224.000000, 1218.000000, 1206.000000, 1196.000000, 1192.000000, 1190.000000, 1168.000000, 1167.000000, 1158.000000, 1147.000000, 1157.000000, 1134.000000, 1109.000000, 1129.000000, 1123.000000, 1101.000000, 1097.000000, 1101.000000, 1081.000000, 1066.000000, 1063.000000, 1054.000000, 1026.000000, 1015.000000'''

# Convert the CSV string to a numpy array and reshape it to (1,200)
input_vals = np.array([float(x) for x in input_str.split(',')])
sample_input = input_vals.reshape(1, -1).astype(np.float32)

# Run inference.
# NOTE: This script assumes that your ONNX export has two outputs:
#   1. "class_probabilities": (1, num_classes)
#   2. "attention_weights": (1, n_heads, query_len, key_len)
outputs = session.run(None, {"spectral_vector": sample_input})
class_probs = outputs[0]  # first output: class probabilities

# Get predicted class index and convert it to a class name.
predicted_class = np.argmax(class_probs, axis=1)[0]
# Define the class names (update as needed for your model/dataset)
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

predicted_class_name = CLASS_NAMES.get(predicted_class)
print("Predicted class:", predicted_class_name)

# ---------------------------
# 1. Plot the Spectral Data
# ---------------------------
plt.figure(figsize=(10, 6))
plt.plot(np.arange(200), sample_input[0], marker='o', linestyle='-', color='blue')
plt.xlabel("Spectral Band (0-indexed)")
plt.ylabel("Spectral Value")
plt.title("Spectral Data")
plt.ylim(top=10000)  # Set the maximum y value to 10000
plt.show()

# ---------------------------
# 2. Plot the Attention Map
# ---------------------------
if len(outputs) > 1:
    attention_weights = outputs[1]
    # Expected shape: (batch, n_heads, query_len, key_len)
    attn = attention_weights[0]
    print("Attention weights shape:", attn.shape)

    # For visualization, focus on the attention from the [CLS] token.
    # Here we assume the first row corresponds to [CLS] and we take the first head for demonstration.
    cls_attn = attn[0, :]  # 1D array of length equal to key_len
    cls_attn_to_bands = cls_attn[1:]  # Exclude the [CLS] token itself

    # Identify the indices of the top 5 important spectral bands
    top5_idx = np.argsort(cls_attn_to_bands)[-5:][::-1]
    print("Top 5 important bands (0-indexed):", top5_idx)
        
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(200), cls_attn_to_bands, color='skyblue', edgecolor='black', label='Attention weight')
    plt.xlabel("Spectral Band (0-indexed)")
    plt.ylabel("Attention Weight")
    plt.title(f"Attention Weights for Sample Input (Predicted Class: {predicted_class_name})")
    
    # Highlight the top 5 bands with a different color
    for idx in top5_idx:
        plt.bar(idx, cls_attn_to_bands[idx], color='orange', edgecolor='black')
    plt.legend()
    plt.show()
else:
    print("Attention weights are not available from the ONNX model export.")



# import numpy as np
# import torch
# import onnxruntime as ort
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# # 1. Ask user for comma separated spectral input
# user_input = '''4020.000000, 4656.000000, 5169.000000, 5181.000000, 5762.000000, 6167.000000, 6358.000000, 6255.000000, 6423.000000, 6105.000000, 6121.000000, 6136.000000, 5968.000000, 6167.000000, 6221.000000, 6225.000000, 6139.000000, 5942.000000, 5919.000000, 5824.000000, 5794.000000, 5805.000000, 5788.000000, 5700.000000, 5695.000000, 5649.000000, 5494.000000, 5626.000000, 5599.000000, 5151.000000, 5238.000000, 5054.000000, 5188.000000, 5413.000000, 5088.000000, 4844.000000, 5291.000000, 5578.000000, 5375.000000, 3690.000000, 5431.000000, 5482.000000, 5297.000000, 5116.000000, 4841.000000, 4438.000000, 4608.000000, 5007.000000, 4904.000000, 4874.000000, 4814.000000, 4770.000000, 4711.000000, 3946.000000, 3799.000000, 3664.000000, 3310.000000, 2125.000000, 2284.000000, 2444.000000, 3086.000000, 2803.000000, 3954.000000, 4181.000000, 4060.000000, 4090.000000, 4008.000000, 3936.000000, 3908.000000, 3850.000000, 3758.000000, 3666.000000, 3603.000000, 2689.000000, 3083.000000, 2566.000000, 1641.000000, 1516.000000, 1764.000000, 1699.000000, 2103.000000, 2620.000000, 2767.000000, 2724.000000, 2793.000000, 2736.000000, 2828.000000, 2865.000000, 2966.000000, 2944.000000, 2856.000000, 2565.000000, 2655.000000, 2758.000000, 2540.000000, 2682.000000, 2681.000000, 2539.000000, 2284.000000, 2126.000000, 1779.000000, 1519.000000, 1101.000000, 1041.000000, 1060.000000, 1099.000000, 1136.000000, 1257.000000, 1344.000000, 1274.000000, 1372.000000, 1629.000000, 1858.000000, 2000.000000, 2103.000000, 2146.000000, 2192.000000, 2174.000000, 2155.000000, 2050.000000, 2056.000000, 2071.000000, 2036.000000, 2010.000000, 2059.000000, 2046.000000, 1993.000000, 1976.000000, 1961.000000, 1947.000000, 1915.000000, 1890.000000, 1852.000000, 1814.000000, 1780.000000, 1719.000000, 1674.000000, 1646.000000, 1599.000000, 1504.000000, 1385.000000, 1269.000000, 1124.000000, 1040.000000, 1026.000000, 1041.000000, 1093.000000, 1194.000000, 1279.000000, 1222.000000, 1098.000000, 1093.000000, 1204.000000, 1333.000000, 1322.000000, 1251.000000, 1244.000000, 1298.000000, 1349.000000, 1368.000000, 1376.000000, 1367.000000, 1363.000000, 1367.000000, 1356.000000, 1322.000000, 1313.000000, 1315.000000, 1307.000000, 1308.000000, 1286.000000, 1304.000000, 1289.000000, 1277.000000, 1259.000000, 1238.000000, 1224.000000, 1218.000000, 1206.000000, 1196.000000, 1192.000000, 1190.000000, 1168.000000, 1167.000000, 1158.000000, 1147.000000, 1157.000000, 1134.000000, 1109.000000, 1129.000000, 1123.000000, 1101.000000, 1097.000000, 1101.000000, 1081.000000, 1066.000000, 1063.000000, 1054.000000, 1026.000000, 1015.000000'''
# spectral_values = [float(x.strip()) for x in user_input.split(",")]
# spectral_array = np.array(spectral_values, dtype=np.float32).reshape(1, -1)

# # 2. Load inference parameters
# inference_params = torch.load("/Users/phani/Desktop/AI/spectra-luma/model/inference_params.pth", map_location="cpu")
# CLASS_NAMES = inference_params["CLASS_NAMES"]
# band_min = inference_params["band_min"]
# band_max = inference_params["band_max"]
# num_bands = len(band_min)
# if spectral_array.shape[1] != num_bands:
#     print(f"Error: Expected {num_bands} bands but got {spectral_array.shape[1]} bands.")
#     exit(1)

# # 3. Load the ONNX model
# ort_session = ort.InferenceSession("/Users/phani/Desktop/AI/spectra-luma/model/SpectralTransformer.onnx")
# ort_inputs = {ort_session.get_inputs()[0].name: spectral_array}

# # 4. Run inference to get logits and attentions
# outputs = ort_session.run(None, ort_inputs)
# logits = outputs[0]      # shape: (1, num_classes)
# attentions = outputs[1]  # shape: (num_layers, batch, num_heads, seq_len, seq_len)

# # 5. Compute class probabilities and print them sorted
# exp_logits = np.exp(logits)
# probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
# sorted_indices = np.argsort(probabilities[0])[::-1]
# print("\nProbabilities (sorted):")
# for idx in sorted_indices:
#     class_name = CLASS_NAMES.get(idx, f"Class {idx}")
#     print(f"{class_name}: {probabilities[0][idx] * 100:.2f}%")

# print("Attentions shape:", np.array(attentions).shape)

# # 6 & 7. Create a single figure with two subplots:
# #      - Left: Attention map from layer 1, head 1
# #      - Right: Input spectra (with y-axis maximum fixed at 10,000)

# # Get the attention map from the first layer and first batch element.
# attn_first_layer = attentions[0][0]  # shape: (seq_len, seq_len)

# # Create a figure with two subplots side-by-side
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # Plot the attention map in the left subplot
# im = axs[0].imshow(attn_first_layer, cmap='viridis')
# axs[0].set_title('Attention Map (Layer 1)')
# axs[0].set_xlabel('Key Positions')
# axs[0].set_ylabel('Query Positions')
# fig.colorbar(im, ax=axs[0])

# # Plot the input spectra in the right subplot
# axs[1].plot(spectral_array[0], marker='o', linestyle='-')
# axs[1].set_title('Input Spectra')
# axs[1].set_xlabel('Band Index')
# axs[1].set_ylabel('Intensity')
# axs[1].grid(True)
# axs[1].set_ylim(top=10000)  # Set the maximum y value to 10000

# plt.tight_layout()
# plt.show()