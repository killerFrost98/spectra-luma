import numpy as np
import onnxruntime
import torch

# Provided CLASS_NAMES mapping
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

def main():
    print("[INFO] Loading inference parameters...")
    try:
        inference_params = torch.load("/Users/phani/Desktop/AI/spectra-luma/model/inference_params.pth", map_location=torch.device('cpu'))
        band_min = inference_params['band_min']  # numpy array of shape (num_bands,)
        band_max = inference_params['band_max']  # numpy array of shape (num_bands,)
        num_bands = band_min.shape[0]
        print("[SUCCESS] Loaded inference parameters successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load inference parameters: {e}")
        return

    print("[INFO] Loading ONNX model...")
    try:
        session = onnxruntime.InferenceSession("/Users/phani/Desktop/AI/spectra-luma/model/SpectralTransformer.onnx")
        print("[SUCCESS] ONNX model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load ONNX model: {e}")
        return

    # Retrieve input and output names from the ONNX model
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"[INFO] Model Input Name: {input_name}")
        print(f"[INFO] Model Output Name: {output_name}")
    except Exception as e:
        print(f"[ERROR] Failed to retrieve model input/output names: {e}")
        return

    # Prompt user for spectral data
    user_input = "2592.000000, 4373.000000, 4501.000000, 4573.000000, 5082.000000, 5409.000000, 5601.000000, 5537.000000, 5598.000000, 5281.000000, 5392.000000, 5379.000000, 5233.000000, 5448.000000, 5565.000000, 5590.000000, 5496.000000, 5390.000000, 5309.000000, 5265.000000, 5274.000000, 5340.000000, 5269.000000, 5248.000000, 5300.000000, 5210.000000, 5094.000000, 5237.000000, 5230.000000, 4842.000000, 5006.000000, 4775.000000, 5019.000000, 5271.000000, 5193.000000, 5095.000000, 5700.000000, 6004.000000, 5897.000000, 4123.000000, 5889.000000, 6018.000000, 5760.000000, 5620.000000, 5345.000000, 4905.000000, 5088.000000, 5538.000000, 5515.000000, 5472.000000, 5499.000000, 5464.000000, 5357.000000, 4487.000000, 4267.000000, 4175.000000, 3759.000000, 2277.000000, 2448.000000, 2699.000000, 2763.000000, 4100.000000, 4595.000000, 4835.000000, 4752.000000, 4743.000000, 4713.000000, 4683.000000, 4588.000000, 4574.000000, 4433.000000, 4334.000000, 4244.000000, 4076.000000, 3646.000000, 2967.000000, 1781.000000, 1641.000000, 1937.000000, 1843.000000, 2443.000000, 3075.000000, 3190.000000, 3202.000000, 3262.000000, 3263.000000, 2564.000000, 2700.000000, 2740.000000, 2716.000000, 2584.000000, 3020.000000, 3119.000000, 2567.000000, 2993.000000, 3226.000000, 3211.000000, 3006.000000, 2693.000000, 2456.000000, 1988.000000, 1650.000000, 1113.000000, 1049.000000, 1078.000000, 1120.000000, 1151.000000, 1318.000000, 1432.000000, 1321.000000, 1446.000000, 1777.000000, 2062.000000, 2271.000000, 2394.000000, 2472.000000, 2512.000000, 2511.000000, 2497.000000, 2357.000000, 2373.000000, 2404.000000, 2359.000000, 2337.000000, 2411.000000, 2391.000000, 2327.000000, 2300.000000, 2297.000000, 2251.000000, 2228.000000, 2183.000000, 2140.000000, 2080.000000, 2036.000000, 1949.000000, 1877.000000, 1863.000000, 1792.000000, 1663.000000, 1491.000000, 1351.000000, 1157.000000, 1056.000000, 1036.000000, 1046.000000, 1132.000000, 1275.000000, 1392.000000, 1319.000000, 1131.000000, 1125.000000, 1294.000000, 1424.000000, 1424.000000, 1329.000000, 1321.000000, 1380.000000, 1454.000000, 1482.000000, 1502.000000, 1488.000000, 1477.000000, 1486.000000, 1469.000000, 1439.000000, 1424.000000, 1422.000000, 1411.000000, 1397.000000, 1379.000000, 1406.000000, 1401.000000, 1396.000000, 1375.000000, 1346.000000, 1332.000000, 1320.000000, 1311.000000, 1289.000000, 1292.000000, 1289.000000, 1254.000000, 1251.000000, 1238.000000, 1208.000000, 1211.000000, 1173.000000, 1168.000000, 1160.000000, 1162.000000, 1128.000000, 1125.000000, 1126.000000, 1110.000000, 1085.000000, 1095.000000, 1073.000000, 1032.000000, 1019.000000"
    try:
        spectral_data = np.array([float(x.strip()) for x in user_input.split(',')])
        print("[INFO] User input parsed successfully.")
    except ValueError:
        print("[ERROR] Invalid input. Please enter numerical values separated by commas.")
        return

    # Validate input length
    if spectral_data.shape[0] != num_bands:
        print(f"[ERROR] Invalid input length. Expected {num_bands} values, but got {spectral_data.shape[0]}.")
        return

    print("[INFO] Normalizing input data...")
    spectral_data_norm = (spectral_data - band_min) / (band_max - band_min + 1e-6)
    print("[SUCCESS] Normalized input data.")

    # Reshape input data for model inference
    input_array = spectral_data_norm.astype(np.float32).reshape(1, -1)
    print(f"[INFO] Input data reshaped to: {input_array.shape}")

    # Run inference with the ONNX model
    print("[INFO] Running inference...")
    try:
        outputs = session.run([output_name], {input_name: input_array})
        logits = outputs[0]
        print("[SUCCESS] Inference completed successfully.")
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return

    # Compute softmax probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    probabilities = probabilities[0]  # remove the batch dimension

    # Get top 3 predictions
    top3_indices = np.argsort(probabilities)[::-1][:3]
    print("[INFO] Top 3 predictions:")
    for idx in top3_indices:
        label = CLASS_NAMES.get(idx, "Unknown")
        confidence = probabilities[idx] * 100
        print(f"{label}: {confidence:.2f}%")

if __name__ == "__main__":
    main()