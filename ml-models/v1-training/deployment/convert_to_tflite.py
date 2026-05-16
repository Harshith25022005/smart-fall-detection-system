import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# =====================================================
# CONFIG — must match train_fixed.py EXACTLY
# =====================================================

WINDOW_SIZE  = 50
NUM_FEATURES = 4
FALL_STRIDE  = 1
STRIDE       = 25

CLASSES = ["Walking", "Jogging", "Sitting", "Standing", "Falling"]

WISDM_PATH = "../../../datasets/wisdm_dataset/WISDM_ar_v1.1_raw.txt"

FALL_PATH = "../../../datasets/synthetic_fall_dataset/synthetic_falling_clean.txt"
# =====================================================
# LOAD + WINDOW (identical to training)
# =====================================================

def parse_line(line):
    line = line.strip().replace(";", "")
    if not line: return None
    parts = line.split(",")
    if len(parts) != 6: return None
    try:
        activity = parts[1].strip()
        x   = float(parts[3])
        y   = float(parts[4])
        z   = float(parts[5])
        mag = np.sqrt(x*x + y*y + z*z)
        return activity, x, y, z, mag
    except:
        return None

def load_raw(path):
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = parse_line(line)
            if rec:
                data.append(rec)
    return data

def build_windows(data):
    grouped = {}
    for activity, x, y_val, z, mag in data:
        if activity not in CLASSES: continue
        if activity not in grouped: grouped[activity] = []
        grouped[activity].append([x, y_val, z, mag])
    X = []
    for activity in grouped:
        arr    = np.array(grouped[activity])
        stride = FALL_STRIDE if activity == "Falling" else STRIDE
        for start in range(0, len(arr) - WINDOW_SIZE, stride):
            window = arr[start:start + WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                X.append(window)
    return np.array(X)

print("Loading data (same files as training)...")
data  = load_raw(WISDM_PATH) + load_raw(FALL_PATH)
X_all = build_windows(data)

# Normalize with the SAME scaler logic as training
X_flat = X_all.reshape(-1, NUM_FEATURES)
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)
X_all  = X_flat.reshape(-1, WINDOW_SIZE, NUM_FEATURES).astype(np.float32)

np.random.shuffle(X_all)
rep_samples = X_all[:200]

print(f"Representative samples : {len(rep_samples)}")
print(f"Scaler mean            : {scaler.mean_}")
print(f"Scaler stdv            : {scaler.scale_}")

# =====================================================
# LOAD MODEL
# =====================================================

model = tf.keras.models.load_model("model_fall_detection.h5")
print("\n✅ Model Loaded")

# =====================================================
# REPRESENTATIVE DATASET — real data, correct scaler
# =====================================================

def representative_data_gen():
    for sample in rep_samples:
        yield [sample[np.newaxis, :, :]]

# =====================================================
# CONVERT
# =====================================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8
converter._experimental_disable_per_channel = True

print("\nConverting to INT8 TFLite...")
tflite_model = converter.convert()

with open("model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(f"✅ model_int8.tflite  —  {len(tflite_model)} bytes")

# =====================================================
# GENERATE model.h
# =====================================================

with open("model.h", "w") as f:
    f.write("#include <pgmspace.h>\n\n")
    f.write("const unsigned char model[] PROGMEM = {\n")
    for i, byte in enumerate(tflite_model):
        f.write(f"0x{byte:02x},")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n\n")
    f.write(f"const unsigned int model_len = {len(tflite_model)};\n")

print("✅ model.h created\nDONE")