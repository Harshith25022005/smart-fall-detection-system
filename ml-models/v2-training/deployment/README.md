# Version 1 TinyML Deployment Pipeline

Converts the trained TensorFlow model into an ESP32-compatible TensorFlow Lite Micro model.

---

# Purpose

This pipeline performs:

- TensorFlow Lite conversion
- INT8 quantization
- representative dataset calibration
- TensorFlow Lite export
- automatic `model.h` generation

for deployment on ESP32.

---

# Pipeline

```text
train.py
   ↓
model_fall_detection.h5
   ↓
convert_to_tflite.py
   ↓
model_int8.tflite
   ↓
model.h
   ↓
ESP32 firmware deployment
```

---

# Features

- Full INT8 quantization
- Representative dataset calibration
- ESP32-compatible inference
- Automatic C header generation
- TensorFlow Lite Micro deployment support

---

# Generated Files

| File | Description |
|---|---|
| model_int8.tflite | Quantized TinyML model |
| model.h | Embedded C array for ESP32 |

---

# Quantization Details

## Input Type
INT8

## Output Type
INT8

## Optimization
Default TensorFlow Lite optimization

## Representative Dataset
Real sensor windows using:
- WISDM activities
- synthetic fall samples

---

# Notes

The generated `model.h` file is copied into:

```text
firmware/model.h
```

for deployment on the ESP32 firmware.

---

# Future Improvements

Planned future upgrades:

- quantization-aware training
- latency benchmarking
- automatic deployment scripts
- memory optimization
- tensor arena auto-sizing
- model version tracking