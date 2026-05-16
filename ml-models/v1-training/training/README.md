# Version 1 Training Pipeline

Initial TinyML training pipeline for ESP32-based fall detection and activity recognition.

---

# Features

- Sliding window preprocessing
- Per-class stride control
- Synthetic fall data integration
- Standard normalization
- Conv1D neural network
- Class imbalance handling
- TensorFlow training pipeline

---

# Activities

- Walking
- Jogging
- Sitting
- Standing
- Falling

---

# Dataset Sources

## WISDM Dataset
Used for normal activity recognition.

## Synthetic Fall Dataset
Custom-generated fall data collected using ESP32 + MPU6050.

---

# Model Architecture

- Conv1D
- Batch Normalization
- MaxPooling
- GlobalAveragePooling
- Dense classifier

---

# Training Strategy

## Window Size
50 samples

## Sampling Rate
20 Hz

## Fall Handling
Fall class uses:
- tighter stride
- class weighting

to reduce imbalance problems.

---

# Output

The script exports:
- trained .h5 model
- normalization statistics
- evaluation metrics

---

# Deployment

Model is later converted to:
- TensorFlow Lite
- Quantized int8 model
- embedded C array (`model.h`)

for ESP32 TinyML inference.