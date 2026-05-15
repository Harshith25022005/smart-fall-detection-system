# Datasets

Datasets used for training and evaluating the fall detection and activity recognition models.

---

# Included Datasets

| Dataset                   | Description                                                |
|---------------------------|------------------------------------------------------------|
| synthetic_fall_dataset    | Custom synthetic fall data collected using ESP32 + MPU6050 |
| wisdm_dataset             | Public WISDM activity recognition dataset                  |

---

# Synthetic Fall Dataset

Custom-generated fall dataset collected using the wearable prototype.

## Activities
- Falling
- Walking
- Sitting
- Standing
- Jogging

## Sensor
- MPU6050

## Sampling
- Approx. 20 Hz

---

# WISDM Dataset

Public human activity recognition dataset.

Source:
https://www.cis.fordham.edu/wisdm/dataset.php

Used for:
- activity recognition pretraining
- motion pattern analysis
- baseline comparison

---

# Notes

Datasets may contain:
- raw sensor values
- processed CSV files
- augmented data
- train/test splits

Future versions may migrate large datasets to external cloud storage.