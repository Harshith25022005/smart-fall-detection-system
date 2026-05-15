# ESP32 TinyML Fall Detection Firmware

ESP32-based wearable fall detection and activity recognition system using:

- MPU6050 accelerometer
- TensorFlow Lite Micro
- Bluetooth Serial communication
- Edge-based TinyML inference

---

# Features

- Real-time activity recognition
- Fall detection using FSM logic
- Quantized TensorFlow Lite model
- BLE activity query support
- Non-blocking LED notifications
- Majority-vote prediction smoothing

---

# Hardware Requirements

## Components

- ESP32 Dev Module
- MPU6050 Accelerometer/Gyroscope
- USB cable
- Breadboard / wearable setup

---

# Wiring

| MPU6050 | ESP32 |
|--------|--------|
| VCC | 3.3V |
| GND | GND |
| SDA | GPIO 21 |
| SCL | GPIO 22 |

---

# Software Requirements

## 1. Install Arduino IDE

Download Arduino IDE 2.x:

https://www.arduino.cc/en/software

Install normally.

---

# ESP32 Board Installation

## Step 1

Open Arduino IDE.

Go to:

File → Preferences

## Step 2

Inside:

"Additional Boards Manager URLs"

Add:

```text
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```

## Step 3

Go to:

Tools → Board → Boards Manager

Search:

```text
ESP32
```

Install:

```text
ESP32 by Espressif Systems
```

---

# Required Libraries

Install these libraries from:

Sketch → Include Library → Manage Libraries

## Install

### TensorFlow Lite ESP32

Search:

```text
TensorFlowLite_ESP32
```

Install latest compatible version.

---

# Built-in Libraries Used

These come with ESP32 package:

- Wire
- BluetoothSerial

---

# Project Files

| File | Description |
|------|-------------|
| main.ino | Main ESP32 firmware |
| model.h | Quantized TensorFlow Lite model |
| README.md | Documentation |

---

# Upload Instructions

## Step 1

Connect ESP32 using USB.

## Step 2

Select board:

Tools → Board → ESP32 Arduino → ESP32 Dev Module

## Step 3

Select COM Port:

Tools → Port

## Step 4

Click Upload.

---

# Serial Monitor

Baud Rate:

```text
115200
```

Expected startup:

```text
ESP32 FALL DETECTOR READY
```

---

# Bluetooth Commands

## Device Name

```text
ESP32_FALL_DETECT
```

## Commands

| Command | Action |
|---------|--------|
| s | Returns current activity |

---

# Current Activities

- Sitting
- Standing
- Walking
- Jogging
- Falling

---

# TinyML Details

## Input Window

- Window Size: 50
- Channels: 4
- Sampling Rate: 20 Hz

## Tensor Arena

```text
40 KB
```

## Model Type

Quantized TensorFlow Lite Micro model.

---

# Notes

This repository currently contains the first stable integrated prototype version of the system.

Future improvements:
- modular firmware architecture
- improved TinyML model
- cloud connectivity
- mobile application
- adaptive thresholding
- edge optimization