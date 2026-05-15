# Hardware Setup

Pocket-sized wearable ESP32 TinyML fall detection prototype.

---

# Hardware Components

| Component                 | Description                       |
|---------------------------|-----------------------------------|
| ESP32 DevKit V1           | Main microcontroller              |
| MPU6050                   | Accelerometer + gyroscope sensor  |
| 18650 Li-ion Cell         | Portable battery source           |
| TP4056 Type-C Module      | Battery charging + protection     |
| MT3608 Boost Converter    | Boosts battery voltage to 5V      |
| Slide Switch              | Power ON/OFF control              |
| LED + 1k Resistor         | Visual status indicator           |
| Buzzer                    | Audio alert output                |

---

# System Architecture

```text
18650 Battery
      ↓
TP4056 Charger Module
      ↓
Power Switch
      ↓
MT3608 Boost Converter (5V)
      ↓
ESP32 VIN
      ↓
MPU6050 + peripherals
```

---

# Power Chain

## Battery → TP4056

| Battery Holder | TP4056 |
|---|---|
| Positive (+) | B+ |
| Negative (-) | B- |

---

## TP4056 → Switch → MT3608

| Connection | Destination |
|---|---|
| TP4056 OUT+ | Switch COM |
| Switch ON pin | MT3608 VIN+ |
| TP4056 OUT- | MT3608 VIN- |

---

## MT3608 → ESP32

| MT3608 | ESP32 |
|---|---|
| VOUT+ (5V) | VIN / 5V |
| VOUT- | GND |

---

# Common Ground

All grounds are connected together:

- TP4056 OUT-
- MT3608 VIN-
- MT3608 VOUT-
- ESP32 GND
- MPU6050 GND
- LED cathode
- Buzzer GND

---

# MPU6050 Wiring

## I2C Connections

| MPU6050 | ESP32 |
|---|---|
| VCC | 3V3 |
| GND | GND |
| SDA | GPIO21 |
| SCL | GPIO22 |

---

## Optional Pins

| Pin | Status |
|---|---|
| AD0 | Not connected (0x68 address) |
| INT | Not connected |

---

# LED Indicator

| Connection | Description |
|---|---|
| GPIO25 → 1k resistor → LED anode | LED control |
| LED cathode → GND | Ground |

---

# Buzzer Wiring

| Buzzer Pin | ESP32 |
|---|---|
| Positive (+) | 3V3 |
| Negative (-) | GPIO5 |

---

# Important Notes

## MT3608 Voltage Adjustment

Before connecting to the ESP32:

- Adjust MT3608 output to exactly 5.0V using a multimeter.
- Incorrect voltage may damage the ESP32.

---

## TP4056 Limitation

This TP4056 module is NOT a true power-path charger.

Recommended practice:

- Turn OFF the device while charging.
- Avoid simultaneous charging + heavy system load.

---

# Prototype Notes

Current prototype is designed as:

- compact pocket-box wearable
- battery-powered edge AI device
- offline TinyML inference system

---

# Future Hardware Improvements

Planned future revisions:

- custom PCB
- LiPo charging power-path IC
- vibration motor alerts
- dedicated buzzer driver transistor
- battery monitoring circuit
- low-power optimization
- wearable enclosure design