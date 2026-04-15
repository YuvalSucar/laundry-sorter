# 🤖 Autonomous Laundry Sorter

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A?logo=raspberrypi)
![Arduino](https://img.shields.io/badge/Arduino-UNO-00979D?logo=arduino)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~90%25-brightgreen)

An end-to-end autonomous robotic system that classifies and sorts laundry items into labeled baskets — completely hands-free.

Built from scratch: 3D-printed robotic arm, custom-wired electronics, self-collected image dataset, and a fine-tuned deep learning model deployed on edge hardware.

---

## Demo

> The arm detects a clothing item via ultrasonic sensor → picks it up → holds it in front of the camera → classifies it using a ResNet18 model running on Raspberry Pi → drops it into the correct basket → waits for the next item.

### The System

<img src="תמונות/system_overview.jpg" width="450"/>

*The complete setup: robotic arm surrounded by 6 sorting baskets*

<img src="תמונות/robotic_arm.jpg" width="400"/>

*3D-printed robotic arm with servo motors and custom wiring*

### Dataset Sample

<img src="תמונות/dataset_sample.png" width="600"/>

*Sample images from the custom dataset — photographed and labeled manually*

---

## System Architecture

<img src="תמונות/system_architecture.png" width="600"/>

```
Camera (CSI)
     │
     ▼
Raspberry Pi 5          ← Python inference (ResNet18 + PyTorch)
     │  Serial (USB)
     ▼
Arduino UNO             ← C++ motion logic
     │  I2C
     ▼
PCA9685 PWM Driver
     │  PWM
     ▼
Servo Motors → Robotic Arm
```

**Hardware stack:** Raspberry Pi 5 · Arduino UNO · PCA9685 PWM driver · Servo motors · Ultrasonic sensor · Camera Module V2 · 5V/8A power supply

**Software stack:** Python · PyTorch · torchvision · C++ (Arduino) · Serial protocol (PICK / HOLD / DROP / ACK / DONE)

---

## ML Model

### Approach: Transfer Learning with ResNet18

Rather than training from scratch, I fine-tuned a pretrained ResNet18 (ImageNet weights) on a custom clothing dataset.

**Training pipeline:**
- **Stage 1** — Freeze all layers, train only the classification head (FC layer) for 15 epochs
- **Stage 2** — Unfreeze `layer4` + FC, fine-tune with a lower learning rate (1e-4) for 8 additional epochs
- **Best model** saved based on validation accuracy

**Data augmentation:** random horizontal flip, rotation (±10°), color jitter

**Classes (6):**
| Label | Description |
|-------|-------------|
| `color_pants` | Colored trousers |
| `color_shirts` | Colored shirts |
| `dresses` | Dresses |
| `jeans` | Denim jeans |
| `towel` | Towels |
| `white` | White laundry |

**Results:**
- Test accuracy: ~90%
- Evaluated with confusion matrix and per-class classification report
- Custom CNN (trained from scratch) was also tested and compared — ResNet18 with transfer learning outperformed it significantly

<img src="תמונות/confusion_matrix.png" width="500"/>

*Confusion matrix on test set*

<img src="תמונות/classification_report.png" width="500"/>

*Per-class precision, recall and F1 scores*

---

## Project Structure

```
laundry-sorter/
├── תמונות/
├── models/
├── arduino/
├── train/
├── val/
├── test/
├── pi_main.py
├── train_custom.py
└── README.md
```

---

## Author

Yuval Sucher
