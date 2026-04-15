# 🤖 Autonomous Laundry Sorter

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A?logo=raspberrypi)
![Arduino](https://img.shields.io/badge/Arduino-UNO-00979D?logo=arduino)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~90%25-brightgreen)

An end-to-end autonomous robotic system that classifies and sorts laundry items into labeled baskets with minimal human intervention.

The project combines a 3D-printed robotic arm, custom electronics, a self-collected image dataset, and a fine-tuned ResNet18 model running on Raspberry Pi.

---

## Demo

The workflow is simple:

1. The ultrasonic sensor detects an item in the input area.
2. The robotic arm picks it up.
3. The Raspberry Pi captures an image.
4. A ResNet18 model classifies the item.
5. The arm drops it into the matching basket.
6. The system waits for the next item.

### Full System

<img src="תמונות/system_overview.jpg" width="650"/>

*Complete setup with the robotic arm and labeled baskets*

<img src="תמונות/robotic_arm.jpg" width="420"/>

*3D-printed robotic arm with servo motors and custom wiring*

### Dataset Example

<img src="תמונות/dataset_sample.png" width="650"/>

*Sample images from the custom dataset*

---

## System Architecture

<img src="תמונות/system_architecture.png" width="650"/>

```text
Camera (CSI)
     │
     ▼
Raspberry Pi 5          ← Python inference (ResNet18 + PyTorch)
     │  Serial (USB)
     ▼
Arduino UNO             ← C++ motion control
     │  I2C
     ▼
PCA9685 PWM Driver
     │  PWM
     ▼
Servo Motors → Robotic Arm
```

### Hardware

- Raspberry Pi 5
- Arduino UNO
- PCA9685 PWM driver
- Servo motors
- Ultrasonic sensor
- Camera Module V2
- 5V/8A power supply

### Software

- Python
- PyTorch
- torchvision
- Arduino C++
- Serial communication protocol between Raspberry Pi and Arduino

---

## Machine Learning Model

### Model Choice

The final deployed model is **ResNet18 with transfer learning**.

Instead of training from scratch, a pretrained ResNet18 model was fine-tuned on a custom laundry dataset.

### Training Process

- **Stage 1:** Freeze all backbone layers and train only the final classification layer for 15 epochs
- **Stage 2:** Unfreeze `layer4` and the classifier head, then continue fine-tuning with a lower learning rate (`1e-4`) for 8 more epochs
- The best checkpoint was selected according to validation accuracy

### Data Augmentation

- Random horizontal flip
- Small random rotation
- Color jitter

### Classes

- `color_pants`
- `color_shirts`
- `dresses`
- `jeans`
- `towel`
- `white`

### Results

- Test accuracy: about **90%**
- Evaluated with confusion matrix and classification report
- A custom CNN and YOLO-based approaches were also explored, but ResNet18 performed best for this setup

<img src="תמונות/confusion_matrix.png" width="520"/>

*Confusion matrix on the test set*

<img src="תמונות/classification_report.png" width="520"/>

*Per-class precision, recall, and F1-score*

---

## Hardware-Software Protocol

```text
Pi → Arduino:   PICK
Arduino → Pi:   HOLD
Pi → Arduino:   DROP <label>
Arduino → Pi:   ACK <label>
Arduino → Pi:   DONE
```

This protocol lets the Raspberry Pi control the motion sequence while the Arduino handles the low-level arm movement.

---

## Project Structure

```text
laundry-sorter/
├── README.md
├── pi_main.py
├── train_custom.py
├── check_data.py
├── arduino/
│   └── arduino.ino
├── models/
│   └── resnet18_best_ft.pt
├── train/
├── val/
├── test/
└── תמונות/
    ├── system_overview.jpg
    ├── robotic_arm.jpg
    ├── dataset_sample.png
    ├── system_architecture.png
    ├── confusion_matrix.png
    └── classification_report.png
```

---

## Setup and Run

### Raspberry Pi

```bash
pip install torch torchvision pillow numpy pyserial picamera2
python pi_main.py
```

### Training

```bash
pip install torch torchvision scikit-learn
python train_custom.py
```

### Arduino

Open `arduino/arduino.ino` in the Arduino IDE and upload it to the Arduino UNO.

---

## Main Challenges

| Challenge | Solution |
|---|---|
| Collecting enough data | Built a custom dataset manually with multiple angles and lighting conditions |
| Limited compute | Used transfer learning instead of training a deep network from scratch |
| Reliable arm control | Split responsibilities between Raspberry Pi and Arduino |
| Power stability | Used a dedicated power supply for the driver and motors |
| Motion precision | Tuned servo positions and basket drop angles manually |

---

## Future Improvements

- Expand the dataset for better generalization
- Improve grip reliability for soft or folded garments
- Add active cooling for long Raspberry Pi runs
- Explore object detection for handling multiple items at once

---

## Author

**Yuval Sucher**  
Computer Science Student, Holon Institute of Technology
