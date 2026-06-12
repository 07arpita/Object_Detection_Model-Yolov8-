# Multi-Domain Object Detection Pipeline using YOLOv8

An end-to-end computer vision perception pipeline utilizing the **YOLOv8 Small (YOLOv8s)** framework. This model is fine-tuned via transfer learning on a multi-domain custom dataset to perform real-time localized object detection and tracking across retail apparel taxonomy and household/kitchen assets.

---

## 🚀 Core Capabilities
* **Granular Multi-Class Taxonomy:** Fine-tuned to detect and segment overlapping apparel types and household items simultaneously.
* **High-Resolution Spatial Scaling:** Trained with an inflated frame resolution of `imgsz=800` to maximize small-object feature retention.
* **Automated Inference Pipeline:** Converts raw, unseen video tracking data (`.mp4`/`.avi`) into frame-by-frame coordinate matrices with dynamic bounding-box overlays.

---

## 🛠️ System Architecture & Training Profile
Training was executed inside an isolated environment with hardware acceleration.

* **Base Architecture:** `yolov8s.pt` (Small variant balancing speed and spatial accuracy)
* **Compute Infrastructure:** NVIDIA Tesla T4 GPU (14,913 MiB VRAM) | 12.7 GB System RAM
* **Training Hyperparameters:**
  * Epochs: `30`
  * Image Resolution: `800 x 800 pixels`
  * Optimization Software: PyTorch 2.11.0 + CUDA 12.8

---

## 📁 Repository Structure
```text
├── Yolov8.ipynb          # End-to-end training, validation, and inference script
├── best.pt               # Final serialized model checkpoints (PyTorch weights)
├── README.md             # Technical documentation
├── sample.mp4            # Raw input test video stream
└── evaluation_metrics/   # Folder containing model verification graphs
    ├── results.png
    ├── confusion_matrix.png
    ├── BoxR_curve.png
    └── val_batch0_pred.jpg
```

---

## 📊 Dataset & Class Taxonomy
The dataset was sourced dynamically via Roboflow (Workspace: `alexa-fiverr`, Project: `alexa-tybnf`), incorporating background-subtracted and real-world environmental scenes. 

| Category | Primary Target Classes Detected |
| :--- | :--- |
| **Apparel & Fashion** | `Tshirt`, `pants`, `shirt`, `midi-dress`, `short-skirt`, `sweater`, `short`, `long-dress` |
| **Kitchen & Home** | `Kitchen Item`, `can-opener`, `chopping-board`, `cooking-pot`, `cooking-strainer`, `dish-cover` |
| **Fitness & Misc** | `Gym Equipment`, `dumbbells`, `electric-kettle` |

---

## 📈 Model Performance & Evaluation

### 1. Loss Convergence & Training Performance
The validation box loss and classification loss curves demonstrate smooth optimization convergence over the training history cycle.

![Training Progress Logs](evaluation_metrics/results.png)

### 2. Localization Precision & Class Mapping
The confusion matrix heatmap showcases the exact localization distributions, validating clean separation across closely related fashion variants and kitchen equipment.

![Confusion Matrix Heatmap](evaluation_metrics/confusion_matrix.png)

### 3. Recall Metrics
The Recall-Confidence curve monitors the model's true positive localization capability across multiple confidence evaluation thresholds.

![Recall Confidence Curve](evaluation_metrics/BoxR_curve.png)

### 4. Sample Validation Batch Predictions
Below are structural sample predictions extracted from model validation batches, demonstrating stable localization bounds on unseen test images:

![Validation Batch Predictions](evaluation_metrics/val_batch0_pred.jpg)

---

## 💻 Quickstart: Local Inference Execution

### 1. Installation
Install the necessary runtime packages directly into your workspace:
```bash
pip install ultralytics roboflow
```

### 2. Run Inference on a Test Video Stream
Execute the script below to run your fine-tuned model weights against any raw video asset:

```python
from ultralytics import YOLO
import os

# Initialize the fine-tuned model weights
model = YOLO("best.pt")

# Configure source video path and custom output folder structure
video_path = "sample.mp4"
custom_output_dir = "outputs/"

# Stream frame-by-frame GPU inference 
results = model(video_path, save=True, save_dir=custom_output_dir)
```

---

## 🔮 Future Development Matrix
* **Weight Compression:** Quantize the native `.pt` checkpoint weights into a hyper-optimized **ONNX** runtime layout to increase processing speeds on local edge processors.
* **Tracking Fusion:** Link an active Kalman-filtering tracking algorithm (e.g., ByteTrack) directly over the bounding-box coordinate arrays to maintain unique IDs for moving objects during sudden occlusions.
