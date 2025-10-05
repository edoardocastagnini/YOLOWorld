# YOLOWorld – PhiNet Extension with Distillation & Pruning

This repository extends the original [Ultralytics YOLOWorld](https://github.com/ultralytics/ultralytics) framework to explore **model compression techniques** — specifically **Knowledge Distillation** and **Pruning** — applied to a **custom YOLOWorld architecture** with a **PhiNet backbone**.

---

## 🔧 Overview

The project is divided into two main modules:

1. **Distillation** – integrates a new *Distillation Loss* into the YOLO training pipeline.  
2. **Pruning** – implements both standard and custom pruning methods for YOLOWorld components.

Both modules build directly on top of the **Ultralytics YOLOWorld codebase**, keeping the same structure, with a few added or modified files.

---

## 🧩 Custom YOLOWorld Architecture

The base model used here is a **custom YOLOWorld** where the **original backbone** has been replaced with a **PhiNet** block.  
This version maintains the standard YOLO head and decoder, ensuring full compatibility with the YOLO training and evaluation routines.

---

## 🔬 1. Knowledge Distillation

A new **Distillation Loss** was implemented inside the YOLO training loop, combined with the three standard YOLO losses:

- **Box loss**  
- **Class loss**  
- **Distribution focal loss**  

The *distillation loss* term encourages the student model to mimic intermediate representations and predictions of the teacher model.

### 🧪 Experiments

Several experiments were carried out:

- Varying the weight of the standard YOLO losses (original, halved, quartered).  
- Testing different strengths of the distillation term.  
- Distilling:  
  - **YOLOWorld-N (student)** from **YOLOWorld-N (teacher)**
  - **YOLOWorld-N (student)** from **YOLOWorld-S (teacher)**  
  - **YOLOWorld-S (student)** from **YOLOWorld-S (teacher)**  

The goal was to find the optimal trade-off between **accuracy**, **stability**, and **student compactness**.

---

## ✂️ 2. Pruning

The pruning stage focused on **reducing model size and complexity** while maintaining detection performance.

Two distinct approaches were used:

- **Backbone + Decoder** → standard *Torch-Pruning* workflow.  
- **Transformer Head (C2fAttn modules)** → custom pruning method developed for this project.

This was necessary because the **C2fAttn layers** in YOLOWorld are **not natively compatible** with Torch-Pruning dependency tracing.  
A custom implementation handles zeroing and structural pruning safely within the transformer head.

### 🧠 Importance Criteria

Three importance metrics were tested:

- **Magnitude** – classical weight-based criterion.  
- **Taylor** – uses first-order Taylor expansion to estimate loss sensitivity.  
- **LAMP** – Layer-Adaptive Magnitude Pruning, balancing pruning across layers.  

Each combination of backbone/decoder and transformer head prune rates was tested to study performance trade-offs.

---

## 📁 Repository Structure

├── distillation/                          # Knowledge distillation experiments
│   ├── datasets/                          # Dataset directory
│   │   └── README.md                      # Instructions for dataset download and structure
│   │
│   ├── ultralytics/                       # Modified Ultralytics YOLO base code
│   │   ├── cfg/                           # Config files (includes distillation.yaml and PhiNet YAML)
│   │   │   └── models/v8/yolov8-worldv2-phinet.yaml
│   │   ├── data/                          # Dataset configs (e.g. coco.yaml)
│   │   ├── distillation/                  # Custom distillation loss and utilities
│   │   ├── engine/                        # Training engine and DDP setup
│   │   ├── hub/                           # Model hub and export tools
│   │   ├── models/                        # YOLO model definitions
│   │   ├── nn/                            # Neural network modules (includes PhiNet backbone)
│   │   ├── solutions/                     # Optional YOLO utilities
│   │   ├── trackers/                      # Object tracking components
│   │   ├── utils/                         # Helper functions
│   │   └── __init__.py
│   │
│   ├── distill.py                         # Main training script for knowledge distillation
│   ├── requirements.txt                   # Dependencies for distillation
│   ├── train_args.yaml                    # Default distillation training args
│   ├── ywv2-n-trained100epochs.pt         # Teacher model (YOLO-WorldV2-N)
│   ├── ywv2-s-trained100epochs.pt         # Teacher model (YOLO-WorldV2-S)
│   └── README.md                          # Distillation instructions
│
├── pruning/                               # Model pruning experiments
│   ├── datasets/                          # Dataset directory
│   │   └── README.md                      # Instructions for dataset download and structure
│   │
│   ├── ultralytics/                       # Modified Ultralytics YOLO base code
│   │   ├── cfg/                           # Config files
│   │   ├── data/                          # Dataset configs
│   │   ├── engine/                        # Training engine
│   │   ├── hub/                           # Model hub and export tools
│   │   ├── models/                        # YOLO model definitions
│   │   ├── nn/                            # Neural network modules (PhiNet backbone)
│   │   ├── solutions/                     # Optional YOLO utilities
│   │   ├── trackers/                      # Object tracking components
│   │   ├── utils/                         # Helper functions
│   │   └── __init__.py
│   │
│   ├── ywv2_pruning_magn.py               # Magnitude-based pruning
│   ├── ywv2_pruning_taylor.py             # Taylor-based pruning
│   ├── ywv2_pruning_lamp.py               # LAMP-based pruning
│   ├── requirements.txt                   # Dependencies for pruning
│   ├── train_args.yaml                    # Training configuration after pruning
│   ├── ywv2-n-trained100epochs.pt         # Pretrained model (YOLO-WorldV2-N)
│   ├── ywv2-s-trained100epochs.pt         # Pretrained model (YOLO-WorldV2-S)
│   └── README.md                          # Pruning instructions
│
└── README.md                              # Main repository documentation
