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

## 💾 3. Pretrained Models and Inference

In the root directory, two pretrained YOLOWorld–PhiNet checkpoints are provided:
  - ywv2-n-trained100epochs.pt
  - ywv2-s-trained100epochs.pt

Both models were trained for 100 epochs using the original [WorldTrainerFromScratch](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py) provided by Ultralytics and can be used directly for evaluation, fine-tuning, pruning or as teacher models in distillation experiments 

An additional script, inference.py, allows you to run inference on custom images using any of the trained or pruned models:

`python inference.py --model path/to/model.pt --classes "class1, class2, class3 ..." --image path/to/image.jpg`.
