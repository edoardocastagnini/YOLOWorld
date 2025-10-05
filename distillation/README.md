# Distillation

This folder contains the implementation of **knowledge distillation** for the custom **YOLOWorld + PhiNet** model, based on the Ultralytics YOLO framework.

---

## ⚙️ Setup

1. **Download the dataset**  
   Place the dataset inside the directory: /datasets

2. **Configure the distillation settings**  
   Edit the file: ultralytics/cfg/distillation.yaml

   In this file, specify:
    - The **teacher checkpoint path** (pretrained model to distill from)  
    - The **student model configuration YAML** (custom YOLO + PhiNet model located at ultralytics/cfg/models/v8/yolov8-worldv2-phinet.yaml)

## 🚀 Training

1. Within the distill.py script, you can modify:
	- The standard YOLO losses (e.g. box, cls, dfl losses)
	- The distillation loss weight, which balances the teacher–student learning

2. To start the distillation process, run:
```bash
python distill.py
   
