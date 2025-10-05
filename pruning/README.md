# ✂️ Pruning

Structured pruning for **YOLOWorld with PhiNet backbone**:
- Backbone & detection decoder pruned with **Torch-Pruning**
- Transformer head (C2fAttn) pruned with a **custom** routine
- Three importance criteria: **Magnitude**, **Taylor**, **LAMP**

---

## 📦 Files

- `ywv2_pruning_magn.py` — pruning with **Magnitude** importance
- `ywv2_pruning_taylor.py` — pruning with **Taylor** (1st-order) importance
- `ywv2_pruning_lamp.py` — pruning with **LAMP** (layer-adaptive magnitude)

---

## ⚙️ Arguments (edit via `argparse` in the script)

- `--model` : path to the trained YOLO-World checkpoint to prune (e.g., `runs/.../weights/best.pt`)
- `--target_prune_rate` : prune rate for **backbone + decoder** (e.g., `0.4` → keep 60%)
- `--head_prune_rate` : prune rate for **transformer head** (e.g., `0.5`)
- `--finetune_epochs` : epochs for **post-pruning fine-tuning** (e.g., `50`)

> These are the only knobs intended to be changed for experiments.

---

## 🚀 Usage

### Magnitude
```bash
python ywv2_pruning_magn.py \
  --model ywv2-n-trained100epochs.pt \
  --target_prune_rate 0.5 \
  --head_prune_rate 0.5 \
  --finetune_epochs 5
