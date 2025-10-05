from ultralytics import YOLOWorld
from ultralytics.distillation.trainer import DistillationTrainer
from ultralytics.utils import DEFAULT_CFG
import yaml


with open("ultralytics/cfg/distillation.yaml", "r") as f:
    hparams = yaml.safe_load(f)

teacher_model = YOLOWorld(hparams["teacher_checkpoint"])

trainer = DistillationTrainer(
    teacher_model = teacher_model,
    cfg=DEFAULT_CFG,
    overrides={"model": hparams["student_model"], "data": hparams["data"], "epochs": hparams["epochs"], "batch":16, "device": [0], "box": 1.875, "cls": 0.125, "dfl": 0.375, "distill_loss":10}
)

trainer.train()