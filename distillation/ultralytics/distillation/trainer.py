from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
#impor torch
import torch
#F
import torch.nn.functional as F
import torch.nn as nn
                          #WorldTrainerFromScratch
class DistillationTrainer(WorldTrainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model  # Modello insegnante
        self.teacher_model.eval()  # Congela il modello insegnante
        # Inizializza hparams con i parametri passati
        self.hparams = kwargs.get("overrides", {})

        # Inizializza un dizionario per salvare le feature map
        self.teacher_features = {}
        self.student_features = {}
        # Registra gli hook per catturare le feature map dai layer 8, 11 e 14
        target_layers = [8,11,14]
        for layer_idx in target_layers:
            self.teacher_model.model.model[layer_idx].register_forward_hook(self.save_teacher_features(layer_idx))
        
    def save_teacher_features(self, layer_idx):
        def hook(module, input, output):
            self.teacher_features[layer_idx] = output
        return hook


