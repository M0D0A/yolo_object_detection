import torch
from pathlib import Path


class SaveController:
    def __init__(self, mode="min", threshold=0, threshold_mode="rel"):
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None


    def calculation(self, value):
        if self.mode == "min" and self.threshold_mode == "abs":
            return value < self.best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "abs":
            return value > self.best + self.threshold
        elif self.mode == "min" and self.threshold_mode == "rel":
            return value < self.best - self.best * self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return value > self.best + self.best * self.threshold
        

    def __call__(self, value):
        if self.best is None:
            self.best = value
            return True
        
        if self.calculation(value):
            self.best = value
            return True
    
        return False


def save_model(
        epoch, EPOCHS, model_state_dict,
        optimizer_state_dict, lr_scheduler_state_dict,
        train_loss, val_loss, metrics, lr_list,
        save_path 
    ):
    save_path = Path(save_path)
    checkpoint = {
        "epoch": f"{epoch}/{EPOCHS}",
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
        "history_of_education": {
            "loss": {
                "train": train_loss,
                "val": val_loss
            },
            "mAP": metrics,
            "lr_list": lr_list
        }
    }
    torch.save(checkpoint, save_path)
