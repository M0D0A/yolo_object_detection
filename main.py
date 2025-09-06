import torch
import random
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data_preparation import create_loader
from src.models.backbone import resnet50
from src.models.yolo_v1 import YoloV1
from src.loss import YoloLoss
from src.traning import train
from src.utils.save_controller import SaveController
from src.utils.train_stoper import TrainStoper


BACKBONE_FREEZE = True
LAYER4_UNFREEZE = True
SEED = 52

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(SEED) # random
    np.random.seed(SEED) # numpy
    torch.manual_seed(SEED) # torch CPU
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED) # torch GPU

    train_loader, val_loader = create_loader(
        path="datasets/VOC2012",
        batch_size=32,
        train_shuffle=True,
        split=[0.8, 0.2],
        num_workers=0,
    )
    
    model = YoloV1(
        S=7, B=2, num_cls=20,
        backbone=resnet50(),
        bn=True, use_layer="conv1x1",
        act="leakyrelu"
    )

    model.load_state_dict(torch.load("models/yolov1_first_stage_46.plt", map_location="cpu", weights_only=False)["model_state_dict"])
    model = model.to(device)

    if BACKBONE_FREEZE:
        for param in model.backbone.parameters():
            param.requires_grad = False

    if LAYER4_UNFREEZE:
        for param in model.backbone[7].parameters():
            param.requires_grad = True
    
    loss_func = YoloLoss(S=7, B=2, C=20)

    opt_sgd = SGD(
        [
            {"params": model.head.parameters(), "lr": 1e-5},
            {"params": model.neck.parameters(), "lr": 1e-5},
            {"params": model.backbone[7].parameters(), "lr": 1e-4}
        ],
        weight_decay=5e-4,
        momentum=0.9
    )

    lr_scheduler = ReduceLROnPlateau(
        opt_sgd, mode="min", factor=0.1,
        patience=4, threshold=1e-4,
        threshold_mode="rel"
    )

    save_controller = SaveController(threshold=1e-4)
    
    train_stoper = TrainStoper(mode="min", patience=10, threshold=1e-4, threshold_mode="rel")

    # ===== запуск цикла обучения =====
    print("start")
    train(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model, loss_func=loss_func,
        opt=opt_sgd, lr_scheduler=lr_scheduler,
        save_controller=save_controller,
        train_stoper=train_stoper,
        EPOCHS=50
    )
