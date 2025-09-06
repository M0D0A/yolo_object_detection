import torch
from tqdm import tqdm
import numpy as np
from ..utils.nms import non_max_suppression
from ..utils.helpers import clip_boxes, process_batch
from ..utils.save_controller import save_model
from ..metrics import mean_average_precision

def train(
        device,
        train_loader, val_loader,
        model, loss_func,
        opt, lr_scheduler,
        save_controller,
        train_stoper,
        EPOCHS
    ):

    train_loss, val_loss, metrics, lr_list = [], [], [], []

    for epoch in range(EPOCHS):
        # ===== train =====
        run_train_loss_list = []
        mean_train_loss_list = None 
        
        train_loop = tqdm(train_loader, leave=False)
        model.train()
        for sample, targets, _ in train_loop:
            preds = model(sample.to(device))

            loss, *loss_list = loss_func(preds, targets.to(device))

            opt.zero_grad()
            loss.backward()
            opt.step()

            run_train_loss_list.append(loss_list)
            mean_train_loss_list = np.array(run_train_loss_list).mean(axis=0)

            train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]; train_loss={mean_train_loss_list[0]:.4f}")

        train_loss.append(mean_train_loss_list)

        # ===== val =====
        run_val_loss_list = []
        mean_val_loss_list = None
        stats = []

        val_loop = tqdm(val_loader, leave=False)
        model.eval()
        with torch.no_grad():
            iou_vector = torch.linspace(0.5, 0.95, 10, device=device)
            num_iou = iou_vector.numel()

            for sample, targets, bnboxes_classes in val_loop:
                preds, bnboxes = model(sample.to(device))

                loss, *loss_list = loss_func(preds, targets.to(device))

                run_val_loss_list.append(loss_list)
                mean_val_loss_list = np.array(run_val_loss_list).mean(axis=0)

                nms_preds = non_max_suppression(bnboxes, score_threshold=0.25, iou_threshold=0.45)

                # преобразование данных для расчёта метрики
                for idx, pred in enumerate(nms_preds):
                    true_lables = bnboxes_classes[idx].to(device)

                    num_true_lables = true_lables.shape[0]
                    num_pred = pred.shape[0]
                    correct = torch.zeros(size=(num_pred, num_iou), dtype=torch.bool, device=device)

                    if num_pred == 0:
                        if num_true_lables:
                            stats.append((
                                correct,
                                *torch.zeros((2,0), device=device),
                                true_lables[:,  4]
                            ))
                        continue

                    pred_clone = pred.clone()

                    if num_true_lables:
                        true_bnboxes = true_lables[:, :4]
                        clip_boxes(true_bnboxes)
                        nslables = torch.cat([true_lables[:, 4:], true_bnboxes], dim=1)
                        correct = process_batch(pred_clone, nslables, iou_vector)

                    stats.append((correct, pred[:, 4], pred[:, 5], true_lables[:, 4]))
                
                val_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]; val_loss={mean_val_loss_list[0]:.4f}")

        val_loss.append(mean_val_loss_list)

        print(f"Epoch [{epoch+1}/{EPOCHS}]; train_loss={mean_train_loss_list[0]:.4f};\
                xywh_loss={mean_train_loss_list[1]:.4f}; conf_loss={mean_train_loss_list[2]:.4f};\
                cls_loss={mean_train_loss_list[3]:.4f}; noobj_loss={mean_train_loss_list[4]:.4f}\n\
                val_loss={mean_val_loss_list[0]:.4f}; xywh_loss={mean_val_loss_list[1]:.4f};\
                conf_loss={mean_val_loss_list[2]:.4f}; cls_loss={mean_val_loss_list[3]:.4f};\
                noobj_loss={mean_val_loss_list[4]:.4f}; lr={lr_scheduler.get_last_lr()}")

        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any(): # есть хоть какая-то статистика
            ap = mean_average_precision(*stats) # распаковываем статистику
            ap50, ap = ap[:, 0], ap.mean(1)
            map50, mAP = ap50.mean(), ap.mean()
            metrics.append((map50, mAP))
            print(f'Epoch [{epoch+1}/{EPOCHS}]: mAP|0.5 = {map50:.4f}, mAP|0.5:0.95 = {mAP:.4f}')
        else:
            print(f"Epoch [{epoch+1}/{EPOCHS}]: метрика не расчитывалась")
        
        if save_controller(mean_val_loss_list[0]):
            save_model(
                epoch=epoch+1, EPOCHS=EPOCHS,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=opt.state_dict(),
                lr_scheduler_state_dict=lr_scheduler.state_dict(),
                train_loss=train_loss, val_loss=val_loss,
                metrics=metrics, lr_list=lr_list,
                save_path=f"models/my_yolov1_{epoch+1}.plt"
            )
            print(f"Epoch [{epoch+1}/{EPOCHS}] === MODEL SAVE ===")

        lr_list.append(lr_scheduler.get_last_lr())
        lr_scheduler.step(mean_val_loss_list[0])
        if train_stoper(mean_val_loss_list[0]):
            print(f"Epoch [{epoch+1}/{EPOCHS}] === STOP LEARNING ===")
            break

    # save_model(
    #     epoch=epoch+1, EPOCHS=EPOCHS,
    #     model_state_dict=model.state_dict(),
    #     optimizer_state_dict=opt.state_dict(),
    #     lr_scheduler_state_dict=lr_scheduler.state_dict(),
    #     train_loss=train_loss, val_loss=val_loss,
    #     metrics=metrics, lr_list=lr_list,
    #     save_path="models/my_yolov1_final.plt"
    # )
    # print(f"Epoch [{epoch+1}/{EPOCHS}] === FINAL MODEL SAVE ===")