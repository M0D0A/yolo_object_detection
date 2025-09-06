import torch
import torch.nn as nn
from ..utils.conver_box import cell_xywh2xyxy
from ..utils.iou import bnboxes_iou


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_cord = 5
        self.lambda_noobj = 0.5


    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred (batch, 7, 7, 30)
        # target (batch, 7, 7, 30)

        device = pred.device
        batch_size = pred.shape[0]

        # === pred ===
        pred_coord_conf = pred[..., :5*self.B].reshape(batch_size, self.S, self.S, self.B, -1)   # (batch, 7, 7, 2, 5)
        pred_xywh = pred_coord_conf[..., :4]   # (batch, 7, 7, 2, 4)
        bnboxes_xy = pred_xywh[..., :2]
        bnboxes_wh = pred_xywh[..., 2:]
        bnboxes_xywh = torch.cat([bnboxes_xy, bnboxes_wh], dim=-1)

        pred_cls = pred[..., 5*self.B:]   # (batch, 7, 7, 20)

        # === target ===
        mask_obj = torch.any(target>0, dim=-1) # (batch, 7, 7)

        target_coord_conf = target[..., :5*self.B].reshape(batch_size, self.S, self.S, self.B, -1)   # (batch, 7, 7, 2, 5)
        target_xywh = target_coord_conf[..., :4]   # (batch, 7, 7, 2, 4)
        target_cls = target[..., 5*self.B:]   # (batch, 7, 7, 20)

        pred_xyxy = cell_xywh2xyxy(bnboxes_xywh)
        target_xyxy = cell_xywh2xyxy(target_xywh)

        iou = bnboxes_iou(pred_xyxy, target_xyxy) # (batch, 7, 7, 2)
        max_idx = iou.argmax(dim=-1, keepdim=True) # Максимальный из двух коробок в каждом векторе (batch, 7, 7, 1)

        box_idx = torch.zeros(iou.shape, device=device).scatter(-1, max_idx, 1.0)

        obj_box_idx = (mask_obj[..., None] * box_idx).bool() # (batch, 7, 7, 2)
        noobj_box_idx = ~obj_box_idx # (batch, 7, 7, 2)

        # === coord loss ===
        obj_pred_xywh = pred_xywh[obj_box_idx]
        obj_target_xywh = target_xywh[obj_box_idx]

        pred_x = obj_pred_xywh[..., 0]
        target_x = obj_target_xywh[..., 0]

        pred_y = obj_pred_xywh[..., 1]
        target_y = obj_target_xywh[..., 1]

        pred_w = obj_pred_xywh[..., 2]
        target_w = obj_target_xywh[..., 2].sqrt()

        pred_h = obj_pred_xywh[..., 3]
        target_h = obj_target_xywh[..., 3].sqrt()

        x_loss = (pred_x - target_x).square().sum()
        y_loss = (pred_y - target_y).square().sum()
        w_loss = (pred_w - target_w).square().sum()
        h_loss = (pred_h - target_h).square().sum()

        xywh_loss = x_loss + y_loss + w_loss + h_loss

        # === conf loss === сравниваем не бинарно а по iou предсказанной с истинной если таковые имеются
        pred_conf = pred_coord_conf[..., 4][obj_box_idx]
        target_conf = iou[obj_box_idx]

        conf_loss = (pred_conf - target_conf).square().sum()

        # === cls loss ===
        obj_pred_cls = pred_cls[mask_obj]
        obj_target_cls = target_cls[mask_obj]

        cls_loss = (obj_pred_cls - obj_target_cls).square().sum()

        # === noobj loss ===
        noobj_pred_box = pred_coord_conf[..., 4][noobj_box_idx]

        noobj_loss = noobj_pred_box.square().sum()

        loss = (self.lambda_cord * xywh_loss + conf_loss + cls_loss + self.lambda_noobj * noobj_loss) / batch_size

        loss_item = loss.item()
        xywh_loss = (xywh_loss / batch_size).item()
        conf_loss = (conf_loss / batch_size).item()
        cls_loss = (cls_loss / batch_size).item()
        noobj_loss = (noobj_loss / batch_size).item()
        
        return loss, loss_item, xywh_loss, conf_loss, cls_loss, noobj_loss
