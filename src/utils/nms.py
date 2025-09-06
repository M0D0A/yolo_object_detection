import torch
from .iou import boxes_iou 


def nms(boxes: torch.Tensor, scores: torch.Tensor, threshold: float=0.45):
    _, sorted_scores_idx = scores.sort(descending=True)

    keep = []
    while sorted_scores_idx.numel() > 0:
        if sorted_scores_idx.numel() == 1:
            keep.append(sorted_scores_idx)
            break

        idx = sorted_scores_idx[0]
        keep.append(idx)

        boxes1 = boxes[sorted_scores_idx[0]].unsqueeze(dim=0)
        boxes2 = boxes[sorted_scores_idx[1:]]
        iou = boxes_iou(boxes1, boxes2)

        i = (iou < threshold).nonzero()[:, 1]
        if i.numel() == 0:
            break

        sorted_scores_idx = sorted_scores_idx[i+1]

    return torch.tensor(keep, dtype=torch.int)


def non_max_suppression(
        pred: torch.Tensor, score_threshold: float=0.25,
        iou_threshold: float=0.45, agnostic:bool=False,
        max_wh: int=7600, classes: list=None
    ):
    # bs - batch_size
    # nb - num_boxes

    if classes is not None:
        classes = torch.tensor(classes, device=pred.device)

    bs = pred.shape[0]
    candidates = pred[..., 4] > score_threshold

    output = [torch.zeros((0,6), device=pred.device)] * bs

    for idx, pr in enumerate(pred):
        pr = pr[candidates[idx]]

        if not pr.shape[0]:
            continue

        pr[:, 5:]*=pr[:, 4:5]

        boxes_pr = pr[:, :4]
        cls_pr = pr[:, 5:]

        score, idx_cls = cls_pr.max(dim=-1, keepdim=True)

        pr = torch.cat((boxes_pr, score, idx_cls.float()), dim=-1)[score.view(-1) > score_threshold]

        if classes is not None:
            i = (pr[:, 5:6] == classes).any(dim=-1)
            pr=pr[i]
            if not pr.shape[0]:
                continue

        scores = pr[:, 4]
        cls_scaling = pr[:, 5:6] * (0 if agnostic else max_wh)
        boxes = pr[:, :4] + cls_scaling

        idx_nms = nms(boxes, scores, iou_threshold)

        output[idx] = pr[idx_nms]

    return output
