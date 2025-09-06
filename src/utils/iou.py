import torch


def bnboxes_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)

    xy_left = torch.max(boxes1[..., :2], boxes2[..., :2])
    xy_right = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh = (xy_right - xy_left).clamp(min=0)

    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter + 1e-6 # 1e-6 предотвращает деление на 0

    return inter / union


def boxes_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    xy_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    xy_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (xy_right - xy_left).clamp(min=0)

    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter + 1e-6

    return inter / union
