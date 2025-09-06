import torch
from .helpers import grid 

def xyxy2xywh(inp_boxes):
    out_boxes = torch.empty_like(inp_boxes)
    out_boxes[..., :2] = (inp_boxes[..., :2] + inp_boxes[..., 2:]) / 2
    out_boxes[..., 2:] = inp_boxes[..., 2:] - inp_boxes[..., :2]

    return out_boxes


def xywh2xyxy(inp_boxes):
    out_boxes = torch.empty_like(inp_boxes)
    out_boxes[..., :2] = inp_boxes[..., :2] - (inp_boxes[..., 2:] / 2)
    out_boxes[..., 2:] = inp_boxes[..., :2] + (inp_boxes[..., 2:] / 2)

    return out_boxes


def cell_xywh2xyxy(inp_boxes):
    batch_size = inp_boxes.shape[0]
    grid_size = (inp_boxes.shape[1], inp_boxes.shape[2])

    out_boxes = torch.empty_like(inp_boxes, dtype=inp_boxes.dtype, device=inp_boxes.device)

    xy = inp_boxes[..., :2] / 7 + grid(grid_size, dtype=inp_boxes.dtype, device=inp_boxes.device)
    wh = inp_boxes[..., 2:] / 2

    out_boxes[..., :2] = xy - (inp_boxes[..., 2:] / 2)
    out_boxes[..., 2:] = xy + (inp_boxes[..., 2:] / 2)

    return out_boxes