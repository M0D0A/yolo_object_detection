import numpy as np
from numpy.typing import NDArray


def compute_ap(recall: NDArray, precision: NDArray):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    mpre = mpre[::-1]
    mpre = np.maximum.accumulate(mpre)
    mpre = mpre[::-1]

    x = np.linspace(0,1,101)

    ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    
    return ap


def mean_average_precision(
        tp: NDArray, conf: NDArray,
        pred_cls: NDArray, target_cls: NDArray,
        eps: float=1e-16
    ):
    idx = np.argsort(conf)[::-1]

    tp  = tp[idx]
    conf = conf[idx]
    pred_cls = pred_cls[idx]

    un_cls, num_cls = np.unique(target_cls, return_counts=True)
    nc = un_cls.shape[0]

    ap = np.zeros((nc, tp.shape[1]))
    for ci, cls in enumerate(un_cls):
        i = pred_cls == cls
        num_gt_cls = num_cls[ci]
        num_pred_cls = i.sum()

        if num_pred_cls == 0 or num_gt_cls == 0:
            continue

        tpc = tp[i].cumsum(axis=0)
        fpc = (1 - tp[i]).cumsum(axis=0)

        precision = tpc / (tpc + fpc)
        recall = tpc / (num_gt_cls + eps)

        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])\
            
    return ap