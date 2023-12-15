import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelBinarizer


def compute_iou(pred, mask, num_classes, average="macro") -> float:
    """
    A function that computes the Intersection over Union (IoU)
    for the pixels with a given label between the prediction and the mask
    """
    assert pred.shape == mask.shape

    lb = LabelBinarizer()
    lb.fit(range(num_classes))
    binarized_mask = lb.transform(mask.ravel())
    binarized_pred = lb.transform(pred.ravel())
    iou = jaccard_score(binarized_mask, binarized_pred, labels=range(num_classes), average=average)

    return iou
