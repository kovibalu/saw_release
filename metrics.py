# Functions directly related to computing precision and recall

import numpy as np


def grouped_confusion_matrix(y_true, y_pred):
    """
    Create "grouped" (3x2) confusion matrix from ground truth and predicted labels.
    The baselines predict only two types of labels (non-smooth/smooth shading),
    but we have 3 types of ground truth labels:
        (0) normal/depth discontinuity non-smooth shading (NS-ND)
        (1) shadow boundary non-smooth shading (NS-SB)
        (2) smooth shading (S)
    Ground truth labels 0, 1 are mapped to predicted label 0 (non-smooth shading).
    Ground truth label 2 is mapped to predicted label 1 (smooth shading).
    """
    # Sanity checks
    assert set(np.unique(y_true)).issubset(set([0, 1, 2]))
    assert set(np.unique(y_pred)).issubset(set([0, 1]))
    assert len(y_pred) == len(y_true)
    assert y_true.ndim == 1
    assert y_pred.ndim == 1

    conf_mx = np.zeros((3, 2), dtype=int)
    for gt_label in xrange(3):
        mask = y_true == gt_label
        for pred_label in xrange(2):
            conf_mx[gt_label, pred_label] = np.sum(y_pred[mask] == pred_label)

    return conf_mx


def get_pr_from_conf_mx(conf_mx, class_weights):
    """
    Compute precision and recall based on a special 3x2 confusion matrix with
    class reweighting.
    The input is not a proper confusion matrix, because the baselines predict
    only two types of labels (non-smooth/smooth shading), but we have 3 types
    of ground truth labels:
        (0) normal/depth discontinuity non-smooth shading (NS-ND)
        (1) shadow boundary non-smooth shading (NS-SB)
        (2) smooth shading (S)
    Ground truth labels 0, 1 are mapped to predicted label 0 (non-smooth shading).
    Ground truth label 2 is mapped to predicted label 1 (smooth shading).
    """
    assert not np.all(conf_mx == 0)
    assert conf_mx.ndim == 2
    assert conf_mx.shape[0] == 3
    assert conf_mx.shape[1] == 2

    # Rebalance confusion matrix rows
    if class_weights:
        assert len(class_weights) == 3
        label_counts = np.sum(conf_mx, axis=1)
        assert np.all(label_counts > 0)
        conf_mx = conf_mx.astype(float)
        conf_mx *= (np.array(class_weights, dtype=float) / label_counts)[:, np.newaxis]

    smooth_count_true = np.sum(conf_mx[2, :])
    smooth_count_pred = np.sum(conf_mx[:, 1])
    smooth_count_correct = float(conf_mx[2, 1])
    assert smooth_count_true != 0
    smooth_recall = smooth_count_correct / smooth_count_true
    if smooth_count_pred:
        smooth_prec = smooth_count_correct / smooth_count_pred
    else:
        smooth_prec = 1

    return smooth_prec, smooth_recall
