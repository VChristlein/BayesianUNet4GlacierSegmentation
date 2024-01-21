import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from scipy.spatial.distance import cdist

def dice_coefficient(u,v):
    """
    For binary vectors the Dice cooefficient can be written as
    2 * |u * v| / (|u**2| + |v**2|)

    | u * v | gives intersecting set
    |u**2|, |v**2| number of (true) elements in set

    :param u:  binary vector
    :param v:  binary vector of same length as u
    :return:   dice coefficient
    """
    c_uv = np.sum(u*v)
    if c_uv == 0:
        return 0
    else:
        c_u = np.sum(u**2)
        c_v = np.sum(v**2)
    return 2 * c_uv / (c_u + c_v)

def dice_coefficient_cutoff(gt, pred, cutoff):
    """
    Binarizes pred using cutoff as threshold
    and computes dice coefficent for boolean vectors
    Used for threshold optimization

    :param gt:  binary vector
    :param pred:  vector
    :param cutoff: cutoff threshold in same range as pred
    :return dice coefficient
    """
    pred_bin = (pred >= cutoff).astype(int)
    return dice_coefficient(gt, pred_bin)

def IOU(y_true, y_pred):
    """
        Returns Intersection over Union (IOU)
    :param y_true:  binary vector of ground truth
    :param y_pred:  binary vector prediction, same length as u
    :return:   IOU
    """

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true.astype(np.bool) | y_pred.astype(np.bool))
    return intersection / union


def specificity(y_true, y_pred):
    """
        Returns specificity

    :param y_true:  binary vector of ground truth
    :param y_pred:  binary vector of prediction, same length as u
    :return:   specificity value
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = np.sum(neg_y_true * y_pred)
    tn = np.sum(neg_y_true * neg_y_pred)
    result = tn / (tn + fp + K.epsilon())
    return result

def euclidian_tf(y_true, y_pred):
    """
    :param y_true:  binary vector of ground truth
    :param y_pred:  binary vector of prediction, same length as u
    :return:   euclidian distance between y_true and y_pred
    """
    return tf.reduce_sum((y_true - y_pred)**2)

def line_graph(y_true, y_pred):
    gt_line = np.where(y_true)
    pred_line = np.where(y_pred)
    gt_y_start = np.argmin(gt_line[0])
    gt_start = (gt_y_start, np.argmin(gt_line[1][gt_y_start]))
    gt_dist = cdist(gt_line, gt_line)


def line_accuracy(y_true, y_pred):
    intersection = np.sum(y_true == y_pred)
    union = np.sum(y_true)
    if union == 0:
        if intersection == 0:
            return 1
        else:
            return 0
    else:
        return intersection / union
