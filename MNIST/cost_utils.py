import numpy as np


def categorical_cross_entropy(A, Y):
    """
    Categorical cross-entropy loss
    """
    J = - np.mean(np.sum(Y * np.log(A + 1e-7) + (1 - Y) * np.log(1-A + 1e-7), axis=0)) # Sum over labels, but mean over examples
    J = np.maximum(J, 0)
    return J

def dZ_categorical_cross_entropy(A, Y):
    """
    The first error dA := dJ/dA is computed using categorical cross-entropy loss as J
    """
    dZ = A - Y # Derivative of categorical cross-entropy loss with respect to A
    return dZ

def relative_difference(Z1, Z2):
    diff = np.linalg.norm(Z1 - Z2)/np.linalg.norm(Z1 + Z2)
    return diff

def compute_accuracy(pred_classes, Y):
    """
    :Y: (classes_number, m) is the one hot encoding of ground truth
    :pred_classes: (1, m) is the predicted output of a model
    """
    true_classes = np.argmax(Y, axis=0)
    TP = np.sum(np.where(true_classes == pred_classes, 1, 0))
    FP = np.sum(np.where(true_classes != pred_classes, 1, 0))
    acc = TP / (TP + FP)
    return acc


    
