import numpy as np
import torch



def features2result(features, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        features (Tensor): shape (n, num_classes)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    print(features.shape)
    print(labels.shape)
    print(labels)
    print(num_classes)
    if features.shape[0] == 0:
        return [np.zeros((0, num_classes), dtype=np.float32) for i in range(num_classes)]
    else:
        features = features.cpu().numpy()
        labels = labels.cpu().numpy()
        return [features[labels == i, :] for i in range(num_classes)]