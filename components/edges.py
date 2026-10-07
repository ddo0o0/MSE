# Edge-generation helpers adapted from BASNet (CVPR 2019), https://github.com/NathanUA/BASNet -- MIT.
from scipy.ndimage import distance_transform_edt
import numpy as np

def onehot_to_multiclass_edges(mask, radius, num_classes):
    if radius < 0:
        return mask


    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):

    if radius < 0:
        return mask


    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])
    for i in range(num_classes):

        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = (edgemap > 0).astype(np.uint8)*255
    return edgemap


def mask_to_onehot(mask, num_classes):
    _mask = [mask == (i) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)
