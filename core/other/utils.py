import numpy as np
from numpy import cos, sin
from datetime import datetime
# from pyqtgraph import makeARGB


def compute_translation_matrix(translation):
    dx, dy, dz = translation
    translation_matrix = np.array([
        [1., 0., 0., dx],
        [0., 1., 0., dy],
        [0., 0., 1., dz],
        [0., 0., 0., 1.]
        ])
    return translation_matrix


def compute_rotation_matrix(angles):
    alpha, beta, gamma = angles
    rotation_matrix = np.array([
        [cos(alpha)*cos(beta),  cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma),    cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma),    0.],
        [sin(alpha)*cos(beta),  sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma),    sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma),    0.],
        [-sin(beta),            cos(beta)*sin(gamma),                                       cos(beta)*cos(gamma),                                       0.],
        [0.,                    0.,                                                         0.,                                                         1.]
        ])
    return rotation_matrix


def unique_with_indices(array):
    return [(uniqueEl, np.array([i for i, element in enumerate(array) if element is uniqueEl])) for uniqueEl in set(array)]


def datetime_from_seconds(seconds):
    zerodatetime = datetime.fromtimestamp(0)
    nowdatetime = datetime.fromtimestamp(seconds)
    return nowdatetime - zerodatetime

def make3DRGBA(array3D, lut=None, levels=None):
    from pyqtgraph import makeARGB
    levels = [np.nanmin(array3D), np.nanmax(array3D)] if levels is None else levels
    arrayRGBA = np.ndarray((*(array3D.shape), 4), dtype=np.ubyte)
    for i, array2D in enumerate(array3D):
        arrayRGBA[i] = makeARGB(array2D, lut=lut, levels=levels, useRGBA=True)[0]
    return arrayRGBA

