import numpy as np
from numpy import cos, sin
from pyqtgraph import makeARGB


def computeTranslationMatrix(translation):
    dx, dy, dz = translation
    translationMatrix = np.array([
        [1., 0., 0., dx],
        [0., 1., 0., dy],
        [0., 0., 1., dz],
        [0., 0., 0., 1.]
        ])
    return translationMatrix


def computeRotationMatrix(angles):
    alpha, beta, gamma = angles
    rotationMatrix = np.array([
        [cos(alpha)*cos(beta),  cos(alpha)*sin(beta)*sin(gamma) - sin(alpha)*cos(gamma),    cos(alpha)*sin(beta)*cos(gamma) + sin(alpha)*sin(gamma),    0.],
        [sin(alpha)*cos(beta),  sin(alpha)*sin(beta)*sin(gamma) + cos(alpha)*cos(gamma),    sin(alpha)*sin(beta)*cos(gamma) - cos(alpha)*sin(gamma),    0.],
        [-sin(beta),            cos(beta)*sin(gamma),                                       cos(beta)*cos(gamma),                                       0.],
        [0.,                    0.,                                                         0.,                                                         1.]
        ])
    return rotationMatrix


def uniqueWithIndices(array):
    return [(uniqueEl, np.array([i for i, element in enumerate(array) if element is uniqueEl])) for uniqueEl in set(array)]


def make3DRGBA(array3D, lut=None, levels=None):
    levels = [array3D.min(), array3D.max()] if levels is None else levels
    arrayOfRGBA = np.ndarray((*(array3D.shape), 4), dtype=np.ubyte)
    for i, array2D in enumerate(array3D):
        arrayOfRGBA[i] = makeARGB(array2D, lut=lut, levels=levels, useRGBA=True)[0]
    return arrayOfRGBA

