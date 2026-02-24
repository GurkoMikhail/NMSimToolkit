from datetime import datetime, timedelta
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from core.other.typing_definitions import Float


def compute_translation_matrix(translation: Union[NDArray[Float], Sequence[Float]]) -> NDArray[Float]:
    dx, dy, dz = translation
    translation_matrix = np.array([
        [1., 0., 0., dx],
        [0., 1., 0., dy],
        [0., 0., 1., dz],
        [0., 0., 0., 1.]
        ])
    return translation_matrix


def compute_rotation_matrix(angles: Union[NDArray[Float], Sequence[Float]]) -> NDArray[Float]:
    alpha, beta, gamma = angles
    rotation_matrix = np.array([
        [np.cos(alpha)*np.cos(beta),  np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma),    np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma),    0.],
        [np.sin(alpha)*np.cos(beta),  np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma),    np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma),    0.],
        [-np.sin(beta),               np.cos(beta)*np.sin(gamma),                                          np.cos(beta)*np.cos(gamma),                                          0.],
        [0.,                          0.,                                                                  0.,                                                                  1.]
        ])
    return rotation_matrix


def unique_with_indices(array: Sequence[Any]) -> List[Tuple[Any, NDArray[np.int64]]]:
    return [(uniqueEl, np.array([i for i, element in enumerate(array) if element is uniqueEl])) for uniqueEl in set(array)]


def datetime_from_seconds(seconds: Float) -> timedelta:
    zerodatetime = datetime.fromtimestamp(0)
    nowdatetime = datetime.fromtimestamp(seconds)
    return nowdatetime - zerodatetime

def make3DRGBA(array3D: NDArray[np.generic], lut: Optional[Any] = None, levels: Optional[Sequence[Float]] = None) -> NDArray[np.ubyte]:
    from pyqtgraph import makeARGB
    levels = [np.nanmin(array3D), np.nanmax(array3D)] if levels is None else levels
    arrayRGBA = np.ndarray((*(array3D.shape), 4), dtype=np.ubyte)
    for i, array2D in enumerate(array3D):
        arrayRGBA[i] = makeARGB(array2D, lut=lut, levels=levels, useRGBA=True)[0]
    return arrayRGBA

