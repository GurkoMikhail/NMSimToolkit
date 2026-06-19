import numpy as np
from core.other.typing_definitions import Float

Matrix3x3DType = np.dtype([
    ('m00', Float), ('m01', Float), ('m02', Float),
    ('m10', Float), ('m11', Float), ('m12', Float),
    ('m20', Float), ('m21', Float), ('m22', Float)
])

Vector3DDType = np.dtype([
    ('x', Float),
    ('y', Float),
    ('z', Float)
])

TransformDType = np.dtype([
    ('rotation', Matrix3x3DType),
    ('translation', Vector3DDType)
])
