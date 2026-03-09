import numpy as np
from core.other.typing_definitions import Float, ShapeID

ShapeDataDType = np.dtype([
    ('shape', ShapeID),
    ('param_0', Float),
    ('param_1', Float),
    ('param_2', Float),
    ('param_3', Float)
])
