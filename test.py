import numpy as np
import core.physics.g4compton as g4compton
from hepunits import*

rng = np.random.default_rng(4815162342)
generator = g4compton.initialize(rng)
# rng.random(3)

for i in range(10):
    theta = generator(140.5*keV, 8)
    print(theta)


