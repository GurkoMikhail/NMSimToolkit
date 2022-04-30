import numpy as np
from volumes import TransformableVolumeWithChild
from geometries import Box
from materials import Material
from hepunits import*


class GammaCamera(TransformableVolumeWithChild):


    def __init__(self, collimator, detector, shieldingThickness=1*cm, name=None):
        maxSize = np.where(collimator.size > detector.size, collimator.size, detector.size)
        detectorBoxSize = maxSize[...]
        detectorBoxSize[2] = collimator.size[2] + detector.size[2]
        super().__init__(
            geometry=Box(detectorBoxSize[0] + 2*shieldingThickness, detectorBoxSize[1] + 2*shieldingThickness, detectorBoxSize[2] + shieldingThickness),
            material=Material.load('Pb'),
            name=name
        )
        detectorBox = TransformableVolumeWithChild(
            geometry=Box(*detectorBoxSize),
            material=Material.load('Air, Dry (near sea level)'),
            name='DetectorBox'
        )
        detectorBox.translate(z=shieldingThickness)
        detectorBox.setParent(self)
        collimator.translate(z=(detectorBoxSize[2] - collimator.size[2])/2)
        detectorBox.addChild(collimator)
        detector.translate(z=-(detectorBoxSize[2] - detector.size[2])/2)
        detectorBox.addChild(detector)

    @property
    def detectorBox(self):
        return self.childs[0]

    @property
    def collimator(self):
        return self.childs[0].childs[0]
        
    @property
    def detector(self):
        return self.childs[0].childs[1]

