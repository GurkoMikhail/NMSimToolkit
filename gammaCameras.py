import numpy as np
from volumes import TransformableVolumeWithChild, TransformableVolume
from geometries import Box
from materials import Material
from hepunits import*


class GammaCamera(TransformableVolumeWithChild):


    def __init__(self, collimator, detector, shieldingThickness=2*cm, glassBackendThickness=5*cm, name=None):
        detectorBoxSize = np.where(collimator.size > detector.size, collimator.size, detector.size)
        detectorBoxSize[2] = collimator.size[2] + detector.size[2] + glassBackendThickness
        detectorBox = TransformableVolumeWithChild(
            geometry=Box(*detectorBoxSize),
            material=Material.load('Air, Dry (near sea level)'),
            name='DetectorBox'
        )
        glassBackendSize = detectorBoxSize.copy()
        glassBackendSize[2] = glassBackendThickness
        glassBackend = TransformableVolume(
            geometry=Box(*glassBackendSize),
            material=Material.load('Glass, Borosilicate (Pyrex)'),
            name='GlassBackend'
        )
        super().__init__(
            geometry=Box(detectorBoxSize[0] + 2*shieldingThickness, detectorBoxSize[1] + 2*shieldingThickness, detectorBoxSize[2] + shieldingThickness),
            material=Material.load('Pb'),
            name=name
        )
        detectorBox.translate(z=shieldingThickness)
        detectorBox.setParent(self)
        collimator.translate(z=(detectorBoxSize[2]/2 - collimator.size[2]/2))
        detectorBox.addChild(collimator)
        detector.translate(z=(detectorBoxSize[2]/2 - collimator.size[2] - detector.size[2]/2))
        detectorBox.addChild(detector)
        glassBackend.translate(z=(detectorBoxSize[2]/2 - collimator.size[2] - detector.size[2] - glassBackend.size[2]/2))
        detectorBox.addChild(glassBackend)

    @property
    def detectorBox(self):
        return self.childs[0]

    @property
    def collimator(self):
        return self.detectorBox.childs[0]

    @property
    def detector(self):
        return self.detectorBox.childs[1]

