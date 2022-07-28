import numpy as np
from core.geometry.volumes import TransformableVolumeWithChild, TransformableVolume
from core.geometry.geometries import Box
import settings.database_setting as database_setting
from hepunits import*


class GammaCamera(TransformableVolumeWithChild):


    def __init__(self, collimator, detector, shielding_thickness=2*cm, glass_backend_thickness=5*cm, name=None):
        detector_box_size = np.where(collimator.size > detector.size, collimator.size, detector.size)
        detector_box_size[2] = collimator.size[2] + detector.size[2] + glass_backend_thickness
        material_database = database_setting.material_database
        detector_box = TransformableVolumeWithChild(
            geometry=Box(*detector_box_size),
            material=material_database['Air, Dry (near sea level)'],
            name='Detector_box'
        )
        glass_backend_size = detector_box_size.copy()
        glass_backend_size[2] = glass_backend_thickness
        glass_backend = TransformableVolume(
            geometry=Box(*glass_backend_size),
            material=material_database['Glass, Borosilicate (Pyrex)'],
            name='Glass_backend'
        )
        super().__init__(
            geometry=Box(detector_box_size[0] + 2*shielding_thickness, detector_box_size[1] + 2*shielding_thickness, detector_box_size[2] + shielding_thickness),
            material=material_database['Pb'],
            name=name
        )
        detector_box.translate(z=shielding_thickness)
        detector_box.set_parent(self)
        collimator.translate(z=(detector_box_size[2]/2 - collimator.size[2]/2))
        detector_box.add_child(collimator)
        detector.translate(z=(detector_box_size[2]/2 - collimator.size[2] - detector.size[2]/2))
        detector_box.add_child(detector)
        glass_backend.translate(z=(detector_box_size[2]/2 - collimator.size[2] - detector.size[2] - glass_backend.size[2]/2))
        detector_box.add_child(glass_backend)

    @property
    def detector_box(self):
        return self.childs[0]

    @property
    def collimator(self):
        return self.detector_box.childs[0]

    @property
    def detector(self):
        return self.detector_box.childs[1]

