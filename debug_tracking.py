import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
import hepunits as units
import numpy as np

# A quick manual trace logic check without processes compiling
from core.geometry.geometries import Box
from core.geometry.volumes import TransformableVolume, VolumeWithChilds
from core.geometry.gamma_cameras import GammaCamera
from core.geometry.geometry_compiler import GeometryCompiler
from core.geometry.parametric_collimators import ParametricParallelCollimator
import settings.database_setting as database_setting
from core.other.vectors_soa import Vector3DSoA
from core.geometry.navigation_state import NavigationState
import core.geometry.geometry_kernels as g_kernels

def run_debug():
    material_database = database_setting.material_database

    detector = TransformableVolume(
        geometry=Box(10*units.cm, 10*units.cm, 1*units.cm),
        material=material_database['Sodium Iodide'],
        name='Detector'
    )

    collimator = ParametricParallelCollimator(
        size=[10*units.cm, 10*units.cm, 4*units.cm],
        hole_diameter=0.4*units.cm,
        septa=0.05*units.cm,
        material=material_database['Pb'],
        name='Collimator'
    )

    camera = GammaCamera(collimator, detector, gap=1*units.mm, shielding_thickness=2*units.cm)
    root_volume = VolumeWithChilds(geometry=Box(100*units.cm, 100*units.cm, 100*units.cm), material=material_database['Air, Dry (near sea level)'], name="World")
    root_volume.add_child(camera)

    compiler = GeometryCompiler()
    geometry_buffer = compiler.compile_scene(root_volume)

    pos = Vector3DSoA.allocate(1, dtype=np.float64)
    dir_vec = Vector3DSoA.allocate(1, dtype=np.float64)

    pos.x[0] = 0.0
    pos.y[0] = 0.0
    pos.z[0] = -19.5000000000001

    dir_vec.x[0] = 0.0
    dir_vec.y[0] = 0.0
    dir_vec.z[0] = 1.0

    nav_state = NavigationState.allocate(1)
    nav_state.current_volume[0] = -1
    nav_state.next_volume[0] = -1
    nav_state.boundary_distance[0] = 0.0

    active_indices = np.array([0], dtype=np.int64)

    print("Testing Z=-19.5000000000001")
    g_kernels.cast_path_kernel(pos, dir_vec, active_indices, geometry_buffer, nav_state)

    print("Result:", nav_state.current_volume[0], nav_state.next_volume[0], nav_state.boundary_distance[0])

if __name__ == '__main__':
    run_debug()
