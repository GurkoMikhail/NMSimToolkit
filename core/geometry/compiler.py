import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Tuple

from core.geometry.geometries import Box
from core.geometry.geometry_buffer import GeometryBuffer, TransformSoA
from core.geometry.volumes import ElementaryVolume, TransformableVolume, VolumeWithChilds
from core.materials.materials import Material
from core.other.typing_definitions import Float, Index
from core.other.vectors_soa import Vector3DSoA


class SceneCompiler:
    """
    Compiler that transforms an Object-Oriented Scene Graph (volumes.py)
    into a Structure of Arrays (SoA) flat GeometryBuffer optimized for
    Numba Frustum Culling and raycasting.
    """

    SHAPE_TYPE_BOX = 0
    # Additional primitives can be mapped here: SHAPE_TYPE_CYLINDER = 1, etc.

    def __init__(self):
        self._volumes: List[ElementaryVolume] = []
        # Maps an ElementaryVolume instance to its assigned integer index
        self._volume_to_id: Dict[ElementaryVolume, int] = {}

        # Maps an Material instance to its assigned integer ID
        self._materials: List[Material] = []
        self._material_to_id: Dict[Material, int] = {}

        self._miss_indices: List[int] = []

    def compile(self, root: ElementaryVolume) -> GeometryBuffer:
        """
        Walks the tree starting from the root using Depth-First Search.
        Populates flat arrays and calculates forward-jump 'miss_index'
        for Frustum Culling.
        """
        # Clear state
        self._volumes.clear()
        self._volume_to_id.clear()
        self._materials.clear()
        self._material_to_id.clear()
        self._miss_indices.clear()

        # 1. Depth-First Traversal
        # Start with the identity matrix as the base World -> Local transform
        base_transform = np.eye(4, dtype=Float)
        self._dfs_traverse(root, base_transform)

        N = len(self._volumes)

        # 2. Allocate Arrays
        volume_ids = np.zeros(N, dtype=Index)
        shape_types = np.zeros(N, dtype=np.int32)

        size_x = np.zeros(N, dtype=Float)
        size_y = np.zeros(N, dtype=Float)
        size_z = np.zeros(N, dtype=Float)

        r00 = np.zeros(N, dtype=Float)
        r01 = np.zeros(N, dtype=Float)
        r02 = np.zeros(N, dtype=Float)
        r10 = np.zeros(N, dtype=Float)
        r11 = np.zeros(N, dtype=Float)
        r12 = np.zeros(N, dtype=Float)
        r20 = np.zeros(N, dtype=Float)
        r21 = np.zeros(N, dtype=Float)
        r22 = np.zeros(N, dtype=Float)

        tx = np.zeros(N, dtype=Float)
        ty = np.zeros(N, dtype=Float)
        tz = np.zeros(N, dtype=Float)

        material_ids = np.zeros(N, dtype=Index)

        # 3. Populate Arrays
        for idx, volume in enumerate(self._volumes):
            volume_ids[idx] = idx

            # Extract material mapping
            if volume.material not in self._material_to_id:
                mat_id = len(self._materials)
                self._materials.append(volume.material)
                self._material_to_id[volume.material] = mat_id
            material_ids[idx] = self._material_to_id[volume.material]

            # Extract geometry properties
            geom = volume.geometry
            if isinstance(geom, Box):
                shape_types[idx] = self.SHAPE_TYPE_BOX
                # For Box, we typically store half_size as it's the core parameter for collisions
                hs = geom.half_size
                size_x[idx] = hs[0]
                size_y[idx] = hs[1]
                size_z[idx] = hs[2]
            else:
                raise NotImplementedError(f"Compilation of shape {type(geom)} is not implemented yet.")

            # Extract Transformation Matrix (World -> Local)
            matrix = volume._accumulated_transform

            # Extract Rotation (3x3)
            # The matrix in `volumes.py` follows convention where `.T` is applied during multiplication
            # Specifically: `np.matmul(local_position, transformation_matrix.T.astype(position.dtype), out=local_position)`
            # So the matrix stores:
            # [R00, R10, R20, TX]
            # [R01, R11, R21, TY]
            # [R02, R12, R22, TZ]
            # [ 0 ,  0 ,  0 ,  1]
            # Let's verify by just using the matrix as is, and we will apply M @ P in our Numba Kernel
            # However, looking closely at `compute_translation_matrix`:
            # [1, 0, 0, dx] -> M[0,3] = dx.
            # Local pos logic: local_position @ M.T
            # For P = [x,y,z,1], P @ M.T =>
            #   x*M[0,0] + y*M[0,1] + z*M[0,2] + 1*M[0,3] => x*1 + dx.
            # So the components map logically. M is truly World->Local.

            # Rotation Elements
            r00[idx], r01[idx], r02[idx] = matrix[0, 0], matrix[0, 1], matrix[0, 2]
            r10[idx], r11[idx], r12[idx] = matrix[1, 0], matrix[1, 1], matrix[1, 2]
            r20[idx], r21[idx], r22[idx] = matrix[2, 0], matrix[2, 1], matrix[2, 2]

            # Translation Elements
            tx[idx] = matrix[0, 3]
            ty[idx] = matrix[1, 3]
            tz[idx] = matrix[2, 3]

        # 4. Miss Indices
        miss_indices = np.array(self._miss_indices, dtype=Index)

        # 5. Build and validate
        buffer = GeometryBuffer(
            volume_id=volume_ids,
            shape_type=shape_types,
            sizes=Vector3DSoA(size_x, size_y, size_z),
            transform=TransformSoA(
                r00, r01, r02, r10, r11, r12, r20, r21, r22,
                tx, ty, tz
            ),
            material_id=material_ids,
            miss_index=miss_indices
        )

        buffer.validate()
        return buffer

    def _dfs_traverse(self, volume: ElementaryVolume, current_transform: NDArray[Float]) -> None:
        """
        Recursively traverse the graph and assign indices.
        Accumulates the World -> Local transform for each node.
        Returns the assigned index of the current volume.
        """
        idx = len(self._volumes)
        self._volumes.append(volume)
        self._volume_to_id[volume] = idx

        # Accumulate transformation. Transformable volumes apply their local matrix.
        if isinstance(volume, TransformableVolume):
            # In volumes.py: total_transform = parent.total_transform @ self.transform
            # Because it is World -> Local, the chain applies recursively.
            accumulated_transform = current_transform @ volume.transformation_matrix
        else:
            accumulated_transform = current_transform

        # Temporarily store the accumulated transform on the volume object to extract it in the loop
        volume._accumulated_transform = accumulated_transform

        # Place holder for miss_index. We will fill it after processing all children.
        self._miss_indices.append(-1)

        if isinstance(volume, VolumeWithChilds):
            for child in volume.childs:
                self._dfs_traverse(child, accumulated_transform)

        # The next element added to the list will be the element AFTER this entire subtree.
        # So if we miss this volume, we jump to the index equal to the CURRENT length of self._volumes.
        # Example: parent (idx=0). Has children 1, 2. Length becomes 3. If we miss parent 0, jump to 3.
        self._miss_indices[idx] = len(self._volumes)
