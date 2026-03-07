import numpy as np
from typing import List, Tuple, Dict
from core.geometry.volumes import ElementaryVolume, TransformableVolume, VolumeWithChilds
from core.geometry.geometries import Box
from core.materials.materials import Material
from core.other.typing_definitions import Float, Index
from core.geometry.geometry_buffer_soa import GeometryBuffer


class SceneCompiler:
    """
    Compiles an Object-Oriented tree of Volumes into a flat Structure of Arrays (SoA).
    Produces `GeometryBuffer` and a list of unique `Material` references.
    """

    def __init__(self) -> None:
        self.materials: List[Material] = []
        self._material_map: Dict[Material, int] = {}

        self.nodes: List[dict] = []

    def compile(self, root_volume: ElementaryVolume) -> Tuple[GeometryBuffer, List[Material]]:
        self.materials.clear()
        self._material_map.clear()
        self.nodes.clear()

        self._traverse(root_volume)

        # Build SoA
        n_nodes = len(self.nodes)
        node_type = np.empty(n_nodes, dtype=np.uint8)
        material_id = np.empty(n_nodes, dtype=np.uint32)
        miss_index = np.empty(n_nodes, dtype=Index)

        hx = np.empty(n_nodes, dtype=Float)
        hy = np.empty(n_nodes, dtype=Float)
        hz = np.empty(n_nodes, dtype=Float)

        r_xx = np.empty(n_nodes, dtype=Float)
        r_xy = np.empty(n_nodes, dtype=Float)
        r_xz = np.empty(n_nodes, dtype=Float)

        r_yx = np.empty(n_nodes, dtype=Float)
        r_yy = np.empty(n_nodes, dtype=Float)
        r_yz = np.empty(n_nodes, dtype=Float)

        r_zx = np.empty(n_nodes, dtype=Float)
        r_zy = np.empty(n_nodes, dtype=Float)
        r_zz = np.empty(n_nodes, dtype=Float)

        tx = np.empty(n_nodes, dtype=Float)
        ty = np.empty(n_nodes, dtype=Float)
        tz = np.empty(n_nodes, dtype=Float)

        for i, node in enumerate(self.nodes):
            node_type[i] = node['type']
            material_id[i] = node['material_id']
            miss_index[i] = node['miss_index']

            hx[i], hy[i], hz[i] = node['half_size']

            inv_mat = node['inverse_transformation']

            # The transformation matrices in volumes.py:
            # np.matmul(local_position, transformation_matrix.T.astype(position.dtype), out=local_position)
            # local_position[:, :3] = position
            # local_position[:, 3] = 1.0 (implied, or actually they construct [N, 4])
            # If the transformation is applied via P @ T.T, this means the matrix T is stored such that
            # translation is in the last column (T[:3, 3]), or last row?
            # volumes.py:
            # translation_matrix = np.array([
            #   [1, 0, 0, x],
            #   [0, 1, 0, y],
            #   [0, 0, 1, z],
            #   [0, 0, 0, 1]
            # ]) -> So translation is in the last column [0:3, 3].

            r_xx[i], r_xy[i], r_xz[i] = inv_mat[0, :3]
            r_yx[i], r_yy[i], r_yz[i] = inv_mat[1, :3]
            r_zx[i], r_zy[i], r_zz[i] = inv_mat[2, :3]

            tx[i], ty[i], tz[i] = inv_mat[:3, 3]

        buffer = GeometryBuffer(
            node_type=node_type,
            material_id=material_id,
            miss_index=miss_index,
            half_size_x=hx, half_size_y=hy, half_size_z=hz,
            R_xx=r_xx, R_xy=r_xy, R_xz=r_xz,
            R_yx=r_yx, R_yy=r_yy, R_yz=r_yz,
            R_zx=r_zx, R_zy=r_zy, R_zz=r_zz,
            T_x=tx, T_y=ty, T_z=tz
        )
        buffer._validate()
        return buffer, self.materials

    def _traverse(self, volume: ElementaryVolume) -> int:
        """
        DFS traversal. Returns the number of nodes processed (including children).
        """
        current_index = len(self.nodes)

        # Register Material
        mat = volume.material
        if mat not in self._material_map:
            self._material_map[mat] = len(self.materials)
            self.materials.append(mat)
        mat_id = self._material_map[mat]

        # Geometry type parsing (Currently assuming Box, will expand if needed)
        geom = volume.geometry
        if isinstance(geom, Box):
            n_type = 0
        else:
            raise NotImplementedError(f"Geometry type {type(geom)} not yet supported in SoA compiler")

        # Transformations
        if isinstance(volume, TransformableVolume):
            # Transformation is defined as P_local = P_world @ transformation_matrix.T
            # The matrix stored in OOP is already the World -> Local transformation matrix.
            inv_matrix = volume.total_transformation_matrix
        else:
            inv_matrix = np.eye(4, dtype=Float)

        node_data = {
            'type': n_type,
            'material_id': mat_id,
            'half_size': geom.half_size,
            'inverse_transformation': inv_matrix,
            'miss_index': -1 # Placeholder, will be updated if it has siblings/parents
        }

        self.nodes.append(node_data)

        nodes_processed = 1

        if isinstance(volume, VolumeWithChilds):
            for child in volume.childs:
                nodes_processed += self._traverse(child)

        # If the ray misses this entire branch (parent + all children),
        # it should jump to the node immediately following the subtree.
        # miss_index = current_index + nodes_processed
        node_data['miss_index'] = current_index + nodes_processed

        return nodes_processed
