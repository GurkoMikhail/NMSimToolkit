import numpy as np
from numba import njit
from numpy.typing import NDArray

from core.other.typing_definitions import Float, Index
from core.geometry.geometry_buffer_soa import GeometryBuffer
from core.particles.particles_soa import ParticleState

@njit(cache=True)
def cast_path_kernel(
    geom_buffer: GeometryBuffer,
    state: ParticleState,
    target_indices: NDArray[Index],
    out_distance: NDArray[Float],
    out_material_id: NDArray[np.uint32],
    distance_epsilon: Float = 1e-3  # 1 micron in mm (assuming units are mm)
) -> None:
    """
    Raycasting engine using jump-loops (Frustum Culling) over flattened GeometryBuffer.
    Updates `out_distance` and `out_material_id` for the provided `target_indices`.
    """
    n_nodes = len(geom_buffer.node_type)

    for i in range(len(target_indices)):
        p_idx = target_indices[i]

        # World coordinates and direction
        px_w = state.position.x[p_idx]
        py_w = state.position.y[p_idx]
        pz_w = state.position.z[p_idx]

        dx_w = state.direction.x[p_idx]
        dy_w = state.direction.y[p_idx]
        dz_w = state.direction.z[p_idx]

        best_distance = np.inf
        best_material = 0xFFFFFFFF  # Invalid/Background

        node_idx = 0
        while node_idx < n_nodes:
            # Load transformation matrix (World -> Local)
            R_xx = geom_buffer.R_xx[node_idx]
            R_xy = geom_buffer.R_xy[node_idx]
            R_xz = geom_buffer.R_xz[node_idx]

            R_yx = geom_buffer.R_yx[node_idx]
            R_yy = geom_buffer.R_yy[node_idx]
            R_yz = geom_buffer.R_yz[node_idx]

            R_zx = geom_buffer.R_zx[node_idx]
            R_zy = geom_buffer.R_zy[node_idx]
            R_zz = geom_buffer.R_zz[node_idx]

            T_x = geom_buffer.T_x[node_idx]
            T_y = geom_buffer.T_y[node_idx]
            T_z = geom_buffer.T_z[node_idx]

            # Apply transformation to Position
            px_l = px_w * R_xx + py_w * R_xy + pz_w * R_xz + T_x
            py_l = px_w * R_yx + py_w * R_yy + pz_w * R_yz + T_y
            pz_l = px_w * R_zx + py_w * R_zy + pz_w * R_zz + T_z

            # Apply transformation to Direction (Rotation only)
            dx_l = dx_w * R_xx + dy_w * R_xy + dz_w * R_xz
            dy_l = dx_w * R_yx + dy_w * R_yy + dz_w * R_yz
            dz_l = dx_w * R_zx + dy_w * R_zy + dz_w * R_zz

            # Box Half-sizes
            hx = geom_buffer.half_size_x[node_idx]
            hy = geom_buffer.half_size_y[node_idx]
            hz = geom_buffer.half_size_z[node_idx]

            # Check inside
            inside_x = np.abs(px_l) <= hx
            inside_y = np.abs(py_l) <= hy
            inside_z = np.abs(pz_l) <= hz
            inside = inside_x and inside_y and inside_z

            # Ray-Box Intersection (Slab Method)
            # Safe division and branching to avoid np.inf * 0.0 = NaN
            if dx_l == 0:
                tmin_x = -np.inf if inside_x else np.inf
                tmax_x = np.inf if inside_x else -np.inf
            else:
                inv_dx = 1.0 / dx_l
                t1_x = (-px_l - hx) * inv_dx
                t2_x = (-px_l + hx) * inv_dx
                tmin_x = min(t1_x, t2_x)
                tmax_x = max(t1_x, t2_x)

            if dy_l == 0:
                tmin_y = -np.inf if inside_y else np.inf
                tmax_y = np.inf if inside_y else -np.inf
            else:
                inv_dy = 1.0 / dy_l
                t1_y = (-py_l - hy) * inv_dy
                t2_y = (-py_l + hy) * inv_dy
                tmin_y = min(t1_y, t2_y)
                tmax_y = max(t1_y, t2_y)

            if dz_l == 0:
                tmin_z = -np.inf if inside_z else np.inf
                tmax_z = np.inf if inside_z else -np.inf
            else:
                inv_dz = 1.0 / dz_l
                t1_z = (-pz_l - hz) * inv_dz
                t2_z = (-pz_l + hz) * inv_dz
                tmin_z = min(t1_z, t2_z)
                tmax_z = max(t1_z, t2_z)

            tmin = max(tmin_x, max(tmin_y, tmin_z))
            tmax = min(tmax_x, min(tmax_y, tmax_z))

            hit = tmax >= tmin and tmax >= 0

            # To match geometries.py ray_casting:
            # distance = np.where(tmax > tmin, tmin, np.inf)
            # distance[inside] = tmax[inside]
            # distance[distance < 0] = np.inf
            # distance += self.distance_epsilon

            # tmax > tmin check is essentially `hit = tmax > tmin and tmax >= 0`.
            # geometries.py implements ray_casting as:
            # distance = np.where(tmax > tmin, tmin, np.inf)
            # distance[inside] = tmax[inside]
            # distance[distance < 0] = np.inf
            # distance += self.distance_epsilon

            # Here, the condition `tmax > tmin` tells if the infinite line intersects the box.
            if tmax > tmin:
                dist = tmax if inside else tmin

                # Check distance >= 0
                if dist >= 0:
                    # In geometries/volumes.py:
                    # `distance_to_child_min = distance_to_child.min(axis=0)`
                    # `distance[inside] = np.where(distance_inside < distance_to_child_min, distance_inside, distance_to_child_min)`
                    # Wait, `inside` in the `volumes.py` logic refers to `current_volume != 0`.
                    # `current_volume` gets set inside `cast_path`: `current_volume[inside] = self`.
                    # Here `inside` means the ray intersects the box (i.e. distance < np.inf).
                    # `cast_path` does:
                    # distance, inside = self.geometry.cast_path(...)
                    # current_volume[inside] = self
                    # So `current_volume != 0` implies that the ray HITS the node (dist != inf).
                    # For rays that HIT the node, the distance is updated to the MINIMUM among the parent's distance and all hit children's distances.

                    if dist < best_distance:
                        best_distance = dist
                        best_material = geom_buffer.material_id[node_idx]

                # We always go to children if we intersect the bounding box (tmax > tmin)
                node_idx += 1
            else:
                # Missed the bounding box completely, skip this node and all its children!
                node_idx = geom_buffer.miss_index[node_idx]
                if node_idx == -1: # Last node
                    break

        # Apply epsilon and write back
        if best_distance != np.inf:
            best_distance += distance_epsilon

        out_distance[i] = best_distance
        out_material_id[i] = best_material
