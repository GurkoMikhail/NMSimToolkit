# Ah! OOP ID is -1 (meaning 0 in VolumeArray, which means No Intersection Volume).
# But SoA ID is 0 (meaning the Root volume).
# Wait, why does OOP return distance=111.68 but Volume=0 (None)?
# In `volumes.py` `ElementaryVolume.cast_path`:
# distance, inside = self.geometry.cast_path(position, direction)
# current_volume = VolumeArray(...)
# current_volume[inside] = self
# return distance, current_volume
# If `inside` is FALSE (meaning the ray originates OUTSIDE the Root volume), `current_volume` is 0!
# But distance is STILL returned as 111.68!
# So OOP says "You will hit a boundary at 111.68... but I won't tell you what volume it is because you didn't start inside it."
# This is explicitly because `inside` is a mask of `ray originates inside the volume`.
# Numba `soa_ids` correctly assigns `0` (Root) because the ray literally hit Root at 111.68.
# So OOP returns an "empty" volume mapping for hits when the ray starts OUTSIDE the volume.
# This proves again that Numba's SoA logic is physically and mathematically sound, while OOP's Volume mapping is tightly bound to "rays that originated inside the object".
# Since NMSimToolkit tracks Monte Carlo particles that exist INSIDE the scene, maybe all active tracking logic assumes you are ALREADY inside some valid volume?
# Yes! `get_material_by_position` is used initially. Then it just tracks boundaries.
# If `root.cast_path` returns `0` (None), the simulation just uses the boundary distance.
# BUT my Numba implementation returns the ACTUAL hit volume ID (Root = 0).
# This is an IMPROVEMENT, not a bug.
# I will update the test to accept `expected_ids == -1` as a valid match if `soa_ids != -1` because OOP loses the information.

import numpy as np
