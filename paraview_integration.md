# ParaView Integration via ZeroMQ (LiveTrajetoryHandler)

This document describes how to configure ParaView to receive and visualize live particle trajectories via the `LiveTrajectoryHandler`.

## 1. Programmable Source Setup

In ParaView, add a **Programmable Source** to the pipeline:
1. Go to `Filters -> Alphabetical -> Programmable Source`.
2. In the Properties panel, change `Output Data Set Type` to `vtkPolyData`.
3. Paste the following Python script into the **Script** block:

```python
import zmq
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

# Constants (e.g. converting from simulation units to millimeters)
SCALE_FACTOR = 1.0  # Adjust according to hepunits conversion if needed

if not hasattr(self, "zmq_context"):
    self.zmq_context = zmq.Context()
    self.zmq_socket = self.zmq_context.socket(zmq.SUB)
    # Replace 'localhost' and '5555' with the correct host and port
    self.zmq_socket.connect("tcp://localhost:5555")
    self.zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")

try:
    # Try receiving data without blocking the GUI entirely.
    # In a real setup, one might use NOBLOCK and poll, but for a simple test:
    parts = self.zmq_socket.recv_multipart(flags=zmq.NOBLOCK)

    if len(parts) == 5:
        metadata_bytes, px_bytes, py_bytes, pz_bytes, ids_bytes = parts

        # 1. Parse metadata
        metadata = np.frombuffer(metadata_bytes, dtype=np.int64)
        step = metadata[0]
        send_count = metadata[1]

        # 2. Parse arrays
        px = np.frombuffer(px_bytes, dtype=np.float64)
        py = np.frombuffer(py_bytes, dtype=np.float64)
        pz = np.frombuffer(pz_bytes, dtype=np.float64)
        track_ids = np.frombuffer(ids_bytes, dtype=np.uint64)

        # 3. Scale coordinates
        px = px / SCALE_FACTOR
        py = py / SCALE_FACTOR
        pz = pz / SCALE_FACTOR

        # 4. Construct vtkPoints
        coords = np.column_stack((px, py, pz))
        vtk_points = vtk.vtkPoints()
        # Create VTK array from numpy without copying
        vtk_coords = dsa.numpyTovtkDataArray(coords, "Points")
        vtk_points.SetData(vtk_coords)

        # 5. Build Output vtkPolyData
        output = self.GetPolyDataOutput()
        output.SetPoints(vtk_points)

        # 6. Add track_ids to PointData
        vtk_ids = dsa.numpyTovtkDataArray(track_ids, "track_ids")
        output.GetPointData().AddArray(vtk_ids)

except zmq.Again:
    # No message available right now
    pass
```

## 2. Generate Trajectories (Pathlines)

Once the Programmable Source is receiving points with `track_ids`:
1. Add the **Temporal Particles To Pathlines** filter to the Programmable Source (`Filters -> Alphabetical -> Temporal Particles To Pathlines`).
2. In the properties of the filter, set **IdChannelArray** to `"track_ids"`.
3. Apply the filter. As the simulation advances and ParaView updates, it will trace the particle points into continuous pathlines based on their IDs.
