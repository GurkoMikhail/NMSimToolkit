# ParaView Integration via Raw TCP Socket (LiveTrajetoryHandler)

This document describes how to configure ParaView to receive and visualize live particle trajectories via the `LiveTrajectoryHandler`.

## 1. Programmable Source Setup

In ParaView, add a **Programmable Source** to the pipeline:
1. Go to `Filters -> Alphabetical -> Programmable Source`.
2. In the Properties panel, change `Output Data Set Type` to `vtkPolyData`.
3. Paste the following Python script into the **Script** block:

```python
import socket
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

# Constants (e.g. converting from simulation units to millimeters)
SCALE_FACTOR = 1.0  # Adjust according to hepunits conversion if needed

def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

if not hasattr(self, "tcp_socket") or self.tcp_socket is None:
    self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Replace 'localhost' and '5555' with the correct host and port
        self.tcp_socket.connect(("localhost", 5555))
        # Optional: set timeout so ParaView doesn't completely freeze if the sim pauses
        self.tcp_socket.settimeout(0.5)
    except ConnectionError:
        self.tcp_socket = None

if self.tcp_socket is not None:
    try:
        # 1. Receive exactly 16 bytes for metadata (2 * 8 bytes for np.int64)
        metadata_bytes = recv_exact(self.tcp_socket, 16)
        if metadata_bytes:
            metadata = np.frombuffer(metadata_bytes, dtype=np.int64)
            step = metadata[0]
            send_count = metadata[1]

            # 2. Calculate expected sizes
            float_bytes = send_count * 8
            uint_bytes = send_count * 8

            # 3. Receive exact bytes for each array
            px_bytes = recv_exact(self.tcp_socket, float_bytes)
            py_bytes = recv_exact(self.tcp_socket, float_bytes)
            pz_bytes = recv_exact(self.tcp_socket, float_bytes)
            ids_bytes = recv_exact(self.tcp_socket, uint_bytes)

            if px_bytes and py_bytes and pz_bytes and ids_bytes:
                # 4. Parse arrays
                px = np.frombuffer(px_bytes, dtype=np.float64)
                py = np.frombuffer(py_bytes, dtype=np.float64)
                pz = np.frombuffer(pz_bytes, dtype=np.float64)
                track_ids = np.frombuffer(ids_bytes, dtype=np.uint64)

                # 5. Scale coordinates
                px = px / SCALE_FACTOR
                py = py / SCALE_FACTOR
                pz = pz / SCALE_FACTOR

                # 6. Construct vtkPoints
                coords = np.column_stack((px, py, pz))
                vtk_points = vtk.vtkPoints()
                # Create VTK array from numpy without copying
                vtk_coords = dsa.numpyTovtkDataArray(coords, "Points")
                vtk_points.SetData(vtk_coords)

                # 7. Build Output vtkPolyData
                output = self.GetPolyDataOutput()
                output.SetPoints(vtk_points)

                # 8. Add track_ids to PointData
                vtk_ids = dsa.numpyTovtkDataArray(track_ids, "track_ids")
                output.GetPointData().AddArray(vtk_ids)
            else:
                # Disconnected mid-stream
                self.tcp_socket.close()
                self.tcp_socket = None

    except socket.timeout:
        # No message available right now
        pass
    except (ConnectionError, OSError):
        self.tcp_socket.close()
        self.tcp_socket = None
```

## 2. Generate Trajectories (Pathlines)

Once the Programmable Source is receiving points with `track_ids`:
1. Add the **Temporal Particles To Pathlines** filter to the Programmable Source (`Filters -> Alphabetical -> Temporal Particles To Pathlines`).
2. In the properties of the filter, set **IdChannelArray** to `"track_ids"`.
3. Apply the filter. As the simulation advances and ParaView updates, it will trace the particle points into continuous pathlines based on their IDs.
