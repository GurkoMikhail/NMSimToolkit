import queue
from threading import Thread
from typing import Any, List

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from hepunits import cm, m


class Visualization:

    def __init__(self, queue: queue.Queue, volumes: List[Any]) -> None:
        self.queue = queue
        self.volumes = volumes
        self.app = pg.mkQApp("Simulation visualization")
        self.widget = gl.GLViewWidget()
        self.widget.setCameraPosition(distance=3*m)
        self.widget.setWindowTitle('Simulation visualization')
        self.scatterPlotItem = gl.GLScatterPlotItem()
        self.widget.addItem(self.scatterPlotItem)
        self.linePlotItem = gl.GLLinePlotItem(mode='lines', width=0.1)
        self.widget.addItem(self.linePlotItem)
        volumeSize = np.array((100*cm, 100*cm, 100*cm))
        self.volumeBox = gl.GLBoxItem()
        self.volumeBox.setSize(*volumeSize)
        self.volumeBox.translate(*(-volumeSize/2))
        self.widget.addItem(self.volumeBox)
        self.axes = gl.GLAxisItem()
        self.axes.setSize(*volumeSize/2)
        self.widget.addItem(self.axes)
        self.widget.show()
        self.thread = Thread(target=self.run, daemon=True)

    def update(self):
        position = []
        data = np.concatenate(self.dataHistory)
        particleID = data['particleID']
        particlePosition = data['position']
        emissionPosition = data['emissionPosition']
        processName = data['processName']
        uniqueParticleID = np.unique(particleID)
        for ID in uniqueParticleID:
            indices = (particleID == ID).nonzero()[0]
            indices = np.column_stack([indices, indices]).ravel()
            pos = np.zeros((indices.size, 3), dtype=float)
            pos[0] = emissionPosition[indices[0]]
            pos[1:] = particlePosition[indices[:-1]]
            position.append(pos)
        position = np.concatenate(position)
        color = (1, 1, 0, 0.1)
        self.linePlotItem.setData(pos=position, color=color)
        color = np.column_stack([
            1*(processName == 'PhotoelectricEffect'), 
            1*(processName == 'ComptonScattering'), 
            1*(processName == 'CoherentScattering'),
            np.full_like(particleID, 1)
        ])
        self.scatterPlotItem.setData(pos=particlePosition, size=2, color=color)

    def start(self):
        self.thread.start()
        self.app.exec()

    def run(self):
        self.dataHistory = []
        for data in iter(self.queue.get, None):
            insideVolumes = np.zeros(data.size, dtype=bool)
            for volume in self.volumes:
                insideVolumes += volume.checkInside(data.position)
            data = data[insideVolumes]
            if data.size > 0:
                self.dataHistory.append(data)
                self.update()
            pass

