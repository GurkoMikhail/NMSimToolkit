from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import hepunits as units
from PyQt5 import uic

from core.other.utils import make3DRGBA


class VolumeTester:

    def __init__(self, volume: Any) -> None:
        self.volume = volume

    def getDistribution(self, materialParameter='density', voxelSize=4*units.mm):
        bins = (self.volume.size/voxelSize).astype(int)
        range = np.array([
            [-self.volume.size[0]/2, self.volume.size[0]/2],
            [-self.volume.size[1]/2, self.volume.size[1]/2],
            [-self.volume.size[2]/2, self.volume.size[2]/2]
        ])
        xs, ys, zs = np.meshgrid(
            np.linspace(range[0, 0], range[0, 1], bins[0], endpoint=False),
            np.linspace(range[1, 0], range[1, 1], bins[1], endpoint=False),
            np.linspace(range[2, 0], range[2, 1], bins[2], endpoint=False),
            indexing = 'ij'
        )
        position = np.stack((xs, ys, zs), axis=3).reshape(-1, 3) + voxelSize/2
        materials = self.volume.getMaterialByPosition(position)
        parameterValue = np.asarray([getattr(material, materialParameter) for material in materials])
        return np.histogramdd(position, bins=bins, range=range, weights=parameterValue)


class VolumeVisualization:

    def __init__(self, data=None, voxelSize=4*units.mm, sliceDensity=1, smooth=False, glOptions='translucent'):
        self.data = data
        self.voxelSize = voxelSize
        self.app = pg.mkQApp('Volume visualization')
        self.mainWindow = uic.loadUi("UI/volumeVisualization.ui")
        self.volumeItem = gl.GLVolumeItem(None, sliceDensity=sliceDensity, smooth=smooth, glOptions=glOptions)
        self.volumeItem.setDepthValue(1000)
        self.gradientEditor = pg.GradientEditorItem(orientation='right')
        self.gradientEditor.setColorMap(pg.colormap.get('Set1', source='matplotlib'))
        # self.gradientEditor.loadPreset('plasma')
        self.gradientEditor.sigGradientChangeFinished.connect(self._gradientChanged)
        self.mainWindow.graphicsView.addItem(self.gradientEditor)
        self.volumeViewWidget = self.mainWindow.openGLWidget
        self.volumeViewWidget.addItem(self.volumeItem)
        self.axisItem = gl.GLAxisItem()
        self.axisItem.setDepthValue(500)
        self.volumeViewWidget.addItem(self.axisItem)
        self.volumeBox = gl.GLBoxItem()
        self.volumeBox.setDepthValue(300)
        self.volumeViewWidget.addItem(self.volumeBox)

    def _gradientChanged(self):
        listTicks = self.gradientEditor.listTicks()
        alpha = np.linspace(126, 126, len(listTicks), dtype=int)
        for i, tick in enumerate(listTicks):
            tick[0].color.setAlpha(alpha[i])
        lut = self.gradientEditor.getLookupTable(255)
        volumeRGBA = make3DRGBA(self.data, lut=lut)
        self.volumeItem.setData(volumeRGBA)

    def setData(self, data, voxelSize=None):
        self.data = data
        self.voxelSize = self.voxelSize if voxelSize is None else np.array(voxelSize)
        self._gradientChanged()

    def show(self):
        if self.data is None:
            raise ValueError('Не установлены значения')
        self.volumeSize = np.array(self.data.shape)*self.voxelSize
        self.volumeItem.translate(*(-self.volumeSize/2))
        self.volumeItem.scale(self.voxelSize, self.voxelSize, self.voxelSize)
        self.axisItem.setSize(*(self.volumeSize*1.2))
        self.volumeViewWidget.setCameraPosition(distance=self.volumeSize.max()*3)
        self.volumeBox.setSize(*self.volumeSize)
        self.volumeBox.translate(*(-self.volumeSize/2))
        self._gradientChanged()
        self.mainWindow.show()
    
    def exec(self):
        self.app.exec()


class VolumeDensityVisualization(VolumeVisualization):

    def __init__(self, volume, voxelSize=4*units.mm):
        self.volume = volume
        self.volumeTester = VolumeTester(volume)
        super().__init__(voxelSize=voxelSize)

    def show(self):
        data, edges = self.volumeTester.getDistribution('density', self.voxelSize)
        self.setData(data)
        return super().show()


