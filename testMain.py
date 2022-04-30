from volumeVisualization import VolumeTester, VolumeVisualization, VolumeDensityVisualization
from volumes import VolumeWithChilds, TransformableVolume
from parametricCollimators import ParametricParallelCollimator
from gammaCameras import GammaCamera
from geometries import Box
from sources import efg3, PointSource
from particlesFlows import ParticleFlow
from materials import MaterialsDataBase
from simulationDataManager import SimulationDataManager
from visualization import Visualization
import numpy as np
from hepunits import*

angles = np.linspace(-pi/4, 3*pi/4, 32)/degree

from voxelVolumes import WoodcockVoxelVolume


if __name__ == '__main__':
    materialsDataBase = MaterialsDataBase()

    simulationVolume = VolumeWithChilds(
        geometry=Box(100*cm, 100*cm, 100*cm),
        material=materialsDataBase['Air, Dry (near sea level)'],
        name='SimulationVolume'
    )
    
    detector = TransformableVolume(
        geometry=Box(51.2*cm, 40*cm, 1*cm),
        material=materialsDataBase['Sodium Iodide'],
        name='Detector'
    )

    collimator = ParametricParallelCollimator(
        size=(51.2*cm, 40*cm, 2*cm),
        holeDiameter=1.1*mm,
        septa=0.16*mm,
        material=materialsDataBase['Pb'],
        name='Collimator'
    )

    spectHead = GammaCamera(collimator, detector, name='GammaCamera')
    spectHead.rotate(gamma=pi/2)
    spectHead.translate(y=51.2/2*cm + spectHead.size[2])

    materialIDDistribution = np.load('Phantoms/ae3.npy')
    materialDistribution = np.ndarray(materialIDDistribution.shape, dtype=object)
    materialDistribution[materialIDDistribution == 1] = materialsDataBase['Air, Dry (near sea level)']
    materialDistribution[materialIDDistribution == 2] = materialsDataBase['Tissue, Soft (ICRU-44)']
    materialDistribution[materialIDDistribution == 3] = materialsDataBase['B-100 Bone-Equivalent Plastic']

    phantom = WoodcockVoxelVolume(
        voxelSize=4*mm,
        materialDistribution=materialDistribution
    )
    phantom.rotate(gamma=pi/2)
    phantom.setParent(simulationVolume)

    simulationVolume.addChild(spectHead)
    spectHead2 = spectHead.dublicate()

    spectHead.rotate(alpha=-pi/4)
    spectHead2.rotate(alpha=pi/4)


    volumeTester = VolumeTester(simulationVolume)
    distribution, edges = volumeTester.getDistribution('density')

    # volumeVisualization = VolumeVisualization(distribution)
    # volumeVisualization.show()
    # volumeVisualization.exec()


    # source = PointSource(
    #     (0*cm, 0*cm, 0*cm),
    #     300*MBq,
    #     140.5*keV
    # )

    source = efg3(
        (-25.6*cm, -25.6*cm, -25.6*cm),
        300*MBq,
        rotationAngles=(0, 0, pi/2)
    )

    particleFlow = ParticleFlow(source, simulationVolume, particlesNumber=10**5, stopTime=0.1*s)
    particleFlow.start()

    simulationDataManager = SimulationDataManager('test.hdf', [spectHead.detector, spectHead2.detector])
    
    for data in iter(particleFlow.queue.get, None):
        simulationDataManager.addInteractionData(data)

    # visualization = Visualization(particleFlow.queue, [spectHead.detector, spectHead2.detector])
    # visualization.start()

    particleFlow.join()

