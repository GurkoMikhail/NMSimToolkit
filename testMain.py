from volumeVisualization import VolumeTester, VolumeVisualization
from volumes import VolumeWithChilds, TransformableVolume
from parametricCollimators import ParametricParallelCollimator
from voxelVolumes import WoodcockVoxelVolume
from gammaCameras import GammaCamera
from geometries import Box
from sources import efg3
from particlesFlows import ParticleFlow
from materials import MaterialsDataBase
from simulationManagers import SimulationDataManager
import numpy as np
from hepunits import*


def modeling(args):
    angle, deltaAngle = args
    materialsDataBase = MaterialsDataBase()

    simulationVolume = VolumeWithChilds(
        geometry=Box(120*cm, 120*cm, 80*cm),
        material=materialsDataBase['Air, Dry (near sea level)'],
        name='SimulationVolume'
    )
    
    detector = TransformableVolume(
        geometry=Box(54.*cm, 40*cm, 0.95*cm),
        material=materialsDataBase['Sodium Iodide'],
        name='Detector'
    )

    collimator = ParametricParallelCollimator(
        size=(detector.size[0], detector.size[2], 3.5*cm),
        holeDiameter=1.5*mm,
        septa=0.2*mm,
        material=materialsDataBase['Pb'],
        name='Collimator'
    )

    spectHead = GammaCamera(
        collimator=collimator,
        detector=detector,
        shieldingThickness=2*cm,
        glassBackendThickness=7.6*cm,
        name='GammaCamera'
    )
    spectHead.rotate(gamma=pi/2)
    spectHead.translate(y=51.2/2*cm + spectHead.size[2])
    spectHead.rotate(alpha=angle)

    materialIDDistribution = np.load('Phantoms/ae3.npy')
    materialDistribution = np.ndarray(materialIDDistribution.shape, dtype=object)
    materialDistribution[materialIDDistribution == 1] = materialsDataBase['Air, Dry (near sea level)']
    materialDistribution[materialIDDistribution == 2] = materialsDataBase['Tissue, Soft (ICRU-44)']
    materialDistribution[materialIDDistribution == 3] = materialsDataBase['B-100 Bone-Equivalent Plastic']

    phantom = WoodcockVoxelVolume(
        voxelSize=4*mm,
        materialDistribution=materialDistribution,
        name='Phantom'
    )
    phantom.rotate(gamma=pi/2)
    phantom.setParent(simulationVolume)

    tableSize = [35*cm, simulationVolume.size[2], 2*cm]
    table = TransformableVolume(
        geometry=Box(*tableSize),
        material=materialsDataBase['Polyvinyl Chloride'],
        name='Table'
    )
    table.transformationMatrix = phantom.transformationMatrix.copy()
    phantomThickness = (128 - phantom.materialDistribution.nonzero()[2].max())*phantom.voxelSize[2] - phantom.voxelSize[2]/2
    table.translate(z=(phantomThickness + table.size[2]/2), inLocal=True)
    simulationVolume.addChild(table)

    simulationVolume.addChild(spectHead)
    
    spectHead2 = spectHead.dublicate()
    spectHead2.rotate(alpha=deltaAngle)

    volumeTester = VolumeTester(simulationVolume)
    distribution, edges = volumeTester.getDistribution('density')

    distribution[distribution == distribution.min()] = np.nan

    volumeVisualization = VolumeVisualization(distribution)
    volumeVisualization.show()
    volumeVisualization.exec()

    source = efg3(
        300*MBq
    )
    source.rotate(gamma=pi/2)

    particleFlow = ParticleFlow(source, simulationVolume, particlesNumber=10**8, stopTime=15*s)
    particleFlow.start()

    simulationDataManager = SimulationDataManager(
        fileName=f'{round(angle/degree, 1)} deg.hdf',
        sensitiveVolumes=[spectHead.detector, spectHead2.detector]
    )
    
    for data in iter(particleFlow.queue.get, 'Finish'):
        simulationDataManager.addInteractionData(data)
    simulationDataManager.saveInteractionData()
    particleFlow.join()


if __name__ == '__main__':
    from multiprocessing import Pool

    angles = np.linspace(-pi/4, 3*pi/4, 32)[:16]
    deltaAngle = np.full_like(angles, 16*pi/(32 - 1))
    pool = Pool(16)
    pool.map(modeling, zip(angles, deltaAngle))



