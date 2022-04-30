from pathlib import Path
from hepunits import*
from h5py import File
import numpy as np


class SimulationDataManager:
    """ 
    Основной класс моделирования
    """

    def __init__(self, fileName, sensitiveVolumes=[], **kwds):
        self.fileName = Path(fileName)
        self.sensitiveVolumes = sensitiveVolumes
        self.saveEmissionDistribution = True
        self.saveDoseDistribution = True
        self.distributionVoxelSize = 4.*mm
        self.clearInteractionData()
        self.iteractionBufferSize = int(10**3)
        self._bufferedInteractionsNumber = 0
        self.args = [
            'saveEmissionDistribution',
            'saveDoseDistribution',
            'distributionVoxelSize',
            'iteractionBufferSize'
            ]

        for arg in self.args:
            if arg in kwds:
                setattr(self, arg, kwds[arg])

    def checkProgressInFile(self):
        try:
            file = File(self.fileName, 'r')
        except Exception:
            lastTime = None
            state = None
        else:
            try:
                lastTime = file['Source timer']
                # state = file['Source state']
                state = None
                lastTime = float(np.array(lastTime))
            except Exception:
                print(f'\tНе удалось восстановить прогресс')
                lastTime = None
                state = None
            finally:
                print(f'\tПрогресс восстановлен')
                file.close()
        finally:
            print(f'\tSource timer: {lastTime}')
            return lastTime, state

    def addInteractionData(self, interactionData):
        for volume in self.sensitiveVolumes:
            inVolume = volume.checkInside(interactionData.position)
            interactionDataInVolume = interactionData[inVolume]
            if interactionDataInVolume.size > 0:
                interactionDataForSave = np.recarray(interactionDataInVolume.shape, dtype=interactionDataDType)

                interactionDataForSave.globalPosition = interactionDataInVolume.position
                interactionDataForSave.globalDirection = interactionDataInVolume.direction
                interactionDataForSave.localPosition = volume.convertToLocalPosition(interactionDataInVolume.position, asParent=False)
                interactionDataForSave.localDirection = volume.convertToLocalDirection(interactionDataInVolume.direction, asParent=False)

                for field in interactionDataInVolume.dtype.fields:
                    if field in interactionDataForSave.dtype.fields:
                        interactionDataForSave[field] = interactionDataInVolume[field]

                self.interactionData[volume.name].append(interactionDataForSave)
                self._bufferedInteractionsNumber += interactionDataForSave.size
        if self._bufferedInteractionsNumber > self.iteractionBufferSize:
            self.saveInteractionData()
            self._bufferedInteractionsNumber = 0

    def concatenateInteractionData(self):
        for volume in self.sensitiveVolumes:
            volumeName = volume.name
            self.interactionData[volumeName] = np.concatenate(self.interactionData[volumeName]).view(np.recarray)

    def clearInteractionData(self):
        self.interactionData = {volume.name: [] for volume in self.sensitiveVolumes}

    def saveInteractionData(self):
        self.concatenateInteractionData()
        try:
            file = File(f'Output data/{self.fileName}', 'a')
        except Exception:
            print(f'Не удалось сохранить {self.fileName.name}!')
        else:
            if not 'interactionData' in file:
                group = file.create_group('interactionData')
                for volumeName, interactionData in self.interactionData.items():
                    volumeGroup = group.create_group(volumeName)
                    for field in interactionData.dtype.fields:
                        maxshape = list(interactionData[field].shape)
                        maxshape[0] = None
                        volumeGroup.create_dataset(
                            field,
                            data=interactionData[field],
                            compression="gzip",
                            chunks=True,
                            maxshape=maxshape
                            )
            else:
                group = file['interactionData']
                for volumeName, interactionData in self.interactionData.items():
                    volumeGroup = group[volumeName]
                    for key in volumeGroup.keys():
                        volumeGroup[key].resize(
                            (volumeGroup[key].shape[0] + interactionData[key].shape[0]),
                            axis=0
                            )
                        volumeGroup[key][-interactionData[key].shape[0]:] = interactionData[key]
            print(f'{self.fileName.name} generated {self._bufferedInteractionsNumber} events')
            file.close()
        self.clearInteractionData()


interactionDataDType = np.dtype([
        ('globalPosition', '3d'),
        ('globalDirection', '3d'),
        ('localPosition', '3d'),
        ('localDirection', '3d'),
        ('processName', 'S30'),
        ('particleType', 'S30'),
        ('particleID', 'u8'),
        ('energyTransfer', 'd'),
        ('materialDensity', 'd'),
        ('scatteringAngles', '2d'),
        ('emissionTime', 'd'),
        ('emissionPosition', '3d'),
        ('distanceTraveled', 'd'),
])

