import numpy as np
from numpy import inf
from processes import processesList
from materials import MaterialsDataBase
from woodcoockVolumes import WoodcockVolume
from hepunits import*


class PropagationWithInteraction:
    """ Класс распространения частиц с взаимодействием """

    def __init__(self, processesList=processesList, materialsDataBase=MaterialsDataBase(), rng=None):
        self.materialsDataBase = materialsDataBase
        self.rng = np.random.default_rng() if rng is None else rng
        self.processes = [process(materialsDataBase, rng) for process in processesList.values()]

    def __call__(self, particles, volume):
        """ Сделать шаг """
        distance, currentVolume  = volume.castPath(particles.position, particles.direction)
        materials = np.array([volume.material for volume in currentVolume], dtype=object)
        processesLAC = self.getProcessesLAC(particles, materials)
        totalLAC = processesLAC.sum(axis=0)
        freePath = self.rng.exponential(1/totalLAC)
        interacted = (freePath < distance).nonzero()[0]
        distance[interacted] = freePath[interacted]
        particles.move(distance)
        if interacted.size > 0:
            currentVolume = currentVolume[interacted]
            materials = materials[interacted]
            interactedParticles = particles[interacted]
            processesLAC = processesLAC[:, interacted]
            totalLAC = totalLAC[interacted]
            woodcockVolume = [i for i, volume in enumerate(currentVolume) if isinstance(volume, WoodcockVolume)]
            if len(woodcockVolume) > 0:
                materials[woodcockVolume] = volume.getMaterialByPosition(interactedParticles.position[woodcockVolume])
                processesLAC[:, woodcockVolume] = self.getProcessesLAC(interactedParticles[woodcockVolume], materials[woodcockVolume])
            interactionData = []
            for process, indices in self.chooseProcess(processesLAC, totalLAC):
                processingParticles = interactedParticles[indices]
                interactionData.append(process(processingParticles, materials[indices]))
                interactedParticles[indices] = processingParticles
            particles[interacted] = interactedParticles
            return np.concatenate(interactionData).view(np.recarray)

    def getProcessesLAC(self, particles, materials):
        LAC = []
        for process in self.processes:
            LAC.append(process.getLAC(particles, materials))
        return np.asarray(LAC)

    def getTotalLAC(self, particles, materials):
        totalLAC = np.zeros(particles.size, dtype=float)
        for process in self.processes:
            totalLAC += process.getLAC(particles, materials)
        return totalLAC

    def generateFreePath(self, particles, materials):
        freePath = np.full((len(self.processes), particles.size), inf, dtype=float)
        for i, process in enumerate(self.processes):
            freePath[i] = process.generateFreePath(particles, materials)
        return freePath.min(axis=0)

    def chooseProcess(self, processesLAC, totalLAC):
        probabilities = processesLAC/totalLAC
        rnd = self.rng.random(totalLAC.size)
        chosenProcess = []
        p0 = 0
        for i, process in enumerate(self.processes):
            p1 = p0 + probabilities[i]
            inDelta = (p0 <= rnd)
            inDelta *= (rnd <= p1)
            indices = inDelta.nonzero()[0]
            p0 = p1
            chosenProcess.append((process, indices))
        return chosenProcess

