import cProfile
from hepunits import*
import numpy as np
from numpy import cos
import multiprocessing as mp
from propagationManagers import PropagationWithInteraction
import threading as mt
import queue
from time import time

Queue = mp.Queue
Thread = mp.Process


class ParticleFlow(Thread):
    """ Класс потока частиц """

    def __init__(self, source, simulationVolume, propagationManager=PropagationWithInteraction(), stopTime=1*s, particlesNumber=10**3, queue=Queue(maxsize=1), solidAngle=None):
        super().__init__()
        self.source = source
        self.simulationVolume = simulationVolume
        self.propagationManager = propagationManager
        self.stopTime = stopTime
        self.particlesNumber = int(particlesNumber)
        self.solidAngle = solidAngle
        self.queue = queue
        self.step = 1
        self.minEnergy = 1.*keV
        self.daemon = True

    def offTheSolidAngle(self, direction):
        vector, angle = self.solidAngle
        cosAlpha = vector[0]*direction[:, 0]
        cosAlpha += vector[1]*direction[:, 1]
        cosAlpha += vector[2]*direction[:, 2]
        return cosAlpha <= cos(angle)

    def invalidParticles(self, particles):
        result = particles.energy <= self.minEnergy
        result += self.simulationVolume.checkOutside(particles.position)
        if self.solidAngle is not None:
            result += self.offTheSolidAngle(particles.direction)
        return result

    def sendData(self, data):
        self.queue.put(data)

    def nextStep(self):
        interactionData = self.propagationManager(self.particles, self.simulationVolume)
        invalidParticles = self.invalidParticles(self.particles)
        if self.source.timer <= self.stopTime:
            newParticles = self.source.generateParticles(np.count_nonzero(invalidParticles))
            self.particles[invalidParticles] = newParticles
        else:
            self.particles = self.particles[~invalidParticles]
        self.step += 1
        # self.sendData(self.particles)
        self.sendData(interactionData)

    def _run(self):
        cProfile.runctx('self._run()', globals(), locals(), 'stats.txt')

    def run(self):
        """ Реализация работы потока частиц """
        print(f'\t{self.name} started')
        start = time()
        self.particles = self.source.generateParticles(self.particlesNumber)
        while self.particles.size > 0:
                self.nextStep()
                print(self.source.timer/s)
        self.queue.put('Finish')
        print(f'\t{self.name} finished\n'
            + f'\t\tTime passed: {time() - start} seconds')

