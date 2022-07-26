from core.other.nonunique_array import NonuniqueArray
from core.materials.materials import MaterialArray
from core.particles.particles import ParticleArray
from core.materials.material_database import MaterialDataBase
from core.physics.processes import PhotoelectricEffect, ComptonScattering
import numpy as np
from hepunits import*


if __name__ == '__main__':

    rng = np.random.default_rng()
    
    material_database = MaterialDataBase()
    
    # for material in material_database.material_array:
    #     print(material)
    
    material_array = material_database.material_array
    # material_array = set(material_array)
    # material_array = np.asarray(material_array)
    
    # material_array[6] = material_database['Pb']
    
    n = material_array.size
    
    energy = np.full(n, 140.5*keV)
    particle_type = np.zeros_like(energy, dtype=int)
    direction = np.ones((energy.size, 3), dtype=float)
    position = np.ones_like(direction)
    
    particle = ParticleArray(particle_type, position, direction, energy)
    
    process = ComptonScattering(material_database=material_database, rng=rng)
    
    LAC = process.get_LAC(particle, material_array)*cm
    
    
    for material_name, coeff in zip(material_array.name, LAC):
        print(material_name, '\t', coeff, '\t', 'cm^-1')

