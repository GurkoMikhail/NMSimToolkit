from core.materials.materials import MaterialArray, dtype_of_material, Material
from core.particles.particles import ParticleArray
from core.materials.material_database import MaterialDataBase
from core.physics.processes import PhotoelectricEffect, ComptonScattering
import numpy as np
from hepunits import*


if __name__ == '__main__':
    
    rng = np.random.default_rng()
    
    array = MaterialArray(1)
    
    material_database = MaterialDataBase()
    
    material = material_database.material_array
    
    n = material.size
    
    energy = np.full(n, 140.5*keV)
    particle_type = np.zeros_like(energy, dtype=int)
    direction = np.ones((energy.size, 3), dtype=float)
    position = np.ones_like(direction)
    
    particle = ParticleArray(particle_type, position, direction, energy)
    
    process = ComptonScattering(material_database=material_database, rng=rng)
    
    LAC = process.get_LAC(particle, material)*cm
    
    for material_name, coeff in zip(material.name, LAC):
        print(material_name, '\t', coeff, '\t', 'cm^-1')

