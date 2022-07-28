from core.materials.attenuation_database import AttenuationDataBase
from core.materials.attenuation_functions import AttenuationFunction
from core.other.nonunique_array import NonuniqueArray
from core.materials.materials import Material, MaterialArray
from core.particles.particles import ParticleArray
from core.materials.material_database import MaterialDataBase
from core.physics.processes import CoherentScattering, PhotoelectricEffect, ComptonScattering
import numpy as np
from h5py import File
from hepunits import*


if __name__ == '__main__':
    
    attenuation_database = AttenuationDataBase()
    material_database = MaterialDataBase()
    attenuation_database.add_material(material_database.values())
    material = material_database['Water, Liquid']
    # material = material_database['Pb']
    print(material.density/(g/cm3))
    process = CoherentScattering(attenuation_database)
    energy = 140.5*keV
    print(f'Energy: {energy}')
    LAC = process.attenuation_function(material, energy)*cm
    print(f'LAC: {LAC}')
    
    # composition = material.composition
    # print(material)
    # print('pass')
    # material = Material()
    material_array = MaterialArray(10)
    material_array[5] = material
    
    print(material_array == 0)
    # print(material in material_array)
    
    # print(material_array[2])
    
    # material_array[:5] = material
    
    # print(material_array[2])
    
    
    
    # print(material_array.nbytes/(1024**2))
    # material_array = material_array[:]
    # print(material_array.nbytes/(1024**2))
    # material_array = np.unique(material_array)
    # print(material_array.restore())

    # rng = np.random.default_rng()
    
    # material_database = MaterialDataBase()
    
    # for material in material_database.material_array:
    #     print(material)
    
    # material_array = material_database.material_array
    # material_array = np.asarray(material_array)
    
    # material_array[6] = material_database['Pb']
    
    # n = material_array.size
    
    # material_array = MaterialArray(n)
    
    # energy = np.full(n, 140.5*keV)
    # particle_type = np.zeros_like(energy, dtype=int)
    # direction = np.ones((energy.size, 3), dtype=float)
    # position = np.ones_like(direction)
    
    # particle = ParticleArray(particle_type, position, direction, energy)
    
    # process = ComptonScattering(material_database=material_database, rng=rng)
    
    # LAC = process.get_LAC(particle, material_array)*cm
    
    
    # for material_name, coeff in zip(material_array.name, LAC):
    #     print(material_name, '\t', coeff, '\t', 'cm^-1')

