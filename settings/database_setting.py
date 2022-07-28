from core.materials.attenuation_database import AttenuationDataBase
from core.materials.material_database import MaterialDataBase


material_database = MaterialDataBase()
attenuation_database = AttenuationDataBase()
attenuation_database.add_material(material_database.values())
