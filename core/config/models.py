from typing import List, Literal, Optional, Union, Annotated, Tuple
from pydantic import BaseModel, Field

class TranslateConfig(BaseModel):
    type: Literal['translate'] = 'translate'
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    in_local: bool = False

class RotateConfig(BaseModel):
    type: Literal['rotate'] = 'rotate'
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    rotation_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    in_local: bool = False

TransformConfig = Annotated[Union[TranslateConfig, RotateConfig], Field(discriminator='type')]

class BoxConfig(BaseModel):
    type: Literal['Box'] = 'Box'
    x: float
    y: float
    z: float

GeometryConfig = Annotated[Union[BoxConfig], Field(discriminator='type')]

class BaseNodeConfig(BaseModel):
    name: Optional[str] = None
    transformations: List[TransformConfig] = Field(default_factory=list)

class VolumeConfig(BaseNodeConfig):
    type: Literal['Volume'] = 'Volume'
    geometry: GeometryConfig
    material: str
    children: List['AnyNodeConfig'] = Field(default_factory=list)

class WoodcockVoxelVolumeConfig(BaseNodeConfig):
    type: Literal['WoodcockVoxelVolume'] = 'WoodcockVoxelVolume'
    voxel_size: float
    material_distribution_path: str
    children: List['AnyNodeConfig'] = Field(default_factory=list)

class GammaCameraConfig(BaseNodeConfig):
    type: Literal['GammaCamera'] = 'GammaCamera'
    collimator: 'AnyNodeConfig'
    detector: 'AnyNodeConfig'
    gap: float = 0.1
    shielding_thickness: float = 2.0
    glass_backend_thickness: float = 5.0
    children: List['AnyNodeConfig'] = Field(default_factory=list)

class ParametricParallelCollimatorConfig(BaseNodeConfig):
    type: Literal['ParametricParallelCollimator'] = 'ParametricParallelCollimator'
    size: Tuple[float, float, float]
    hole_diameter: float
    septa_thickness: float
    material: str
    children: List['AnyNodeConfig'] = Field(default_factory=list)

class ParametricParallelSquareCollimatorConfig(BaseNodeConfig):
    type: Literal['ParametricParallelSquareCollimator'] = 'ParametricParallelSquareCollimator'
    size: Tuple[float, float, float]
    hole_size: float
    septa_thickness: float
    material: str
    children: List['AnyNodeConfig'] = Field(default_factory=list)

class SourceConfig(BaseNodeConfig):
    type: Literal['Source'] = 'Source'
    activity: Optional[float] = None
    distribution_path: str
    voxel_size: float = 0.4
    radiation_type: str = 'Gamma'
    energy: Union[float, List[List[float]]] = 140.5e3
    half_life: float = 21600.0
    children: List['AnyNodeConfig'] = Field(default_factory=list)

AnyNodeConfig = Annotated[
    Union[
        VolumeConfig,
        WoodcockVoxelVolumeConfig,
        GammaCameraConfig,
        ParametricParallelCollimatorConfig,
        ParametricParallelSquareCollimatorConfig,
        SourceConfig,
    ],
    Field(discriminator='type')
]

# We must use model_rebuild to resolve forward references for Self or mutual recursion in pydantic V2
VolumeConfig.model_rebuild()
WoodcockVoxelVolumeConfig.model_rebuild()
GammaCameraConfig.model_rebuild()
ParametricParallelCollimatorConfig.model_rebuild()
ParametricParallelSquareCollimatorConfig.model_rebuild()
SourceConfig.model_rebuild()

class DataHandlerConfig(BaseModel):
    type: Literal['DirectStreamHandler', 'SensitiveVolumeHandler', 'HistoryAssemblerHandler']
    sensitive_volumes: List[str] = Field(default_factory=list)

class DataManagerConfig(BaseModel):
    filename: str
    handlers: List[DataHandlerConfig] = Field(default_factory=list)
    buffer_capacity: int = 100000

class SimulationManagerConfig(BaseModel):
    stop_time: float = 1.0
    particles_number: int = 1000
    min_energy: float = 1000.0

class SimulationConfig(BaseModel):
    settings: SimulationManagerConfig
    data_manager: DataManagerConfig
    scene: AnyNodeConfig
