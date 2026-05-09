from typing import List, Literal, Optional, Union, Annotated, Tuple
from pydantic import BaseModel, Field, model_validator
from core.config.units import LengthConfig, EnergyConfig, TimeConfig, ActivityConfig, AngleConfig

class TranslateConfig(BaseModel):
    type: Literal['translate'] = 'translate'
    x: LengthConfig = 0.0
    y: LengthConfig = 0.0
    z: LengthConfig = 0.0
    in_local: bool = False

class RotateConfig(BaseModel):
    type: Literal['rotate'] = 'rotate'
    alpha: AngleConfig = 0.0
    beta: AngleConfig = 0.0
    gamma: AngleConfig = 0.0
    rotation_center: Tuple[LengthConfig, LengthConfig, LengthConfig] = (0.0, 0.0, 0.0)
    in_local: bool = False

TransformConfig = Annotated[Union[TranslateConfig, RotateConfig], Field(discriminator='type')]

class BoxConfig(BaseModel):
    type: Literal['Box'] = 'Box'
    x: LengthConfig
    y: LengthConfig
    z: LengthConfig

GeometryConfig = Annotated[Union[BoxConfig], Field(discriminator='type')]

class SpatialNodeConfig(BaseModel):
    name: Optional[str] = None
    transformations: List[TransformConfig] = Field(default_factory=list)

class CompositeNodeConfig(SpatialNodeConfig):
    children: List['AnyNodeConfig'] = Field(default_factory=list)

class VolumeConfig(CompositeNodeConfig):
    type: Literal['Volume'] = 'Volume'
    geometry: GeometryConfig
    material: str

class BaseDistributionConfig(BaseModel):
    path: str
    mapping: dict[float, Union[float, str]] = {}
    fill_value: Optional[Union[float, str]] = None

class NumpyDistributionConfig(BaseDistributionConfig):
    format: Literal['numpy'] = 'numpy'

class RawDistributionConfig(BaseDistributionConfig):
    format: Literal['raw'] = 'raw'
    shape: Tuple[int, int, int]
    order: Literal['C', 'F'] = 'C'

AnyDistributionConfig = Annotated[
    Union[
        NumpyDistributionConfig,
        RawDistributionConfig
    ],
    Field(discriminator='format')
]

class WoodcockVoxelVolumeConfig(CompositeNodeConfig):
    type: Literal['WoodcockVoxelVolume'] = 'WoodcockVoxelVolume'
    voxel_size: LengthConfig
    distribution: AnyDistributionConfig

class GammaCameraConfig(CompositeNodeConfig):
    type: Literal['GammaCamera'] = 'GammaCamera'
    collimator: 'AnyNodeConfig'
    detector: 'AnyNodeConfig'
    gap: LengthConfig = 0.1
    shielding_thickness: LengthConfig = 2.0
    glass_backend_thickness: LengthConfig = 5.0

class ParametricParallelCollimatorConfig(CompositeNodeConfig):
    type: Literal['ParametricParallelCollimator'] = 'ParametricParallelCollimator'
    size: Tuple[LengthConfig, LengthConfig, LengthConfig]
    hole_diameter: LengthConfig
    septa_thickness: LengthConfig
    material: str

class ParametricParallelSquareCollimatorConfig(CompositeNodeConfig):
    type: Literal['ParametricParallelSquareCollimator'] = 'ParametricParallelSquareCollimator'
    size: Tuple[LengthConfig, LengthConfig, LengthConfig]
    hole_size: LengthConfig
    septa_thickness: LengthConfig
    material: str

class SourceConfig(CompositeNodeConfig):
    type: Literal['Source'] = 'Source'
    activity: Optional[ActivityConfig] = None
    distribution: AnyDistributionConfig
    voxel_size: LengthConfig = 4.0 # mm
    radiation_type: str = 'Gamma'
    # TODO: Pydantic union discrimination with deeply nested pint validators might be complex
    # but for simple types we can redefine it. Let's see if Union[EnergyConfig, List[List[float]]] works.
    energy: Union[EnergyConfig, List[List[float]]] = 0.1405 # 140.5 keV in MeV
    half_life: TimeConfig = 2.16e13 # 6 hours in ns

class BaseSpatialNodeConfig(SpatialNodeConfig):
    type: Literal['SpatialNode'] = 'SpatialNode'

class BaseCompositeNodeConfig(CompositeNodeConfig):
    type: Literal['CompositeNode'] = 'CompositeNode'

AnyNodeConfig = Annotated[
    Union[
        BaseSpatialNodeConfig,
        BaseCompositeNodeConfig,
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

class DirectStreamHandlerConfig(BaseModel):
    type: Literal['DirectStreamHandler'] = 'DirectStreamHandler'

class SensitiveVolumeHandlerConfig(BaseModel):
    type: Literal['SensitiveVolumeHandler'] = 'SensitiveVolumeHandler'
    sensitive_volumes: List[str] = Field(default_factory=list)

class HistoryAssemblerHandlerConfig(BaseModel):
    type: Literal['HistoryAssemblerHandler'] = 'HistoryAssemblerHandler'
    sensitive_volumes: List[str] = Field(default_factory=list)

AnyDataHandlerConfig = Annotated[
    Union[
        DirectStreamHandlerConfig,
        SensitiveVolumeHandlerConfig,
        HistoryAssemblerHandlerConfig
    ],
    Field(discriminator='type')
]

class DataManagerConfig(BaseModel):
    filename: str
    handlers: List[AnyDataHandlerConfig] = Field(default_factory=list)
    buffer_capacity: int = 100000

class SimulationManagerConfig(BaseModel):
    start_time: TimeConfig = 0.0
    stop_time: TimeConfig = 1.0
    particles_number: int = 1000
    min_energy: EnergyConfig = 1000.0

class BaseProtocolConfig(BaseModel):
    pass

class CustomSweepProtocolConfig(BaseProtocolConfig):
    type: Literal['CustomSweep'] = 'CustomSweep'
    grid_variables: dict[str, List[float]] = Field(default_factory=dict)
    zipped_variables: dict[str, List[float]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_zipped_lengths(self) -> 'CustomSweepProtocolConfig':
        if self.zipped_variables:
            lengths = {len(v) for v in self.zipped_variables.values()}
            if len(lengths) > 1:
                raise ValueError("All arrays in 'zipped_variables' must have the exact same length.")
        return self

class StepAndShootProtocolConfig(BaseProtocolConfig):
    type: Literal['StepAndShoot'] = 'StepAndShoot'
    views: int
    gamma_cameras: int = 1
    start_angle: AngleConfig
    end_angle: AngleConfig
    time_per_view: TimeConfig

AnyProtocolConfig = Annotated[
    Union[
        CustomSweepProtocolConfig,
        StepAndShootProtocolConfig
    ],
    Field(discriminator='type')
]

class SimulationConfig(BaseModel):
    protocol: Optional[AnyProtocolConfig] = None
    simulation_manager: SimulationManagerConfig
    data_manager: DataManagerConfig
    scene: AnyNodeConfig
