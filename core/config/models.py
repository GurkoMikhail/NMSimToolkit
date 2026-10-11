from typing import Any, List, Literal, Optional, Union, Annotated, Tuple
from pydantic import BaseModel, Field, model_validator, field_serializer
from core.config.units import (
    LengthConfig, EnergyConfig, TimeConfig, ActivityConfig, AngleConfig, CountConfig
)
import hepunits as units

class TranslateConfig(BaseModel):
    type: Literal['translate'] = 'translate'
    x: LengthConfig = 0.0 * units.mm
    y: LengthConfig = 0.0 * units.mm
    z: LengthConfig = 0.0 * units.mm
    in_local: bool = False

class RotateConfig(BaseModel):
    type: Literal['rotate'] = 'rotate'
    alpha: AngleConfig = 0.0 * units.rad
    beta: AngleConfig = 0.0 * units.rad
    gamma: AngleConfig = 0.0 * units.rad
    rotation_center: Tuple[LengthConfig, LengthConfig, LengthConfig] = (0.0 * units.mm, 0.0 * units.mm, 0.0 * units.mm)
    in_local: bool = False

class MatrixTransformConfig(BaseModel):
    type: Literal['matrix'] = 'matrix'
    matrix: Union[List[float], List[List[float]]]
    in_local: bool = False

    @model_validator(mode='after')
    def validate_matrix_shape(self) -> 'MatrixTransformConfig':
        raw = self.matrix
        if len(raw) == 16 and all(isinstance(x, (int, float)) for x in raw):
            self.matrix = [
                [float(raw[i * 4 + j]) for j in range(4)]
                for i in range(4)
            ]
            return self
        if len(raw) == 4 and all(len(row) == 4 and all(isinstance(x, (int, float)) for x in row) for row in raw):
            self.matrix = [[float(x) for x in row] for row in raw]
            return self
        raise ValueError(
            f"Матрица трансформации должна иметь размерность 4x4 (16 элементов) или список 4x4, получено {len(raw)} строк"
        )

TransformConfig = Annotated[Union[TranslateConfig, RotateConfig, MatrixTransformConfig], Field(discriminator='type')]


class BoxConfig(BaseModel):
    type: Literal['Box'] = 'Box'
    x: LengthConfig
    y: LengthConfig
    z: LengthConfig

GeometryConfig = Annotated[Union[BoxConfig], Field(discriminator='type')]

class SpatialNodeConfig(BaseModel):
    name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    transformations: List[TransformConfig] = Field(default_factory=list)

class CompositeNodeConfig(SpatialNodeConfig):
    children: List['AnyNodeConfig'] = Field(default_factory=list)

class VolumeConfig(CompositeNodeConfig):
    type: Literal['Volume'] = 'Volume'
    geometry: GeometryConfig
    material: str

class BaseDistributionConfig(BaseModel):
    path: str
    mapping: Optional[dict[float, Union[float, str]]] = None
    fill_value: Optional[Union[float, str]] = None

class NumpyDistributionConfig(BaseDistributionConfig):
    format: Literal['numpy'] = 'numpy'

class RawDistributionConfig(BaseDistributionConfig):
    format: Literal['raw'] = 'raw'
    shape: Tuple[int, int, int]
    order: Literal['C', 'F'] = 'C'
    encoding: Literal['text', 'binary'] = 'text'

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

class GammaCameraSlotsConfig(BaseModel):
    casing: Optional[str] = None
    detector_box: Optional[str] = None
    collimator: Optional[str] = None
    crystal: Optional[str] = None
    glass_backend: Optional[str] = None

class GammaCameraConfig(CompositeNodeConfig):
    type: Literal['GammaCamera'] = 'GammaCamera'
    slots: GammaCameraSlotsConfig = Field(default_factory=GammaCameraSlotsConfig)

class ParametricParallelCollimatorConfig(CompositeNodeConfig):
    type: Literal['ParametricParallelCollimator'] = 'ParametricParallelCollimator'
    size: Tuple[LengthConfig, LengthConfig, LengthConfig]
    hole_diameter: LengthConfig
    septa: LengthConfig
    material: str
    hole_shape: str = 'hexagonal'

class DirectParallelCollimatorConfig(CompositeNodeConfig):
    type: Literal['DirectParallelCollimator'] = 'DirectParallelCollimator'
    size: Tuple[LengthConfig, LengthConfig, LengthConfig]
    hole_diameter: LengthConfig
    septa: LengthConfig
    material: str = 'Pb'
    hole_material: Optional[str] = None
    hole_shape: str = 'hexagonal'

class SourceConfig(CompositeNodeConfig):
    type: Literal['Source'] = 'Source'
    activity: Optional[ActivityConfig] = None
    distribution: AnyDistributionConfig
    voxel_size: LengthConfig = 4.0 * units.mm
    radiation_type: str = 'Gamma'
    energy: Union[EnergyConfig, List[List[Any]], List[Tuple[Any, float]]] = 140.5 * units.keV
    half_life: TimeConfig = 6.0 * units.hour
    radionuclide: Optional[str] = None
    phantom_type: Optional[str] = None

    @field_serializer('activity', when_used='unless-none')
    def serialize_activity(self, value: Any) -> Any:
        if isinstance(value, (int, float)):
            act_mbq = float(value) / float(units.MBq)
            return f"{act_mbq:.6g} MBq"
        return value

    @field_serializer('energy', when_used='unless-none')
    def serialize_energy(self, value: Any) -> Any:
        if isinstance(value, (int, float)):
            energy_kev = float(value) / float(units.keV)
            return [[f"{energy_kev:.6g} keV", 1.0]]
        if isinstance(value, str):
            return [[str(value), 1.0]]
        if isinstance(value, list):
            res = []
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    e_val, p_val = item[0], item[1]
                    if isinstance(e_val, (int, float)):
                        e_str = f"{float(e_val) / float(units.keV):.6g} keV"
                    else:
                        e_str = str(e_val)
                    res.append([e_str, float(p_val)])
                else:
                    res.append(item)
            return res
        return value

class DoseGridNodeConfig(CompositeNodeConfig):
    type: Literal['DoseGridNode'] = 'DoseGridNode'
    size: Tuple[LengthConfig, LengthConfig, LengthConfig] = (100.0 * units.mm, 100.0 * units.mm, 100.0 * units.mm)
    dose_voxel_size: LengthConfig = 5.0 * units.mm
    is_active: bool = True

class GantryConfig(CompositeNodeConfig):
    type: Literal['Gantry'] = 'Gantry'

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
        DirectParallelCollimatorConfig,
        SourceConfig,
        DoseGridNodeConfig,
        GantryConfig,
    ],
    Field(discriminator='type')
]

# We must use model_rebuild to resolve forward references for Self or mutual recursion in pydantic V2
VolumeConfig.model_rebuild()
WoodcockVoxelVolumeConfig.model_rebuild()
GammaCameraConfig.model_rebuild()
ParametricParallelCollimatorConfig.model_rebuild()
DirectParallelCollimatorConfig.model_rebuild()
SourceConfig.model_rebuild()
DoseGridNodeConfig.model_rebuild()
GantryConfig.model_rebuild()

class DirectStreamHandlerConfig(BaseModel):
    type: Literal['DirectStreamHandler'] = 'DirectStreamHandler'
    show_escaped_tracks: bool = False

class SensitiveVolumeHandlerConfig(BaseModel):
    type: Literal['SensitiveVolumeHandler'] = 'SensitiveVolumeHandler'
    sensitive_volumes: List[str] = Field(default_factory=list)

class HistoryAssemblerHandlerConfig(BaseModel):
    type: Literal['HistoryAssemblerHandler'] = 'HistoryAssemblerHandler'
    sensitive_volumes: List[str] = Field(default_factory=list)
    save_initial_states: bool = True

class DoseMapHandlerConfig(BaseModel):
    type: Literal['DoseMapHandler'] = 'DoseMapHandler'
    grid_names: List[str] = Field(default_factory=list)
    shm_name: str = 'nmsim_dose_shm'

AnyDataHandlerConfig = Annotated[
    Union[
        DirectStreamHandlerConfig,
        SensitiveVolumeHandlerConfig,
        HistoryAssemblerHandlerConfig,
        DoseMapHandlerConfig
    ],
    Field(discriminator='type')
]

class DataManagerConfig(BaseModel):
    filename: str
    handlers: List[AnyDataHandlerConfig] = Field(default_factory=list)
    buffer_capacity: Optional[int] = None
    floating_point_precision: Literal['float32', 'float64'] = 'float32'
    compression: Optional[Literal['gzip', 'lzf']] = None
    compression_opts: int = Field(default=4, ge=0, le=9)

class SimulationManagerConfig(BaseModel):
    start_time: TimeConfig = 0.0 * units.ns
    stop_time: TimeConfig = 1.0 * units.s
    particles_number: CountConfig = 10_000
    min_energy: EnergyConfig = 1.0 * units.keV
    physics_list: Optional[str] = None

class BaseProtocolConfig(BaseModel):
    pass

class CustomSweepProtocolConfig(BaseProtocolConfig):
    type: Literal['CustomSweep'] = 'CustomSweep'
    grid_variables: dict[str, List[float]] = Field(default_factory=dict)
    zipped_variables: dict[str, List[float]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def check_zipped_lengths(self) -> 'CustomSweepProtocolConfig':
        if self.zipped_variables:
            lengths = {len(var_values) for var_values in self.zipped_variables.values()}
            if len(lengths) > 1:
                raise ValueError("All arrays in 'zipped_variables' must have the exact same length.")
        return self

class StepAndShootProtocolConfig(BaseProtocolConfig):
    type: Literal['StepAndShoot'] = 'StepAndShoot'
    steps: int = Field(ge=1, description="Количество физических угловых положений (шагов) гантри")
    start_angle: AngleConfig = 0.0 * units.deg
    end_angle: AngleConfig = 360.0 * units.deg
    endpoint: bool = False
    rotation_direction: Literal['CW', 'CCW'] = 'CW'
    time_per_step: TimeConfig = 1.0 * units.s
    radius: Optional[Union[LengthConfig, List[LengthConfig]]] = None
    grid_variables: dict[str, List[float]] = Field(default_factory=dict)
    zipped_variables: dict[str, List[float]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_variables_consistency(self) -> 'StepAndShootProtocolConfig':
        if isinstance(self.radius, list) and len(self.radius) != self.steps:
            raise ValueError(
                f"Длина массива радиусов ({len(self.radius)}) обязана строго совпадать "
                f"с числом шагов steps ({self.steps})"
            )
        if self.zipped_variables:
            for var_name, var_vals in self.zipped_variables.items():
                if len(var_vals) != self.steps:
                    raise ValueError(
                        f"Длина массива переменной '{var_name}' ({len(var_vals)}) "
                        f"обязана строго совпадать с числом шагов steps ({self.steps})"
                    )
        return self

AnyProtocolConfig = Annotated[
    Union[
        StepAndShootProtocolConfig,
        CustomSweepProtocolConfig,
    ],
    Field(discriminator='type')
]

class SimulationConfig(BaseModel):
    pool_size: int = Field(default=1, ge=1)
    protocol: Optional[AnyProtocolConfig] = None
    simulation_manager: SimulationManagerConfig
    data_manager: DataManagerConfig
    scene: AnyNodeConfig


