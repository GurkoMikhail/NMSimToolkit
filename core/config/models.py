from typing import Any, List, Literal, Optional, Union, Annotated, Tuple
from pydantic import BaseModel, Field, model_validator, field_serializer, field_validator
from core.config.units import (
    LengthConfig, EnergyConfig, TimeConfig, ActivityConfig, AngleConfig, CountConfig, DataSizeConfig, format_data_size
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
        if len(raw) == 16 and all(isinstance(elem, (int, float)) for elem in raw):
            flat_values: List[float] = [float(elem) for elem in raw if isinstance(elem, (int, float))]
            self.matrix = [
                [flat_values[i * 4 + j] for j in range(4)]
                for i in range(4)
            ]
            return self
        if len(raw) == 4 and all(
            isinstance(row, list) and len(row) == 4 and all(isinstance(elem, (int, float)) for elem in row)
            for row in raw
        ):
            self.matrix = [
                [float(elem) for elem in row]
                for row in raw
                if isinstance(row, list)
            ]
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
    dtype: str = 'float32'

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

class SourceMetadataConfig(BaseModel):
    radionuclide: Optional[str] = None
    phantom_type: Optional[str] = None

class SourceConfig(CompositeNodeConfig):
    type: Literal['Source'] = 'Source'
    activity: Optional[ActivityConfig] = None
    distribution: Optional[AnyDistributionConfig] = None
    voxel_size: LengthConfig = 4.0 * units.mm
    radiation_type: str = 'Gamma'
    energy: Union[EnergyConfig, List[List[Any]], List[Tuple[Any, float]]] = 140.5 * units.keV
    half_life: TimeConfig = 6.0 * units.hour
    radionuclide: Optional[str] = None
    phantom_type: Optional[str] = None

    @field_serializer('activity', when_used='unless-none')
    def serialize_activity(self, value: Any) -> Any:
        if isinstance(value, (int, float)):
            activity_value = float(value) / float(units.MBq)
            return f"{activity_value:.6g} MBq"
        return value

    @field_serializer('energy', when_used='unless-none')
    def serialize_energy(self, value: Any) -> Any:
        if isinstance(value, (int, float)):
            energy_value = float(value) / float(units.keV)
            return [[f"{energy_value:.6g} keV", 1.0]]
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
    dtype: Literal['float32', 'float64'] = 'float32'

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

# Пересборка моделей для разрешения рекурсивных ссылок в Pydantic V2
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
    buffer_size: DataSizeConfig = Field(default="3 MiB", validate_default=True)
    floating_point_precision: Literal['float32', 'float64'] = 'float32'
    compression: Optional[Literal['gzip', 'lzf']] = None
    compression_opts: int = Field(default=4, ge=0, le=9)

    @field_serializer('buffer_size')
    def serialize_buffer_size(self, value: Any) -> Any:
        """
        Выгружает объём буфера в читаемом виде: '3 MiB' при кратности двоичному префиксу
        (GiB, MiB, KiB), иначе '<N> B'. Строковые макросы ${var} выгружаются без изменений.
        """
        if isinstance(value, str):
            return value
        return format_data_size(value)

class SimulationManagerConfig(BaseModel):
    start_time: TimeConfig = 0.0 * units.ns
    stop_time: TimeConfig = 1.0 * units.s
    particles_number: CountConfig = 10_000
    batch_capacity: Optional[CountConfig] = None
    buffer_capacity: Optional[CountConfig] = None
    min_energy: EnergyConfig = 1.0 * units.keV
    physics_list: Optional[str] = None
    seed: Optional[int] = None

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
    initial_time: TimeConfig = 0.0 * units.s
    sequential_time: bool = True
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

class LoggingConfig(BaseModel):
    """
    Конфигурация файлового и консольного логирования процесса моделирования.
    """
    output_dir: Optional[str] = "logs"
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    console: bool = False
    per_task_files: bool = True
    format: str = "[%(asctime)s: %(levelname)s] [%(name)s] %(message)s"
    filename_pattern: str = "task_{task_id}.log"

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("output_dir", mode="before")
    @classmethod
    def normalize_output_dir(cls, value: Any) -> Any:
        if value is None:
            return None
        string_value = str(value).strip()
        if not string_value:
            return None
        return string_value

    @field_validator("filename_pattern")
    @classmethod
    def validate_filename_pattern(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Шаблон имени файла лога не может быть пустым.")
        try:
            formatted_name = value.format(task_id=0)
        except (KeyError, IndexError, ValueError) as format_error:
            raise ValueError(
                f"Некорректный шаблон имени файла лога '{value}'. "
                f"Шаблон обязан поддерживать подстановку ключа 'task_id': {format_error}"
            )
        if not formatted_name.strip():
            raise ValueError("Сформированное имя файла лога не может быть пустым.")
        return value.strip()


class SimulationConfig(BaseModel):
    pool_size: int = Field(default=1, ge=1)
    protocol: Optional[AnyProtocolConfig] = None
    simulation_manager: SimulationManagerConfig
    data_manager: DataManagerConfig
    scene: AnyNodeConfig
    logging: Optional[LoggingConfig] = None

    @model_validator(mode="after")
    def validate_logging_concurrency(self) -> "SimulationConfig":
        if self.pool_size > 1 and self.logging is not None and not self.logging.per_task_files:
            raise ValueError(
                "При многопроцессном моделировании (pool_size > 1) параметр "
                "logging.per_task_files обязан быть True во избежание конфликтов "
                "блокировок файлов в ОС Windows."
            )
        if isinstance(self.protocol, StepAndShootProtocolConfig):
            sim_fields_set = self.simulation_manager.model_fields_set
            if "stop_time" not in sim_fields_set:
                self.simulation_manager.stop_time = "${stop_time}"
            if (
                "start_time" not in sim_fields_set
                and self.simulation_manager.stop_time == "${stop_time}"
            ):
                self.simulation_manager.start_time = "${start_time}"
        return self


__all__ = [
    "TranslateConfig",
    "RotateConfig",
    "MatrixTransformConfig",
    "TransformConfig",
    "BoxConfig",
    "GeometryConfig",
    "SpatialNodeConfig",
    "CompositeNodeConfig",
    "VolumeConfig",
    "BaseDistributionConfig",
    "NumpyDistributionConfig",
    "RawDistributionConfig",
    "AnyDistributionConfig",
    "WoodcockVoxelVolumeConfig",
    "GammaCameraSlotsConfig",
    "GammaCameraConfig",
    "ParametricParallelCollimatorConfig",
    "DirectParallelCollimatorConfig",
    "SourceMetadataConfig",
    "SourceConfig",
    "DoseGridNodeConfig",
    "GantryConfig",
    "BaseSpatialNodeConfig",
    "BaseCompositeNodeConfig",
    "AnyNodeConfig",
    "DirectStreamHandlerConfig",
    "SensitiveVolumeHandlerConfig",
    "HistoryAssemblerHandlerConfig",
    "DoseMapHandlerConfig",
    "AnyDataHandlerConfig",
    "DataManagerConfig",
    "SimulationManagerConfig",
    "BaseProtocolConfig",
    "CustomSweepProtocolConfig",
    "StepAndShootProtocolConfig",
    "AnyProtocolConfig",
    "LoggingConfig",
    "SimulationConfig",
]


