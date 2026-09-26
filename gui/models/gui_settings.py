from typing import Any, Dict, Iterator
from pydantic import BaseModel, ConfigDict, Field


class GuiSimulationSettings(BaseModel):
    """
    Строго типизированная модель конфигурационных настроек симуляции графического интерфейса.
    Исключает антипаттерн неструктурированного словаря Dict[str, Any] в MainWindow
    и обеспечивает строгую LBYL-валидацию физических диапазонов параметров.
    """
    model_config = ConfigDict(validate_assignment=True)

    views_number: int = Field(default=1, ge=1, description="Количество ракурсов ОФЭКТ")
    stop_time: float = Field(default=1.0, gt=0.0, description="Время моделирования (с)")
    angular_range: float = Field(default=360.0, gt=0.0, le=360.0, description="Угловой диапазон вращения (град)")
    particles_number: int = Field(default=5000, ge=1, description="Число частиц на задачу")
    pool_size: int = Field(default=1, ge=1, description="Размер пула параллельных процессов")
    min_energy: float = Field(default=1.0, ge=0.0, description="Порог энергии частиц (кэВ)")
    buffer_capacity: int = Field(default=10000, ge=100, description="Емкость буфера частиц")
    max_tracks_per_batch: int = Field(default=2000, ge=1, description="Максимум треков в пачке")
    max_tracks_points: int = Field(default=50000, ge=1, description="Максимум точек треков")
    render_as_lines: bool = Field(default=True, description="Отрисовка треков линиями")
    show_escaped_tracks: bool = Field(default=False, description="Отображение вылетевших треков")
    dose_accumulation_enabled: bool = Field(default=True, description="Включение накопления дозы")
    dose_voxel_size: float = Field(default=5.0, gt=0.0, description="Размер вокселя дозы (мм)")

    def update(self, new_settings: Any) -> None:
        """
        Обновление параметров конфигурации с валидацией инвариантов Pydantic.
        """
        if not isinstance(new_settings, (GuiSimulationSettings, dict)):
            raise TypeError(f"Ожидался GuiSimulationSettings или dict, получено {type(new_settings).__name__}")
        source_data = new_settings.to_dict() if isinstance(new_settings, GuiSimulationSettings) else new_settings
        for param_key, param_value in source_data.items():
            if param_key == 'views_number':
                self.views_number = int(param_value)
            elif param_key == 'stop_time':
                self.stop_time = float(param_value)
            elif param_key == 'angular_range':
                self.angular_range = float(param_value)
            elif param_key == 'particles_number':
                self.particles_number = int(param_value)
            elif param_key == 'pool_size':
                self.pool_size = int(param_value)
            elif param_key == 'min_energy':
                self.min_energy = float(param_value)
            elif param_key == 'buffer_capacity':
                self.buffer_capacity = int(param_value)
            elif param_key == 'max_tracks_per_batch':
                self.max_tracks_per_batch = int(param_value)
            elif param_key == 'max_tracks_points':
                self.max_tracks_points = int(param_value)
            elif param_key == 'render_as_lines':
                self.render_as_lines = bool(param_value)
            elif param_key == 'show_escaped_tracks':
                self.show_escaped_tracks = bool(param_value)
            elif param_key == 'dose_accumulation_enabled':
                self.dose_accumulation_enabled = bool(param_value)
            elif param_key == 'dose_voxel_size':
                self.dose_voxel_size = float(param_value)
            else:
                raise KeyError(f"Неизвестный параметр конфигурации: {param_key}")

    def to_dict(self) -> Dict[str, Any]:
        """Экспорт настроек в плоский словарь."""
        return self.model_dump()

    def __getitem__(self, key: str) -> Any:
        dump_data = self.model_dump()
        if key not in dump_data:
            raise KeyError(f"Неизвестный параметр конфигурации: {key}")
        return dump_data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.__class__.model_fields:
            raise KeyError(f"Неизвестный параметр конфигурации: {key}")
        self.update({key: value})

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump().get(key, default)

    def __iter__(self) -> Iterator[str]:
        return iter(self.model_dump())

    def keys(self):
        return self.model_dump().keys()

    def items(self):
        return self.model_dump().items()

    def values(self):
        return self.model_dump().values()
