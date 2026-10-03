from typing import Any, Dict, Iterator
from pydantic import BaseModel, ConfigDict, Field


class GuiSimulationSettings(BaseModel):
    """
    Строго типизированная модель конфигурационных настроек симуляции графического интерфейса.
    Исключает антипаттерн неструктурированного словаря Dict[str, Any] в MainWindow
    и обеспечивает строгую LBYL-валидацию физических диапазонов параметров.
    """
    model_config = ConfigDict(validate_assignment=True)

    particles_number: int = Field(default=5000, ge=1, description="Число частиц на задачу")
    pool_size: int = Field(default=1, ge=1, description="Размер пула параллельных процессов")
    min_energy: float = Field(default=1.0, ge=0.0, description="Порог энергии частиц (кэВ)")
    max_tracks_per_batch: int = Field(default=2000, ge=1, description="Максимум треков в пачке")
    max_tracks_points: int = Field(default=50000, ge=1, description="Максимум точек треков")
    render_as_lines: bool = Field(default=True, description="Отрисовка треков линиями")
    grid_snap_step: float = Field(default=10.0, gt=0.0, description="Шаг сетки перемещения Gizmo (мм)")
    angle_snap_step: float = Field(default=15.0, gt=0.0, description="Шаг угловой привязки Gizmo (град)")
    scale_snap_step: float = Field(default=1.0, gt=0.0, description="Шаг привязки масштаба Gizmo (мм)")
    xray_energy_kev: float = Field(default=140.0, gt=0.0, description="Энергия фотонов для расчета ослабления и прозрачности (кэВ)")
    pseudo_xray_mode: bool = Field(default=False, description="Режим отображения в псевдорентгене")

    def update(self, new_settings: Any) -> None:
        """
        Обновление параметров конфигурации с валидацией инвариантов Pydantic.
        """
        if not isinstance(new_settings, (GuiSimulationSettings, dict)):
            raise TypeError(f"Ожидался GuiSimulationSettings или dict, получено {type(new_settings).__name__}")
        source_data = new_settings.to_dict() if isinstance(new_settings, GuiSimulationSettings) else new_settings
        for param_key, param_value in source_data.items():
            if param_key == 'particles_number':
                self.particles_number = int(param_value)
            elif param_key == 'pool_size':
                self.pool_size = int(param_value)
            elif param_key == 'min_energy':
                self.min_energy = float(param_value)
            elif param_key == 'max_tracks_per_batch':
                self.max_tracks_per_batch = int(param_value)
            elif param_key == 'max_tracks_points':
                self.max_tracks_points = int(param_value)
            elif param_key == 'render_as_lines':
                self.render_as_lines = bool(param_value)
            elif param_key == 'grid_snap_step':
                self.grid_snap_step = float(param_value)
            elif param_key == 'angle_snap_step':
                self.angle_snap_step = float(param_value)
            elif param_key == 'scale_snap_step':
                self.scale_snap_step = float(param_value)
            elif param_key == 'xray_energy_kev':
                self.xray_energy_kev = float(param_value)
            elif param_key == 'pseudo_xray_mode':
                self.pseudo_xray_mode = bool(param_value)
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


# Псевдоним модели для явного отражения ее доменной зоны ответственности
ExecutionAndRenderSettings = GuiSimulationSettings

__all__ = [
    'GuiSimulationSettings',
    'ExecutionAndRenderSettings',
]
