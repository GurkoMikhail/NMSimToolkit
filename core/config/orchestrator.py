"""
Модуль параллельного оркестратора симуляций (Orchestrator).
Обеспечивает распределение параметрических задач моделирования по пулу процессов
с контролем межпроцессных блокировок файлов HDF5 и генерацией независимых PRNG seeds.
"""

from copy import deepcopy
from multiprocessing import Manager, Pool
from typing import Any, Callable, Dict, List, Optional, Tuple
from numpy.random import SeedSequence

from core.config.models import (
    CustomSweepProtocolConfig,
    SimulationConfig,
)
from core.config.simulation_worker import simulation_worker_task
from core.config.sweep_compiler import SweepCompiler
from core.data.data_handlers import BaseDataHandler


class Orchestrator:
    """
    Диспетчер параллельного пула симуляций.
    """

    def __init__(self, raw_config_dict: Any) -> None:
        """
        Инициализирует оркестратор сырым словарем конфигурации или объектом SimulationConfig.
        """
        if isinstance(raw_config_dict, SimulationConfig):
            self.raw_config_dict: Dict[str, Any] = raw_config_dict.model_dump()
            self.parsed_config: SimulationConfig = raw_config_dict
        else:
            self.raw_config_dict = dict(raw_config_dict)
            self.parsed_config = SimulationConfig.model_validate(raw_config_dict)

    def compile_protocol(self) -> CustomSweepProtocolConfig:
        """
        Компилирует протокол исследования в параметрический CustomSweepProtocolConfig
        через SweepCompiler.
        """
        return SweepCompiler.compile_protocol(self.parsed_config.protocol)

    def _generate_job_list(self, sweep_config: CustomSweepProtocolConfig) -> List[Dict[str, float]]:
        """
        Генерирует матрицу задач на основе переданной конфигурации параметров.
        """
        return SweepCompiler.generate_job_matrix(sweep_config)

    def generate_tasks(self) -> List[Dict[str, float]]:
        """
        Генерирует плоский список словарей параметров для каждой задачи согласно скомпилированному протоколу.
        """
        return SweepCompiler.generate_job_matrix(self.compile_protocol())

    @staticmethod
    def inject_variables(node: Any, context: Dict[str, float]) -> Any:
        """
        Подставляет значения переменных из контекста в шаблоны строк конфигурации.
        """
        return SweepCompiler.inject_variables(node, context)

    def run(
        self,
        telemetry_queue: Optional[Any] = None,
        extra_handlers: Optional[List[BaseDataHandler]] = None,
        extra_handler_factory: Optional[Callable[..., List[BaseDataHandler]]] = None,
        extra_handler_args: Tuple[Any, ...] = (),
    ) -> List[Any]:
        """
        Запускает параллельный цикл вычислений оркестратора, распределяя задачи по пулу процессов.
        Поддерживает общую очередь телеметрии и внедрение внешних обработчиков данных
        (extra_handlers / extra_handler_factory) для динамического сбора результатов.
        """
        pool_size = int(self.parsed_config.pool_size)
        task_contexts = self.generate_tasks()

        multiprocessing_manager = Manager()
        try:
            file_locks: Dict[str, Any] = {}
            seed_sequence = SeedSequence()
            seeds = seed_sequence.spawn(len(task_contexts))

            payloads: List[Tuple[Any, ...]] = []
            for task_index, (context, seed_item) in enumerate(zip(task_contexts, seeds)):
                task_dict = deepcopy(self.raw_config_dict)
                task_dict["_task_id"] = task_index
                task_dict["_context"] = context
                injected_dict = self.inject_variables(task_dict, context)

                # Группировка файловых блокировок по имени выходного HDF5-файла
                filename_str = injected_dict.get("data_manager", {}).get("filename", "default.hdf")
                if filename_str not in file_locks:
                    file_locks[filename_str] = multiprocessing_manager.Lock()

                seed_value = seed_item.generate_state(1)[0]
                payloads.append((
                    injected_dict,
                    seed_value,
                    file_locks[filename_str],
                    telemetry_queue,
                    extra_handlers,
                    extra_handler_factory,
                    extra_handler_args,
                    task_index,
                ))

            if pool_size > 1:
                with Pool(pool_size) as worker_pool:
                    return worker_pool.map(simulation_worker_task, payloads)
            else:
                results_list = []
                for single_payload in payloads:
                    results_list.append(simulation_worker_task(single_payload))
                return results_list
        finally:
            try:
                multiprocessing_manager.shutdown()
            except (OSError, ValueError):
                pass


__all__ = ["Orchestrator"]
