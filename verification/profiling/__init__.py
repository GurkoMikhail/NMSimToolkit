"""
Фасадный пакет профилирования и анализа узких мест расчетного конвейера NMSimToolkit.
"""
from verification.profiling.batch_size_benchmark import (
    DEFAULT_BATCH_SIZES,
    BatchSizeBenchmarkRecord,
    run_batch_size_sweep_suite,
    run_single_batch_size_benchmark,
)
from verification.profiling.cprofile_harness import (
    CProfileDomainBreakdown,
    CProfileFunctionRecord,
    DualThreadCProfileResult,
    EndToEndCProfileResult,
    run_cprofile_suite,
    run_dual_thread_cprofile,
    run_end_to_end_cprofile,
)
from verification.profiling.data_handlers_benchmark import (
    HANDLER_SPECS,
    BufferCapacityScalingRecord,
    DataHandlerBenchmarkRecord,
    NullDrainHandler,
    run_data_handlers_comparison_suite,
    run_single_handler_benchmark,
)
from verification.profiling.micro_benchmarks import (
    MicroBenchmarkRecord,
    run_micro_benchmarks_suite,
)
from verification.profiling.orchestrator_scaling import (
    OrchestratorScalingRecord,
    run_orchestrator_scaling_benchmark,
)
from verification.profiling.plotting import (
    plot_batch_size_scaling,
    plot_cprofile_domain_breakdown,
    plot_data_handlers_comparison,
    plot_micro_benchmarks_speedup,
    plot_orchestrator_scaling,
    plot_step_phase_breakdown,
    plot_transport_telemetry_comparison,
)
from verification.profiling.step_profiler import (
    PhysicsProcessesBreakdown,
    SourceSubphaseBreakdown,
    StepPhaseBreakdownResult,
    run_step_phase_profiler,
    run_step_profiling_suite,
)
from verification.profiling.transport_telemetry import (
    TransportTelemetryMetrics,
    collect_transport_telemetry,
    run_transport_telemetry_suite,
)
from verification.profiling.workloads import (
    ALL_COLLIMATOR_KINDS,
    CollimatorKind,
    ProfilingWorkload,
    build_all_workloads_matrix,
    build_ge_discovery_camera_head,
    build_heart_spect_workload,
    build_nema_tb161_workload,
    build_point_source_workload,
    warmup_all_profiling_kernels,
    warmup_propagator_instance,
)

__all__ = [
    "ALL_COLLIMATOR_KINDS",
    "BatchSizeBenchmarkRecord",
    "BufferCapacityScalingRecord",
    "CProfileDomainBreakdown",
    "CProfileFunctionRecord",
    "CollimatorKind",
    "DEFAULT_BATCH_SIZES",
    "DataHandlerBenchmarkRecord",
    "DualThreadCProfileResult",
    "EndToEndCProfileResult",
    "HANDLER_SPECS",
    "MicroBenchmarkRecord",
    "NullDrainHandler",
    "OrchestratorScalingRecord",
    "PhysicsProcessesBreakdown",
    "ProfilingWorkload",
    "SourceSubphaseBreakdown",
    "StepPhaseBreakdownResult",
    "TransportTelemetryMetrics",
    "build_all_workloads_matrix",
    "build_ge_discovery_camera_head",
    "build_heart_spect_workload",
    "build_nema_tb161_workload",
    "build_point_source_workload",
    "collect_transport_telemetry",
    "plot_batch_size_scaling",
    "plot_cprofile_domain_breakdown",
    "plot_data_handlers_comparison",
    "plot_micro_benchmarks_speedup",
    "plot_orchestrator_scaling",
    "plot_step_phase_breakdown",
    "plot_transport_telemetry_comparison",
    "run_batch_size_sweep_suite",
    "run_cprofile_suite",
    "run_data_handlers_comparison_suite",
    "run_dual_thread_cprofile",
    "run_end_to_end_cprofile",
    "run_micro_benchmarks_suite",
    "run_orchestrator_scaling_benchmark",
    "run_single_batch_size_benchmark",
    "run_single_handler_benchmark",
    "run_step_phase_profiler",
    "run_step_profiling_suite",
    "run_transport_telemetry_suite",
    "warmup_all_profiling_kernels",
    "warmup_propagator_instance",
]
