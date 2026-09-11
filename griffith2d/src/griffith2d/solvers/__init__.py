"""Stable solver components for Griffith 2D."""

from .degraded_elastic_dem import (
	DegradedElasticDEMTrainingConfig,
	DegradedElasticDEMTrainingResult,
	DegradedElasticEnergyComponents,
	degraded_elastic_potential_energy,
	expand_uniform_phase,
	train_degraded_elastic_dem,
)
from .elastic_dem import (
	ElasticDEMTrainingConfig,
	ElasticDEMTrainingResult,
	ElasticEnergyComponents,
	elastic_potential_energy,
	train_elastic_dem,
)
from .phase_field_dem import (
	PhaseFieldDEMTrainingConfig,
	PhaseFieldDEMTrainingResult,
	PhaseFieldEnergyComponents,
	phase_field_energy,
	train_phase_field_dem,
)
from .staggered import (
	LoadPathResult,
	LoadStepResult,
	StaggeredCouplingConfig,
	StaggeredIterationRecord,
	StaggeredState,
	UniformPhaseSolveResult,
	active_energy_at_points,
	initial_staggered_state,
	run_staggered_load_path,
	solve_staggered_load_step,
	solve_uniform_phase,
)

__all__ = [
	"ElasticDEMTrainingConfig",
	"ElasticDEMTrainingResult",
	"ElasticEnergyComponents",
	"DegradedElasticEnergyComponents",
	"DegradedElasticDEMTrainingConfig",
	"DegradedElasticDEMTrainingResult",
	"PhaseFieldDEMTrainingConfig",
	"PhaseFieldDEMTrainingResult",
	"PhaseFieldEnergyComponents",
	"LoadPathResult",
	"LoadStepResult",
	"StaggeredCouplingConfig",
	"StaggeredIterationRecord",
	"StaggeredState",
	"UniformPhaseSolveResult",
	"active_energy_at_points",
	"elastic_potential_energy",
	"expand_uniform_phase",
	"initial_staggered_state",
	"degraded_elastic_potential_energy",
	"phase_field_energy",
	"run_staggered_load_path",
	"solve_staggered_load_step",
	"solve_uniform_phase",
	"train_elastic_dem",
	"train_degraded_elastic_dem",
	"train_phase_field_dem",
]
