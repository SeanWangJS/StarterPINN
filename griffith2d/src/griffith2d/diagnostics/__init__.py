"""Physical diagnostics and analytic references for Griffith 2D."""

from .coupling import (
	CoupledPhaseRoot,
	DegradedElasticBenchmarkMetrics,
	UniformDegradedTensionBenchmark,
	continue_low_damage_root,
	coupled_phase_residual,
	coupled_phase_roots,
	degraded_boundary_reaction,
	degraded_equilibrium_residual,
	dimensionless_traction,
	evaluate_degraded_elastic_benchmark,
	plane_strain_tension_modulus,
)
from .elasticity import (
	ElasticBenchmarkMetrics,
	PlaneStrainTensionBenchmark,
	equilibrium_residual,
	evaluate_elastic_benchmark,
)
from .phase_field import (
	PhaseFieldBenchmarkMetrics,
	PhaseFieldConstraintViolations,
	PrecrackProfileBenchmark,
	evaluate_phase_field_benchmark,
	natural_phase_flux,
	phase_field_constraint_violations,
	phase_field_residual,
	uniform_history_phase,
)

__all__ = [
	"ElasticBenchmarkMetrics",
	"DegradedElasticBenchmarkMetrics",
	"CoupledPhaseRoot",
	"PhaseFieldBenchmarkMetrics",
	"PhaseFieldConstraintViolations",
	"PlaneStrainTensionBenchmark",
	"PrecrackProfileBenchmark",
	"UniformDegradedTensionBenchmark",
	"equilibrium_residual",
	"degraded_boundary_reaction",
	"degraded_equilibrium_residual",
	"continue_low_damage_root",
	"coupled_phase_residual",
	"coupled_phase_roots",
	"dimensionless_traction",
	"evaluate_elastic_benchmark",
	"evaluate_degraded_elastic_benchmark",
	"evaluate_phase_field_benchmark",
	"natural_phase_flux",
	"plane_strain_tension_modulus",
	"phase_field_constraint_violations",
	"phase_field_residual",
	"uniform_history_phase",
]
