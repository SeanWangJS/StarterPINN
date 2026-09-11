"""Continuum physics definitions for the Griffith phase-field model."""

from .degradation import quadratic_degradation, quadratic_degradation_derivative
from .elasticity import (
	AmorEnergySplit,
	amor_energy_split,
	degraded_elastic_energy,
	degraded_stress,
	undamaged_elastic_energy,
	undamaged_stress,
)
from .kinematics import PlaneStrainDecomposition, plane_strain_decomposition, small_strain
from .history import update_history
from .materials import IsotropicMaterial
from .phase_field import PhaseFieldEnergyDensity, PhaseFieldMaterial, at2_crack_density, frozen_history_energy_density
from .scaling import dimensional_history, nondimensional_history

__all__ = [
	"AmorEnergySplit",
	"IsotropicMaterial",
	"PhaseFieldEnergyDensity",
	"PhaseFieldMaterial",
	"PlaneStrainDecomposition",
	"amor_energy_split",
	"at2_crack_density",
	"degraded_elastic_energy",
	"degraded_stress",
	"dimensional_history",
	"frozen_history_energy_density",
	"nondimensional_history",
	"plane_strain_decomposition",
	"quadratic_degradation",
	"quadratic_degradation_derivative",
	"small_strain",
	"undamaged_elastic_energy",
	"undamaged_stress",
	"update_history",
]


