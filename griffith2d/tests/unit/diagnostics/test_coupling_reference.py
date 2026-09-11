import pytest

from griffith2d.diagnostics import (
    PlaneStrainTensionBenchmark,
    continue_low_damage_root,
    coupled_phase_roots,
    dimensionless_traction,
    plane_strain_tension_modulus,
)
from griffith2d.physics import IsotropicMaterial


def test_effective_modulus_matches_validated_plane_strain_solution() -> None:
    benchmark = PlaneStrainTensionBenchmark(
        1.0, 1.0, 0.1, IsotropicMaterial(young_modulus=1.0, poisson_ratio=0.3)
    )

    effective_modulus = plane_strain_tension_modulus(benchmark)

    assert effective_modulus == pytest.approx(benchmark.traction / benchmark.strain_xx)
    assert dimensionless_traction(0.1, effective_modulus, 2.4, 0.2) == pytest.approx(
        0.1 * (0.2 / (effective_modulus * 2.4)) ** 0.5
    )


def test_low_load_equation_exposes_all_three_roots_and_stability() -> None:
    roots = coupled_phase_roots(tau=0.1, residual_stiffness=1e-3)

    assert len(roots) == 3
    assert [root.fixed_point_stable for root in roots] == [True, False, True]
    assert roots[0].phase == pytest.approx(0.010294267348147625, abs=1e-13)
    assert roots[1].phase == pytest.approx(0.7675548009214062, abs=1e-13)
    assert roots[2].phase == pytest.approx(0.9999000079997098, abs=1e-13)
    assert max(abs(root.residual) for root in roots) < 1e-14


def test_continuation_selects_low_damage_branch_from_zero() -> None:
    previous_phase = 0.0
    selected_phases = []

    for tau in (0.0, 0.025, 0.05, 0.075, 0.1, 0.125):
        selected, _ = continue_low_damage_root(tau, 1e-3, previous_phase)
        selected_phases.append(selected.phase)
        previous_phase = selected.phase

    assert all(next_phase >= phase for phase, next_phase in zip(selected_phases, selected_phases[1:]))
    assert selected_phases[-1] == pytest.approx(0.01638503363387401, abs=1e-13)
    assert all(phase < 0.02 for phase in selected_phases)