import numpy as np
import pytest
import json
import base64
import hashlib
from pathlib import Path
import griffith2d.benchmarks.center_crack_fem as center_crack_fem
from skfem import Basis, ElementTriP1, ElementVector, MeshTri, asm
from skfem.models.elasticity import linear_elasticity

from griffith2d.benchmarks.center_crack_fem import (
    FrozenPhaseFEMResult,
    PlateFEMPathError,
    _amor_tangent,
    solve_frozen_precrack_fem,
    solve_frozen_phase_fem,
    solve_uncracked_plate_fem,
    solve_staggered_plate_fem,
    solve_plate_fem_path,
    transfer_active_energy_to_history_fem,
)
from griffith2d.discretization import CenterCrackPlate
from griffith2d.physics import IsotropicMaterial


@pytest.mark.parametrize("cells", [12, 24])
def test_plate_fem_path_is_irreversible_and_reproducible(cells: int) -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    loads = (0.0, 0.01, 0.02, 0.01, 0.0)
    first = solve_plate_fem_path(plate, material, loads, 1.0, 0.2, 1e-3, cells, cells)
    second = solve_plate_fem_path(plate, material, loads, 1.0, 0.2, 1e-3, cells, cells)
    assert tuple(step.imposed_displacement for step in first) == loads
    for index, (step, repeated) in enumerate(zip(first, second), start=1):
        assert step.load_index == index
        assert step.phase_violation <= 1e-12
        assert step.history_violation <= 1e-12
        assert step.state.displacement.free_residual_relative <= 1e-9
        assert step.state.phase.projected_gradient <= 1e-7
        np.testing.assert_allclose(step.state.phase.phase, repeated.state.phase.phase, atol=1e-12)
        np.testing.assert_allclose(step.state.history, repeated.state.history, atol=1e-12)
        assert step.fracture_energy == pytest.approx(repeated.fracture_energy, abs=1e-12)
        assert step.cumulative_work == pytest.approx(repeated.cumulative_work, abs=1e-12)
        if index > 1:
            previous = first[index - 2]
            expected_work = previous.cumulative_work + (
                previous.state.displacement.top_reaction + step.state.displacement.top_reaction
            ) * (step.imposed_displacement - previous.imposed_displacement)
            assert step.cumulative_work == pytest.approx(expected_work, abs=1e-14)
    np.testing.assert_allclose(first[-1].state.displacement.displacement, 0.0, atol=1e-14)
    assert first[-1].cumulative_work < first[2].cumulative_work


def test_plate_fem_reference_cli_exports_named_fields(tmp_path, monkeypatch) -> None:
    config = Path(__file__).resolve().parents[2] / "configs/references/center_crack_path.toml"
    output = tmp_path / "reference/path.json"
    failure_output = tmp_path / "failure.json"
    monkeypatch.setattr("sys.argv", [
        "center_crack_fem", "--config", str(config), "--output", str(output),
        "--failure-output", str(failure_output),
    ])
    center_crack_fem.main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert not failure_output.exists()
    assert payload["schema_version"] == 2
    provenance = payload["provenance"]
    assert provenance["runtime"]["randomness"] == "deterministic_FEM_no_random_initialization"
    snapshot = provenance["source_snapshot"]["src/griffith2d/benchmarks/center_crack_fem.py"]
    content = base64.b64decode(snapshot["base64"])
    assert hashlib.sha256(content).hexdigest() == snapshot["sha256"]
    assert content == Path(center_crack_fem.__file__).read_bytes()
    config_content = base64.b64decode(provenance["config_snapshot"]["base64"])
    assert config_content == config.read_bytes()
    assert hashlib.sha256(config_content).hexdigest() == provenance["config_snapshot"]["sha256"]
    triangles = np.asarray(payload["triangles"])
    assert triangles.shape[1] == 3
    assert triangles.min() >= 0
    assert triangles.max() < len(payload["nodes"])
    assert payload["status"] == "converged"
    assert payload["reference_scope"] == "nodal_history_proxy_not_certified_propagation"
    assert len(payload["steps"]) == 5
    assert set(payload["versions"]) == {"numpy", "scipy", "scikit-fem", "torch"}
    for step in payload["steps"]:
        assert len(step["phase"]) == len(payload["nodes"])
        assert len(step["history"]) == len(payload["nodes"])
        assert np.asarray(step["displacement"]).shape == np.asarray(payload["nodes"]).shape
        assert step["phase_violation"] <= 1e-12
        assert step["history_violation"] <= 1e-12
        assert step["total_energy"] == pytest.approx(step["elastic_energy"] + step["fracture_energy"])
    assert payload["steps"][0]["work_minus_energy_increment"] == 0.0


def test_plate_fem_reference_cli_does_not_write_success_after_budget_failure(tmp_path, monkeypatch) -> None:
    config = Path(__file__).resolve().parents[2] / "configs/references/center_crack_path.toml"
    output = tmp_path / "failed/path.json"
    solver = center_crack_fem.solve_staggered_plate_fem

    def exhaust_budget(*args, **kwargs):
        kwargs["max_iterations"] = 1
        return solver(*args, **kwargs)

    monkeypatch.setattr(center_crack_fem, "solve_staggered_plate_fem", exhaust_budget)
    monkeypatch.setattr("sys.argv", [
        "center_crack_fem", "--config", str(config), "--output", str(output),
    ])
    with pytest.raises(PlateFEMPathError) as caught:
        center_crack_fem.main()
    assert caught.value.load_index == 1
    assert caught.value.committed_steps == ()
    assert "iteration budget exhausted" in str(caught.value.__cause__)
    assert not output.exists()
    archives = list(output.parent.glob("*.failure.json"))
    assert len(archives) == 1
    failure = json.loads(archives[0].read_text(encoding="utf-8"))
    assert failure["status"] == "failed"
    assert failure["load_index"] == 1
    assert failure["committed_steps"] == []
    assert "iteration budget exhausted" in failure["cause"]


def test_reference_cli_archives_successful_prefix_and_preserves_old_output(tmp_path, monkeypatch) -> None:
    config = Path(__file__).resolve().parents[2] / "configs/references/center_crack_path.toml"
    output = tmp_path / "success.json"
    output.write_text("previous successful result", encoding="utf-8")
    archive = tmp_path / "failure.json"
    solver = center_crack_fem.solve_staggered_plate_fem
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-step failure")
        return solver(*args, **kwargs)

    monkeypatch.setattr(center_crack_fem, "solve_staggered_plate_fem", fail_second)
    monkeypatch.setattr("sys.argv", [
        "center_crack_fem", "--config", str(config), "--output", str(output),
        "--failure-output", str(archive),
    ])
    with pytest.raises(PlateFEMPathError):
        center_crack_fem.main()
    failure = json.loads(archive.read_text(encoding="utf-8"))
    assert failure["load_index"] == 2
    assert len(failure["committed_steps"]) == 1
    assert failure["committed_steps"][0]["load_index"] == 1
    assert len(failure["committed_steps"][0]["state"]["phase"]["phase"]) > 0
    assert "injected second-step failure" in failure["cause"]
    assert failure["provenance"]["source_snapshot"]
    assert failure["provenance"]["runtime"]["executable"]
    assert output.read_text(encoding="utf-8") == "previous successful result"
    assert calls == 2


def test_reference_provenance_snapshots_all_sources_without_git(tmp_path, monkeypatch) -> None:
    package = tmp_path / "project/src/griffith2d"
    entry = package / "benchmarks/center_crack_fem.py"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"pass\n")
    untracked = package / "untracked.py"
    untracked.write_bytes(b"# coding: latin-1\nvalue = '\xe9'\n")
    build = tmp_path / "project/pyproject.toml"
    build.write_bytes(b'[project]\nname = "snapshot-fixture"\n')
    config = tmp_path / "reference.toml"
    config.write_bytes(b'[benchmark]\nkind = "center_crack_fem_path"\n')

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(center_crack_fem, "__file__", str(entry))
    monkeypatch.setattr(center_crack_fem.subprocess, "run", missing_git)
    provenance = center_crack_fem._reference_provenance(config)
    assert provenance["git"]["available"] is False
    assert "git unavailable" in provenance["git"]["error"]
    for name, path in {
        "src/griffith2d/benchmarks/center_crack_fem.py": entry,
        "src/griffith2d/untracked.py": untracked,
        "pyproject.toml": build,
    }.items():
        snapshot = provenance["source_snapshot"][name]
        restored = base64.b64decode(snapshot["base64"])
        assert restored == path.read_bytes()
        assert hashlib.sha256(restored).hexdigest() == snapshot["sha256"]


def test_reference_run_uses_captured_config_when_file_changes(tmp_path) -> None:
    original = Path(__file__).resolve().parents[2] / "configs/references/center_crack_path.toml"
    config = tmp_path / "config.toml"
    config.write_bytes(original.read_bytes())
    provenance = center_crack_fem._reference_provenance(config)
    config.write_text("invalid TOML [", encoding="utf-8")
    payload = center_crack_fem.run(config, provenance=provenance)
    assert payload["status"] == "converged"
    assert len(payload["steps"]) == 5
    assert base64.b64decode(payload["provenance"]["config_snapshot"]["base64"]) == original.read_bytes()


def test_reference_cli_rejects_existing_failure_archive_before_solving(tmp_path, monkeypatch) -> None:
    archive = tmp_path / "failure.json"
    archive.write_text("existing evidence", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "center_crack_fem", "--config", "unused.toml", "--failure-output", str(archive),
    ])
    with pytest.raises(SystemExit) as caught:
        center_crack_fem.main()
    assert caught.value.code == 2
    assert archive.read_text(encoding="utf-8") == "existing evidence"


def test_reference_cli_rejects_success_failure_path_collision(tmp_path, monkeypatch) -> None:
    output = tmp_path / "same.json"
    monkeypatch.setattr("sys.argv", [
        "center_crack_fem", "--config", "unused.toml", "--output", str(output),
        "--failure-output", str(output),
    ])
    with pytest.raises(SystemExit) as caught:
        center_crack_fem.main()
    assert caught.value.code == 2
    assert not output.exists()


def test_reference_cli_archives_invalid_config_before_any_step(tmp_path, monkeypatch) -> None:
    config = tmp_path / "invalid.toml"
    config.write_text("not valid TOML [", encoding="utf-8")
    output = tmp_path / "success.json"
    archive = tmp_path / "failure.json"
    monkeypatch.setattr("sys.argv", [
        "center_crack_fem", "--config", str(config), "--output", str(output),
        "--failure-output", str(archive),
    ])
    with pytest.raises(ValueError):
        center_crack_fem.main()
    failure = json.loads(archive.read_text(encoding="utf-8"))
    assert failure["exception_type"] == "TOMLDecodeError"
    assert failure["load_index"] is None
    assert failure["committed_steps"] == []
    assert base64.b64decode(failure["provenance"]["config_snapshot"]["base64"]) == config.read_bytes()
    assert not output.exists()


def test_reference_cli_preserves_solver_error_when_archive_io_fails(tmp_path, monkeypatch) -> None:
    config = Path(__file__).resolve().parents[2] / "configs/references/center_crack_path.toml"
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    archive = blocked / "failure.json"
    output = tmp_path / "success.json"

    def fail_path(*args, **kwargs):
        raise RuntimeError("original solver error")

    monkeypatch.setattr(center_crack_fem, "solve_plate_fem_path", fail_path)
    monkeypatch.setattr("sys.argv", [
        "center_crack_fem", "--config", str(config), "--output", str(output),
        "--failure-output", str(archive),
    ])
    with pytest.raises(RuntimeError, match="original solver error") as caught:
        center_crack_fem.main()
    assert "Failure archive could not be saved" in caught.value.__notes__[0]
    assert not output.exists()


def test_plate_fem_path_stops_at_failure_with_unchanged_committed_prefix(monkeypatch) -> None:
    solver = center_crack_fem.solve_staggered_plate_fem
    calls = []
    snapshots = []

    def fail_second_step(*args, **kwargs):
        calls.append(args[2])
        if len(calls) == 2:
            raise RuntimeError("injected path failure")
        state = solver(*args, **kwargs)
        snapshots.append((state.phase.phase.copy(), state.history.copy()))
        return state

    monkeypatch.setattr(center_crack_fem, "solve_staggered_plate_fem", fail_second_step)
    with pytest.raises(PlateFEMPathError) as caught:
        solve_plate_fem_path(
            CenterCrackPlate(2.0, 1.0, 0.5), IsotropicMaterial(1.0, 0.3),
            (0.0, 0.01, 0.02), 1.0, 0.2, 1e-3, 12, 12,
        )
    failure = caught.value
    assert failure.load_index == 2
    assert failure.imposed_displacement == 0.01
    assert len(failure.committed_steps) == 1
    assert calls == [0.0, 0.01]
    assert "injected path failure" in str(failure.__cause__)
    np.testing.assert_array_equal(failure.committed_steps[0].state.phase.phase, snapshots[0][0])
    np.testing.assert_array_equal(failure.committed_steps[0].state.history, snapshots[0][1])


@pytest.mark.parametrize("loads", [(), (0.01,), (0.0, -0.1), (0.0, float("nan")), (False, 0.01)])
def test_plate_fem_path_validates_all_loads_before_solving(loads: tuple, monkeypatch) -> None:
    def unexpected_solver(*args, **kwargs):
        pytest.fail("invalid path must be rejected before any load step runs")

    monkeypatch.setattr(center_crack_fem, "solve_staggered_plate_fem", unexpected_solver)
    with pytest.raises(ValueError, match="imposed_displacements"):
        solve_plate_fem_path(
            CenterCrackPlate(2.0, 1.0, 0.5), IsotropicMaterial(1.0, 0.3),
            loads, 1.0, 0.2, 1e-3, 12, 12,
        )


@pytest.mark.parametrize("load", [0.0, 0.01])
@pytest.mark.parametrize("cells", [12, 24])
def test_staggered_plate_fixed_load_returns_consistent_fields(load: float, cells: int) -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    result = solve_staggered_plate_fem(plate, material, load, 1.0, 0.2, 1e-3, cells, cells)
    assert len(result.records) >= 2
    assert result.displacement.free_residual_relative < 1e-9
    assert result.phase.projected_gradient <= 1e-7
    np.testing.assert_array_equal(result.phase.nodes, result.displacement.nodes)
    recovered = transfer_active_energy_to_history_fem(
        result.displacement.nodes, result.displacement.displacement, material
    )
    np.testing.assert_allclose(result.history, recovered, atol=1e-14, rtol=0.0)
    repeated_phase = solve_frozen_phase_fem(plate, 1.0, 0.2, result.history, cells, cells)
    np.testing.assert_allclose(result.phase.phase, repeated_phase.phase, atol=1e-7, rtol=0.0)
    repeated_displacement = solve_frozen_precrack_fem(
        plate, material, load, 0.2, 1e-3, cells, cells,
        frozen_phase=result.phase, quadrature_order=4,
    )
    np.testing.assert_allclose(
        result.displacement.displacement, repeated_displacement.displacement,
        atol=1e-12, rtol=0.0,
    )
    final = result.records[-1]
    assert final.phase_change <= 1e-7
    assert final.displacement_relative_change <= 1e-7
    assert final.history_relative_change <= 1e-7
    assert final.energy_relative_change <= 1e-7


def test_staggered_plate_keeps_committed_bounds_during_unloading(monkeypatch) -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    loaded = solve_staggered_plate_fem(plate, material, 0.01, 1.0, 0.2, 1e-3, 12, 12)
    phase_snapshot = loaded.phase.phase.copy()
    history_snapshot = loaded.history.copy()
    phase_bounds = []
    history_bounds = []
    phase_solver = center_crack_fem.solve_frozen_phase_fem
    transfer = center_crack_fem.transfer_active_energy_to_history_fem

    def track_phase(*args, **kwargs):
        bound = kwargs.get("previous_phase")
        if bound is not None:
            phase_bounds.append(bound.copy())
        return phase_solver(*args, **kwargs)

    def track_history(*args, **kwargs):
        history_bounds.append(kwargs["previous_history"].copy())
        return transfer(*args, **kwargs)

    monkeypatch.setattr(center_crack_fem, "solve_frozen_phase_fem", track_phase)
    monkeypatch.setattr(center_crack_fem, "transfer_active_energy_to_history_fem", track_history)
    unloaded = solve_staggered_plate_fem(
        plate, material, 0.0, 1.0, 0.2, 1e-3, 12, 12,
        previous_phase=loaded.phase, previous_history=loaded.history,
    )
    for bound in phase_bounds:
        np.testing.assert_array_equal(bound, phase_snapshot)
    for bound in history_bounds:
        np.testing.assert_array_equal(bound, history_snapshot)
    np.testing.assert_array_equal(unloaded.history, history_snapshot)
    assert np.all(unloaded.phase.phase >= phase_snapshot - 1e-12)
    np.testing.assert_allclose(unloaded.displacement.displacement, 0.0, atol=1e-14)
    assert unloaded.displacement.top_reaction == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_array_equal(loaded.phase.phase, phase_snapshot)
    np.testing.assert_array_equal(loaded.history, history_snapshot)


def test_staggered_plate_failure_does_not_modify_previous_state(monkeypatch) -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    previous = solve_staggered_plate_fem(plate, material, 0.01, 1.0, 0.2, 1e-3, 12, 12)
    phase_snapshot = previous.phase.phase.copy()
    history_snapshot = previous.history.copy()
    with pytest.raises(RuntimeError, match="iteration budget exhausted"):
        solve_staggered_plate_fem(
            plate, material, 0.02, 1.0, 0.2, 1e-3, 12, 12,
            previous_phase=previous.phase, previous_history=previous.history,
            max_iterations=1,
        )
    np.testing.assert_array_equal(previous.phase.phase, phase_snapshot)
    np.testing.assert_array_equal(previous.history, history_snapshot)

    displacement_solver = center_crack_fem.solve_frozen_precrack_fem
    calls = 0

    def fail_displacement(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected displacement failure")
        return displacement_solver(*args, **kwargs)

    monkeypatch.setattr(center_crack_fem, "solve_frozen_precrack_fem", fail_displacement)
    with pytest.raises(RuntimeError, match="injected displacement failure"):
        solve_staggered_plate_fem(
            plate, material, 0.02, 1.0, 0.2, 1e-3, 12, 12,
            previous_phase=previous.phase, previous_history=previous.history,
        )
    np.testing.assert_array_equal(previous.phase.phase, phase_snapshot)
    np.testing.assert_array_equal(previous.history, history_snapshot)


@pytest.mark.parametrize("options", [
    {"max_iterations": 0}, {"max_iterations": 1.5},
    {"phase_tolerance": float("nan")}, {"relative_tolerance": 0.0},
])
def test_staggered_plate_rejects_invalid_convergence_options(options: dict) -> None:
    with pytest.raises(ValueError):
        solve_staggered_plate_fem(
            CenterCrackPlate(2.0, 1.0, 0.5), IsotropicMaterial(1.0, 0.3),
            0.01, 1.0, 0.2, 1e-3, 12, 12, **options,
        )


@pytest.mark.parametrize("phase_value", [0.0, 0.4, 1.0])
def test_constant_phase_return_scales_tensile_reaction_and_energy(phase_value: float) -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    baseline = solve_frozen_phase_fem(plate, 1.0, 0.2, 0.0, 12, 12)
    frozen = FrozenPhaseFEMResult(
        baseline.nodes, np.full(len(baseline.nodes), phase_value), 0.0
    )
    result = solve_frozen_precrack_fem(
        plate, material, 0.01, 0.2, 1e-3, 12, 12, frozen_phase=frozen
    )
    effective_modulus = material.young_modulus / (1.0 - material.poisson_ratio**2)
    expected_reaction = ((1.0 - phase_value)**2 + 1e-3) * effective_modulus * 0.01 * 4.0
    assert result.top_reaction == pytest.approx(expected_reaction, rel=1e-10)
    assert result.strain_energy == pytest.approx(0.01 * expected_reaction, rel=1e-10)
    expected = np.column_stack((
        -material.poisson_ratio / (1.0 - material.poisson_ratio) * 0.01 * result.nodes[:, 0],
        0.01 * result.nodes[:, 1],
    ))
    np.testing.assert_allclose(result.displacement, expected, atol=1e-12, rtol=0.0)
    assert result.free_residual_relative < 1e-9


def test_nonuniform_phase_feedback_matches_integrated_amor_energy() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    initial = solve_frozen_phase_fem(plate, 1.0, 0.2, 0.0, 12, 12)
    first = solve_frozen_precrack_fem(
        plate, material, 0.01, 0.2, 1e-3, 12, 12, frozen_phase=initial
    )
    history = transfer_active_energy_to_history_fem(first.nodes, first.displacement, material)
    updated = solve_frozen_phase_fem(
        plate, 1.0, 0.2, history, 12, 12, previous_phase=initial.phase
    )
    phase_snapshot = updated.phase.copy()
    feedback = solve_frozen_precrack_fem(
        plate, material, 0.01, 0.2, 1e-3, 12, 12,
        frozen_phase=updated, quadrature_order=4,
    )
    np.testing.assert_array_equal(feedback.nodes, updated.nodes)
    assert np.all(updated.phase >= initial.phase - 1e-12)
    assert updated.projected_gradient < 1e-7
    assert feedback.free_residual_relative < 1e-9

    mesh = MeshTri.init_tensor(
        np.unique(feedback.nodes[:, 0]), np.unique(feedback.nodes[:, 1])
    )
    vector_basis = Basis(mesh, ElementVector(ElementTriP1()), intorder=4)
    scalar_basis = Basis(mesh, ElementTriP1(), quadrature=vector_basis.quadrature)
    dofs = np.zeros(vector_basis.N)
    dofs[vector_basis.nodal_dofs] = feedback.displacement.T
    gradient = vector_basis.interpolate(dofs).grad
    strain = 0.5 * (gradient + np.swapaxes(gradient, 0, 1))
    trace = np.trace(strain, axis1=0, axis2=1)
    active = (
        0.5 * material.bulk_modulus * np.maximum(trace, 0.0)**2
        + material.shear_modulus * (np.sum(strain**2, axis=(0, 1)) - trace**2 / 3.0)
    )
    inactive = 0.5 * material.bulk_modulus * np.minimum(trace, 0.0)**2
    phase_quadrature = scalar_basis.interpolate(updated.phase)
    density = ((1.0 - phase_quadrature)**2 + 1e-3) * active + inactive
    integral = float(np.sum(density * vector_basis.dx))
    assert feedback.strain_energy == pytest.approx(integral, rel=1e-12)
    assert feedback.strain_energy == pytest.approx(0.01 * feedback.top_reaction, rel=1e-10)
    np.testing.assert_array_equal(updated.phase, phase_snapshot)


def test_phase_feedback_rejects_mismatched_nodes_and_invalid_values() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    phase = solve_frozen_phase_fem(plate, 1.0, 0.2, 0.0, 12, 12)
    invalid_fields = (
        FrozenPhaseFEMResult(phase.nodes[::-1], phase.phase[::-1], 0.0),
        FrozenPhaseFEMResult(phase.nodes, phase.phase[:, None], 0.0),
        FrozenPhaseFEMResult(phase.nodes, np.full(len(phase.nodes), np.nan), 0.0),
        FrozenPhaseFEMResult(phase.nodes, np.full(len(phase.nodes), 1.1), 0.0),
        FrozenPhaseFEMResult(phase.nodes, np.full(len(phase.nodes), -0.1), 0.0),
    )
    for invalid in invalid_fields:
        with pytest.raises(ValueError, match="frozen_phase"):
            solve_frozen_precrack_fem(
                plate, material, 0.01, 0.2, 1e-3, 12, 12, frozen_phase=invalid
            )


@pytest.mark.parametrize("gradient", [
    np.array([[0.01, 0.0], [0.0, 0.02]]),
    np.array([[-0.01, 0.0], [0.0, -0.02]]),
    np.array([[0.0, 0.03], [0.0, 0.0]]),
])
def test_active_energy_transfer_recovers_affine_plane_strain(gradient: np.ndarray) -> None:
    mesh = MeshTri.init_tensor(np.linspace(-2.0, 2.0, 5), np.linspace(-1.0, 1.0, 5))
    nodes = mesh.p.T.copy()
    material = IsotropicMaterial(1.0, 0.3)
    strain = 0.5 * (gradient + gradient.T)
    trace = np.trace(strain)
    expected = (
        0.5 * material.bulk_modulus * max(trace, 0.0)**2
        + material.shear_modulus * (np.sum(strain**2) - trace**2 / 3.0)
    )
    history = transfer_active_energy_to_history_fem(nodes, nodes @ gradient.T, material)
    np.testing.assert_allclose(history, expected, atol=1e-14, rtol=0.0)
    previous = history + 0.1
    unloaded = transfer_active_energy_to_history_fem(
        nodes, np.zeros_like(nodes), material, previous_history=previous
    )
    np.testing.assert_array_equal(unloaded, previous)
    np.testing.assert_array_equal(previous, history + 0.1)


def test_active_energy_transfer_is_positive_and_integral_preserving() -> None:
    mesh = MeshTri.init_tensor(np.array([-2.0, -0.5, 0.0, 2.0]), np.array([-1.0, 0.0, 0.2, 1.0]))
    nodes = mesh.p.T.copy()
    displacement = np.column_stack((0.01 * nodes[:, 0]**2, 0.02 * nodes[:, 1]**2))
    material = IsotropicMaterial(1.0, 0.3)
    lumped_mass = np.zeros(len(nodes))
    total_active_energy = 0.0
    for triangle in mesh.t.T:
        positions = nodes[triangle]
        affine = np.column_stack((np.ones(3), positions))
        coefficients = np.linalg.solve(affine, displacement[triangle])
        gradient = coefficients[1:].T
        strain = 0.5 * (gradient + gradient.T)
        trace = np.trace(strain)
        active = (
            0.5 * material.bulk_modulus * max(trace, 0.0)**2
            + material.shear_modulus * (np.sum(strain**2) - trace**2 / 3.0)
        )
        area = 0.5 * abs(np.linalg.det(positions[1:] - positions[0]))
        lumped_mass[triangle] += area / 3.0
        total_active_energy += area * active

    history = transfer_active_energy_to_history_fem(nodes, displacement, material)
    assert np.all(history >= 0.0)
    assert np.ptp(history) > 0.0
    assert lumped_mass @ history == pytest.approx(total_active_energy, rel=1e-12)
    for bad_nodes, bad_displacement in (
        (nodes[::-1], displacement[::-1]), (nodes, displacement[:-1]),
    ):
        with pytest.raises(ValueError):
            transfer_active_energy_to_history_fem(bad_nodes, bad_displacement, material)
    for invalid in (np.zeros((len(nodes), 1)), np.full(len(nodes), -1.0),
                    np.full(len(nodes), np.nan)):
        with pytest.raises(ValueError, match="previous_history"):
            transfer_active_energy_to_history_fem(
                nodes, displacement, material, previous_history=invalid
            )


def test_frozen_precrack_displacement_drives_phase_on_identical_mesh() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    displacement = solve_frozen_precrack_fem(plate, material, 0.01, 0.2, 1e-3, 12, 12)
    history = transfer_active_energy_to_history_fem(
        displacement.nodes, displacement.displacement, material
    )
    phase = solve_frozen_phase_fem(plate, 1.0, 0.2, history, 12, 12)
    np.testing.assert_array_equal(phase.nodes, displacement.nodes)
    distance = np.hypot(
        np.maximum(np.abs(phase.nodes[:, 0]) - 0.5, 0.0), phase.nodes[:, 1]
    )
    assert np.all(phase.phase >= np.exp(-distance / 0.2) - 1e-12)
    assert np.all(phase.phase <= 1.0 + 1e-12)
    assert phase.projected_gradient < 1e-7
    assert np.ptp(history) > 0.0


def test_independent_fem_recovers_uncracked_plane_strain_plate() -> None:
    plate = CenterCrackPlate(half_width=2.0, half_height=1.0, crack_half_length=0.5)
    material = IsotropicMaterial(young_modulus=1.0, poisson_ratio=0.3)
    imposed_displacement = 0.01
    lame = material.young_modulus * material.poisson_ratio / (
        (1.0 + material.poisson_ratio) * (1.0 - 2.0 * material.poisson_ratio)
    )
    vertical_strain = imposed_displacement / plate.half_height
    horizontal_strain = -lame * vertical_strain / (lame + 2.0 * material.shear_modulus)
    effective_modulus = 4.0 * material.shear_modulus * (lame + material.shear_modulus) / (
        lame + 2.0 * material.shear_modulus
    )
    reaction = effective_modulus * vertical_strain * 2.0 * plate.half_width
    energy = reaction * imposed_displacement

    for cells in (4, 8, 16):
        result = solve_uncracked_plate_fem(
            plate, material, imposed_displacement, cells_x=cells, cells_y=cells
        )
        np.testing.assert_allclose(
            result.displacement,
            np.stack(
                (horizontal_strain * result.nodes[:, 0],
                 vertical_strain * result.nodes[:, 1]),
                axis=-1,
            ),
            atol=1e-12,
            rtol=0.0,
        )
        assert result.top_reaction == pytest.approx(reaction, abs=1e-12)
        assert result.strain_energy == pytest.approx(energy, abs=1e-12)


def test_reference_mesh_requires_center_node_and_finite_load() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)

    with pytest.raises(ValueError, match="cells_x"):
        solve_uncracked_plate_fem(plate, material, 0.01, cells_x=3, cells_y=4)
    with pytest.raises(ValueError, match="imposed_displacement"):
        solve_uncracked_plate_fem(plate, material, float("inf"), cells_x=4, cells_y=4)


@pytest.mark.parametrize("cells", [12, 40, 64])
def test_frozen_precrack_reference_resolves_free_equilibrium(cells: int) -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    result = solve_frozen_precrack_fem(
        plate, material, imposed_displacement=0.01,
        length_scale=0.2, residual_stiffness=1e-3, cells_x=cells, cells_y=cells,
    )
    uncracked = solve_uncracked_plate_fem(
        plate, material, imposed_displacement=0.01, cells_x=cells, cells_y=cells
    )

    assert result.free_residual_relative < 1e-9
    assert 0.0 < result.top_reaction < uncracked.top_reaction
    assert result.strain_energy > 0.0
    assert result.iterations <= 30
    assert np.diff(np.unique(result.nodes[:, 1])).min() > 1e-3


def test_amor_tangent_matches_independent_linear_elastic_forms() -> None:
    basis = Basis(
        MeshTri.init_tensor(np.linspace(-1.0, 1.0, 3), np.linspace(-1.0, 1.0, 3)),
        ElementVector(ElementTriP1()),
    )
    material = IsotropicMaterial(1.0, 0.3)
    bulk, shear = material.bulk_modulus, material.shear_modulus
    degradation = 0.35
    values = np.full((basis.mesh.nelements, len(basis.W)), degradation)

    tensile = asm(
        _amor_tangent, basis, bulk_modulus=bulk, shear_modulus=shear,
        degradation=values, volume_degradation=values,
    )
    compressive = asm(
        _amor_tangent, basis, bulk_modulus=bulk, shear_modulus=shear,
        degradation=values, volume_degradation=np.ones_like(values),
    )
    lame = bulk - 2.0 * shear / 3.0
    np.testing.assert_allclose(
        tensile.toarray(),
        degradation * asm(linear_elasticity(Lambda=lame, Mu=shear), basis).toarray(),
        atol=1e-14, rtol=0.0,
    )
    np.testing.assert_allclose(
        compressive.toarray(),
        asm(
            linear_elasticity(
                Lambda=bulk - 2.0 * degradation * shear / 3.0,
                Mu=degradation * shear,
            ),
            basis,
        ).toarray(),
        atol=1e-14, rtol=0.0,
    )


def test_frozen_precrack_reaction_and_energy_change_less_under_refinement() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    solutions = [
        solve_frozen_precrack_fem(plate, material, 0.01, 0.2, 1e-3, cells, cells)
        for cells in (12, 24, 40, 64)
    ]
    reactions = [solution.top_reaction for solution in solutions]
    energies = [solution.strain_energy for solution in solutions]

    assert 0.0 < reactions[3] < reactions[2] < reactions[1] < reactions[0]
    assert reactions[0] - reactions[1] > reactions[1] - reactions[2] > reactions[2] - reactions[3] > 0.0
    assert energies[0] - energies[1] > energies[1] - energies[2] > energies[2] - energies[3] > 0.0
    assert all(solution.free_residual_relative < 1e-9 for solution in solutions)
    for solution in solutions:
        assert solution.strain_energy == pytest.approx(
            0.01 * solution.top_reaction, rel=1e-10
        )
    for point in (np.array([0.0, 0.2]), np.array([0.5, 0.2])):
        displacements = []
        for solution in solutions:
            distances_squared = np.sum((solution.nodes - point) ** 2, axis=1)
            assert distances_squared.min() == pytest.approx(0.0, abs=1e-25)
            displacements.append(solution.displacement[int(np.argmin(distances_squared))])
        assert np.linalg.norm(displacements[1] - displacements[2]) > np.linalg.norm(
            displacements[2] - displacements[3]
        )


def test_frozen_precrack_quadrature_refinement_and_zero_load() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    material = IsotropicMaterial(1.0, 0.3)
    solutions = [
        solve_frozen_precrack_fem(
            plate, material, 0.01, 0.2, 1e-3, 24, 24, quadrature_order=order
        )
        for order in (2, 4, 6)
    ]
    reactions = [solution.top_reaction for solution in solutions]
    assert abs(reactions[1] - reactions[2]) < 0.1 * abs(reactions[0] - reactions[1])
    assert all(solution.free_residual_relative < 1e-9 for solution in solutions)

    unloaded = solve_frozen_precrack_fem(plate, material, 0.0, 0.2, 1e-3, 12, 12)
    np.testing.assert_allclose(unloaded.displacement, 0.0, atol=1e-14)
    assert unloaded.top_reaction == pytest.approx(0.0, abs=1e-14)
    assert unloaded.strain_energy == pytest.approx(0.0, abs=1e-14)


def test_frozen_phase_fem_preserves_precrack_at_zero_history() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    result = solve_frozen_phase_fem(plate, 1.0, 0.2, 0.0, 12, 12)
    distance = np.hypot(
        np.maximum(np.abs(result.nodes[:, 0]) - 0.5, 0.0), result.nodes[:, 1]
    )
    initial = np.exp(-distance / 0.2)
    assert np.all(result.phase >= initial - 1e-12)
    assert np.all(result.phase <= 1.0 + 1e-12)
    assert np.all(result.phase[distance == 0.0] == 1.0)
    assert result.projected_gradient < 1e-7


def test_frozen_phase_fem_matches_uniform_at2_solution_without_precrack() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    for history in (0.0, 0.5, 2.0):
        result = solve_frozen_phase_fem(
            plate, 1.0, 0.2, history, 12, 12, initial_precrack=False
        )
        expected = 2.0 * history / (1.0 / 0.2 + 2.0 * history)
        np.testing.assert_allclose(result.phase, expected, atol=1e-6, rtol=0.0)
        assert result.projected_gradient < 1e-7


def test_frozen_phase_fem_interpolates_nodal_history_without_broadcasting() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    uniform = solve_frozen_phase_fem(
        plate, 1.0, 0.2, 0.6, 12, 12, initial_precrack=False
    )
    constant_nodes = solve_frozen_phase_fem(
        plate, 1.0, 0.2, np.full(len(uniform.nodes), 0.6), 12, 12,
        initial_precrack=False,
    )
    np.testing.assert_array_equal(uniform.nodes, constant_nodes.nodes)
    np.testing.assert_allclose(uniform.phase, constant_nodes.phase, atol=1e-10)

    asymmetric = solve_frozen_phase_fem(
        plate, 1.0, 0.2, 0.6 + 0.2 * uniform.nodes[:, 0], 12, 12,
        initial_precrack=False,
    )
    middle = np.isclose(asymmetric.nodes[:, 1], 0.0)
    right = middle & np.isclose(asymmetric.nodes[:, 0], plate.half_width)
    left = middle & np.isclose(asymmetric.nodes[:, 0], -plate.half_width)
    assert asymmetric.phase[right].item() > asymmetric.phase[left].item()
    assert asymmetric.projected_gradient < 1e-7

    for invalid in (np.array([0.6]), np.full(len(uniform.nodes), -0.1)):
        with pytest.raises(ValueError, match="history"):
            solve_frozen_phase_fem(
                plate, 1.0, 0.2, invalid, 12, 12, initial_precrack=False
            )


def test_nonuniform_history_solution_is_stationary_for_at2_energy() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    reference = solve_frozen_phase_fem(
        plate, 1.0, 0.2, 0, 12, 12, initial_precrack=False
    )
    history_nodes = 0.6 + 0.2 * reference.nodes[:, 0]
    result = solve_frozen_phase_fem(
        plate, 1.0, 0.2, history_nodes, 12, 12, initial_precrack=False
    )
    mesh = MeshTri.init_tensor(
        np.unique(result.nodes[:, 0]), np.unique(result.nodes[:, 1])
    )
    np.testing.assert_array_equal(mesh.p.T, result.nodes)
    basis = Basis(mesh, ElementTriP1(), intorder=4)
    history_quadrature = basis.interpolate(history_nodes)

    def energy(phase_nodes: np.ndarray) -> float:
        phase = basis.interpolate(phase_nodes)
        density = (
            (1.0 - phase) ** 2 * history_quadrature
            + phase**2 / 0.4
            + 0.1 * np.sum(phase.grad**2, axis=0)
        )
        return float(np.sum(density * basis.dx))

    for point in (np.array([0.0, 0.0]), np.array([0.5, 0.2])):
        index = int(np.argmin(np.sum((result.nodes - point) ** 2, axis=1)))
        assert 0.0 < result.phase[index] < 1.0
        positive = result.phase.copy()
        negative = result.phase.copy()
        positive[index] += 1e-5
        negative[index] -= 1e-5
        assert abs((energy(positive) - energy(negative)) / 2e-5) < 5e-7


def test_nonuniform_history_precrack_phase_refines_without_healing() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    tip_band_phase = []
    for cells in (12, 24, 40, 64):
        baseline = solve_frozen_phase_fem(plate, 1.0, 0.2, 0.0, cells, cells)
        history_nodes = 0.6 + 0.2 * baseline.nodes[:, 0]
        result = solve_frozen_phase_fem(
            plate, 1.0, 0.2, history_nodes, cells, cells
        )
        np.testing.assert_array_equal(result.nodes, baseline.nodes)
        assert np.all(result.phase >= baseline.phase - 1e-12)
        assert np.all(result.phase <= 1.0 + 1e-12)
        assert result.projected_gradient < 1e-7
        tip_band = np.sum((result.nodes - np.array([0.5, 0.2])) ** 2, axis=1)
        assert tip_band.min() == pytest.approx(0.0, abs=1e-25)
        tip_band_phase.append(result.phase[int(np.argmin(tip_band))])

    assert abs(tip_band_phase[2] - tip_band_phase[3]) < abs(
        tip_band_phase[1] - tip_band_phase[2]
    )


def test_frozen_phase_fem_does_not_heal_on_unloading() -> None:
    plate = CenterCrackPlate(2.0, 1.0, 0.5)
    loaded = solve_frozen_phase_fem(plate, 1.0, 0.2, 2.0, 12, 12)
    unloaded = solve_frozen_phase_fem(
        plate, 1.0, 0.2, 0.0, 12, 12, previous_phase=loaded.phase
    )
    np.testing.assert_array_equal(unloaded.nodes, loaded.nodes)
    assert np.all(unloaded.phase >= loaded.phase - 1e-12)
    assert np.any(loaded.phase > 0.5)
    assert unloaded.projected_gradient < 1e-7

    with pytest.raises(ValueError, match="mesh node order"):
        solve_frozen_phase_fem(
            plate, 1.0, 0.2, 0.0, 12, 12, previous_phase=loaded.phase[:-1]
        )