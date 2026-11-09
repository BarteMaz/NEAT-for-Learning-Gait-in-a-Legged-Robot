"""World configuration: validation, applying it to a server, and its physical effect.

Values used in the tests are deliberately different from PyBullet's defaults
(dt = 1/240, 50 solver iterations, gravity (0, 0, 0) after connect), so a
configuration that is silently not applied cannot pass.
"""

import dataclasses

import pybullet as p
import pytest

from neat_walking_robot.simulation import (
    SimulationData,
    connect_and_configure_simulation,
    prepare_simulation,
)

CUSTOM = SimulationData(
    dt=1 / 500, gravity=(0.5, -0.25, -9.0), num_solver_iterations=17
)


def _engine_params(client: int) -> dict:
    return p.getPhysicsEngineParameters(physicsClientId=client)


def _assert_configured(client: int, config: SimulationData) -> None:
    params = _engine_params(client)
    assert params["fixedTimeStep"] == pytest.approx(config.dt)
    assert params["numSolverIterations"] == config.num_solver_iterations
    gravity = (
        params["gravityAccelerationX"],
        params["gravityAccelerationY"],
        params["gravityAccelerationZ"],
    )
    assert gravity == pytest.approx(config.gravity)


# --- SimulationData -----------------------------------------------------------------


def test_defaults_are_valid():
    config = SimulationData()
    assert config.dt == pytest.approx(1 / 240)
    assert config.gravity == pytest.approx((0.0, 0.0, -9.81))


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CUSTOM.dt = 0.1


@pytest.mark.parametrize("dt", [0.0, -1 / 240])
def test_non_positive_time_step_is_rejected(dt):
    with pytest.raises(ValueError):
        SimulationData(dt=dt)


@pytest.mark.parametrize("gravity", [(0.0, -9.81), (0.0, 0.0, -9.81, 0.0)])
def test_gravity_must_have_three_components(gravity):
    with pytest.raises(ValueError):
        SimulationData(gravity=gravity)


@pytest.mark.parametrize("iterations", [0, -5])
def test_non_positive_solver_iterations_are_rejected(iterations):
    with pytest.raises(ValueError):
        SimulationData(num_solver_iterations=iterations)


# --- Applying the configuration ------------------------------------------------------


def test_prepare_simulation_applies_every_parameter(client):
    prepare_simulation(CUSTOM, client)
    _assert_configured(client, CUSTOM)


def test_prepare_simulation_touches_only_the_given_server(two_clients):
    first, second = two_clients
    before = _engine_params(first)
    prepare_simulation(CUSTOM, second)
    assert _engine_params(first) == before
    _assert_configured(second, CUSTOM)


def test_reset_simulation_discards_the_configuration(client):
    # documents PyBullet's behaviour that the reset of an episode must account for
    prepare_simulation(CUSTOM, client)
    p.resetSimulation(physicsClientId=client)
    assert _engine_params(client)["gravityAccelerationZ"] == pytest.approx(0.0)


def test_configuration_can_be_reapplied_after_reset(client):
    prepare_simulation(CUSTOM, client)
    p.resetSimulation(physicsClientId=client)
    prepare_simulation(CUSTOM, client)
    _assert_configured(client, CUSTOM)


def test_connect_and_configure_returns_a_configured_server():
    client = connect_and_configure_simulation(CUSTOM)
    try:
        assert client >= 0
        assert p.isConnected(physicsClientId=client)
        _assert_configured(client, CUSTOM)
    finally:
        p.disconnect(client)


def test_connect_and_configure_uses_direct_mode_by_default():
    client = connect_and_configure_simulation(SimulationData())
    try:
        assert (
            p.getConnectionInfo(physicsClientId=client)["connectionMethod"] == p.DIRECT
        )
    finally:
        p.disconnect(client)


# --- Physical effect (the numbers are actually used by the engine) -----------------------


def _free_fall_velocity(
    client: int, seconds: float, dt: float
) -> tuple[float, float, float]:
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=client)
    ball = p.createMultiBody(
        1.0, shape, basePosition=[0, 0, 100], physicsClientId=client
    )
    # remove the default damping so the result can be compared with v = g * t (block 2 of stage 1)
    p.changeDynamics(
        ball, -1, linearDamping=0.0, angularDamping=0.0, physicsClientId=client
    )
    for _ in range(round(seconds / dt)):
        p.stepSimulation(physicsClientId=client)
    velocity, _ = p.getBaseVelocity(ball, physicsClientId=client)
    return velocity


def test_free_fall_follows_configured_gravity_and_time_step(client):
    prepare_simulation(CUSTOM, client)
    seconds = 0.5
    velocity = _free_fall_velocity(client, seconds, CUSTOM.dt)
    expected = tuple(g * seconds for g in CUSTOM.gravity)
    assert velocity == pytest.approx(expected, rel=1e-6)


def test_time_step_sets_simulated_time_per_step(client):
    # 0.5 s of simulated time takes 250 steps at 1/500 s; with a wrong dt the ball would fall faster or slower
    config = SimulationData(dt=1 / 500, gravity=(0.0, 0.0, -10.0))
    prepare_simulation(config, client)
    velocity = _free_fall_velocity(client, 0.5, config.dt)
    assert velocity[2] == pytest.approx(-5.0, rel=1e-6)
