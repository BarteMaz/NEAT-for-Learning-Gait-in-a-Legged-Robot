"""Environment factory and the public API of the environments package."""

import pybullet as p
import pytest

from neat_walking_robot.environments import (
    EnvironmentType,
    FlatEnvironment,
    RoughEnvironment,
    SlopeEnvironment,
    create_environment,
)

# One entry per environment type: (expected class, name of the params fixture).
EXPECTED = {
    EnvironmentType.FLAT: (FlatEnvironment, "make_base_params"),
    EnvironmentType.ROUGH: (RoughEnvironment, "make_rough_params"),
    EnvironmentType.SLOPE: (SlopeEnvironment, "make_slope_params"),
}


def test_every_environment_type_is_covered_by_this_test():
    # adding a type to the enum without a test entry (or registry entry) must fail loudly
    assert set(EXPECTED) == set(EnvironmentType)


@pytest.mark.parametrize(
    "env_type", list(EnvironmentType), ids=lambda t: t.name.lower()
)
def test_factory_builds_the_matching_class(env_type, request, client):
    env_class, params_fixture = EXPECTED[env_type]
    params = request.getfixturevalue(params_fixture)()
    env = create_environment(env_type, params, client, seed=0)
    assert type(env) is env_class
    assert env.params is params
    assert env.client == client


@pytest.mark.parametrize(
    ("env_type", "wrong_params_fixture"),
    [
        # subclasses of BaseEnvironmentData must not be accepted by the flat environment
        (EnvironmentType.FLAT, "make_rough_params"),
        (EnvironmentType.FLAT, "make_slope_params"),
        (EnvironmentType.ROUGH, "make_base_params"),
        (EnvironmentType.ROUGH, "make_slope_params"),
        (EnvironmentType.SLOPE, "make_base_params"),
        (EnvironmentType.SLOPE, "make_rough_params"),
    ],
)
def test_factory_rejects_params_of_another_environment(
    env_type, wrong_params_fixture, request, client
):
    params = request.getfixturevalue(wrong_params_fixture)()
    with pytest.raises(TypeError):
        create_environment(env_type, params, client, seed=0)


def test_rejected_params_build_nothing(make_rough_params, client):
    # the type check must happen before any body is created
    with pytest.raises(TypeError):
        create_environment(EnvironmentType.FLAT, make_rough_params(), client, seed=0)
    assert p.getNumBodies(physicsClientId=client) == 0


@pytest.mark.parametrize(
    "env_type", list(EnvironmentType), ids=lambda t: t.name.lower()
)
def test_environment_type_round_trips_through_its_value(env_type):
    # experiment configs and logs will store the type as text
    assert EnvironmentType(env_type.value) is env_type


def test_public_api_lists_only_existing_names():
    from neat_walking_robot import environments, simulation

    for package in (environments, simulation):
        for name in package.__all__:
            assert hasattr(package, name), (
                f"{package.__name__}.__all__ lists missing name {name}"
            )
