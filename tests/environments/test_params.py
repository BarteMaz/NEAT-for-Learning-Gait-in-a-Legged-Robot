"""Validation of environment parameter dataclasses.

Invalid parameters must fail at construction time (fail fast), so a broken
experiment never starts.
"""

import dataclasses

import pytest

# --- BaseEnvironmentData -----------------------------------------------------


def test_typical_params_are_accepted(make_base_params):
    params = make_base_params()
    assert params.track_length == 6.0


def test_params_are_immutable(make_base_params):
    params = make_base_params()
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.track_length = 10.0


@pytest.mark.parametrize(
    "field",
    [
        "track_length",
        "track_width",
        "wall_height",
        "wall_thickness",
        "ground_thickness",
        "friction",
    ],
)
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_non_positive_dimensions_are_rejected(make_base_params, field, value):
    with pytest.raises(ValueError):
        make_base_params(**{field: value})


def test_start_beyond_goal_is_rejected(make_base_params):
    # start x = -3 + 4 = 1, goal x = 3 - 4 = -1: the goal would be behind the robot
    with pytest.raises(ValueError):
        make_base_params(start_distance=4.0, end_distance=4.0)


def test_start_must_be_strictly_before_goal(make_base_params):
    # start x = -3 + 3 = 0 and goal x = 3 - 3 = 0: zero distance to walk
    with pytest.raises(ValueError):
        make_base_params(start_distance=3.0, end_distance=3.0)


def test_start_and_goal_positions_follow_from_distances(make_base_params):
    params = make_base_params(start_distance=1.5, end_distance=0.5)
    start_x = -params.track_length / 2 + params.start_distance
    goal_x = params.track_length / 2 - params.end_distance
    assert start_x < goal_x


# --- RoughEnvironmentData -----------------------------------------------------


def test_typical_rough_params_are_accepted(make_rough_params):
    params = make_rough_params()
    assert params.amplitude == 0.1


def test_rough_params_keep_base_validation(make_rough_params):
    # checks that RoughEnvironmentData.__post_init__ calls super().__post_init__()
    with pytest.raises(ValueError):
        make_rough_params(track_length=-1.0)


def test_grid_that_does_not_fit_the_track_is_rejected(make_rough_params):
    # 6.0 / 0.07 is not an integer, so the terrain would not cover the track
    with pytest.raises(ValueError):
        make_rough_params(cell_size=0.07)


def test_wavelength_shorter_than_four_cells_is_rejected(make_rough_params):
    with pytest.raises(ValueError):
        make_rough_params(cell_size=0.05, wavelength_range=(0.1, 2.0))


def test_terrain_deeper_than_lowest_ground_point_is_rejected(make_rough_params):
    with pytest.raises(ValueError):
        make_rough_params(amplitude=0.2, lowest_ground_point=-0.15)


@pytest.mark.parametrize(
    "overrides",
    [
        {"cell_size": 0.0},
        {"amplitude": 0.0},
        {"amplitude": -0.1},
        {"n_waves": 0},
        {"transition_distance": -0.5},
        {"wavelength_range": (2.0, 0.5)},
    ],
    ids=[
        "zero_cell",
        "zero_amplitude",
        "negative_amplitude",
        "no_waves",
        "negative_transition",
        "reversed_range",
    ],
)
def test_invalid_terrain_params_are_rejected(make_rough_params, overrides):
    with pytest.raises(ValueError):
        make_rough_params(**overrides)
