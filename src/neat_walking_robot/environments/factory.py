from enum import Enum

from neat_walking_robot.environments.base import BaseEnvironment, BaseEnvironmentData
from neat_walking_robot.environments.flat import FlatEnvironment
from neat_walking_robot.environments.rough import RoughEnvironment, RoughEnvironmentData
from neat_walking_robot.environments.slope import SlopeEnvironment, SlopeEnvironmentData


class EnvironmentType(Enum):
    FLAT = "flat_environment"
    ROUGH = "rough_environment"
    SLOPE = "slope_environment"

_REGISTRY: dict[EnvironmentType, tuple[type[BaseEnvironment], type[BaseEnvironmentData]]] = {
    EnvironmentType.FLAT: (FlatEnvironment, BaseEnvironmentData),
    EnvironmentType.ROUGH: (RoughEnvironment, RoughEnvironmentData),
    EnvironmentType.SLOPE: (SlopeEnvironment, SlopeEnvironmentData),
}

def create_environment(env_type: EnvironmentType, env_data: BaseEnvironmentData, client: int, seed: int) -> BaseEnvironment:
    env_class, data_class = _REGISTRY[env_type]
    if type(env_data) is not data_class:
        raise TypeError(
            f"{env_type} expects {data_class.__name__} but got {type(env_data).__name__}"
        )
    return env_class(env_data, client, seed)