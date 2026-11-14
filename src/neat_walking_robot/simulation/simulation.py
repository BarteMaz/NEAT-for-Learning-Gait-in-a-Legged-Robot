from dataclasses import dataclass

import pybullet as p
import pybullet_data


@dataclass(frozen=True, kw_only=True)
class SimulationData:
    dt: float = 1/240 # default value for Bullet's solver
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    num_solver_iterations: int = 50 # default value for Bullet's solver

    def __post_init__(self):
        if self.dt <= 0.0:
            raise ValueError(f"Time step ({self.dt}) must be greater than 0.0")
        if len(self.gravity) != 3:
            raise ValueError("Gravity vector must be of length 3 (value for each axis: x, y, z)")
        if self.num_solver_iterations <= 0:
            raise ValueError(f"Number of iterations ({self.num_solver_iterations}) must be positive")


def prepare_simulation(simulation_params: SimulationData, client: int) -> None:
    gx, gy, gz = simulation_params.gravity
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(gx, gy, gz, physicsClientId=client)
    p.setPhysicsEngineParameter(
        fixedTimeStep=simulation_params.dt,
        numSolverIterations=simulation_params.num_solver_iterations,
        deterministicOverlappingPairs=1,
        physicsClientId=client
    )


def connect_and_configure_simulation(simulation_params: SimulationData, client_type=p.DIRECT) -> int:
    client = p.connect(client_type)
    prepare_simulation(simulation_params, client)
    return client


