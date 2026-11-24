from dataclasses import dataclass

import numpy as np
import pybullet as p


from neat_walking_robot.environments.base import BaseEnvironment, BaseEnvironmentData


@dataclass(frozen=True, kw_only=True)
class RoughEnvironmentData(BaseEnvironmentData):
    cell_size: float
    amplitude: float
    transition_distance: float
    n_waves: int
    wavelength_range: tuple[float, float]

    def __post_init__(self):
        super().__post_init__()
        if -self.amplitude < self.lowest_ground_point:
            raise ValueError(
                f"Negative amplitude ({-self.amplitude} m) must be within the lowest ground point ({self.lowest_ground_point} m)")

        if self.amplitude <= 0.0:
            raise ValueError(f"Amplitude ({self.amplitude}) must be greater than 0")

        if self.cell_size <= 0.0:
            raise ValueError(f"Cell size ({self.cell_size}) must be greater than 0")

        if self.transition_distance <= 0.0:
            raise ValueError(f"Transition distance ({self.transition_distance}) must be greater than 0")

        if self.n_waves <= 0:
            raise ValueError(f"Number of waves ({self.n_waves}) must be at least 1")

        rows = int(np.round(self.track_length / self.cell_size)) + 1
        cols = int(np.round(self.track_width / self.cell_size)) + 1
        l_eff = (rows - 1) * self.cell_size
        w_eff = (cols - 1) * self.cell_size
        if abs(l_eff - self.track_length) > 1e-9 or abs(w_eff - self.track_width) > 1e-9:
            raise ValueError(
                f"Error: track_length / cell_size or track_width / cell_size is not an integer, track has {l_eff:.3f} x {w_eff:.3f} m")

        lam_min, lam_max = self.wavelength_range
        if lam_min > lam_max:
            raise ValueError(
                f"Incorrect wavelength range ({lam_min}, {lam_max}) the start of range is greater than the end of range")
        if lam_min < 4 * self.cell_size:
            raise ValueError(f"The shortest wave {lam_min} m is less then 4 cells ({4 * self.cell_size} m)")


def smoothstep(t):
    return t * t * (3 - 2 * t)


class RoughEnvironment(BaseEnvironment[RoughEnvironmentData]):
    def __init__(self, params: RoughEnvironmentData, client: int, seed: int):
        super().__init__(params, client, seed)
        self.rng = np.random.default_rng(seed)
        self.flat_area = self.params.start_distance + self.params.buffer_space
        self.rows = int(np.round(self.params.track_length / self.params.cell_size)) + 1
        self.cols = int(np.round(self.params.track_width / self.params.cell_size)) + 1
        self.max_slope_deg = None
        self.ground = self.build_ground()

    def make_grid(self):
        rows = self.rows
        cols = self.cols

        l_eff = (rows - 1) * self.params.cell_size
        w_eff = (cols - 1) * self.params.cell_size

        x = np.linspace(-l_eff / 2, l_eff / 2, rows)
        y = np.linspace(-w_eff / 2, w_eff / 2, cols)
        X, Y = np.meshgrid(x, y, indexing="ij")
        return X, Y

    def generate_terrain(self, X, Y):
        lam_min, lam_max = self.params.wavelength_range

        h = np.zeros_like(X)
        for _ in range(self.params.n_waves):
            lam = self.rng.uniform(lam_min, lam_max)  # długość fali [m]
            k = 2 * np.pi / lam  # liczba falowa [rad/m]
            theta = self.rng.uniform(0, 2 * np.pi)  # kierunek fali
            phi = self.rng.uniform(0, 2 * np.pi)  # przesunięcie fazy
            a = lam  # dłuższe fale = wyższe pagórki
            h += a * np.sin(k * (X * np.cos(theta) + Y * np.sin(theta)) + phi)

        return h * self.params.amplitude / np.abs(h).max()  # skalowanie: max |h| = amplitude

    def flatten_left_strip(self, h, X, target=0.0):
        s = X - X.min()
        if self.params.transition_distance > 0:
            t = np.clip((s - self.flat_area) / self.params.transition_distance, 0.0, 1.0)
            b = smoothstep(t)  # 0 w pasie, 1 poza przejściem
        else:
            b = (s > self.flat_area).astype(float)  # bez przejścia: twardy uskok
        return target * (1 - b) + h * b

    def count_max_slope_deg(self, h):
        gx, gy = np.gradient(h, self.params.cell_size, self.params.cell_size)
        return np.degrees(np.arctan(np.hypot(gx, gy).max()))

    def build_ground(self):
        x_dist, y_dist = self.make_grid()
        height_data_0 = self.generate_terrain(x_dist, y_dist)
        height_data = self.flatten_left_strip(height_data_0, x_dist)

        self.max_slope_deg = self.count_max_slope_deg(height_data)

        ground_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.params.cell_size, self.params.cell_size, 1.0],
            heightfieldData=height_data.flatten(order="F").tolist(),
            numHeightfieldRows=self.rows,
            numHeightfieldColumns=self.cols,
            heightfieldTextureScaling=(self.rows - 1) / 2,
            flags=p.GEOM_CONCAVE_INTERNAL_EDGE,
            physicsClientId=self.client
        )
        ground = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=ground_shape, physicsClientId=self.client)
        p.changeDynamics(ground, -1, lateralFriction=self.params.friction, physicsClientId=self.client)
        z_offset = (height_data.min() + height_data.max()) / 2
        p.resetBasePositionAndOrientation(ground, [0, 0, z_offset], [0, 0, 0, 1], physicsClientId=self.client)
        return ground

    def get_ground_level(self, x: float, y: float) -> float:
        hit = p.rayTest([x, y, -5], [x, y, 5], physicsClientId=self.client)[0]
        return hit[3][2] if hit[0] == self.ground else -1.0
