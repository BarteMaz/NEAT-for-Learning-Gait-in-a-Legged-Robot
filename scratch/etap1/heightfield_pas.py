"""
Przykład poglądowy (nie kod projektu): heightfield o wymiarach L x W z komórką C
oraz wypłaszczony pas o długości D przy lewej krawędzi terenu (najmniejsze x).

Uruchomienie:
    python heightfield_pas.py          # okno GUI
    python heightfield_pas.py --check  # bez GUI, tylko sprawdzenie raycastem
"""
import sys

import numpy as np
import pybullet as p
import pybullet_data

# ---------------- PARAMETRY ----------------
L = 10.0   # długość pola wzdłuż osi X [m]
W = 4.0    # szerokość pola wzdłuż osi Y [m]
C = 0.05   # odstęp między punktami siatki (rozmiar komórki) [m]
D = 2.0    # długość płaskiego pasa od lewej krawędzi [m]
M = 0.5    # długość strefy przejścia za pasem [m]

AMPLITUDE = 0.15                 # maksymalne |h| terenu przed wypłaszczeniem [m]
N_WAVES = 12                     # ile fal sinusoidalnych sumujemy
WAVELENGTH_RANGE = (0.5, 3.0)    # zakres długości fal [m]
SEED = 0


# ---------------- KROK 1: SIATKA ----------------
def make_grid(L, W, C):
    """Zwraca współrzędne X, Y (w metrach, układ wycentrowany jak w Bullecie)."""
    rows = int(round(L / C)) + 1          # punktów jest o 1 więcej niż komórek
    cols = int(round(W / C)) + 1
    L_eff = (rows - 1) * C                # faktyczne wymiary po zaokrągleniu
    W_eff = (cols - 1) * C
    if abs(L_eff - L) > 1e-9 or abs(W_eff - W) > 1e-9:
        print(f"Uwaga: L/C lub W/C nie jest całkowite, teren ma {L_eff:.3f} x {W_eff:.3f} m")

    x = np.linspace(-L_eff / 2, L_eff / 2, rows)
    y = np.linspace(-W_eff / 2, W_eff / 2, cols)
    X, Y = np.meshgrid(x, y, indexing="ij")   # X[i, j] = x_i, Y[i, j] = y_j
    return X, Y


# ---------------- KROK 2: TEREN ----------------
def generate_terrain(X, Y, C, amplitude, n_waves, wl_range, rng):
    """Suma fal płaskich o losowym kierunku, długości i fazie."""
    lam_min, lam_max = wl_range
    if lam_min < 4 * C:
        raise ValueError(f"Najkrótsza fala {lam_min} m to mniej niż 4 komórki ({4 * C} m)")

    h = np.zeros_like(X)
    for _ in range(n_waves):
        lam = rng.uniform(lam_min, lam_max)    # długość fali [m]
        k = 2 * np.pi / lam                    # liczba falowa [rad/m]
        theta = rng.uniform(0, 2 * np.pi)      # kierunek fali
        phi = rng.uniform(0, 2 * np.pi)        # przesunięcie fazy
        a = lam                                # dłuższe fale = wyższe pagórki
        h += a * np.sin(k * (X * np.cos(theta) + Y * np.sin(theta)) + phi)

    return h * amplitude / np.abs(h).max()     # skalowanie: max |h| = amplitude


# ---------------- KROK 3: WYPŁASZCZENIE PASA ----------------
def smoothstep(t):
    return t * t * (3 - 2 * t)


def flatten_left_strip(h, X, D, M, target=0.0):
    """Płasko dla s in [0, D], przejście dla s in [D, D+M], oryginał dalej.
    s to odległość od lewej krawędzi wzdłuż osi X."""
    s = X - X.min()
    if M > 0:
        t = np.clip((s - D) / M, 0.0, 1.0)
        b = smoothstep(t)                       # 0 w pasie, 1 poza przejściem
    else:
        b = (s > D).astype(float)               # bez przejścia: twardy uskok
    return target * (1 - b) + h * b


def max_slope_deg(h, C):
    """Największe nachylenie terenu (różnice skończone)."""
    gx, gy = np.gradient(h, C, C)
    return np.degrees(np.arctan(np.hypot(gx, gy).max()))


# ---------------- KROK 4: PYBULLET ----------------
def build_heightfield(h, C):
    rows, cols = h.shape
    shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[C, C, 1.0],
        heightfieldData=h.flatten(order="F").tolist(),   # indeks i + j*rows
        numHeightfieldRows=rows,
        numHeightfieldColumns=cols,
        flags=p.GEOM_CONCAVE_INTERNAL_EDGE,
    )
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape)
    z_offset = (h.min() + h.max()) / 2                    # Bullet centruje AABB w Z
    p.resetBasePositionAndOrientation(body, [0, 0,  z_offset], [0, 0, 0, 1])
    p.changeVisualShape(body, -1, rgbaColor=[0.6, 0.7, 0.6, 1])
    return body


def ground_z(body, x, y):
    hit = p.rayTest([x, y, -5], [x, y, 5])[0]
    return hit[3][2] if hit[0] == body else None


def main():
    check_only = "--check" in sys.argv
    if D > L:
        raise ValueError("D nie może być większe niż L")

    X, Y = make_grid(L, W, C)
    rng = np.random.default_rng(SEED)
    h0 = generate_terrain(X, Y, C, AMPLITUDE, N_WAVES, WAVELENGTH_RANGE, rng)
    h = flatten_left_strip(h0, X, D, M)

    print(f"Siatka: {h.shape[0]} x {h.shape[1]} punktów")
    print(f"Max nachylenie przed: {max_slope_deg(h0, C):.1f} st., po: {max_slope_deg(h, C):.1f} st.")

    p.connect(p.DIRECT if check_only else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.5, 0.5, 0.5],
    )
    visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.5, 0.5, 0.5],
        rgbaColor=[0, 0 ,1,1],
    )
    object_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0, 0, 0],
    )

    body = build_heightfield(h, C)
    aabb = p.getAABB(body)
    z_min, z_max = aabb[0][2], aabb[1][2]
    print(f"z_min: {z_min}, z_max: {z_max}")

    x_left = X.min()
    for s in [0.1, D / 2, D - 0.1, D + M / 2, D + M + 1.0]:
        i = int(round(s / C))
        z_sim = ground_z(body, x_left + s, 0.0)
        print(f"s = {s:4.2f} m  ->  z w symulacji = {z_sim:+.4f}, w danych = {h[i, h.shape[1] // 2]:+.4f}")

    if check_only:
        p.disconnect()
        return

    p.resetDebugVisualizerCamera(6, 30, -35, [0, 0, 0])
    p.loadURDF("r2d2.urdf", [x_left + D / 2, 0, 0.5])   # robot na środku pasa
    while p.isConnected():
        p.stepSimulation()


if __name__ == "__main__":
    main()
