import numpy as np


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
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    X, Y = np.meshgrid(x, y, indexing="ij")   # X[i, j] = x_i, Y[i, j] = y_j
    return X, Y

L = 20
W = 10
C= 0.05
X, Y = make_grid(L, W, C)

print(f"X shape: {X.dtype}")

for i in range(4):
    for j in range(4):
        print(X[i, j], end=", ")
    print("")

print("")

for i in range(4):
    for j in range(4):
        print(Y[i, j], end=", ")
    print("\n")