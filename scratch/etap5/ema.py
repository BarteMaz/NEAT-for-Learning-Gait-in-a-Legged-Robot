import numpy as np

def ema(u, alpha):
    y = np.empty_like(u)
    prev = 0.0
    for n, value in enumerate(u):
        prev = alpha * value + (1 - alpha) * prev
        y[n] = prev
    return y

fs, alpha = 60.0, 0.3
t = np.arange(600) / fs                      # 10 s

# step = ema(np.ones(10), alpha)
# print("step:", step.round(3))                # 0.3, 0.51, 0.657, ...
#
# flip = ema(np.where(np.arange(600) % 2 == 0, 1.0, -1.0), alpha)
# print("flip amplitude:", np.abs(flip[-100:]).max().round(3), "theory:", round(alpha / (2 - alpha), 3))
#
# sine = ema(np.sin(2 * np.pi * 3 * t), alpha)
# print("3 Hz gain:", np.abs(sine[-120:]).max().round(3))   # ~0.754



def filter1(a, u, y):
    return y + a*(u - y)

def filter2(a, u, y):
    return a*u + (1 - a)*y

y = 1.0
u = 2.0

print(filter1(0.3, u, y))
print(filter2(0.3, u, y))

if 0.1:
    print(2)