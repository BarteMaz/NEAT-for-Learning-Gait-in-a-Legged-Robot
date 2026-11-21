import os
import multiprocessing as mp
from time import perf_counter


print(__name__, os.getpid())

UINT = 7000000
T1 = 4.006349600000249

def busy(n):
    sum = 0
    for i in range(n):
        sum += i * i


def task(k):
    busy(k * UINT)
    return os.getpid()


def trivial(n):
    return n


if __name__ == "__main__":
    ctx = mp.get_context("spawn")

    pool = ctx.Pool(4)
    pool.map(trivial, range(4))

    start = perf_counter()
    pool.map(task, [1] * 15 + [8], chunksize=1)
    end = perf_counter()
    print(end - start)

    start = perf_counter()
    pool.map(task, [8] + [1] * 15, chunksize=1)
    end = perf_counter()
    print(end - start)




# Krok 2
# 1.5963930999987497
# Krok 3
# 1: (3.868971899999451, 1.0355075465915937, 1.0355075465915937),
# 2: (2.020598199998858, 1.9827542160546876, 0.9913771080273438),
# 4: (1.0064301999991585, 3.9807525648610294, 0.9951881412152573),
# 24: (0.6011527999999089, 6.664444713558443, 0.27768519639826844)
# Krok 4
# pool.map(task, [1] * 15 + [8], chunksize=1) : 2.72431510000024 sec
# pool.map(task, [8] + [1] * 15, chunksize=1) : 1.9747007999994821
