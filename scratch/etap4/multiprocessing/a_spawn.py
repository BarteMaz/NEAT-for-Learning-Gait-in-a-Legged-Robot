import os
import multiprocessing as mp


print(__name__, os.getpid())

def busy(n):
    count = 0
    for i in range(n):
        count += 1


def work(x):
    busy(10000)
    return x, os.getpid()


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(2, initializer=work, initargs=(1,))
    print(pool.map(work, range(6)))

