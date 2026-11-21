import os
import multiprocessing as mp

_state = {}

print(__name__, os.getpid())

def init(tag):
    _state.update(tag=tag, pid=os.getpid(), counter=0)

def work(x):
    _state["counter"] += 1
    return _state["tag"], _state["pid"], _state["counter"]

if __name__ == '__main__':
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(3, initializer=init, initargs=("A",))
    print(pool.map(work, range(12)))
    print(_state)