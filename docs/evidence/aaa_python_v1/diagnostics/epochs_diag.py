import sys; sys.argv=["x","0"]
exec(open("hash_diag.py").read().split("train_tasks =")[0])
from research.aaa_python.rng import derive_seed
from research.aaa_python.experiment import train
train_tasks = {f: P[("train", f)] for f in FAM}
import numpy as np
for epochs in (2, 6, 20):
    res = {f: [] for f in FAM}
    for init in range(3):
        L = build(init, 1024, None)
        train(L, train_tasks, epochs, derive_seed("aaa.python.v0", "train-order", init), spec)
        for f in FAM: res[f].append(evaluate(L, f)[0])
    print("epochs", epochs, {f: round(float(np.mean(v)),3) for f, v in res.items()}, flush=True)
