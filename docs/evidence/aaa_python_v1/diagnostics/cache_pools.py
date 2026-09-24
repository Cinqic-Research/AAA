import pickle, time
from research.aaa_python import generator, spec as S
spec = S.load()
t = time.time()
out = {}
for split in ("train", "development"):
    for fam in spec["families"]:
        out[(split, fam)] = list(generator.pool(split, fam))
for fam in spec["families"]:
    out[("probe", fam)] = generator.build("probe", fam, range(60), spec)
pickle.dump(out, open("v0_pools.pkl", "wb"))
print("done", time.time() - t, {k: len(v) for k, v in out.items()})
