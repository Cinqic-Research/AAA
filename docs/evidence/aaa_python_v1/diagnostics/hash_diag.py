"""Parameter-neutral hashing diagnostic on the unmodified v0 learner (train/dev/probe identities only)."""
import hashlib, json, math, pickle, sys, time
import numpy as np
from research.aaa_python import spec as S, representation as R
from research.aaa_python.experiment import make_learner, train, run_stream, Plan
from research.aaa_python.learners import OnlineLinear
from research.aaa_python.episode import view_of, label_space
P = pickle.load(open("v0_pools.pkl", "rb"))
spec = S.load()
FAM = list(spec["families"])

def h64(feature):
    return int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")

def variant_vector(signed, dedicated_bias, D):
    def vec(source, representation, dimensions):
        width = D - 1 if dedicated_bias else D
        out = np.zeros(D)
        for f in R._features(source, representation):
            h = h64(f)
            idx = h % width
            s = (1.0 if (h >> 63) & 1 == 0 else -1.0) if signed else 1.0
            out[idx] += s
        n = float(np.linalg.norm(out))
        if n > 0: out /= n
        if dedicated_bias: out[D - 1] += 1.0
        else: out[R._stable("<bias>", D) if D == dimensions else h64("<bias>") % D] += 1.0
        return out
    return vec

class Variant(OnlineLinear):
    vecfn = None
    def _vector(self, text, family):
        rep = R.effective_representation(self.representation, family)
        key = (rep, text)
        c = self._cache.get(key)
        if c is None:
            c = type(self).vecfn(text, rep, self.dimensions)
            self._cache[key] = c
        return c

def build(init, D, vecfn, rep="lexical"):
    base = make_learner(spec, init, rep)
    cls = type("V", (Variant,), {"vecfn": staticmethod(vecfn)}) if vecfn else OnlineLinear
    L = cls(seed=base.seed, representation=rep, dimensions=D, learning_rate=0.2, init_scale=0.01,
            abstain_below=0.0, label_spaces={f: label_space(__import__("research.aaa_python.experiment", fromlist=["x"])._label_probe(f), spec) for f in FAM})
    return L

def evaluate(L, fam):
    """Frozen accuracy and mean log-loss on all 400 development tasks of a family (diagnostic)."""
    from research.aaa_python.episode import Environment
    T = P[("development", fam)]
    acc, nll = 0, 0.0
    for i, t in enumerate(T):
        v = view_of(t, i, spec)
        labels, p = L._probabilities(v)
        acc += labels[int(np.argmax(p))] == t.answer
        j = labels.index(t.answer) if t.answer in labels else None
        nll += -math.log(max(p[j], 1e-12)) if j is not None else 27.6
    return acc / len(T), nll / len(T)

train_tasks = {f: P[("train", f)] for f in FAM}
from research.aaa_python.rng import derive_seed
if sys.argv[1:] == ["verify"]:
    L = build(0, 1024, None)
    train(L, train_tasks, 2, derive_seed("aaa.python.v0", "train-order", 0), spec)
    print("v0 reproduction state hash", L.state_hash())
    sys.exit()
rows = []
inits = range(int(sys.argv[1]) if sys.argv[1:] else 5)
for D in (256, 1024, 4096):
    for signed in (False, True):
        for ded in (False, True):
            for init in inits:
                L = build(init, D, variant_vector(signed, ded, D))
                train(L, train_tasks, 2, derive_seed("aaa.python.v0", "train-order", init), spec)
                for fam in FAM:
                    a, n = evaluate(L, fam)
                    rows.append(dict(D=D, signed=signed, dedicated_bias=ded, init=init, family=fam, acc=a, nll=n))
            sub = [r for r in rows if r["D"] == D and r["signed"] == signed and r["dedicated_bias"] == ded]
            print(D, "signed" if signed else "unsigned", "dedbias" if ded else "hashbias",
                  {f: (round(np.mean([r["acc"] for r in sub if r["family"] == f]), 3), round(np.mean([r["nll"] for r in sub if r["family"] == f]), 3)) for f in FAM}, flush=True)
json.dump(rows, open("hash_diag.json", "w"))
