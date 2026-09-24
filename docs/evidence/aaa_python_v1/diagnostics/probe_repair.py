import pickle, collections
from research.aaa_python import spec as S
from research.aaa_python.oracle import run_many
spec = S.load()
P = pickle.load(open("v0_pools.pkl", "rb"))

def tokdist(a, b):
    ta, tb = a.split(" "), b.split(" ")
    if len(ta) != len(tb):
        return 99
    return sum(x != y for x, y in zip(ta, tb))

def centroid(t):
    cur = t.source.rstrip("\n").split("\n")[t.repair_line - 1]
    scores = []
    for i, c in enumerate(t.candidates):
        scores.append((sum(tokdist(c, o) for o in t.candidates), i))
    return min(scores)[1]

def centroid_excl_current(t):
    cur = t.source.rstrip("\n").split("\n")[t.repair_line - 1]
    scores = []
    for i, c in enumerate(t.candidates):
        if c == cur: continue
        scores.append((sum(tokdist(c, o) for o in t.candidates), i))
    return min(scores)[1]

def visible_tool(tasks):
    # Legitimate tool: run each candidate on the VISIBLE tests only.
    jobs = []
    for t in tasks:
        lines = t.source.rstrip("\n").split("\n")
        for c in t.candidates:
            body = [*lines[: t.repair_line - 1], c, *lines[t.repair_line:]]
            jobs.append(("exec", "\n".join(body + [f"print(f({i}))" for i, _ in t.visible_tests]) + "\n"))
    outs = run_many(jobs, spec)
    preds, npass = [], []
    k = 0
    for t in tasks:
        passing = []
        for i, c in enumerate(t.candidates):
            o = outs[k]; k += 1
            exp = [str(v) for _, v in t.visible_tests]
            if o.status == "ok" and o.stdout.split() == exp:
                passing.append(i)
        npass.append(len(passing))
        cur = t.source.rstrip("\n").split("\n")[t.repair_line - 1]
        # prefer a passing candidate that is not the current (buggy) line; first such
        choice = next((i for i in passing if t.candidates[i] != cur), passing[0] if passing else 0)
        preds.append(choice)
    return preds, npass

for split in ("train", "development"):
    T = P[(split, "repair")]
    acc = lambda f: sum(f(t) == t.answer for t in T) / len(T)
    print(split, "centroid", round(acc(centroid), 3), "centroid_excl_current", round(acc(centroid_excl_current), 3))
    cur_is = sum(t.candidates.index(t.source.rstrip("\n").split("\n")[t.repair_line-1]) != t.answer for t in T)
    print("  buggy line always among candidates:", all(t.source.rstrip('\n').split('\n')[t.repair_line-1] in t.candidates for t in T))
    print("  answer index distribution", collections.Counter(t.answer for t in T))
    preds, npass = visible_tool(T)
    print("  visible-test tool acc", round(sum(p == t.answer for p, t in zip(preds, T)) / len(T), 3), "n passing visible dist", collections.Counter(npass))
    by = collections.defaultdict(list)
    for p, t in zip(preds, T): by[t.slice].append(p == t.answer)
    print("  tool by slice", {k: round(sum(v)/len(v),3) for k, v in by.items()})
    byc = collections.defaultdict(list)
    for t in T: byc[t.slice].append(centroid_excl_current(t) == t.answer)
    print("  centroid by slice", {k: round(sum(v)/len(v),3) for k, v in byc.items()})
