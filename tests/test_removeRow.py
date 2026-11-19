"""
Randomized consistency test of vectorSet.insertRow / vectorSet.removeRow against a brute-force model.

A vectorSet keeps every row it is given (duplicates included) and marks exactly one row of each group of
duplicates (rows that are equal up to a non-zero scalar multiple) as the "unique" representative. After every
operation this test checks the internal bookkeeping (sort order, reverse sort order, unique-row index list and
dict) and the return values of insertRow/removeRow.

Usage: python3 tests/test_removeRow.py [numSeeds]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import vectorSet
    if not callable(vectorSet.vectorSet): raise ImportError
except Exception:
    import vectorSet.vectorSet as vectorSet


def sameDir(u, v):
    # equal up to a non-zero scalar multiple
    u = u / np.linalg.norm(u[1:]); v = v / np.linalg.norm(v[1:])
    return np.allclose(u, v, atol=1e-7) or np.allclose(u, -v, atol=1e-7)

def groupOf(rows, i):
    return [j for j in range(len(rows)) if sameDir(rows[i], rows[j])]

def check(vs, rows, uniq, what):
    """ rows: list of the original rows; uniq: set of row indices expected to be the unique representatives
        (None: only require one representative per group of duplicates) """
    errs = []
    N = len(rows)
    vs.deserialize()
    if not (vs.N == N == len(vs.rows) == len(vs.scale) == len(vs.sortOrd) == len(vs.revSortOrd)):
        errs.append(f'inconsistent lengths: N={vs.N} rows={len(vs.rows)} scale={len(vs.scale)} sortOrd={len(vs.sortOrd)} revSortOrd={len(vs.revSortOrd)}; expected {N}')
        return errs
    if sorted(vs.sortOrd) != list(range(N)):
        errs.append(f'sortOrd is not a permutation: {list(vs.sortOrd)}')
    elif any(vs.revSortOrd[vs.sortOrd[i]] != i for i in range(N)):
        errs.append(f'revSortOrd is not the inverse of sortOrd: {list(vs.sortOrd)} / {vs.revSortOrd}')
    else:
        for i in range(1, N):
            a, b = vs.rows[vs.sortOrd[i-1]], vs.rows[vs.sortOrd[i]]
            if not vs.vecCompare(a, b) and not vs.vecEqual(a, b):
                errs.append(f'rows are not sorted at sorted position {i}')
                break
    got = vs.getRows() if N > 0 else np.zeros((0, vs.d))
    if N > 0 and not np.allclose(got, np.array(rows), atol=1e-9):
        errs.append('getRows() does not return the rows that were put in')
    if list(vs.uniqRowIdx) != sorted(vs.uniqRowIdx):
        errs.append(f'uniqRowIdx not sorted: {vs.uniqRowIdx}')
    if vs.uniqRowIdxSet != {int(i): idx for idx, i in enumerate(vs.uniqRowIdx)}:
        errs.append(f'uniqRowIdxSet {vs.uniqRowIdxSet} does not match uniqRowIdx {vs.uniqRowIdx}')
    if vs.Nunique != len(vs.uniqRowIdx):
        errs.append(f'Nunique {vs.Nunique} != len(uniqRowIdx) {len(vs.uniqRowIdx)}')
    seen = set()
    for i in range(N):
        if i in seen: continue
        g = groupOf(rows, i); seen |= set(g)
        reps = [j for j in g if j in set(vs.uniqRowIdx)]
        if len(reps) != 1:
            errs.append(f'group of duplicates {g} has {len(reps)} unique representatives {reps}')
    if uniq is not None and set(int(i) for i in vs.uniqRowIdx) != set(uniq):
        errs.append(f'unique rows {sorted(vs.uniqRowIdx)} != expected {sorted(uniq)}')
    return [f'[{what}] {e}' for e in errs]


def runSeed(seed, nOps=40, verbose=False):
    rng = np.random.default_rng(seed)
    d = int(rng.integers(3, 5))
    def randRow():
        r = rng.integers(-2, 3, size=d).astype(np.float64)
        if r[-1] == 0: r[-1] = 1.0          # (rows with a zero last entry are a separate, known limitation)
        r[0] = float(rng.integers(-1, 2))
        return r * float(rng.choice([-2.0, -1.0, 0.5, 1.0, 3.0]))
    rows = [randRow() for _ in range(int(rng.integers(1, 8)))]
    vs = vectorSet.vectorSet(np.array(rows))
    errs = check(vs, rows, None, 'init')
    uniq = set(int(i) for i in vs.uniqRowIdx)
    log = [f'init {len(rows)} rows (d={d})']
    for op in range(nOps):
        if errs: break
        kind = rng.choice(['insert', 'insertNoDup', 'removeIdx', 'removeIdxDup', 'removeVec', 'removeVecDup', 'serialize'])
        try:
            if kind in ('insert', 'insertNoDup') or len(rows) == 0:
                v = randRow() if (len(rows) == 0 or rng.random() < 0.5) else rows[int(rng.integers(len(rows)))] * float(rng.choice([-1.0, 1.0, 2.0, -0.5]))
                inclDup = (kind != 'insertNoDup')
                expNew = not any(sameDir(v, r) for r in rows)
                log.append(f'insertRow({v.tolist()}, includeDup={inclDup}) expect isNew={expNew}')
                isNew = vs.insertRow(v.copy(), includeDup=inclDup)
                if bool(isNew) != expNew: errs.append(f'[{log[-1]}] returned {isNew}')
                if expNew or inclDup:
                    rows.append(v)
                    if expNew: uniq.add(len(rows) - 1)
            elif kind == 'serialize':
                log.append('serialize/deserialize')
                vs.serialize(); vs.deserialize()
            else:
                i = int(rng.integers(len(rows)))
                inclDup = kind.endswith('Dup')
                g = groupOf(rows, i)
                if kind.startswith('removeIdx'):
                    expRem = g if inclDup else [i]
                    log.append(f'removeRow(idxOrigOrder={i}, includeDup={inclDup}) group {g}')
                    remd, added = vs.removeRow(idxOrigOrder=i, includeDup=inclDup)
                else:
                    # by vector: rows equal to the vector up to sign (all duplicates if includeDup)
                    expRem = g if inclDup else [j for j in g if np.allclose(rows[j], rows[i], atol=1e-9) or np.allclose(rows[j], -rows[i], atol=1e-9)]
                    log.append(f'removeRow(vec=row {i}, includeDup={inclDup}) group {g}')
                    remd, added = vs.removeRow(vec=rows[i].copy(), includeDup=inclDup)
                remaining = [j for j in g if j not in expRem]
                newIdx = lambda j: j - sum(1 for r in expRem if r < j)
                remUniq = len(uniq & set(expRem)) > 0
                expAdded = [newIdx(min(remaining))] if (remUniq and len(remaining) > 0) else []
                if sorted(int(r) for r in remd) != sorted(expRem): errs.append(f'[{log[-1]}] removed {remd}, expected {expRem}')
                if [int(a) for a in added] != expAdded: errs.append(f'[{log[-1]}] added {added}, expected {expAdded}')
                uniq = set(newIdx(j) for j in uniq if j not in expRem) | set(expAdded)
                rows = [r for j, r in enumerate(rows) if j not in expRem]
        except Exception as e:
            errs.append(f'[{log[-1]}] raised {type(e).__name__}: {e}')
            break
        errs += check(vs, rows, uniq, log[-1])
    if errs and verbose:
        print(f'seed {seed}:'); [print('   ', l) for l in log[-4:]]; [print('    ->', e) for e in errs[:4]]
    return errs


if __name__ == '__main__':
    nSeeds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    failed = {}
    for seed in range(nSeeds):
        errs = runSeed(seed, verbose=(len(failed) < 3))
        if errs: failed[seed] = errs
    kinds = {}
    for errs in failed.values():
        ky = errs[0].split('] ', 1)[-1].split(':')[0][:60]
        kinds[ky] = kinds.get(ky, 0) + 1
    print(f'{nSeeds - len(failed)} of {nSeeds} random operation sequences passed')
    for ky, cnt in sorted(kinds.items(), key=lambda kv: -kv[1]): print(f'   {cnt:4d} x first failure: {ky}')
    print(f'######## {"ALL TESTS PASSED" if not failed else "SOME TESTS FAILED"} ########')
    sys.exit(0 if not failed else 1)
