"""
Test of the compiled duplicate scan (vectorSet.selectUniqueRowsNb, called through vectorSet.selectUniqueRows by
the vectorSet constructor) against the plain Python scan it replaces, which is kept here as the reference.

Checked:
  * identical output (the mask of non-duplicate rows and their positions in the sorted order) on random sets of
    rows with exact duplicates, negated and rescaled copies, and near-duplicates on either side of the absolute
    and of the relative tolerance; in 1 to 8 dimensions, for 1 to 300 rows, for sorted and for arbitrary orders;
  * rows with NaN or infinite entries are treated as the reference treats them (a NaN equals nothing);
  * no rows: the same (empty) result, without calling the compiled function;
  * vectorSet objects built with the compiled scan and with the reference scan have the same unique rows, for
    direction-independent and direction-dependent sets;
  * the time of the constructor with either scan (reported, not checked).

Usage: python3 tests/test_selectUniqueRows.py
"""
import sys, os, time
import numpy as np
import numba as nb
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vectorSet as vsmod
from vectorSet import vectorSet, quickSortRowwise, selectUniqueRows, vecEqualNb

failures = []
def check(name, cond, detail=''):
    print(f'{"PASS" if cond else "FAIL"}  {name}{("   " + str(detail)) if detail != "" and not cond else ""}', flush=True)
    if not cond:
        failures.append(name)

def referenceScan(mat, sortOrd, tol, rTol):
    # (the scan as it was before it was compiled)
    n = sortOrd.shape[0]
    subRows = []
    selIdx = np.ones(n,dtype=np.bool_)
    if n == 0:
        return selIdx, tuple(subRows)
    subRows.append(0)
    for r in range(1,n):
        if vecEqualNb(mat[sortOrd[r-1]], mat[sortOrd[r]], tol, rTol):
            selIdx[r] = False
        else:
            subRows.append(r)
    return selIdx, subRows

def sameResult(a, b):
    return np.array_equal(a[0], b[0]) and list(a[1]) == list(b[1]) and a[0].dtype == b[0].dtype

rng = np.random.default_rng(0)
tol = rTol = 1e-9

def randomRows(n, d):
    # a few base rows, repeated exactly, negated, rescaled, and perturbed around the tolerances
    base = rng.normal(size=(max(1, n // 3), d))
    rows = []
    for _ in range(n):
        v = base[rng.integers(0, base.shape[0])].copy()
        kind = rng.integers(0, 7)
        if kind == 1: v = -v
        elif kind == 2: v = v + rng.choice([0.5, 0.99, 1.01, 2.0]) * tol * rng.choice([-1.0, 1.0], size=d)
        elif kind == 3: v = v * (1.0 + rng.choice([0.5, 0.99, 1.01, 2.0]) * rTol)
        elif kind == 4: v = rng.normal(size=d)
        elif kind == 5: v = -v + rng.choice([0.5, 2.0]) * tol
        rows.append(v)
    return np.array(rows).reshape(n, d)

# ---- 1. the scan itself, on the sorted order the constructor uses and on arbitrary orders
nCases = 0; nDiff = 0; nDup = 0
for case in range(1500):
    d = int(rng.integers(1, 9)); n = int(rng.choice([1, 2, 3, 5, 8, 20, 60, 300], p=[.1, .1, .1, .15, .2, .2, .1, .05]))
    M = randomRows(n, d)
    mat = nb.typed.List([M[i].copy() for i in range(n)])
    for order in ('sorted', 'random'):
        sortOrd = np.arange(n)
        if order == 'sorted': quickSortRowwise(mat, sortOrd, tol, rTol)
        else: rng.shuffle(sortOrd)
        ref = referenceScan(mat, sortOrd, tol, rTol); new = selectUniqueRows(mat, sortOrd, tol, rTol)
        nCases += 1; nDiff += (not sameResult(ref, new)); nDup += int(n - np.sum(ref[0]))
check(f'{nCases} random row sets ({nDup} duplicates found): compiled scan identical to the reference', nDiff == 0, f'{nDiff} differ')

# ---- 2. tolerances exactly as the reference applies them (relative to the SECOND row of each pair)
v = np.array([1.0, -2.0, 3.0e6])
pairs = [v + np.array([0.9e-9, 0, 0]), v + np.array([1.1e-9, 0, 0]), v * (1 + 0.9e-9), v * (1 + 2.1e-9), -v, -v + np.array([0, 0, 2.0e-3]), -v + np.array([0, 0, 4.0e-3]), v + np.array([0, 0, 2.9e-3])]
nDiff = 0
for w in pairs:
    for a, b in ((v, w), (w, v)):
        mat = nb.typed.List([a.copy(), b.copy()]); so = np.arange(2)
        nDiff += (not sameResult(referenceScan(mat, so, tol, rTol), selectUniqueRows(mat, so, tol, rTol)))
check('pairs on either side of the absolute and the relative tolerance, in both orders: identical', nDiff == 0, f'{nDiff} differ')

# ---- 3. NaN and infinite entries
special = [np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), np.array([np.inf, 1.0]), np.array([np.inf, 1.0]), np.array([-np.inf, 1.0]), np.array([1.0, 1.0]), np.array([1.0, np.nan])]
nDiff = 0
with np.errstate(invalid='ignore'):
    for trial in range(50):
        idx = rng.permutation(len(special)); mat = nb.typed.List([special[i].copy() for i in idx]); so = np.arange(len(special))
        ref = referenceScan(mat, so, tol, rTol); new = selectUniqueRows(mat, so, tol, rTol)
        nDiff += (not sameResult(ref, new))
check('rows with NaN and infinite entries: identical (a NaN equals nothing)', nDiff == 0, f'{nDiff} differ')

# ---- 4. no rows
res = selectUniqueRows(nb.typed.List([np.zeros(3)]), np.arange(0), tol, rTol)
check('no rows: empty mask and no positions', res[0].shape == (0,) and len(res[1]) == 0, res)

# ---- 5. whole vectorSet objects, with the compiled and with the reference scan
compiled = vsmod.selectUniqueRows
nDiff = 0; nObj = 0
for case in range(300):
    d = int(rng.integers(2, 7)); n = int(rng.integers(1, 60))
    M = randomRows(n, d)
    for dirIndep in (True, False):
        vsmod.selectUniqueRows = compiled; a = vectorSet(M, dirIndep=dirIndep)
        vsmod.selectUniqueRows = referenceScan; b = vectorSet(M, dirIndep=dirIndep)
        nObj += 1
        nDiff += not (a.uniqRowIdx == b.uniqRowIdx and a.Nunique == b.Nunique and a.uniqRowIdxSet == b.uniqRowIdxSet and np.array_equal(a.getUniqueRows(), b.getUniqueRows()))
vsmod.selectUniqueRows = compiled
check(f'{nObj} vectorSet objects: the same unique rows with either scan', nDiff == 0, f'{nDiff} differ')

# ---- 6. time of the constructor
print()
for (n, d) in [(1, 4), (8, 4), (26, 4), (66, 4), (160, 6), (416, 9)]:
    mats = [randomRows(n, d) for _ in range(max(20, 3000 // (n + 5)))]
    line = f'   {n:4d} rows x {d}:'
    for label, fn in (('reference scan', referenceScan), ('compiled scan', compiled)):
        vsmod.selectUniqueRows = fn
        t = time.perf_counter()
        for M in mats: vectorSet(M)
        line += f'  {label} {1e3 * (time.perf_counter() - t) / len(mats):.3f} ms per vectorSet'
    print(line)
vsmod.selectUniqueRows = compiled

print(f'\n{"ALL TESTS PASSED" if len(failures) == 0 else str(len(failures)) + " TEST(S) FAILED"}')
sys.exit(0 if len(failures) == 0 else 1)
