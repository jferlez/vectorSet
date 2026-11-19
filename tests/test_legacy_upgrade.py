"""
Checks that vectorSet objects pickled by an older version of the class (format 0: the sign of a stored row was
taken from its last entry alone) are upgraded correctly when they are unpickled with the current code.

tests/legacy_format0.pkl was written by the code at commit 5b502d0, and contains pickled vectorSet objects
together with the rows that were put into them. Many of the rows have a zero last entry, for which the old sign
rule stored v and -v as different rows.

Usage: python3 tests/test_legacy_upgrade.py [fixture.pkl]
"""
import sys, os, pickle, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import vectorSet
    if not callable(vectorSet.vectorSet): raise ImportError
except Exception:
    import vectorSet.vectorSet as vectorSet
V = vectorSet.vectorSet

def sameDir(u, v):
    u = u / np.linalg.norm(u[1:]); v = v / np.linalg.norm(v[1:])
    return np.allclose(u, v, atol=1e-7) or np.allclose(u, -v, atol=1e-7)

# Record what the upgrade does (it runs inside pickle.loads, via __setstate__)
upgrades = []
_upgrade = V.upgradeFromLegacy
def _recordingUpgrade(self):
    changed = _upgrade(self)
    upgrades.append(len(changed))
    return changed
V.upgradeFromLegacy = _recordingUpgrade

fixture = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'legacy_format0.pkl')
objs = pickle.load(open(fixture, 'rb'))
problems = []
stats = {'objects': 0, 'rows': 0, 'rows whose stored sign changed': 0, 'objects with a duplicate warning': 0, 'lookups': 0}
for k, rec in enumerate(objs):
    rows = [np.array(r, dtype=np.float64) for r in rec['rows']]
    N = len(rows)
    groups = [[j for j in range(N) if sameDir(rows[i], rows[j])] for i in range(N)]
    upgrades.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        vs = pickle.loads(rec['pickle'])
    errs = []
    if len(upgrades) != 1: errs.append(f'upgradeFromLegacy ran {len(upgrades)} times on load (expected once)')
    stats['rows whose stored sign changed'] += sum(upgrades)
    if vs.__dict__.get('formatVersion') != V.FORMAT_VERSION: errs.append('formatVersion not set')
    # The unique rows are what users of the object refer to: the upgrade must not change them
    oldUniq = set(rec['uniq'])
    if set(int(i) for i in vs.uniqRowIdx) != oldUniq: errs.append('the set of unique rows changed')
    vs.deserialize()
    if not (vs.N == N == len(vs.rows) == len(vs.sortOrd) == len(vs.revSortOrd) == len(vs.scale)):
        errs.append('inconsistent lengths')
    elif sorted(vs.sortOrd) != list(range(N)) or any(vs.revSortOrd[vs.sortOrd[i]] != i for i in range(N)):
        errs.append('sortOrd/revSortOrd inconsistent')
    else:
        if not np.allclose(vs.getRows(), np.array(rows), atol=1e-9): errs.append('getRows() does not return the original rows')
        if not np.allclose(np.array(list(vs.rows)), vs.dirScales(np.array(rows)).reshape(-1, 1) * np.array(rows), atol=1e-12):
            errs.append('stored rows are not normalised with the current rule')
        for i in range(1, N):
            a, b = vs.rows[vs.sortOrd[i-1]], vs.rows[vs.sortOrd[i]]
            if not vs.vecCompare(a, b) and not vs.vecEqual(a, b): errs.append('rows are not sorted'); break
        # Every row has to be found again, with all its duplicates, whatever multiple of it is asked for
        for i in range(N):
            for mult in [1.0, -1.0, -3.0, 0.25]:
                stats['lookups'] += 1
                if not vs.isElem(mult * rows[i]): errs.append(f'isElem fails for {mult} x row {i}'); break
                if sorted(vs.expandDuplicates(vec=mult * rows[i])) != groups[i]: errs.append(f'expandDuplicates({mult} x row {i}) != {groups[i]}'); break
            if sorted(vs.expandDuplicates(i)) != groups[i]: errs.append(f'expandDuplicates({i}) != {groups[i]}')
        # A warning is expected exactly when the old code stored duplicates as distinct unique rows
        merged = any(len(oldUniq & set(g)) > 1 for g in groups)
        if merged != (len(w) > 0): errs.append(f'warning {"missing" if merged else "unexpected"}: {[str(x.message)[:80] for x in w]}')
        stats['objects with a duplicate warning'] += int(len(w) > 0)
        # A second load of the upgraded object must not upgrade again
        upgrades.clear()
        vs.serialize(); again = pickle.loads(pickle.dumps(vs)); vs.deserialize()
        if len(upgrades) != 0: errs.append('an upgraded object was upgraded again when re-loaded')
        # Upgraded objects must keep working: remove every group of duplicates, one after the other
        try:
            while vs.N > 0:
                before = vs.N
                remd, added = vs.removeRow(idxOrigOrder=0, includeDup=True)
                if len(remd) == 0 or vs.N != before - len(remd) or len(added) > 0: errs.append('removeRow(includeDup=True) inconsistent'); break
            if vs.N == 0 and len(vs.uniqRowIdx) != 0: errs.append(f'unique rows left after removing everything: {vs.uniqRowIdx}')
        except Exception as e:
            errs.append(f'removeRow raised {type(e).__name__}: {e}')
    stats['objects'] += 1; stats['rows'] += N
    if errs: problems.append((k, errs))
for k, errs in problems[:5]: print(f'object {k}: {errs[:3]}')
print(f'{len(objs) - len(problems)} of {len(objs)} legacy objects upgraded correctly')
print('   ' + ', '.join(f'{v} {ky}' for ky, v in stats.items()))
print(f'######## {"ALL TESTS PASSED" if not problems else "SOME TESTS FAILED"} ########')
sys.exit(0 if not problems else 1)
