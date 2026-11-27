"""
Checks the array storage of vectorSet (format 2: self.rows is an (N, d) float64 array and self.sortOrd an int64
array, where formats 0 and 1 kept Numba typed Lists, and Python lists while serialized).

tests/legacy_format1.pkl was written by the format-1 code (commit b6a33d9). Each record has the rows a vectorSet
was built from, the insertRow/removeRow calls made on it afterwards with what they returned, the object itself
pickled in serialized form (as the compiler sends and stores it), and what the format-1 code answered for every
query method.

Checked:
  1. every format-1 pickle is upgraded on load and then answers exactly as the format-1 code did;
  2. the current code, given the same rows and the same operations, returns the same from every operation and
     ends in the same state (bit for bit: the stored rows, the scale factors, the orders, the unique rows);
  3. the representation: C-contiguous arrays of the right types, serialize()/deserialize() change nothing, a
     pickle round trip and a deep copy keep the state;
  4. a set that has lost all of its rows still works, also after a pickle round trip (with typed Lists it did
     not: the list came back untyped);
  5. nothing is compiled by Numba while the methods run. (Typed-List methods called from the interpreter were
     compiled again in every process, about 0.6 s per process; the compiled helpers of this module have explicit
     signatures and a disk cache, so they are compiled when the module is imported, and only when its source has
     changed.)

Usage: python3 tests/test_array_storage.py [fixture.pkl]
"""
import sys, os, pickle, copy, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import vectorSet
    if not callable(vectorSet.vectorSet): raise ImportError
except Exception:
    import vectorSet.vectorSet as vectorSet
V = vectorSet.vectorSet

# Count Numba compilations from here on (the module is imported, so its own helpers are compiled or loaded)
compiled = []
try:
    from numba.core import event as nbEvent
    class _Listener(nbEvent.Listener):
        def on_start(self, event): pass
        def on_end(self, event): compiled.append(str(event.data.get('dispatcher')) + ' ' + str(event.data.get('args')))
    nbEvent.register('numba:compile', _Listener())
    haveEvents = True
except Exception:
    haveEvents = False

nFail = 0
def check(label, ok, detail=''):
    global nFail
    print(('PASS  ' if ok else 'FAIL  ') + label + ('' if ok or detail == '' else f'   [{detail}]'))
    nFail += (not ok)

def ints(x):
    # index results as plain Python structures (the format-1 code returned NumPy integers in a few places)
    if x is None: return None
    if isinstance(x, (list, tuple)): return [ints(i) for i in x]
    if isinstance(x, (bool, np.bool_)): return bool(x)
    return int(x)

def answers(vs, probes):
    return {'N': int(vs.N), 'Nunique': int(vs.Nunique), 'uniqRowIdx': ints(vs.uniqRowIdx), 'scale': [float(s) for s in vs.scale], \
            'sortOrd': ints(list(vs.sortOrd)), 'revSortOrd': ints(list(vs.revSortOrd)), \
            'storedRows': np.array([np.asarray(r) for r in vs.rows]).reshape(-1, vs.d), \
            'getRows': vs.getRows(), 'getRowsSorted': vs.getRowsSorted(), 'getUniqueRows': vs.getUniqueRows(), 'getUniqueRowsSorted': vs.getUniqueRowsSorted(), \
            'isElem': [bool(vs.isElem(p.copy())) for p in probes], 'selectRow': [ints(vs.selectRow(p.copy())) for p in probes], \
            'listParallel': [ints(vs.listParallel(p.copy())) for p in probes], \
            'expandDuplicates': [ints(vs.expandDuplicates(i)) for i in range(vs.N)]}

def differences(got, exp):
    out = []
    for ky in exp:
        if ky == 'subtractSet': continue
        g, e = got[ky], exp[ky]
        if isinstance(e, np.ndarray):
            same = isinstance(g, np.ndarray) and g.shape == e.shape and g.dtype == e.dtype and g.tobytes() == e.tobytes()
        elif ky == 'scale':
            same = len(g) == len(e) and all(float(a).hex() == float(b).hex() for a, b in zip(g, e))
        else:
            same = ints(g) == ints(e)
        if not same: out.append(ky)
    return out

fixture = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'legacy_format1.pkl')
recs = pickle.load(open(fixture, 'rb'))['records']

# ---- 1. format-1 pickles, upgraded on load
bad = []; nRows = 0
for k, rec in enumerate(recs):
    vs = pickle.loads(rec['pickle']); minus = pickle.loads(rec['minus'])
    errs = []
    if vs.formatVersion != V.FORMAT_VERSION or minus.formatVersion != V.FORMAT_VERSION: errs.append('formatVersion')
    if not (isinstance(vs.rows, np.ndarray) and isinstance(vs.sortOrd, np.ndarray) and vs.serialized is False and vs.rowsPy is None and vs.sortOrdPy is None): errs.append('representation')
    else:
        errs += differences(answers(vs, rec['probes']), rec['answers'])
        if ints(vs.subtractSet(minus)) != rec['answers']['subtractSet']: errs.append('subtractSet')
    nRows += vs.N
    if errs: bad.append((k, errs))
check(f'{len(recs)} format-1 pickles ({nRows} rows): upgraded on load, every answer as the format-1 code gave it', len(bad) == 0, str(bad[:3]))

# ---- 2. the current code on the same rows and operations
bad = []; nOps = 0
for k, rec in enumerate(recs):
    vs = V(rec['rows'].copy(), dirIndep=rec['dirIndep'])
    errs = []
    for op, vec, ret in rec['ops']:
        nOps += 1
        if op == 'insertRow':
            if bool(vs.insertRow(vec.copy())) != ret: errs.append('insertRow return')
        else:
            r = vs.removeRow(vec=vec.copy())
            if (ints(r[0]), ints(r[1])) != (ret[0], ret[1]): errs.append('removeRow return')
    errs += differences(answers(vs, rec['probes']), rec['answers'])
    minus = V(pickle.loads(rec['minus']).getRows(), dirIndep=rec['dirIndep'])
    if ints(vs.subtractSet(minus)) != rec['answers']['subtractSet']: errs.append('subtractSet')
    if errs: bad.append((k, errs))
check(f'{len(recs)} sets built by the current code, {nOps} insertRow/removeRow calls: same returns and the same final state, bit for bit', len(bad) == 0, str(bad[:3]))

# ---- 3. the representation
rng = np.random.default_rng(0)
M = rng.standard_normal((26, 4)); M[7] = -2.0 * M[3]; M[9] = M[3]
vs = V(M.copy())
def arraysOk(v):
    return isinstance(v.rows, np.ndarray) and v.rows.dtype == np.float64 and v.rows.shape == (v.N, v.d) and v.rows.flags['C_CONTIGUOUS'] \
        and isinstance(v.sortOrd, np.ndarray) and v.sortOrd.dtype == np.int64 and v.sortOrd.shape == (v.N,) and v.sortOrd.flags['C_CONTIGUOUS']
check('constructor: rows is a C-contiguous (N, d) float64 array, sortOrd an int64 array', arraysOk(vs))
vec = rng.standard_normal(4)
vs.insertRow(vec.copy()); okIns = arraysOk(vs) and vs.N == 27
vs.removeRow(vec=vec.copy()); okRem = arraysOk(vs) and vs.N == 26
vs.removeRow(idxOrigOrder=3, includeDup=True)
check('insertRow and removeRow keep that representation', okIns and okRem and arraysOk(vs) and vs.N == 23, f'{vs.N}')
vs = V(M.copy()); probes = [M[3], -M[3], rng.standard_normal(4)]
before = answers(vs, probes); rowsId, ordId = id(vs.rows), id(vs.sortOrd)
ret = vs.serialize()
okSer = ret is vs and vs.serialized is False and id(vs.rows) == rowsId and id(vs.sortOrd) == ordId
ret = vs.deserialize()
check('serialize() and deserialize() return the object and change nothing', okSer and ret is vs and differences(answers(vs, probes), before) == [])
check('pickle round trip keeps the state', differences(answers(pickle.loads(pickle.dumps(vs)), probes), before) == [] and arraysOk(pickle.loads(pickle.dumps(vs))))
dc = copy.deepcopy(vs); dc.insertRow(vec.copy())
check('a deep copy is independent of the original', differences(answers(vs, probes), before) == [] and dc.N == vs.N + 1)

# ---- 4. a set without rows
vs = V(M[:3].copy())
while vs.N > 0: vs.removeRow(idxOrigOrder=0, includeDup=True)
ok = True; detail = ''
try:
    for v in (vs, pickle.loads(pickle.dumps(vs.serialize())).deserialize()):
        full = V(M.copy())
        # (full.subtractSet(v) is not checked: rowwiseSetComplement answers "no rows" when the set to subtract is
        # empty, as it did before the array storage)
        ok = ok and v.N == 0 and v.Nunique == 0 and arraysOk(v) and v.isElem(M[0].copy()) is False and v.subtractSet(full) == [] \
            and v.getUniqueRows().size == 0 and v.expandDuplicates(vec=M[0].copy()) == []
        ok = ok and v.insertRow(M[0].copy()) is True and v.N == 1 and v.isElem(-3.0 * M[0]) is True
except Exception as e:
    ok = False; detail = f'{type(e).__name__}: {e}'
check('a set that has lost all its rows answers queries and takes new rows, also after a pickle round trip', ok, detail)

# ---- 5. nothing was compiled while the methods ran
if haveEvents:
    check('no Numba compilation while any of the above ran', len(compiled) == 0, f'{len(compiled)} compilations, e.g. {compiled[:2]}')
else:
    print('SKIP  numba.core.event is not available: run-time compilations not counted')

print('\n' + ('######## ALL TESTS PASSED ########' if nFail == 0 else f'######## {nFail} CHECK(S) FAILED ########'))
sys.exit(0 if nFail == 0 else 1)
