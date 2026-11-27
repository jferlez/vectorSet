import numpy as np
import math
import bisect
# Numba imports
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict
from numba.np.unsafe.ndarray import to_fixed_tuple
from copy import copy, deepcopy
import warnings

class vectorSet:
    # Version of the stored representation (the attributes of an instance). Instances record the version they
    # were created with in self.formatVersion; an instance without that attribute is a "legacy" object that was
    # pickled by an older version of this class, and is upgraded when it is unpickled (see __setstate__).
    #    0 (legacy): the sign of a direction-independent row was taken from its last entry alone
    #    1:          the sign is taken from the last entry that is non-zero (see dirScales)
    #    2:          self.rows is a C-contiguous (N, d) float64 array and self.sortOrd an int64 array (before:
    #                Numba typed Lists, with Python-list copies rowsPy/sortOrdPy while "serialized"). Arrays
    #                pickle as they are, so serialize()/deserialize() have nothing left to do. The arrays are
    #                never changed in place after the constructor: a change replaces them.
    FORMAT_VERSION = 2

    def __init__(self, rows, tol=1e-9, rTol=1e-9, dirIndep=True, tailMask=1):
        self.formatVersion = vectorSet.FORMAT_VERSION
        self.tol = tol
        self.rTol = rTol
        self.dirIndep = dirIndep
        if not (isinstance(rows,np.ndarray) and len(rows.shape) == 2 and rows.dtype == np.float64):
            raise ValueError('Argument must be a 2-d NumPy array of float64\'s')
        self.d = rows.shape[1]
        self.N = rows.shape[0]
        if not isinstance(tailMask,int) and tailMask >=0 and tailMask < self.d - 1:
            raise ValueError(f'tailMask argument must be an integer between 0 and {self.d}')
        self.tailMask = tailMask
        if self.dirIndep:
            scale = self.dirScales(rows)
        else:
            scale = np.ones(self.N,dtype=np.float64)
        self.scale = scale.tolist()
        self.rows = np.ascontiguousarray(scale.reshape(-1,1) * rows)
        self.rowsPy = None
        self.sortOrd = np.arange(self.N,dtype=np.int64)
        quickSortRowwise(self.rows, self.sortOrd, self.tol, self.rTol)
        self.revSortOrd = getReverseOrder(self.sortOrd).tolist()
        _ , self.uniqRowIdx = selectUniqueRows(self.rows,self.sortOrd,self.tol,self.rTol)
        self.sortOrdPy = None
        self.uniqRowIdx = sorted(self.sortOrd[np.array(self.uniqRowIdx,dtype=np.int64)].tolist())
        self.uniqRowIdxSet = {i:idx for idx, i in enumerate(self.uniqRowIdx)}
        self.Nunique = len(self.uniqRowIdx)
        self.uniqRowSorted = True
        self.serialized = False

    # The scale factors that map each row of a 2-d array to its direction-independent ("dirIndep") representative,
    # i.e. to the same stored row for v and for any non-zero multiple of v (in particular -v):
    #   magnitude: 1 / (norm of the entries from tailMask on), or 1 if that norm is zero;
    #   sign:      such that the last entry of the scaled row that is non-zero (larger than tol in absolute
    #              value) is positive. NB: the sign cannot be taken from the last entry alone: if that entry is
    #              zero, it is zero for both v and -v, and the two would be stored as different rows.
    def dirScales(self, rows):
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, self.d)
        with np.errstate(divide='ignore', invalid='ignore'):
            mag = np.nan_to_num(1/np.linalg.norm(rows[:,self.tailMask:self.d],axis=1), nan=1.0, posinf=1.0)
        scaled = mag.reshape(-1,1) * rows
        nonZero = np.abs(scaled) > self.tol
        # Column of the last non-zero entry of each row (rows without one keep a positive sign)
        lastNonZero = self.d - 1 - np.argmax(nonZero[:,::-1], axis=1)
        sgn = np.where(scaled[np.arange(rows.shape[0]), lastNonZero] < 0, -1.0, 1.0)
        sgn[np.logical_not(np.any(nonZero, axis=1))] = 1.0
        return sgn * mag

    def dirScale(self, vec):
        return float(self.dirScales(vec.reshape(1,-1))[0])

    # Called by pickle (and copy/deepcopy) instead of updating __dict__ directly. Pickle stores only the
    # attributes of an instance, never the code of its class, so an object pickled by an older version of this
    # class comes back with the *current* methods and the *old* data; that data is upgraded here.
    def __setstate__(self, state):
        self.__dict__.update(state)
        if state.get('formatVersion', 0) < vectorSet.FORMAT_VERSION:
            self.upgradeFromLegacy()

    # Formats 0 and 1 kept the rows and the sort order as typed Lists (or as Python lists while serialized)
    def rowsToArrays(self):
        rows = self.rowsPy if self.__dict__.get('serialized', False) else self.rows
        sortOrd = self.sortOrdPy if self.__dict__.get('serialized', False) else self.sortOrd
        if not isinstance(rows, np.ndarray):
            rows = np.array([np.asarray(r,dtype=np.float64) for r in rows],dtype=np.float64).reshape(-1, self.d)
        if not isinstance(sortOrd, np.ndarray):
            sortOrd = np.array([int(i) for i in sortOrd],dtype=np.int64)
        self.rows, self.sortOrd = np.ascontiguousarray(rows,dtype=np.float64), np.ascontiguousarray(sortOrd,dtype=np.int64)
        self.rowsPy, self.sortOrdPy, self.serialized = None, None, False

    # Brings the data of an object created by an older version of this class up to FORMAT_VERSION (in place).
    # Returns the list of row indices whose stored sign changed.
    def upgradeFromLegacy(self):
        version = self.__dict__.get('formatVersion', 0)
        changed = []
        if version < 2:
            self.rowsToArrays()
        if version < 1 and self.dirIndep and self.N > 0:
            rows = self.rows.copy()
            # The stored rows are already scaled, so only the sign of the new scale factor matters
            sgn = self.dirScales(rows)
            changed = [i for i in range(self.N) if sgn[i] < 0]
            if len(changed) > 0:
                oldUniq = list(self.uniqRowIdx)
                for i in changed:
                    rows[i] = -rows[i]
                    self.scale[i] = -self.scale[i]
                # The changed rows have a new place in the sort order
                sortOrd = np.arange(self.N,dtype=np.int64)
                quickSortRowwise(rows, sortOrd, self.tol, self.rTol)
                self.revSortOrd = getReverseOrder(sortOrd).tolist()
                self.rows, self.sortOrd = rows, sortOrd
                # The set of unique rows is deliberately left as it is: users of this object (e.g. the
                # non-redundant hyperplanes of a flipConstraints object) refer to those rows. If two of them
                # are now recognized as duplicates of each other, all that can be done here is to say so.
                uniq = set(oldUniq)
                idx = 0
                while idx < self.N:
                    grp = [int(sortOrd[idx])]
                    while idx + 1 < self.N and vecEqualNb(rows[sortOrd[idx]], rows[sortOrd[idx+1]], self.tol, self.rTol):
                        idx += 1
                        grp.append(int(sortOrd[idx]))
                    idx += 1
                    if len(uniq & set(grp)) > 1:
                        warnings.warn(f'vectorSet (legacy object): rows {sorted(uniq & set(grp))} are duplicates of each other, ' \
                                        + 'but were stored as distinct unique rows by an older version of vectorSet')
        self.formatVersion = vectorSet.FORMAT_VERSION
        return changed

    # Kept for the callers: since format 2 there is nothing to convert (self.serialized stays False)
    def serialize(self):
        return self
    def deserialize(self):
        return self

    # The rows with the given indices as they were supplied, i.e. with the scaling undone. (An empty selection
    # gives an array of shape (0,), as before.)
    def unscaledRows(self, idx):
        if len(idx) == 0:
            return np.array([])
        idx = np.asarray(idx,dtype=np.int64)
        return (1/np.array(self.scale,dtype=np.float64)[idx]).reshape(-1,1) * self.rows[idx]

    def getRows(self):
        return self.unscaledRows(np.arange(self.N))

    def getRowsSorted(self):
        return self.unscaledRows(self.sortOrd)

    def getUniqueRows(self):
        if self.serialized:
            self.deserialize()
        if not self.uniqRowSorted:
            self.uniqRowIdx = sorted([int(self.sortOrd[i]) for i in self.uniqRowIdx])
            self.uniqRowIdxSet = {i:idx for idx, i in enumerate(self.uniqRowIdx)}
            self.uniqRowSorted = True
        return self.unscaledRows(self.uniqRowIdx)

    def getUniqueRowsSorted(self):
        if self.serialized:
            self.deserialize()
        if not self.uniqRowSorted:
            self.uniqRowIdx = sorted([int(self.sortOrd[i]) for i in self.uniqRowIdx])
            self.uniqRowIdxSet = {i:idx for idx, i in enumerate(self.uniqRowIdx)}
            self.uniqRowSorted = True
        return self.unscaledRows(self.sortOrd[np.array(self.uniqRowIdx,dtype=np.int64)])

    def insertRow(self, vec, includeDup=True):
        if self.serialized:
            self.deserialize()
        # includeDup=True will append the row to the full list of rows,
        # even if it is a duplicate
        if not ( isinstance(vec,np.ndarray) and self.d == math.prod(vec.shape) and vec.dtype == np.float64 ):
            raise ValueError(f'Can only insert floating point numpy vectors of length {self.d}')
        iVec = vec.flatten()
        if self.dirIndep:
            scale = self.dirScale(iVec)
            iVec = scale * iVec
        else:
            scale = 1.0
        _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
        if isNew or includeDup:
            self.N += 1
            self.rows = np.ascontiguousarray(np.vstack((self.rows, iVec.reshape(1,-1))))
            self.scale.append(scale)
            self.sortOrd = np.ascontiguousarray(np.insert(self.sortOrd, insertionPoint, self.N - 1))
            for rowIdx in self.sortOrd[(insertionPoint+1):].tolist():
                self.revSortOrd[rowIdx] += 1
            self.revSortOrd.append(insertionPoint)
            if isNew:
                newRowIdx = int(self.sortOrd[insertionPoint])
                if self.uniqRowSorted:
                    rowIdxIP = bisect.bisect(self.uniqRowIdx, newRowIdx)
                    self.uniqRowIdx.insert(rowIdxIP, newRowIdx)
                    self.uniqRowIdxSet[newRowIdx] = rowIdxIP
                    for i in range(rowIdxIP+1,len(self.uniqRowIdx)):
                        self.uniqRowIdxSet[self.uniqRowIdx[i]] += 1
                else:
                    self.uniqRowIdx.append(newRowIdx)
                    self.uniqRowIdx.sort()
                    self.uniqRowIdxSet = {i:idx for idx, i in enumerate(self.uniqRowIdx)}
                    self.uniqRowSorted = True
                self.Nunique = len(self.uniqRowIdx)
        return isNew

    def removeRow(self, idxOrigOrder=None, vec=None, includeDup=False):
        if self.serialized:
            self.deserialize()
        if vec is None and idxOrigOrder is not None:
            if idxOrigOrder < 0 or idxOrigOrder >= self.N:
                raise ValueError(f'Specified row index {idxOrigOrder} must be < {self.N}')
            vec = self.rows[idxOrigOrder]/self.scale[idxOrigOrder]
            # print(f'vec = {vec} scale = {self.scale[idxOrigOrder]}')
        if not ( isinstance(vec,np.ndarray) and self.d == math.prod(vec.shape) and vec.dtype == np.float64 ):
            raise ValueError(f'Can only insert floating point numpy vectors of length {self.d}')
        origVec = vec.flatten().copy()
        iVec = vec.flatten()
        if self.dirIndep:
            scale = self.dirScale(iVec)
            iVec = scale * iVec
        else:
            scale = 1.0
        # print(f'iVec = {iVec}')
        _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
        if isNew:
            return [], []
        else:
            toRemove = []
            # print((insertionPoint, self.sortOrd[insertionPoint-1]))
            # print(self.expandDuplicates(self.sortOrd[insertionPoint-1]))
            assert insertionPoint > 0, f'Non-new hyperplanes should always have an insertionPoint > 0'
            for idx in self.expandDuplicates(int(self.sortOrd[insertionPoint-1])):
                # print(self.rows[idx]/self.scale[idx])
                if includeDup or (idxOrigOrder is not None and idx == idxOrigOrder) \
                        or (idxOrigOrder is None and vecEqualNb(self.rows[idx]/self.scale[idx],origVec,self.tol,self.rTol)):
                    toRemove.append([idx,self.revSortOrd[idx],0])
            removed = []
            # remUniqRow: the row removed in the current iteration is a unique row; remAnyUniqRow: any of the
            # removed rows was a unique row (in which case a remaining duplicate has to take its place below)
            remAnyUniqRow = False
            while len(toRemove) > 0:
                rowIdx, sortIdx, cnt = toRemove.pop()
                removed.append(rowIdx + cnt)
                remUniqRow = False
                for i in range(len(toRemove)):
                    if toRemove[i][0] > rowIdx:
                        toRemove[i][0] -= 1
                        toRemove[i][2] += 1
                    if toRemove[i][1] > sortIdx:
                        toRemove[i][1] -= 1
                for i in range(len(self.sortOrd)):
                    if i in self.uniqRowIdxSet and i >= rowIdx:
                        temp = self.uniqRowIdxSet[i]
                        del self.uniqRowIdxSet[i]
                        if i == rowIdx:
                            remUniqRow = True
                            remAnyUniqRow = True
                        else:
                            self.uniqRowIdxSet[i-1] = temp if not remUniqRow else temp - 1
                sortOrd = np.delete(self.sortOrd, sortIdx)
                self.sortOrd = np.ascontiguousarray(np.where(sortOrd > rowIdx, sortOrd - 1, sortOrd))
                self.revSortOrd.pop(rowIdx)
                self.rows = np.ascontiguousarray(np.delete(self.rows, rowIdx, axis=0))
                self.scale.pop(rowIdx)
                for i in range(len(self.sortOrd)):
                    if self.revSortOrd[i] > sortIdx:
                        self.revSortOrd[i] -= 1
            self.N = self.rows.shape[0]
            self.uniqRowIdx = sorted(list(self.uniqRowIdxSet.keys()))
            added = []
            if remAnyUniqRow:
                _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
                remaining = [] if isNew else self.expandDuplicates(int(self.sortOrd[insertionPoint-1]))
                # (the second condition can only fail for an upgraded legacy object; see upgradeFromLegacy)
                if len(remaining) > 0 and len(self.uniqRowIdxSet.keys() & set(remaining)) == 0:
                    self.uniqRowIdx.append(min(remaining))
                    added.append(self.uniqRowIdx[-1])
                    self.uniqRowIdx.sort()
                    self.uniqRowIdxSet = {i:idx for idx, i in enumerate(self.uniqRowIdx)}
            self.Nunique = len(self.uniqRowIdx)
            return sorted(removed), added

    def expandDuplicates(self,idxOrigOrder=None,vec=None,tailMask=0):
        if self.serialized:
            self.deserialize()
        before = []
        after = []
        if vec is None and idxOrigOrder is not None:
            if idxOrigOrder >= self.N or idxOrigOrder < 0:
                raise ValueError(f'expandDuplicates: argument must be between {0} and {self.N}')
            origIdx = self.revSortOrd[idxOrigOrder]
            ref = self.rows[idxOrigOrder]
        elif vec is not None and idxOrigOrder is None:
            assert isinstance(vec,np.ndarray) and vec.flatten().size == self.d
            ref = vec.copy().flatten()
            if self.dirIndep:
                scale = self.dirScale(ref)
                ref = scale * ref
            _, origIdx, isNew = findInsertionPoint(self.rows, ref, self.sortOrd, self.tol, self.rTol)
            origIdx -= 1
            if isNew:
                return []
        else:
            raise ValueError('Must supply either an index or a vector (but not both)')
        idx = origIdx
        while idx >= 0 and vecEqualNb(self.rows[self.sortOrd[idx]][tailMask:self.d],ref[tailMask:self.d],self.tol,self.rTol):
            before.append(int(self.sortOrd[idx]))
            idx -= 1
        idx = origIdx + 1
        while idx < self.N and vecEqualNb(self.rows[self.sortOrd[idx]][tailMask:self.d],ref[tailMask:self.d],self.tol,self.rTol):
            after.append(int(self.sortOrd[idx]))
            idx += 1
        return sorted(before + after)

    # Returns (isNew, index into self.rows, index into self.uniqRowIdx) for the duplicate of the
    # provided input vector
    def findUniqueDuplicate(self,vec,tailMask=0):
        if not ( isinstance(vec,np.ndarray) and self.d == math.prod(vec.shape) and vec.dtype == np.float64 ):
            raise ValueError(f'Can only search for floating point numpy vectors of length {self.d}')
        iVec = vec.flatten()
        if self.dirIndep:
            scale = self.dirScale(iVec)
            iVec = scale * iVec
        else:
            scale = 1.0
        _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
        if isNew:
            return isNew, None, None
        else:
            idx = insertionPoint - 1
            if idx >= 0 and vecEqualNb(self.rows[self.sortOrd[idx]][tailMask:self.d],iVec[tailMask:self.d],self.tol,self.rTol):
                origIdx = list( self.uniqRowIdxSet.keys() & set(self.expandDuplicates(int(self.sortOrd[idx]),tailMask=tailMask)) )[0]
                if self.uniqRowSorted:
                    i = bisect.bisect_left(self.uniqRowIdx, origIdx)
                    if i < len(self.uniqRowIdx) and self.uniqRowIdx[i] == origIdx:
                        return isNew, origIdx, [i]
                    else:
                        return isNew, origIdx, [i for i in range(self.uniqRowIdx) if self.uniqRowIdx[i] == origIdx]
            idx = insertionPoint + 1
            if idx < self.N and vecEqualNb(self.rows[self.sortOrd[idx]][tailMask:self.d],iVec[tailMask:self.d],self.tol,self.rTol):
                origIdx = list( self.uniqRowIdxSet.keys() & set(self.expandDuplicates(int(self.sortOrd[idx]),tailMask=tailMask)) )[0]
                if self.uniqRowSorted:
                    i = bisect.bisect_left(self.uniqRowIdx, origIdx)
                    if i < len(self.uniqRowIdx) and self.uniqRowIdx[i] == origIdx:
                        return isNew, origIdx, [i]
                    else:
                        return isNew, origIdx, [i for i in range(self.uniqRowIdx) if self.uniqRowIdx[i] == origIdx]
            return isNew, None, [-1]

    def subtractSet(self, minusSet, subUniqueRows=True):
        if not isinstance(minusSet, vectorSet):
            raise ValueError(f'Argument must be a vectorSet')
        if self.serialized:
            self.deserialize()
        subRows = rowwiseSetComplement(minusSet.rows, minusSet.sortOrd, self.rows, self.tol, self.rTol)
        if subUniqueRows:
            return sorted(list(self.uniqRowIdxSet.keys() & set(np.nonzero(subRows)[0])))
        else:
            return sorted(list(np.nonzero(subRows)[0]))

    def isElem(self, vec):
        if not ( isinstance(vec,np.ndarray) and self.d == math.prod(vec.shape) and vec.dtype == np.float64 ):
            raise ValueError(f'Can only check floating point numpy vectors of length {self.d}')
        if self.serialized:
            self.deserialize()
        iVec = vec.flatten()
        if self.dirIndep:
            scale = self.dirScale(iVec)
            iVec = scale * iVec
        else:
            scale = 1.0
        _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
        return not isNew

    def selectRow(self, vec):
        if not ( isinstance(vec,np.ndarray) and self.d == math.prod(vec.shape) and vec.dtype == np.float64 ):
            raise ValueError(f'Can only check floating point numpy vectors of length {self.d}')
        if self.serialized:
            self.deserialize()
        iVec = vec.flatten()
        if self.dirIndep:
            scale = self.dirScale(iVec)
            iVec = scale * iVec
        else:
            scale = 1.0
        _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
        return (not isNew, insertionPoint)

    def listParallel(self, vec):
        if not ( isinstance(vec,np.ndarray) and self.d == math.prod(vec.shape) and vec.dtype == np.float64 ):
            raise ValueError(f'Can only check floating point numpy vectors of length {self.d}')
        if self.serialized:
            self.deserialize()
        iVec = vec.flatten()
        if self.dirIndep:
            scale = self.dirScale(iVec)
            iVec = scale * iVec
        else:
            scale = 1.0
        _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
        cnt = 0
        foundMatch = False
        for off in [insertionPoint-1,insertionPoint,insertionPoint+1]:
            if off >=0 and off < self.N and vecEqualNb(self.rows[self.sortOrd[off]][0:self.d],iVec[0:self.d],self.tol,self.rTol):
                insertionPoint = off
                foundMatch = True
                break
            cnt += 1
        # Can obviously be optimized
        if not isNew and foundMatch:
            identicalHyperplanes = set(self.expandDuplicates(int(self.sortOrd[insertionPoint]),tailMask=0))
        else:
            identicalHyperplanes = set()
        cnt = 0
        foundMatch = False
        for off in [insertionPoint-1,insertionPoint,insertionPoint+1]:
            if off >=0 and off < self.N and vecEqualNb(self.rows[self.sortOrd[off]][self.tailMask:self.d],iVec[self.tailMask:self.d],self.tol,self.rTol):
                insertionPoint = off
                foundMatch = True
                break
            cnt += 1
        if foundMatch:
            parallelHyperplanes = set(self.expandDuplicates(int(self.sortOrd[insertionPoint]),tailMask=self.tailMask)) - identicalHyperplanes
        else:
            parallelHyperplanes = set()
        if not isNew or len(parallelHyperplanes) > 0:
            return (sorted(list(identicalHyperplanes)), sorted(list(parallelHyperplanes)))
        else:
            return None

    def vecEqual(self, vec1, vec2):
        return vecEqualNb(vec1, vec2, self.tol, self.rTol)

    def vecCompare(self, vec1, vec2):
        return vecCompareNb(vec1, vec2, self.tol, self.rTol)

# Numba helper methods go here...

searchType = types.Tuple((types.int64[::1],types.int64[::1]))

@njit( \
    types.boolean \
    ( \
       types.float64[::1], \
       types.float64[::1], \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def vecCompareNb(vec1, vec2, tol, rTol):
    inOrder = True
    for i in range(vec1.shape[0]-1,-1,-1):
        if vec1[i] >= tol + rTol * abs(vec2[i]) + vec2[i]:
            inOrder = False
            break
        elif vec1[i] <= -tol - rTol * abs(vec2[i]) + vec2[i]:
            break
    return inOrder

def vecEqualNb(vec1,vec2, tol, rTol):
    return np.all(np.abs(vec1.reshape(vec2.shape) - vec2) <= tol + rTol * np.abs(vec2)) or np.all(np.abs(vec1.reshape(vec2.shape) + vec2) <= tol + rTol * np.abs(vec2))

@njit( \
    types.int64 \
    ( \
       types.float64[:,::1], \
       types.int64[::1], \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def quickSortRowwise(mat, reorder, tol, rTol):
    pivot = 0
    term = 0
    idx = 0
    n = reorder.shape[0]
    d = 0
    temp = np.zeros((d,),dtype=np.float64)
    tempOrd = np.int64(0)
    if n <= 1:
        return -1
    elif n == 2:
        if not vecCompareNb(mat[reorder[0]],mat[reorder[1]],tol,rTol):
            tempOrd = reorder[0]
            reorder[0] = reorder[1]
            reorder[1] = tempOrd
        return -1
    d = mat.shape[1]
    # Copy pivot to end of array
    tempOrd = reorder[pivot]
    reorder[pivot] = reorder[-1]
    reorder[-1] = tempOrd
    pivotRow = mat[reorder[-1]]

    idx = 0
    # Start term at the last non-pivot row (after copy above)
    term = n - 2
    while idx < term:
        if not vecCompareNb(mat[reorder[idx]],pivotRow,tol,rTol):
            tempOrd = reorder[idx]
            reorder[idx] = reorder[term]
            reorder[term] = tempOrd
            term = term - 1
        else:
            idx = idx + 1
    if not vecCompareNb(mat[reorder[term]],pivotRow,tol,rTol):
        tempOrd = reorder[term]
        reorder[term] = reorder[-1]
        reorder[-1] = tempOrd
        pivot = term
    else:
        if term + 1 < n - 1:
            tempOrd = reorder[term+1]
            reorder[term+1] = reorder[-1]
            reorder[-1] = tempOrd
        pivot = term + 1
    return quickSortRowwise(mat,reorder[:pivot],tol,rTol) + quickSortRowwise(mat,reorder[(pivot+1):],tol,rTol)

def getReverseOrder(reorder):
    revOrd = np.empty(len(reorder),dtype=np.int64)
    revOrd[np.asarray(reorder,dtype=np.int64)] = np.arange(len(reorder),dtype=np.int64)
    return revOrd

@njit( \
    searchType \
    ( \
       types.float64[:,::1], \
       types.float64[::1], \
       types.int64[::1], \
       types.int64, \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def getRowsBinarySearch(mat, row, revSortOrd, col, tol, rTol):
    n = len(revSortOrd)
    d = row.shape[0]
    lb = -1
    ub = n
    idx = n // 2
    windL = 0
    windR = n
    if n == 0:
        return revSortOrd[0:0], np.array([0, 0],dtype=np.int64)
    if row[col] > mat[revSortOrd[-1]][col] + tol + rTol * abs(mat[revSortOrd[-1]][col]):
        return revSortOrd[0:0], np.array([n, n],dtype=np.int64)
    if row[col] < mat[revSortOrd[0]][col] - tol - rTol * abs(mat[revSortOrd[0]][col]):
        return revSortOrd[0:0], np.array([0, 0],dtype=np.int64)

    windL = 0
    windR = n - 1
    while True:
        if abs(windL - windR) <= 1:
            lb = windL if mat[revSortOrd[windL]][col] >= row[col] - tol - rTol * abs(row[col]) else windR
            break
        idx = (windR + windL) // 2
        if mat[revSortOrd[idx]][col] > row[col] - tol - rTol * abs(row[col]):
            windR = idx
        else:
            windL = idx

    windL = 0
    windR = n - 1
    while True:
        if abs(windL - windR) <= 1:
            ub = windR if mat[revSortOrd[windR]][col] <= row[col] + tol + rTol * abs(row[col]) else windL
            break
        idx = (windR + windL) // 2
        if mat[revSortOrd[idx]][col] > row[col] + tol + rTol * abs(row[col]):
            windR = idx
        else:
            windL = idx

    return revSortOrd[lb:(ub+1)], np.array([lb, ub+1],dtype=np.int64)

@njit( \
    types.Tuple((types.int64, types.int64, types.boolean)) \
    ( \
       types.float64[:,::1], \
       types.float64[::1], \
       types.int64[::1], \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def findInsertionPoint(mat, row, revSortOrd, tol, rTol):
    col = 0
    temp = revSortOrd
    loc = 0
    relLoc = np.zeros(2,dtype=np.int64)
    if mat.shape[0] == 0:
        return 0, 0, True
    d = mat.shape[1]
    col = d-1
    while len(temp) > 0 and col >= 0:
        loc += relLoc[0]
        revSortOrd, relLoc = getRowsBinarySearch(mat, row, revSortOrd, col, tol, rTol)
        col = col - 1
    # Leave here
    return  loc + relLoc[0], loc + relLoc[1], (True if relLoc[1] - relLoc[0] == 0 else False)

# The duplicate scan of selectUniqueRows below, compiled: row r (in sorted order) is a duplicate if it equals
# the row before it, or its negative, within the tolerances (the same test as vecEqualNb, entry by entry: with
# v1 the row before and v2 this row, all |v1 - v2| <= tol + rTol |v2|, or all |v1 + v2| <= tol + rTol |v2|; a NaN
# never passes). Returns the mask of rows that are not duplicates and their positions in the sorted order.
@njit( \
    types.Tuple((types.boolean[::1], types.int64[::1])) \
    ( \
       types.float64[:,::1], \
       types.int64[::1], \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def selectUniqueRowsNb(mat, sortOrd, tol, rTol):
    n = sortOrd.shape[0]
    selIdx = np.ones(n,dtype=np.bool_)
    subRows = np.empty(n,dtype=np.int64)
    if n == 0:
        return selIdx, subRows
    d = mat.shape[1]
    subRows[0] = 0
    cnt = 1
    for r in range(1,n):
        v1 = mat[sortOrd[r-1]]
        v2 = mat[sortOrd[r]]
        same = True
        opposite = True
        for k in range(d):
            bound = tol + rTol * abs(v2[k])
            if not (abs(v1[k] - v2[k]) <= bound):
                same = False
            if not (abs(v1[k] + v2[k]) <= bound):
                opposite = False
            if not (same or opposite):
                break
        if same or opposite:
            selIdx[r] = False
        else:
            subRows[cnt] = r
            cnt += 1
    return selIdx, subRows[:cnt].copy()

def selectUniqueRows(mat, sortOrd, tol, rTol):
    n = sortOrd.shape[0]
    if n == 0:
        return np.ones(n,dtype=np.bool_), tuple()
    selIdx, subRows = selectUniqueRowsNb(mat, sortOrd, tol, rTol)
    return selIdx, subRows.tolist()

@njit( \
    types.boolean \
    ( \
       types.float64[:,::1], \
       types.int64[::1], \
       types.float64[:,::1], \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def isSubsetApprox(mat1, sortOrd1, mat2, tol, rTol):
    n1 = mat1.shape[0]
    n2 = mat2.shape[0]
    if n2 == 0:
        return True
    if n1 == 0:
        return False
    d = mat1.shape[1]
    if n2 > n1:
        return False
    foundCnt = 0
    for idx in range(n2):
        temp = sortOrd1
        col = d-1
        while len(temp) > 0 and col >= 0:
            temp, _ = getRowsBinarySearch(mat1, mat2[idx], temp, col, tol, rTol)
            col = col - 1
        if len(temp) >= 1 and np.all(np.abs(mat1[temp[0]] - mat2[idx]) < tol):
            foundCnt = foundCnt + 1
    if foundCnt == n2:
        return True
    else:
        return False

# Returns a boolean list of rows that are in mat2 but NOT in mat1, provided len(mat2) <= len(mat1)
# i.e. it is a set complement mat2 \ mat1
@njit( \
    types.boolean[:] \
    ( \
       types.float64[:,::1], \
       types.int64[::1], \
       types.float64[:,::1], \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def rowwiseSetComplement(mat1, sortOrd1, mat2, tol, rTol):
    n1 = mat1.shape[0]
    n2 = mat2.shape[0]
    retVal = np.ones((n2,),dtype=np.bool_)
    if n2 == 0:
        return retVal
    if n1 == 0:
        retVal = np.zeros((n2,),dtype=np.bool_)
        return retVal
    d = mat1.shape[1]
    foundCnt = 0
    for idx in range(n2):
        temp = sortOrd1
        col = d-1
        while len(temp) > 0 and col >= 0:
            temp, _ = getRowsBinarySearch(mat1, mat2[idx], temp, col, tol, rTol)
            col = col - 1
        if len(temp) >= 1 and np.all(np.abs(mat1[temp[0]] - mat2[idx]) < tol):
            retVal[idx] = False
    return retVal

# def removeDupRows(mat, tol, rTol):
#     n = mat.shape[0]
#     d = mat.shape[1]
#     matIdx = mat.copy()
#     quickSortRowwise(matIdx,np.arange(n),tol,rTol)
#     lut = {}
#     selIdx = np.ones(n,dtype=np.bool_)
#     for r in range(n):
#         col = 0
#         temp = matIdx
#         while temp.shape[0] > 0 and col < d:
#             temp, _ = getRowsBinarySearch(temp, mat[r,:], col, tol, rTol)
#             col = col + 1
#         if temp.shape[0] > 1:
#             lutVal = tuple(temp[0,:].tolist())
#             if lutVal in lut:
#                 if lut[lutVal] > 1:
#                     selIdx[r] = False
#                     lut[lutVal] = lut[lutVal] - 1
#             else:
#                 lut[lutVal] = temp.shape[0] - 1
#                 selIdx[r] = False
#     return mat[selIdx,:], selIdx

