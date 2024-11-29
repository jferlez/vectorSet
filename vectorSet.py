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

class vectorSet:
    def __init__(self, rows, tol=1e-9, rTol=1e-9, dirIndep=True, tailMask=1):
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
            self.scale = ( np.sign(np.sign(rows[:,-1]) + 0.1) * np.nan_to_num(1/np.linalg.norm(rows[:,self.tailMask:self.d],axis=1), nan=1.0, posinf=1.0)  ).tolist()
        else:
            self.scale = [1.0 for r in range(self.N)]
        self.rows = nb.typed.List( [ self.scale[i] * rows[i].copy() for i in range(self.N) ] )
        self.rowsPy = None
        self.sortOrd = np.arange(self.N)
        quickSortRowwise(self.rows, self.sortOrd, self.tol, self.rTol)
        self.revSortOrd = getReverseOrder(self.sortOrd).tolist()
        _ , self.uniqRowIdx = selectUniqueRows(self.rows,self.sortOrd,self.tol,self.rTol)
        self.sortOrd = nb.typed.List( self.sortOrd )
        self.sortOrdPy = None
        self.uniqRowIdx = sorted([self.sortOrd[i] for i in self.uniqRowIdx])
        self.uniqRowIdxSet = {i:idx for idx, i in enumerate(self.uniqRowIdx)}
        self.Nunique = len(self.uniqRowIdx)
        self.uniqRowSorted = True
        self.serialized = False

    def serialize(self):
        if not self.serialized:
            self.rowsPy = list(self.rows)
            self.rows = None
            self.sortOrdPy = list(self.sortOrd)
            self.sortOrd = None
            self.serialized = True
        return self
    def deserialize(self):
        if self.serialized:
            self.rows = nb.typed.List(self.rowsPy)
            self.rowsPy = None
            self.sortOrd = nb.typed.List(self.sortOrdPy)
            self.sortOrdPy = None
            self.serialized = False
        return self

    def getRows(self):
        if self.serialized:
            self.deserialize()
        return np.array( [(1/self.scale[i])*self.rows[i] for i in range(self.N)] )

    def getRowsSorted(self):
        if self.serialized:
            self.deserialize()
        return np.array( [(1/self.scale[self.sortOrd[i]])*self.rows[self.sortOrd[i]] for i in range(self.N)] )

    def getUniqueRows(self):
        if self.serialized:
            self.deserialize()
        if not self.uniqRowSorted:
            self.uniqRowIdx = sorted([self.sortOrd[i] for i in self.uniqRowIdx])
            self.uniqRowSorted = True
        return np.array( [(1/self.scale[i]) * self.rows[i] for i in self.uniqRowIdx] )

    def getUniqueRowsSorted(self):
        if self.serialized:
            self.deserialize()
        if not self.uniqRowSorted:
            self.uniqRowIdx = sorted([self.sortOrd[i] for i in self.uniqRowIdx])
            self.uniqRowSorted = True
        return np.array( [(1/self.scale[self.sortOrd[i]]) * self.rows[self.sortOrd[i]] for i in self.uniqRowIdx] )

    def insertRow(self, vec, includeDup=True):
        if self.serialized:
            self.deserialize()
        # includeDup=True will append the row to the full list of rows,
        # even if it is a duplicate
        if not ( isinstance(vec,np.ndarray) and self.d == math.prod(vec.shape) and vec.dtype == np.float64 ):
            raise ValueError(f'Can only insert floating point numpy vectors of length {self.d}')
        iVec = vec.flatten()
        if self.dirIndep:
            scale = (1.0 if iVec[-1] >= 0 else -1.0) / np.nan_to_num( np.linalg.norm(iVec[self.tailMask:self.d]), nan=1.0, posinf=1.0 )
            iVec = scale * iVec
        else:
            scale = 1.0
        _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
        if isNew or includeDup:
            self.N += 1
            self.rows.append(iVec)
            self.scale.append(scale)
            self.sortOrd.insert(insertionPoint, self.N - 1)
            for idx in range(insertionPoint+1, self.N):
                self.revSortOrd[self.sortOrd[idx]] += 1
            self.revSortOrd.append(insertionPoint)
            if isNew:
                if self.uniqRowSorted:
                    rowIdxIP = bisect.bisect(self.uniqRowIdx, self.sortOrd[insertionPoint])
                    self.uniqRowIdx.insert(rowIdxIP, self.sortOrd[insertionPoint]  )
                    for i in range(rowIdxIP+1,len(self.uniqRowIdx)):
                        self.uniqRowIdxSet[self.uniqRowIdx[i]] += 1
                else:
                    self.uniqRowIdx.append(insertionPoint)
                    self.uniqRowIdx.sort()
                    self.uniqRowIdxSet = {i:idx for idx, i in enumerate(self.uniqRowIdx)}
                    self.uniqRowSorted = True
                self.Nunique = len(self.uniqRowIdx)
        return isNew

    def expandDuplicates(self,idxOrigOrder,ref=None,tailMask=0):
        if self.serialized:
            self.deserialize()
        if idxOrigOrder >= self.N or idxOrigOrder < 0:
            raise ValueError(f'expandDuplicates: argument must be between {0} and {self.N}')
        before = []
        after = []
        origIdx = self.revSortOrd[idxOrigOrder]
        if ref is None:
            ref = self.rows[idxOrigOrder]
        else:
            assert isinstance(ref,np.ndarray) and ref.size == self.d
            ref = ref.copy().flatten()
        idx = origIdx
        while idx >= 0 and vecEqualNb(self.rows[self.sortOrd[idx]][tailMask:self.d],ref[tailMask:self.d],self.tol,self.rTol):
            before.append(self.sortOrd[idx])
            idx -= 1
        idx = origIdx + 1
        while idx < self.N and vecEqualNb(self.rows[self.sortOrd[idx]][tailMask:self.d],self.rows[idxOrigOrder][tailMask:self.d],self.tol,self.rTol):
            after.append(self.sortOrd[idx])
            idx += 1
        return sorted(before + after)

    # Returns (isNew, index into self.rows, index into self.uniqRowIdx) for the duplicate of the
    # provided input vector
    def findUniqueDuplicate(self,vec,tailMask=0):
        if not ( isinstance(vec,np.ndarray) and self.d == math.prod(vec.shape) and vec.dtype == np.float64 ):
            raise ValueError(f'Can only search for floating point numpy vectors of length {self.d}')
        iVec = vec.flatten()
        if self.dirIndep:
            scale = (1.0 if iVec[-1] >= 0 else -1.0) * np.nan_to_num( 1/np.linalg.norm(iVec[self.tailMask:self.d]), nan=1.0, posinf=1.0 )
            iVec = scale * iVec
        else:
            scale = 1.0
        _, insertionPoint, isNew = findInsertionPoint(self.rows, iVec, self.sortOrd, self.tol, self.rTol)
        if isNew:
            return isNew, None, None
        else:
            idx = insertionPoint - 1
            if idx >= 0 and vecEqualNb(self.rows[self.sortOrd[idx]][tailMask:self.d],iVec[tailMask:self.d],self.tol,self.rTol):
                origIdx = list( self.uniqRowIdxSet.keys() & set(self.expandDuplicates(self.sortOrd[idx],tailMask=tailMask)) )[0]
                if self.uniqRowSorted:
                    i = bisect.bisect_left(self.uniqRowIdx, origIdx)
                    if i < len(self.uniqRowIdx) and self.uniqRowIdx[i] == origIdx:
                        return isNew, origIdx, i
                    else:
                        return isNew, origIdx, [i for i in range(self.uniqRowIdx) if self.uniqRowIdx[i] == origIdx]
            idx = insertionPoint + 1
            if idx < self.N and vecEqualNb(self.rows[self.sortOrd[idx]][tailMask:self.d],iVec[tailMask:self.d],self.tol,self.rTol):
                origIdx = list( self.uniqRowIdxSet.keys() & set(self.expandDuplicates(self.sortOrd[idx],tailMask=tailMask)) )[0]
                if self.uniqRowSorted:
                    i = bisect.bisect_left(self.uniqRowIdx, origIdx)
                    if i < len(self.uniqRowIdx) and self.uniqRowIdx[i] == origIdx:
                        return isNew, origIdx, i
                    else:
                        return isNew, origIdx, [i for i in range(self.uniqRowIdx) if self.uniqRowIdx[i] == origIdx]
            return isNew, None, -1

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
            scale = (1.0 if iVec[-1] >= 0 else -1.0) * np.nan_to_num( 1/np.linalg.norm(iVec[self.tailMask:self.d]), nan=1.0, posinf=1.0 )
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
            scale = (1.0 if iVec[-1] >= 0 else -1.0) * np.nan_to_num( 1/np.linalg.norm(iVec[self.tailMask:self.d]), nan=1.0, posinf=1.0 )
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
            scale = (1.0 if iVec[-1] >= 0 else -1.0) * np.nan_to_num( 1/np.linalg.norm(iVec[self.tailMask:self.d]), nan=1.0, posinf=1.0 )
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
            identicalHyperplanes = set(self.expandDuplicates(self.sortOrd[insertionPoint],tailMask=0))
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
            parallelHyperplanes = set(self.expandDuplicates(self.sortOrd[insertionPoint],ref=iVec,tailMask=self.tailMask)) - identicalHyperplanes
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

searchType = types.Tuple((types.ListType( types.int64 ),types.int64[::1]))

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
       types.ListType( types.float64[::1] ), \
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
    d = mat[0].shape[0]
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
    return np.array([r[1] for r in sorted(list(zip(reorder,np.arange(len(reorder)))))],dtype=np.int64)

@njit( \
    searchType \
    ( \
       types.ListType( types.float64[::1] ), \
       types.float64[::1], \
       types.ListType( types.int64 ), \
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
        retVal = nb.typed.List.empty_list(nb.int64)
        return retVal, np.array([0, 0],dtype=np.int64)
    if row[col] > mat[revSortOrd[-1]][col] + tol + rTol * abs(mat[revSortOrd[-1]][col]):
        retVal = nb.typed.List.empty_list(nb.int64)
        return retVal, np.array([n, n],dtype=np.int64)
    if row[col] < mat[revSortOrd[0]][col] - tol - rTol * abs(mat[revSortOrd[0]][col]):
        retVal = nb.typed.List.empty_list(nb.int64)
        return retVal, np.array([0, 0],dtype=np.int64)

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

def findInsertionPoint(mat, row, revSortOrd, tol, rTol):
    col = 0
    temp = revSortOrd
    loc = 0
    relLoc = (0,0)
    if len(mat) == 0:
        return 0, False
    d = mat[0].shape[0]
    col = d-1
    while len(temp) > 0 and col >= 0:
        loc += relLoc[0]
        revSortOrd, relLoc = getRowsBinarySearch(mat, row, revSortOrd, col, tol, rTol)
        col = col - 1
    # Leave here
    return  loc + relLoc[0], loc + relLoc[1], (True if relLoc[1] - relLoc[0] == 0 else False)

def selectUniqueRows(mat, sortOrd, tol, rTol):
    n = sortOrd.shape[0]

    subRows = []
    selIdx = np.ones(n,dtype=np.bool_)
    if n == 0:
        return selIdx, tuple(subRows)
    d = mat[0].shape[0]
    subRows.append(0)
    for r in range(1,n):
        if vecEqualNb(mat[sortOrd[r-1]], mat[sortOrd[r]], tol, rTol):
            selIdx[r] = False
        else:
            subRows.append(r)
    return selIdx, subRows

@njit( \
    types.boolean \
    ( \
       types.ListType( types.float64[::1] ), \
       types.ListType(types.int64), \
       types.ListType( types.float64[::1] ), \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def isSubsetApprox(mat1, sortOrd1, mat2, tol, rTol):
    n1 = len(mat1)
    n2 = len(mat2)
    if n2 == 0:
        return True
    if n1 == 0:
        return False
    d = mat1[0].shape[0]
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
       types.ListType( types.float64[::1] ), \
       types.ListType(types.int64), \
       types.ListType( types.float64[::1] ), \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def rowwiseSetComplement(mat1, sortOrd1, mat2, tol, rTol):
    n1 = len(mat1)
    n2 = len(mat2)
    retVal = np.ones((n2,),dtype=np.bool_)
    if n2 == 0:
        return retVal
    if n1 == 0:
        retVal = np.zeros((n2,),dtype=np.bool_)
        return retVal
    d = mat1[0].shape[0]
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

