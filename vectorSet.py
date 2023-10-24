import numpy as np
import math

# Numba imports
import numba as nb
from numba import njit
from numba.core import types
from numba.typed import Dict
from numba.np.unsafe.ndarray import to_fixed_tuple

class vectorSet:
    def __init__(self, rows, tol=1e-9, rTol=1e-9):
        self.tol = tol
        self.rTol = rTol
        if not (isinstance(rows,np.ndarray) and len(rows.shape) == 2 and rows.dtype == np.float64):
            raise ValueError('Argument must be a 2-d NumPy array of float64\'s')
        self.d = rows.shape[1]
        self.N = rows.shape[0]
        self.rows = nb.typed.List( [ rows[i].copy() for i in range(self.N) ] )
        self.sortOrd = np.arange(self.N)
        quickSortRowwise(self.rows, self.sortOrd, self.tol, self.rTol)
        self.revSortOrd = getReverseOrder(self.sortOrd)
        self.uniqRowNP, self.uniqRows = selectUniqueRows(self.rows,self.sortOrd,self.tol,self.rTol)

    def insertRow(self, vec):
        pass

    def vecEqual(self, vec1, vec2):
        return vecEqualNb(vec1, vec2, self.tol, self.rTol)

    def vecCompare(self, vec1, vec2):
        return vecCompareNb(vec1, vec2, self.tol, self.rTol)

# Numba helper methods go here...

searchType = types.Tuple((types.int64[:],types.int64[:]))

@njit( \
    types.boolean \
    ( \
       types.float64[:], \
       types.float64[:], \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def vecCompareNb(vec1, vec2, tol, rTol):
    inOrder = True
    for i in range(vec1.shape[0]):
        if vec1[i] >= tol + rTol * vec2[i] + vec2[i]:
            inOrder = False
            break
        elif vec1[i] <= tol + rTol * vec2[i] + vec2[i]:
            break
    return inOrder

def vecEqualNb(vec1,vec2, tol, rTol):
    return np.all(np.abs(vec1 - vec2) <= tol + rTol * np.abs(vec2)) or np.all(np.abs(vec1 + vec2) <= tol + rTol * np.abs(vec2))

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
       types.List( types.float64[:] ), \
       types.float64[:], \
       types.int64[:], \
       types.int64, \
       types.float64, \
       types.float64 \
    ), \
    cache=True \
)
def getRowsBinarySearch(mat, row, revSortOrd, col, tol, rTol):
    n = revSortOrd.shape[0]
    d = row.shape[0]
    lb = -1
    ub = n
    idx = n // 2
    windL = 0
    windR = n
    if n == 0:
        return np.empty(d,dtype=np.int64), np.array([0, 0],dtype=np.int64)
    if row[col] > mat[revSortOrd[-1]][col] + tol + rTol * abs(mat[revSortOrd[-1]][col]):
        return np.empty(d,dtype=np.int64), np.array([n, n],dtype=np.int64)
    if row[col] < mat[revSortOrd[0]][col] - tol - rTol * abs(mat[revSortOrd[0]][col]):
        return np.empty(d,dtype=np.int64), np.array([0, 0],dtype=np.int64)

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
    while temp.shape[0] > 0 and col < d:
        loc += relLoc[0]
        revSortOrd, relLoc = getRowsBinarySearch(mat, row, revSortOrd, col, tol, rTol)
        col = col + 1
    # Leave here
    return  loc + relLoc[1], (True if relLoc[1] - relLoc[0] == 0 else False)

def selectUniqueRows(mat, sortOrd, tol, rTol):
    n = sortOrd.shape[0]

    subRows = []
    selIdx = np.ones(n,dtype=np.bool_)
    if n == 0:
        return selIdx, tuple(subRows)
    d = mat[0].shape[0]
    for r in range(1,n):
        if vecEqualNb(mat[sortOrd[r-1]], mat[sortOrd[r]], tol, rTol):
            selIdx[r] = False
        else:
            subRows.append(r)
    return selIdx, tuple(subRows)

@njit( \
    types.boolean \
    ( \
       types.List( types.float64[:] ), \
       types.int64[:], \
       types.List( types.float64[:] ), \
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
        col = 0
        while temp.shape[0] > 0 and col < d:
            temp, _ = getRowsBinarySearch(mat1, mat2[idx], temp, col, tol, rTol)
            col = col + 1
        if temp.shape[0] >= 1 and np.all(np.abs(mat1[temp[0]] - mat2[idx]) < tol):
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
       types.List( types.float64[:] ), \
       types.int64[:], \
       types.List( types.float64[:] ), \
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
        col = 0
        while temp.shape[0] > 0 and col < d:
            temp, _ = getRowsBinarySearch(mat1, mat2[idx], sortOrd1, col, tol, rTol)
            col = col + 1
        if temp.shape[0] >= 1 and np.all(np.abs(mat1[temp[0]] - mat2[idx]) < tol):
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

