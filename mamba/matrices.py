import scipy.sparse as sparse
from collections import Sequence
import ctypes as ct
import numpy as np
from ctypes import byref
from typing import Union, TypeVar, Tuple
import numbers


class MambaMatrix(object):
    def issymmetric(self):
        raise NotImplementedError(
            "issymmetric not implemented for {}".format(self.__class__.__name__))

    def __getattr__(self, attr):
        for obj in [self] + self.__class__.mro():
            if attr in obj.__dict__:
                return obj.__dict__[attr]
        return super().__getattr__(attr)


class scipymkl_csr(sparse.csr.csr_matrix):
    # a sketch of a mamba matrix
    # this class pairs a SQUARE ctypes csr sparse matrix with a scipy csr sparse matrix
    # if a property is upper case it returns a ctypes instance.
    # origin controls the indexing of the ctypes data and the python data.
    # the scipy data is indexed from 0
    # if ia,ja,and a=values are shortcuts to indptr, indices, with offset
    def __init__(self, M=None, offset=1, n=None,
                 a=None, values=None, ia=None, indptr=None,
                 ja=None, indices=None, **kwargs):
        self.offset = offset  # default to 1 for ilu
        self.__build(M, n, a, values, ia, indptr, ja, indices, **kwargs)

    def __build(self, M=None, n=None,
                a=None, values=None, ia=None,
                indptr=None, ja=None, indices=None, **kwargs):

        # Assume that data coming in is always 0 indexed.
        ix = self.offset
        if M is None:
            # take
            if a is not None:
                values = a
            if ia is not None:
                assert ja is not None
                indptr = ia - ix
                indices = ja - ix

            matvars = {'values': values, 'indptr': indptr, 'indices': indices}
            for var in matvars.items():
                assert isinstance(var[1], (Sequence, np.ndarray, np.matrix)), \
                    "{} must be an array when M is None".format(var[0])
#             if isinstance(n, numbers.Number):
#                 rowsize = np.rint(n) == [np.rint(len(ia))]
#                 assert any(rowsize), \
#                     "Array size variable n did not match rows ia"
            assert len(values) == len(indices), \
                "Length of values does not match length of indices"

            if n is None:
                n = len(values)

            super().__init__((values, indices, indptr), shape=(n, n), **kwargs)
        else:
            super().__init__(M, **kwargs)
        self.sort_indices()

    def copy_update(self, dat):
        M = self.copy()
        M.data = dat
        return M

    @property
    def NNZ(self):
        return ct.c_int(self.nnz)

    @property
    def N(self):
        # get ctypes matrix shape
        return ct.c_int(self.n)

    @property
    def n(self):
        # get matrix axis shape as python int
        return self.shape[0]

    @property
    def DATA(self):
        # get matrix values in ctypes double
        return (ct.c_double * (self.nnz))(*self.data)

    @property
    def IA(self):
        # get row pointers in ctypes int with indexing offset controlled by origin
        # last element is nnz.
        return (ct.c_int * (self.n + 1))(*(self.ia))

    @property
    def ia(self):
        return self.indptr + self.offset

    @property
    def JA(self):
        # get column indices for data in ctypes int with indexing offset
        return (ct.c_int * (self.nnz))(*(self.ja))

    @property
    def ja(self):
        return self.indices + self.offset
