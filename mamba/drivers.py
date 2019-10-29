import ctypes as ct
import os
import sys


def full_lib_path(lib: string, path: string) -> List[string]:
    """ Scan `path` for shared object library `lib`

    Args:
        lib: String shared object library name without extension
        path: Path to shared object libraries with ending slash `/`

    Returns:
        list: Path to valid shared object library(ies)
    """
    exts = ['so', 'dll', 'dylib']
    if path is None:  # just going to scan LD_LIBRARY_PATH or equivalent
        if sys.platform.startswith('freebsd') \
                or sys.platform.startswith('linux') \
                or sys.platform.startswith('aix'):  # *nix not mac
            ext = '.so'
        elif sys.platform.startswith('win32') \
                or sys.platform.startswith('cygwin'):  # windows
            ext = '.dll'
        elif sys.platform.startswith('darwin'):  # osx
            ext = '.dylib'
        return [lib + ext]
    os.chdir(path)
    candidates = [file for file in glob.glob(lib + '*')
                  if file.lower().split('.')[-1] in exts]
    if len(candidates) == 0:
        return False
    elif len(candidates) == 1:
        return [path + candidates[0]]
    else:
        return [path + candidate for candidate in candidates]


def allbyref(f, *args):
    # this function calls all args in args using byref.
    return f(*[byref(arg) for arg in args])


class Driver(object):
    __lib = "default_library_here"

    def __init__(self, path: string=None):
        libs = full_lib_path(self.__lib, path)
        if len(libs) > 1:
            raise Exception("More than one library found, unable to resolve")
        try:
            lib = ct.cdll.LoadLibrary(libs[0])
        except Exception as e:
            raise type(e), \
                type(e)(e.message +
                        "Unable to load {} for {}".format(
                            self.__lib, self.__class__.__name__)), \
                sys.exc_info()[2]

    def ilu(self, *args):
        raise NotImplementedError(
            "Incomplete LU not implemented for {}".format(self.__class__.__name__))

    def ichol(self, *args):
        raise NotImplementedError(
            "Incomplete Cholesky not implemented for {}".format(self.__class__.__name__))

    def BiCG(self, *args):
        raise NotImplementedError(
            "BiCG not implemented for {}".format(self.__class__.__name__))

    def PCG(self, *args):
        raise NotImplementedError(
            "BiCG not implemented for {}".format(self.__class__.__name__))


class MKL(Driver):
    _lib = "libmkl_rt"

    def __init__(self, path: string=None):
        super().init(path)

    def gmres_init(n: int, x: np.ndarray = None, b: np.ndarray = None) -> Tuple[cintArray128, cdoubleArray128]:
        """Load default parameter arrays for MKL sparse linear algebra

        The `Intel MKL
        <https://software.intel.com/en-us/mkl-developer-reference-c-dgmres-init>`_
        C function signature is
        ::
            void dgmres_init (const MKL_INT *n , const double *x ,const double *b ,
                MKL_INT *RCI_request , MKL_INT *ipar , double *dpar , double *tmp );

        See:
        `Intel MKL Docs
        <https://software.intel.com/en-us/mkl-developer-reference-c-dgmres-init>`_

        Args:
            n : Size of matrix problem.
            x : Initial estimate of x. Defaults to np.zeros(n).
            b : Initial estimate of y. Defaults to np.zeros(n).
        Returns:
            ``(ctypes.c_int*128)(ipar), (ctypes.c_double*128)(dpar)`` :
            MKL integer and double parameter arrays `ipar, dpar`
        """
        try:
            gmres_init = mkl.dgmres_init
        except:
            raise
        assert isinstance(
            n, int) and n > 0, "Problem size must be positive integer"

        if x is None:
            x = np.zeros(n)
        if b is None:
            b = np.zeros(n)
        if all([isinstance(x, np.ndarray), isinstance(b, np.ndarray)]):
            x = (ct.c_double * n)(*x)
            b = (ct.c_double * n)(*b)
            # the temp array is set according to ipar[14] which defaults to
            # min(150,n)
            tmp = min([150, n])
            tmp = ((2 * tmp + 1) * n + tmp * (tmp + 9) / 2 + 1)
            tmp = int(tmp)
            tmp = (ct.c_double * tmp)(*np.zeros(tmp))
            n = ct.c_int(n)
            rci = ct.c_int(0)
            ipar = (ct.c_int * 128)(*np.zeros(128, dtype=int))
            dpar = (ct.c_double * 128)(*np.zeros(128, dtype=ct.c_double))
            allbyref(gmres_init, n, x, b, rci, ipar, dpar, tmp)
            return ipar, dpar
        else:
            raise exception("gmres failed")

    def ilu(M: scipymkl_csr, intel_params: Tuple[cintArray128, cdoubleArray128] = None) -> scipymkl_csr:
        """Incomplete LU factorization with zero threshold using MKL dscrilu0

        This function produces an incomplete LU-factorization of the 
        scipy-coupled MKL CSR matrix X.  It uses the zero pattern of the original matrix
        and produces a new scipymkl_csr instance.
        The intel_params array tuple should be obtained by calling gmres_init.
        The `Intel MKL
        <https://software.intel.com/en-us/mkl-developer-reference-c-dcsrilu0>`_
        C function signature is
        ::
            void dcsrilu0 (const MKL_INT *n , const double *a , const MKL_INT *ia , 
                const MKL_INT *ja , double *bilu0 , const MKL_INT *ipar , 
                const double *dpar , MKL_INT *ierr );

        See:
        `Intel MKL Docs
        <https://software.intel.com/en-us/mkl-developer-reference-c-dcsrilu0>`_

        Args:
            M : Square sparse MKL-coupled matrix
            intel_params : (ipar,dpar) Integer and double parameters
                    for MKL gmres/ilu0. Defaults to output of gmres_init
            b : Initial estimate of y. Defaults to np.zeros(n).
        Returns:
            ``(ctypes.c_int*128)(ipar), (ctypes.c_double*128)(dpar)`` :
            MKL integer and double parameter arrays `ipar, dpar`
        """
        try:
            ilu = mkl.dcsrilu0
        except:
            raise
        assert isinstance(M, scipymkl_csr), "ilu0 is not defined for "\
            "non-MKL paired sparsed matrices"

        if intel_params is None:
            ipar, dpar = gmres_init(M.n)
        else:
            ipar, dpar = intel_params

        size_output = M.nnz
        bilu0 = ct.c_double * (size_output)
        bilu0 = bilu0(*np.zeros(size_output, dtype=ct.c_double))
        err = ct.c_int()
        allbyref(ilu, M.N, M.DATA, M.IA, M.JA, bilu0, ipar, dpar, err)
        dat = bilu0[:]
        L = M.copy_update(np.array(dat))
        return L
