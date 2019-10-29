import mamba
from mamba import cptr, DefaultTypes
import ctypes as ct
from ctypes import POINTER
import numpy as np
from numpy.ctypeslib import as_ctypes

d = DefaultTypes()
c_types = d._simple_types


def c_equivalent(a, b):
    assert isinstance(a, type(b))
    if hasattr(a, 'value'):
        return a.value == b.value
    if hasattr(a, "__getitem__"):
        return a[:] == b[:]
    raise Exception("Not able to compare a and b")


def test_DefaultTypes():
    d = DefaultTypes()

    # check error messaging
    base_error = d._errormsg
    try:
        d['foo']
    except KeyError as e:
        pass
    d._errormsg = 'foo'
    try:
        d['foo{}']
    except KeyError as e:
        pass

    # check that everything is a type
    for k in d.keys():
        assert isinstance(k, (type, np.dtype))
        assert isinstance(d[k], type)
    return True


def test_cptr_ptr():
    # check that cptr wraps POINTER for ctypes type inputs
    for simpletype in c_types:
        assert isinstance(cptr(simpletype),
                          type(POINTER(simpletype)))
        assert isinstance(cptr(simpletype * 2),
                          type(POINTER(simpletype * 2)))


def test_cptr_c_data():
    # test second clause of cptr(obj), pointers to data
    for simpletype in c_types:
        cdata = simpletype(0)
        varptr = POINTER(simpletype)(cdata)
        assert isinstance(cptr(cdata), type(varptr))
        cdata = (simpletype * 2)(0, 0)
        arrptr = POINTER(simpletype * 2)(cdata)
        assert isinstance(cptr(cdata), type(arrptr))


def test_cptr_void_ptr():
    assert isinstance(cptr(None), type(POINTER(ct.c_void_p)))


def test_cptr_py_int():
    val = 0
    c_T = ct.c_int
    c_PT = POINTER(c_T)
    c_PV = c_PT(c_T(val))
    assert isinstance(cptr(0), type(c_PV))


def test_cptr_py_list_int():
    val = [1, 2]
    c_T = ct.c_int * 2
    c_PT = POINTER(c_T)
    c_PV = c_PT(c_T(*val))
    assert isinstance(cptr([1, 2]), type(c_PV))


def test_cptr_py_float():
    val = 0.0
    c_T = ct.c_double
    c_PT = POINTER(ct.c_double)
    c_PV = c_PT(c_T(val))
    assert isinstance(cptr(0.0), type(c_PV))


def test_cptr_py_list_float():
    val = [1., 2.]
    c_T = ct.c_double * 2
    c_PT = POINTER(c_T)
    c_PV = c_PT(c_T(*val))
    assert isinstance(cptr([1., 2.]), type(c_PV))


def test_cptr_nparr_alltypes():
    for pytype in d.keys():
        if pytype.__class__.__name__.startswith('d'):
            # check for the dtype because
            # coercing a byte array of floats will go to zero and fail
            # because we expect a byte array and get ints.
            coerce_np = False
        else:
            coerce_np = True

        arr = np.random.randn(3).astype(pytype)
        c_T = (d[pytype] * 3)
        c_PT = POINTER(c_T)
        c_PV = c_PT(c_T(*arr))
        assert isinstance(
            cptr(arr, coerce_np=coerce_np), type(c_PV))


def test_monkey_np_ctypeslib_as_ctypes_scalar():
    assert c_equivalent(as_ctypes(1), ct.c_int(1))
    assert c_equivalent(as_ctypes(1.0), ct.c_double(1))


def test_monkey_np_ctypeslib_as_ctypes_list():
    assert c_equivalent(as_ctypes([1, 2]), (ct.c_int * 2)(*[1, 2]))
    assert c_equivalent(as_ctypes([1., 2.]), (ct.c_double * 2)(*[1., 2.]))


def test_monkey_np_ctypeslib_as_ctypes_with_cast():
    assert c_equivalent(
        as_ctypes([1, 2], T=ct.c_float), (ct.c_float * 2)(*[1., 2.]))


def test_monkey_np_ctypeslib_as_ctypes_nparray():
    for pytype in d.keys():
        if pytype.__class__.__name__.startswith('d'):
            # check for the dtype because
            # coercing a byte array of floats will go to zero and fail
            # because we expect a byte array and get ints.
            coerce_np = False
        else:
            coerce_np = True
        arr = np.random.randn(3).astype(pytype)
        carr = as_ctypes(arr, coerce_np=coerce_np)
        assert isinstance(carr, type((d[pytype] * 3)(*[1, 2, 3])))
        assert np.allclose(carr[:], arr)
        carr = as_ctypes(arr, T=ct.c_int)
        assert isinstance(carr, type((ct.c_int * 3)(*[1, 2, 3])))


def test_monkey_np_ctypeslib_as_ctypes_not_implemented_type():
    try:
        as_ctypes('str')
        raise Exception("monkey np.ctypeslib did not catch a not implemented type")
    except NotImplementedError:
        pass


def test_monkey_np_ctypeslib_as_ctypes_invalid_T():
    try:
        as_ctypes(1, T=str)
        raise Exception("monkey np.ctypeslib did not catch T = non-ctypes type")
    except ValueError:
        pass


def test_monkey_np_ctypeslib_as_ctypes_ambiguous_list():
    try:
        as_ctypes([1., 2])
        raise Exception("Monkey np.ctypeslib.as_ctypes did not catch ambiguous list")
    except NotImplementedError:
        pass
