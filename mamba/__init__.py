import numpy as np
import ctypes
from numbers import Number
from typing import Dict, Type, NewType, Any
from collections import Sequence
ct = ctypes

# Calling type(foo) for foo = ctypes.c_<type> or foo = ptr(ctypes.c_<type>)
# returns types _ctypes.PyCSimpleType, _ctypes.PyCPointerType
# But we can't access these so for now we will just make fake types.
# Maybe wrap them?
cType = NewType('cType', Type)  # _ctypes.PyCSimpleType
cPtr = NewType('cPtr', Type)  # _ctypes.PyCPointerType
# cVar = NewType('cVar', ) #similar problem


class DefaultTypes(Dict[Type, cType]):
    # a static class intended for
    _errormsg = "{} is not implemented with a ctypes pair."
    # defaults are c_int and c_double because MKL
    _pairs = {int: ct.c_int, float: ct.c_double}
    # use this to get the numpy default types
    _simple_types = [
        ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong,
        ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong,
        ct.c_float, ct.c_double,
    ]
    _pairs.update({np.dtype(ctype): ctype for ctype in _simple_types})

    def __init__(self):
        super().__init__(self._pairs)

    def __getitem__(self, key: Type) -> cType:
        try:
            return super().__getitem__(key)
        except:
            raise KeyError(self._errormsg.format(key))


def cptr(obj: Any, **kwargs) -> cPtr:
    # get a cPtr from any Python object for which np.as_ctypes is defined.
    from ctypes import POINTER as ptr
    if obj.__class__.__name__ in ['PyCArrayType', 'PyCSimpleType']:
        return ptr(obj)
    elif obj.__class__.__name__.startswith('c_'):
        return ptr(type(obj))(obj)
    elif obj is None:
        return ptr(ct.c_void_p)
    else:
        cvar = np.ctypeslib.as_ctypes(obj, **kwargs)
        return ptr(type(cvar))(cvar)


# overload np.as_ctypes
_npc_as_ctypes = np.ctypeslib.as_ctypes


def as_ctypes_non_np(obj: Any, T: cType=None,
                     D: Dict[Type, cType]=DefaultTypes(),
                     coerce_np: bool=True):
    # Add docstring, output type, support for strings, and ctypes-recasting
    # extend as_ctypes to include lists and scalars
    if isinstance(obj, list):  # lists logic
        ambiguous = False  # flag for ambiguous / mixed types
        objtypes = [type(obj[0])]  # the types in the list
        for ele in obj:
            if (isinstance(ele, Sequence)):
                raise NotImplementedError("Casting lists of Sequence is not supported")
            if not isinstance(ele, type(obj[0])):
                objtypes.append(type(ele))
                ambiguous = True

        if T is None:  # Get a default type
            if ambiguous:
                raise NotImplementedError("Input is a List{} with ambiguous types."
                                          "To cast, provide T=ctypes.<type>".format(
                                              objtypes))
            T = D[type(obj[0])]
        if not T.__class__.__name__.endswith('PyCSimpleType'):
            raise ValueError("Supplied T is not a ctypes type")

        sz = len(obj)
        c_T = T * sz  # c array of type T of length sz
        cvar = (c_T(*obj))  # instantiate c_T with elements of obj.

    elif isinstance(obj, Number):  # scalars
        if T is None:  # Get a default type
            T = D[type(obj)]
        if not T.__class__.__name__.endswith('PyCSimpleType'):
            raise ValueError("Supplied T is not a ctypes type")

        c_T = T
        cvar = (c_T(obj))

    elif hasattr(obj, '__array_interface__'):  # has the numpy array interface
        if T is None:
            if coerce_np:
                if np.allclose(obj.astype(int), obj):  # default to c_int
                    T = int
                elif np.allclose(obj.astype(float), obj):  # default to c_double
                    T = float
            else:
                T = obj.dtype
            T = D[T]
        if not T.__class__.__name__.endswith('PyCSimpleType'):
            raise ValueError("Supplied T is not a ctypes type")
        obj = obj.astype(T)
        cvar = _npc_as_ctypes(obj)
    else:
        raise NotImplementedError("Cannot cast object of type {}".format(type(obj)))
    return cvar


np.ctypeslib.as_ctypes = as_ctypes_non_np
