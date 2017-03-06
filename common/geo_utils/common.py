#
# Some common useful functions
#

import os
import numpy as np
import gdal


def get_basename(filepath):
    """
    Get filename basename without extension
    For example:
    >>> get_basename("/path/to/a/file.123sdufg.sdfs.tiff")
    'file.123sdufg.sdfs'

    """
    bfn = os.path.basename(filepath)
    splt = bfn.split('.')
    return '.'.join(splt[:-1]) if len(splt) > 1 else splt[0]


def get_dtype(depth, is_complex, signed=True):
    """
    Method to convert the pair (depth={1,2,4,8}, is_complex={True,False})
    into numpy dtype
    For example,
    >>> get_type(4, False)
    <type 'numpy.float32'>
    """
    if depth == 1 and not is_complex:
        return np.uint8
    elif depth == 2 and not is_complex:
        return np.uint16 if signed else np.int16
    elif depth == 4 and not is_complex:
        return np.float32
    elif depth == 8 and not is_complex:
        return np.float64
    elif depth == 8 and is_complex:
        return np.complex64
    elif depth == 16 and is_complex:
        return np.complex128
    else:
        raise AssertionError("Data type is not recognized")


def get_gdal_dtype(depth, is_complex, signed=True):
    """
    Method to convert the pair (depth={1,2,4,8}, is_complex={True,False})
    If is_complex == True, depth corresponds real and imaginary parts
    to GDAL data type : gdal.GDT_Byte, ...
    >>> get_gdal_dtype(4, False) == gdal.GDT_Float32
    True
    >>> get_gdal_dtype(8, True) == gdal.GDT_CFloat32
    True
    """
    if depth == 1 and not is_complex:
        return gdal.GDT_Byte
    elif depth == 2 and not is_complex:
        return gdal.GDT_UInt16 if signed else gdal.GDT_Int16
    elif depth == 4 and not is_complex:
        return gdal.GDT_Float32
    elif depth == 8 and not is_complex:
        return gdal.GDT_Float64
    elif depth == 8 and is_complex:
        return gdal.GDT_CFloat32
    elif depth == 16 and is_complex:
        return gdal.GDT_CFloat64
    else:
        raise AssertionError("Data type is not recognized")


def gdal_to_numpy_datatype(gdal_datatype):
    """
    Method to convert gdal data type to numpy dtype
    >>> gdal_to_numpy_datatype(gdal.GDT_Float32) == np.float32
    True
    """
    if gdal_datatype == gdal.GDT_Byte:
        return np.uint8
    elif gdal_datatype == gdal.GDT_Int16:
        return np.int16
    elif gdal_datatype == gdal.GDT_Int32:
        return np.int32
    elif gdal_datatype == gdal.GDT_UInt16:
        return np.uint16
    elif gdal_datatype == gdal.GDT_UInt32:
        return np.uint32
    elif gdal_datatype == gdal.GDT_Float32:
        return np.float32
    elif gdal_datatype == gdal.GDT_Float64:
        return np.float64
    elif gdal_datatype == gdal.GDT_CInt16:
        # No associated type -> cast to complex64
        return np.complex64
    elif gdal_datatype == gdal.GDT_CInt32:
        # No associated type -> cast to complex64
        return np.complex64
    elif gdal_datatype == gdal.GDT_CFloat32:
        return np.complex64
    elif gdal_datatype == gdal.GDT_CFloat64:
        return np.complex128
    else:
        raise AssertionError("Data type '%i' is not recognized" % gdal_datatype)