from enum import Enum
import numpy as np


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices
    Matched with Megatron-LM specifications.
    """

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8
    uint32 = 9

    @classmethod
    def code_from_dtype(cls, value: "DType") -> int:
        """Get the code from the numpy dtype"""
        if value == np.uint8:
            return cls.uint8.value
        elif value == np.int8:
            return cls.int8.value
        elif value == np.int16:
            return cls.int16.value
        elif value == np.int32:
            return cls.int32.value
        elif value == np.int64:
            return cls.int64.value
        elif value == np.float64:
            return cls.float64.value
        elif value == np.float32:
            return cls.float32.value
        elif value == np.uint16:
            return cls.uint16.value
        elif value == np.uint32:
            return cls.uint32.value
        else:
            raise ValueError(f"Unsupported dtype: {value}")

    @classmethod
    def dtype_from_code(cls, value: int) -> "np.dtype":
        """Get the numpy dtype from the code"""
        if value == cls.uint8.value:
            return np.uint8
        elif value == cls.int8.value:
            return np.int8
        elif value == cls.int16.value:
            return np.int16
        elif value == cls.int32.value:
            return np.int32
        elif value == cls.int64.value:
            return np.int64
        elif value == cls.float64.value:
            return np.float64
        elif value == cls.float32.value:
            return np.float32
        elif value == cls.uint16.value:
            return np.uint16
        elif value == cls.uint32.value:
            return np.uint32
        else:
            raise ValueError(f"Unsupported code: {value}")
