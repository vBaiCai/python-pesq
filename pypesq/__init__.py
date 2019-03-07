import numpy as np
from pesq_core import _pesq
from math import fabs

def pesq(ref, deg, fs):
    '''
    params:
        ref: ref signal,
        deg: deg signal, 
        fs: sample rate,
    '''
    ref = np.array(ref, copy=True)
    deg = np.array(deg, copy=True)

    ref = 0.999*ref/np.max(np.abs(ref))
    deg = 0.999*deg/np.max(np.abs(deg))

    if ref.ndim != 1 or deg.ndim != 1:
        raise ValueError("signals must be 1-D array ")

    if fs not in [16000, 8000]:
        raise ValueError("sample rate must be 16000 or 8000")

    if fabs(ref.shape[0] - deg.shape[0]) > fs / 4:
        raise ValueError("ref and deg signals should be in same length.")

    if ref.dtype != np.int16:
        ref *= 32768
        ref = ref.astype(np.int16)

    if deg.dtype != np.int16:
        deg *= 32768
        deg = deg.astype(np.int16)

    score = _pesq(ref, deg, fs)

    return score
