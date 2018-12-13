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
    if fs not in [16000, 8000]:
        raise "sample rate must be 16000 or 8000"

    if fabs(len(ref) - len(deg)) > fs / 4:
        raise "ref and deg signals should be in same length."

    if ref.dtype != np.int16:
        ref *= 32768
        ref = ref.astype(np.int16)

    if deg.dtype != np.int16:
        deg *= 32768
        deg = deg.astype(np.int16)

    score = _pesq(ref, deg, fs)

    return score