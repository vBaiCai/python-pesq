import warnings
import numpy as np
from pesq_core import _pesq
from math import fabs
EPSILON = 1e-6

def pesq(ref, deg, fs=16000, normalize=False):
    '''
    params:
        ref: ref signal,
        deg: deg signal, 
        fs: sample rate,
    '''
    ref = np.array(ref, copy=True)
    deg = np.array(deg, copy=True)

    if normalize:
        ref = ref/np.max(np.abs(ref)) if np.abs(ref) > EPSILON else ref 
        deg = deg/np.max(np.abs(deg)) if np.abs(deg) > EPSILON else deg

    max_sample = np.max(np.abs(np.array([ref, deg])))
    if max_sample > 1:
        c = 1 / max_sample
        ref = ref * c
        deg = deg * c

    if ref.ndim != 1 or deg.ndim != 1:
        raise ValueError("signals must be 1-D array ")

    if fs not in [16000, 8000]:
        raise ValueError("sample rate must be 16000 or 8000")

    if fabs(ref.shape[0] - deg.shape[0]) > fs / 4:
        raise ValueError("ref and deg signals should be in same length.")

    if np.count_nonzero(ref==0) == ref.size:
        raise ValueError("ref is all zeros, processing error! ")

    if np.count_nonzero(deg==0) == deg.size:
        raise ValueError("deg is all zeros, pesq score is nan! ")

    if ref.dtype != np.int16:
        ref *= 32767
        ref = ref.astype(np.int16)

    if deg.dtype != np.int16:
        deg *= 32767
        deg = deg.astype(np.int16)

    try:
        score = _pesq(ref, deg, fs)
    except:
        warnings.warn('Processing Error! return NaN')
        score = np.NAN

    return score
