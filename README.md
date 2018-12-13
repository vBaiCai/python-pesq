# PyPESQ
Pypesq is a python wrapper for the PESQ score calculation C routine. It's only can be used in evaluation purpose.

## INSTALL
```
python setup.py install
```

## HOW TO USE
```python
import soundfile as sf
from pypesq import pesq

ref, sr = sf.read(...)
deg, sr = sf.read(...)

score = pesq(ref, deg, sr)
print(score)
```

# NOTICE:
OWNERS of PESQ ARE:
1.	British Telecommunications plc (BT), all rights assigned to Psytechnics Limited
2.	Royal KPN NV, all rights assigned to OPTICOM GmbH

# REFERENCES:
* [ITU-T P.862](https://www.itu.int/rec/T-REC-P.862/en)
* [C_Extensions_NumPy_arrays](https://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html)
* [kennethreitz/setup.py](https://github.com/kennethreitz/setup.py)
* [massover/accel](https://github.com/massover/accel)