from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import os

try:
    import numpy as np
    includes = [os.path.join(np.get_include(), 'numpy')] 
except:
    includes = []

extension = Extension("pesq_core",
                      sources=["pypesq/pesq.c", "pypesq/dsp.c", "pypesq/pesqdsp.c", "pypesq/pesqio.c", "pypesq/pesqmain.c", "pypesq/pesqmod.c"],
                      include_dirs=includes, 
                      language='c++')

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


setup(name='pypesq',
    version='1.1',
    description="A package to compute pesq score.",
    url='https://github.com/vBaiCai/python-pesq',
    author_email='zhuroubaicai@gmail.com',
    keywords=['pesq', 'speech', 'speech quality'],
    license='MIT',
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={'build_ext': build_ext},
    setup_requires=['numpy'],
    py_modules=['numpy'],
    zip_safe=False,
    install_requires=['numpy'],
    python_requires='!=3.0.*, !=3.1.*, !=3.2.*, <4',
)
