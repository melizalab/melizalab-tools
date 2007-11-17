from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib
import os,sys

nxdir = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')

_pcmseqio = Extension('_pcmseqio',
                    include_dirs = [nxdir],
                    sources = ['pcmseqio.c','pcmio.c']
                    )
_readklu = Extension('_readklu',
                     sources = ['readklu.cc'],
                     )

	
setup(name = "dlab",
      version = "1.0",
      description = "Some small extension modules",
      maintainer = "CD Meliza",
      maintainer_email = "dmeliza@uchicago.edu",
      ext_modules = [ _pcmseqio, _readklu ]
      )
