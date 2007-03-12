from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib
import os

nxdir = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')

_pcmseqio = Extension('_pcmseqio',
                    include_dirs = [nxdir],
                    sources = ['pcmseqio.c','pcmio.c']
                    )
	
setup(name = "_pcmseqio",
      version = "1.0",
      description = "Wrapper for pcm_seq2 read/write functions",
      maintainer = "CD Meliza",
      maintainer_email = "dmeliza@uchicago.edu",
      ext_modules = [ _pcmseqio ]
      )
