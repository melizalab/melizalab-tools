from distutils.core import setup, Extension
from distutils.sysconfig import get_python_lib
import os,sys

nxdir = os.path.join(get_python_lib(plat_specific=1), 'numpy/core/include')
cxxsupp = os.path.normpath(os.path.join(sys.prefix,'share',
                                        'python%d.%d' % (sys.version_info[0],sys.version_info[1]),
                                        'CXX') )
if os.name == 'posix':
        CXX_libraries = ['stdc++','m']
else:
        CXX_libraries = []

_pcmseqio = Extension('_pcmseqio',
                    include_dirs = [nxdir],
                    sources = ['pcmseqio.c','pcmio.c']
                    )

_readklu = Extension('_readklu',
                     include_dirs = [nxdir],
                     sources = ['readklu.cc',
                                os.path.join(cxxsupp,'cxxsupport.cxx'),
                                os.path.join(cxxsupp,'cxx_extensions.cxx'),
                                os.path.join(cxxsupp,'cxxextensions.c')])

	
setup(name = "dlab",
      version = "1.0",
      description = "Some small extension modules",
      maintainer = "CD Meliza",
      maintainer_email = "dmeliza@uchicago.edu",
      ext_modules = [ _pcmseqio, _readklu ]
      )
