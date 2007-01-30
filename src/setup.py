from distutils.core import setup, Extension

dataio = Extension('dataio',
                    include_dirs = ['/usr/lib/python2.4/site-packages/numpy/core/include'],
                    libraries = ['dataio'],
                    library_dirs = ['libdataio'],
                    sources = ['dataio.c']
                    )
	
setup(name = "datio",
      version = "1.0",
      description = "Wrapper for libdataio",
      maintainer = "CD Meliza",
      maintainer_email = "dmeliza@uchicago.edu",
      ext_modules = [ dataio ]
      )
