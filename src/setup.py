from distutils.core import setup, Extension

_pcmseqio = Extension('_pcmseqio',
                    include_dirs = ['/usr/lib/python2.4/site-packages/numpy/core/include'],
                    sources = ['pcmseqio.c','pcmio.c']
                    )
	
setup(name = "_pcmseqio",
      version = "1.0",
      description = "Wrapper for pcm_seq2 read/write functions",
      maintainer = "CD Meliza",
      maintainer_email = "dmeliza@uchicago.edu",
      ext_modules = [ _pcmseqio ]
      )
