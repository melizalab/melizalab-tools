dlab
----

A collection of python code and scripts used by the Meliza Lab.

To install: ``pip install melizalab-tools``

modules
~~~~~~~

-  ``dlab.pprox``: functions for working with
   `pprox <https://meliza.org/spec:2/pprox/>`__ objects, a data format
   for storing multi-trial point process data (e.g. spike times evoked
   by stimulus presentation).
-  ``dlab.neurobank``: some convenient wrappers for interacting with a
   `neurobank <https://github.com/melizalab/neurobank/>`__ repository and archives.
-  ``dlab.signal``:  signal processing functions

console scripts
~~~~~~~~~~~~~~~

-  ``group-kilo-spikes``: sort spike times output from
   `kilosort <https://github.com/MouseLand/Kilosort>`__ and
   `phy2 <https://github.com/cortex-lab/phy/>`__ into pprox files.
- ``get-songs``: extract segments from arf files, rescale, resample, save into
   wave files, and optionally deposit back into neurobank. Usually the first
   step in generating a stimulus set, keeps a nice provenance trail.

other stuff
~~~~~~~~~~~

