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

console scripts
~~~~~~~~~~~~~~~

-  ``group-kilo-spikes``: sort spike times output from
   `kilosort <https://github.com/MouseLand/Kilosort>`__ and
   `phy2 <https://github.com/cortex-lab/phy/>`__ into pprox files.

other stuff
~~~~~~~~~~~

-  ``scripts/extract_waveforms.py``: extracts spike waveforms from a raw
   recording (in ARF format) using spike times stored in a file (pprox
   format). This script is mostly only used to verify that spike sorting
   is working properly, because the ``group-kilo-spikes`` script has an
   option to store average waveforms.
