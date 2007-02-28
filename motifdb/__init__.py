#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""

* Features and motifs

This module was written with the purpose of unifying how motifs and
features are stored and symbolically referenced.  One of the main
design considerations is that the same motif (which in this context is
simply a chunk of sound data from a recording of a bird) may need to
have different names in different experiments. Consequently, the
database stores several lookup tables which are called _stimsets_, and
the choice of stimset controls how symbols are resolved by the module.

Features are components of motifs; a motif can be divided into a set
of disjoint spectrotemporal regions; this is called a _feature
map_. This feature map can be thought of as a matrix of integers, with
dimensions equal in size to some spectrotemporal representation of the
motif.  The (x,y) locations in this matrix having a certain value
comprise a mask which can be used in an ifft process to reconstruct
the signal giving rise to that region of the STFT. Of course, there is
more than one way to divide up the spectrotemporal space, which means
that multiple feature maps may be associated with a single motif.

Thus, to unequivocally refer to a particular feature of a motif (assuming
that a particular stimset defines the base motif symbol), we use the symbol:

<motif>_<featmap>.<feature>

For example A2_0.0 refers to the first feature in the first featuremap
associated with motif A2.


* Data storage

The database consists of a set of tables that (a) represent the
mappings between symbols and motifs, featuremaps, and features, (b)
store various bits of metadata about the entities, and (c) store the
integer matrices for feature maps and pcm data for features.  The last
of these functions is not strictly necessary, as this data can easily
be stored in the filesystem, but the hdf5 format used to store the
tables can also store numerical arrays in a compressed format.  Both
of these types of data tend to be very low-entropy.


* Recombining signals

This sections deals with the grammar of specifying complex signal
recombinations.  The math is extremely straightforward, given the
time-series data for all the features and their relative offsets
within their source motif (at least until we start considering
frequency shifting). The grammar needs to strike a balance between
compactness and completeness. Compact because the resulting sequence
names need to be reasonable in length (since they will probably be
used to name files), and complete because changing the grammar will
make it difficult to compare results across multiple naming schemes.

The most common kind of artificial stimulus will be based on a single
motif.  These include, in order of increasing complexity:
	- complete reconstructions (all features)
	- partial reconstructions (subset of features)
	- reconstructions with time or frequency shifted features
	- substitutions or insertions of features from other motifs
	- substitutions or insertions of artificial noises
	- random or pseudorandom combinations of features

All but the last of these can be taken care with a modified slice
notation. The notion has a couple of problematic shell characters, but
I don't really see any way around it. Using A2 as an example:

A2_0(:)		- a complete reconstruction of A2 based on featuremap 0

A2_0(1)		- feature 1 from A2. This differs from A2_0.1 in that feature
		1 is embedded at its original temporal offset

A2_0(1:3)	- features 1 through 3 from A2

A2_0(1,2,5) 	- features 1,2,5 from A2

A2_0(-1)	- all the features from A2 except 1

A2_0(-1:4) 	- all the features except 1..4.  The minus sign has lower
		precedence than ..

Comma and slice notation can be combined, as in A2_0(1:2,5). However, only
individual atoms can take advantage of the following syntax for modifying
feature placement

A2_0(1,2t50)    - consists of feature 1 from A2, placed at its original
		location, and feature 2, placed 50 ms later than its
		original location.  Negative offsets work as well.

A2_0(1,2f50)	- same as above, but feature 2 is upshifted 50 Hz.

Modifiers may be combined, and other modifiers can be defined.

Finally, the notation can accept "foreign" features and artificial
sounds. For example, the following replaces feature 1 with feature 2
from B2. The offset of any foreign feature is considered to be 0.

A2_0(-1,B2_0.2t100)

"""

# load the main module

from db import *
import schema


__all__ = ['db', 'schema','importer','combiner']
