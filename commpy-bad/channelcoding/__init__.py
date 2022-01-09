"""
============================================
Channel Coding (:mod:`channelcoding`)
============================================

.. module:: channelcoding

Galois Fields
=============

.. autosummary::
    :toctree: generated/

    GF   -- Class representing a Galois Field object.

Algebraic Codes
===============

.. autosummary::
    :toctree: generated/

    cyclic_code_genpoly -- Generate a cylic code generator polynomial.


Convolutional Codes
===================

.. autosummary::
    :toctree: generated/

    Trellis          -- Class representing convolutional code trellis.
    conv_encode      -- Convolutional Encoder.
    viterbi_decode   -- Convolutional Decoder using the Viterbi algorithm.


Turbo Codes
===========

.. autosummary::
    :toctree: generated/

    turbo_encode    -- Turbo Encoder.
    map_decode      -- Convolutional Code decoder using MAP algorithm.
    turbo_decode    -- Turbo Decoder.

LDPC Codes
==========

.. autosummary::
    :toctree: generated/

    get_ldpc_code_params    -- Extract parameters from LDPC code design file.
    ldpc_bp_decode          -- LDPC Code Decoder using Belief propagation.

Interleavers and De-interleavers
================================

.. autosummary::
    :toctree: generated/

    RandInterlv    -- Random Interleaver.

"""

from channelcoding.convcode import Trellis, conv_encode, viterbi_decode
from channelcoding.interleavers import *
from channelcoding.turbo import turbo_encode, map_decode, turbo_decode
# from channelcoding.ldpc import get_ldpc_code_params, ldpc_bp_decode
from channelcoding.gfields import *
from channelcoding.algcode import *

try:
    from numpy.testing import Tester
    test = Tester().test
except:
    pass
