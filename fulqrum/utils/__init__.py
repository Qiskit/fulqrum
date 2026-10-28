# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities
=========

IO
--

.. autosummary::
   :toctree: ../stubs/

   dict_to_json
   json_to_dict

"""

from .matrix import kron_str, qubitoperator_to_matrix
from .io import dict_to_json, json_to_dict

__all__ = ["dict_to_json", "json_to_dict", "kron_str", "qubitoperator_to_matrix"]
