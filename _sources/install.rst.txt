############
Installation
############

Requirements
============

Fulqrum is a mix of C++ and Python, glued together by a Cython interface.  

Python
~~~~~~

Required
********

.. csv-table::
   :header: "Package", "Minimum version", "Description"
   :widths: 15, 10, 30

   "cython", 3.0.5, "The glue between C++ and Python"
   "numpy", 2.0, "Used as array interface between Python and C++"
   "orjson", "any", "IO for saving operators"
   "psutil", "any", "Get system information such as available memory"
   "scipy", "1.14.0", "Needed for `scipy.sparse.linalg.LinearOperator` class"


Optional
********

.. csv-table::
   :header: "Package", "Minimum version", "Description"
   :widths: 15, 10, 30

   "qiskit", 1.0, "Conversion between Qiskit operators and Fulqrum types"
   "openfermion", 1.6.1, "Conversion between OpenFermion operators and Fulqrum types"
   "primme", 3.2.3, "Eigensolver for Hermitian systems"

C++
~~~~

Fulqrum is based on C++17 and requires having OpenMP 3.0+.
Getting OpenMP is straightforward on Linux but does require installing LLVM via Homebrew on OSX.

Note that the runtime of Fulqrum is compiler dependent.  In practice the ``clang`` compiler 
works better than ``gcc``, and vendor versions of ``clang``, such as the Intel and AMD compilers, show
further gains in performance.


Build files locally
===================

For testing we only require building the source files locally using:

.. code-block:: bash

    python setup.py build_ext --inplace

you can add also use env flags on Linux and OSX for such things as specifying the compiler, 
e.g. ``CC=clang CXX=clang++`` or setting the target architecture like ``FQ_ARCH=znver4``.

Parallel builds can be activated using the ``FQ_BUILD_PARALLEL=N`` flag where ``N`` is the number 
of threads to run.


Installation on Linux
=====================

Installation on Linux is simple:

.. code-block:: bash

    pip install .


Installation on OSX
===================

On OSX, to get OpenMP, one should install llvm using homebrew:

.. code-block:: bash

    brew install llvm

Then installation of Fulqrum with openmp can be accomplished using a call like:

.. code-block:: bash

    CC=clang CXX=clang++ pip install .


Installation on Windows
=======================

Windows is currently not supported and there is currently no plan to do so.
