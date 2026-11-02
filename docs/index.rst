.. raw:: html

   <div style="height: 0; visibility: hidden;">

#######
Fulqrum
#######

.. raw:: html

   </div>


.. figure:: images/fulqrum_logo.png
    :align: center


Fulqrum is a set of tools for enabling the solution to general large-scale Hamiltonian subspace eigenproblems over subspaces defined by samples returned from executing circuits on quantum devices.  Such methods work if the target eigenstate has compact support over the underlying full Hilbert space; the target state has nonzero components in a polynomial number of bit-strings.  Fulqrum works over an extended alphabet of operators that includes projector and ladder operators in addition to the usually Pauli set, and because of this, Fulqrum is able to solve both fermionic and spin systems using the same code base.  Moreover, there is no intrinsic limit on the number of qubits that Fulqrum can handle, allowing users to go up to the full-scale of current and future quantum systems.


As shown below, a typical eigenvalue problem is broken up into three pieces: 1) Bit-strings are collected from a quantum computer (classical methods also work), and optionally post-processed.  2) Fulqrum converts these bit-strings into a ``Subspace`` that together with the Hamiltonian (``QubitOperator``) that describes the system are used to generate a ``SubspaceHamiltonian`` from which numerical representations can be generated. Fermionic systems are first transformed to qubit Hamiltonians via an efficient Jordan-Wigner mapping.  Finally, step (3) takes these representations and uses them in standard eigensolving packages.  In this way, Fulqrum allows for flexibility in generating solutions that other methods lack. 

.. figure:: images/workflow.png
    :align: center


.. toctree::
    :maxdepth: 1
    :hidden:

    self
    Installation <install>
    Citing <citing>

.. toctree::
    :maxdepth: 1
    :caption: User guide
    :hidden:
    
    Getting started  <started>
    Using QubitOperators  <qubit_operators.ipynb>
    Using FermionicOperators  <fermi_operators.ipynb>
    Defining Subspaces <subspaces.ipynb>
    Operator IO <operator_io.ipynb>
    Subspace Hamiltonians <subspace_hamiltonians.ipynb>
    Loading Molecular data <molecule_loading.ipynb>

.. toctree::
    :maxdepth: 1
    :caption: Tools
    :hidden:
    
    Logging  <logging.ipynb>

.. toctree::
    :maxdepth: 2
    :caption: Tutorials
    :hidden:
    :glob:

    tutorials/*


.. toctree::
    :maxdepth: 1
    :caption: API Docs
    :hidden:
    
    Core <apidocs/core>
    Utilities <apidocs/utils>
    RAMPS <apidocs/ramps>
    SQD addon support <apidocs/addon_sqd_support>
