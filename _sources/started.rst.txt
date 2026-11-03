.. _started:

###############
Getting started
###############

Simple example
##############

Here we demonstrate the standard workflow using a simple example solving a Heisenberg problem on a 5x5 grid.  

To begin, we import the required libraries, and build our Hamiltonian as a ``QubitOperator``:

.. jupyter-execute::

    import scipy.sparse.linalg as spla
    from qiskit.transpiler import CouplingMap
    import fulqrum as fq

    # Build 16-qubit coupling map
    cmap = CouplingMap.from_grid(4, 4)
    num_qubits = cmap.size()

    # Generate Hamiltonian
    H = fq.QubitOperator(num_qubits, [])
    touched_edges = set({})
    coeffs = [-1/2, -1/2, -1]
    for edge in cmap.get_edges():
        if edge[::-1] not in touched_edges:
            touched_edges.add(edge)
            H += fq.QubitOperator(num_qubits, [("XX", edge, coeffs[0]), 
                                               ("YY", edge, coeffs[1]), 
                                               ("ZZ", edge, coeffs[2])])


The subspace in which our Hamiltonian will be defined is derived from the counts sampled from a quantum computer
and stored as a ``Subspace`` object (note that we do not use the counts data itself):

.. jupyter-execute::

    counts = []
    for kk in range(2**num_qubits):
        counts.append(bin(kk)[2:].zfill(num_qubits))

    S = fq.Subspace([counts])

Right now we have a Hamiltonian and a subspace, but they know nothing about each other.  We can combine
them into a ``SubspaceHamiltonian`` object, which is where all the magic happens, and is what we pass on
to the eigensolver:


.. jupyter-execute::

    Hsub = fq.SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=1, which='SA')


In passing the ``SubspaceHamiltonian`` object directly we have solved the problem using a matrix-free method,
but one can build a collection of different, more performant, matrix-representations from this object.
The found eigenenergy is:

.. jupyter-execute::

    print("Eigenvalue:", evals[0])


there is of course an associated eigenvector, but because of the way Fulqrum solves the problem, it is not
immediately usable.  Instead, we can use ``SubspaceHamiltonian.interpret_vector()`` to cast the solution
as a dictionary the statevector amplitudes as values:

.. jupyter-execute::

    Hsub.interpret_vector(evecs)

