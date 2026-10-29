
![](https://raw.githubusercontent.com/Qiskit/fulqrum/main/docs/images/fulqrum_logo.png)

# Fulqrum

![GitHub Release](https://img.shields.io/github/v/release/Qiskit/fulqrum)
![PyPI Downloads](https://img.shields.io/pypi/dm/fulqrum)
![GitHub License](https://img.shields.io/github/license/Qiskit/fulqrum)

A generalized framework for quantum subspace eigensolving.

Fulqrum is a set of tools for enabling the solution to large-scale Hamiltonian subspace eigenproblems over extended alphabets.  Fulqrum was designed specifically with the goals of: (1) Providing a unified code base for solving both spin and fermionic systems. (2) Working for arbitrary numbers of qubits. (3) Reducing memory consumption, as compared to existing methods. And finally, (4) Decoupling operator construction from eigensolving itself.  In addition to satisfying these goals, Fulqrum is performant, and beats other eigensolvers in terms of total runtime.

In addition to eigensolving itself, Fulqrum provides tools for generating compact subspaces that yield accurate solutions while potentially  using orders of magnitude fewer bit-strings than standard quantum subspace methods alone.


## Documentation

Docs: https://qiskit.github.io/fulqrum


## Installation


### Requirements

Outside of standard Python packages Fulqrum requires some Boost headers and OpenMP 3+.

Required Boost header files are included in the ``third-party/boost`` directory, and therefore, no explicit installation of Boost is needed.

### Git submodule

Fulqrum uses [`qiskit-addon-sqd-hpc`](https://github.com/Qiskit/qiskit-addon-sqd-hpc) as a Git submodule. Therefore, we can do either of the following to get actual and required files from that repo:

Clone Fulqrum with `--recurse-submodules` flag, and it will also clone files from the submodule.
```bash
git clone --recurse-submodules https://github.com/Qiskit/fulqrum.git
```

OR

After cloning Fulqrum normally, do a submodule update.
```bash
git clone https://github.com/Qiskit/fulqrum.git
cd fulqrum
git submodule update --init --recursive
```

### Building files locally

In order to run the unittests locally, it is only necessary to build the Cython files inplace:

```bash
python setup.py build_ext --inplace
```
you can add also use env flags such as `CC=clang CXX=clang++` or set the target architecture using `FQ_ARCH=znver4` (or whatever you have)

### Installation on Linux

Installation on Linux is simple:

```bash
pip install .
```

### Installation on OSX

On OSX, to get OpenMP, one should install llvm using homebrew:

```bash
brew install llvm
```
and the following, or something similar, added to the users `.zshrc` file:

```bash
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
```

Then installation of Fulqrum with openmp can be accomplished using a call like:

```bash
CC=clang CXX=clang++ pip install .
```

### Parallel build

You can turn on parallel compilation by setting the environment variable
`FQ_BUILD_PARALLEL=<num threads>`, which can accelerate the build time. Example
with `8` parallel threads:

```bash
# OSX
FQ_BUILD_PARALLEL=8 CC=clang CXX=clang++ pip install .

# Linux
FQ_BUILD_PARALLEL=8 pip install .

# local
FQ_BUILD_PARALLEL=8 python setup.py build_ext --inplace
```

### Runtime performance tuning

The type-2 build/matvec scripts (`csrlike_builder2`, `csr2`,
`matvec2`) allows two optional **runtime** environment variables for tuning. Both have defaults and are read at run time (no rebuild needed), so they are safe to leave unset.

| Variable | Default | Description |
| --- | --- | --- |
| `FQ_BLK` | `128` | The number of rows in a block of rows. FQ_BLK number of rows reuses a chunk of group_ladder_ptrs. Larger blocks give more reuse but risk overflowing the per-core cache; smaller blocks reduce reuse. `128` is a good middle ground. Values `<= 0` are ignored. |
| `FQ_LADDER_WIDTH` | `2` | Number of ladder bits used to bucket each group's terms, giving `2^FQ_LADDER_WIDTH` bins per group (so `2` → 4 bins, `4` → 16 bins). |

Both variables tune the speed only. They change no result.

Example: run a workload with a larger row block and finer ladder bucketing:

```bash
FQ_BLK=256 FQ_LADDER_WIDTH=4 python your_workload.py
```

## Running benchmarks

See the README.md in the `benchmarks` directory.


## Notes on `Subspace` input format

Two modes of input to subspace is supported:
1. **Half-string mode:** In this mode, the input `subspace_strs` to the `Subspace()` is a length-2
`tuple[list[str], list[str]]`, where the first element represents _alpha_ strings, and the second
one represents _beta_ strings. Internally, `fulqrum` will sort each list and perform a Cartesian product between the two lists to construct the full subspace. A full bitstring will be represented by `beta half str + alpha half str`, and the final length of subspace bitstring will be `len(beta half string) + len(alpha half string)`. For example, `[['100', '001, '010'], ['110', '000']]` is a valid input `subspace_strs` in half-string mode. The final subspace size will be `2 * 3 = 6`, with each bitstrings with `3 + 3 = 6` characters/bits. This mode is useful for chemistry applications. For instance, many popular chemistry packages such as PySCF takes alpha and beta strings separately and extends the subspace by taking Cartesian product. This mode enables PySCF-like functionality and keeps the memory requirement in check.

2. **Full-string mode:** In this mode, `subspace_strs` has to be length-1 `tuple[list[str]]`. In this mode, no Cartesian product is taken internally, and the supplied _full_ bitstrings are used as is after sorting.

**Note:** There is no special flag to denote half-string vs. full-string mode. `Fulqrum` will compute the length of the input `subspace_strs` and detect half-string or full-string mode automatically to construct the subspace. If `subspace_strs` has a different length other than 1 or 2 or its element(s) is not `list[str]`, it will throw an `TypeError`.
