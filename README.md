# lcp-physics
[![Build Status][travis-image]][travis] [![License][license-image]][license]

[travis-image]: https://api.travis-ci.org/locuslab/lcp-physics.svg?branch=master
[travis]: http://travis-ci.org/locuslab/lcp-physics

[license-image]: http://img.shields.io/badge/license-Apache--2-blue.svg?style=flat
[license]: LICENSE

A modular, differentiable 2D rigid-body physics engine built on PyTorch. The
engine formulates contacts, friction, and joint constraints as a linear
complementarity problem (LCP), allowing gradients to propagate through a
simulation for learning and control experiments.

Created by Filipe de Avila Belbute Peres and J. Zico Kolter. See the
[NeurIPS paper](https://papers.nips.cc/paper/7948-end-to-end-differentiable-physics-for-learning-and-control)
for the underlying method.

## Tested environment

The core package and standard demos are tested on CPU with:

- Python 3.11
- PyTorch 2.7.1 (CPU build)
- NumPy 1.26.4
- pygame 2.6.1

The production package targets Python 3.11. The legacy development solver at
`lcp_physics/lcp/solvers/dev_pdipm.py` is not part of the verified compatibility
surface.

## Installation

Use the existing Conda environment named `lcp_physics`. From the repository
root, install the tested CPU dependencies and the current checkout in editable
mode:

```bash
conda activate lcp_physics
python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 pygame==2.6.1 pytest==8.4.1
python -m pip install -e . --no-deps
```

If the environment does not exist yet, create it once before running the
commands above:

```bash
conda create -n lcp_physics python=3.11 -y
```

The editable install makes source edits in this checkout immediately available
to Python. `--no-deps` keeps the explicitly installed CPU PyTorch build instead
of resolving a different wheel through project metadata.

## Tests

Run the complete suite from the repository root:

```bash
python -m pytest -q
```

Using `python -m pytest` ensures the active Conda environment's interpreter and
editable package are used.

## Demos

Run the standard scenes without opening a window by passing `-nd`:

```bash
python demos/demo.py -nd
python demos/fixed_joint_demo.py -nd
```

Omit `-nd` to render with pygame. The gradient example in
`demos/grad_demo.py` optimizes an applied force through a differentiable
simulation and then replays the learned scene.

## Collision and experiment scope

The core engine has no `py3ode` dependency. Collision candidate selection uses
a deterministic all-pairs broad phase implemented in Python and Torch, followed
by the differentiable narrow phase. The broad phase is O(n^2) in the number of
bodies, which is suitable for the repository's small research scenes but is not
intended as a large-scene spatial index.

The Breakout code under `experiments/breakout` remains optional legacy research
code and is outside the tested core package. It needs a separate migration of
its Gym API, Atari environment and ROM setup, and external MPC dependencies
before it can be expected to run on this stack.
