# Python 3.11 and PyTorch 2.7.1 Migration Design

## Goal

Run the core `lcp_physics` engine, standard demos, gradient demo, and automated tests on Python 3.11 and PyTorch 2.7.1 without `py3ode`.

## Scope

The migration covers the core physics package, the production PDIPM solver, demos, packaging metadata, and tests. The Atari Breakout experiments remain source-compatible where practical, but their separate Gym, Atari ROM, and external MPC dependencies are outside this migration.

## Dependency target

- Python 3.11
- PyTorch 2.7.1
- NumPy 1.26.4
- pygame 2.6.1
- pytest 8.x

The `lcp_physics` Conda environment is the installation and verification environment. The core engine remains CPU-first and uses double precision by default.

## Collision architecture

Remove all imports and objects from `py3ode`. Bodies retain their Torch position and shape data, and maintain a set of other bodies excluded from contact. Each shape exposes a conservative bounding radius.

`World.find_contacts()` iterates over unique body pairs. It skips excluded pairs, then uses the sum of bounding radii plus the world's contact epsilon as a conservative broad-phase test. Candidate pairs are passed by body index to `DiffContactHandler`, which retains the existing differentiable circle-circle, circle-hull, and hull-hull narrow-phase algorithms. The ODE-only contact handler is removed from the public implementation because it cannot operate without ODE and is not the default path.

This all-pairs broad phase is quadratic in body count, but it is deterministic, dependency-free, and adequate for the repository's small research scenes. A spatial index is intentionally deferred.

## PyTorch architecture

Convert `LCPFunction` to the current static `torch.autograd.Function` protocol. A small callable factory preserves the existing engine-facing configuration (`eps`, verbosity, iteration limit), while the actual invocation uses `LCPFunction.apply`. Tensor inputs required for the backward pass and non-tensor solver state are stored on `ctx`.

Replace removed `Tensor.btrifact` and `Tensor.btrisolve` calls in the production solver with `torch.linalg.lu_factor` and `torch.linalg.lu_solve`. A focused helper accepts the solver's existing row-vector right-hand sides, converts them to column form for `lu_solve`, and converts results back. Existing KKT assembly and physical equations remain unchanged.

Modernize demo-only uses of `Variable`, `.data[0]`, and manual parameter replacement so gradient examples use leaf tensors, `.item()`, and `torch.no_grad()`.

## Testing

Tests are added before each production change and must first fail on the legacy implementation. Coverage includes:

- importing the package without an `ode` module;
- contact exclusions and pure-Python broad-phase candidate selection;
- circle-circle and hull contact simulation;
- LU factor/solve equivalence against `torch.linalg.solve`;
- LCP forward execution and `torch.autograd.gradcheck`;
- headless demo smoke tests and the existing test suite.

Final verification runs in the `lcp_physics` Conda environment and records exact Python, Torch, NumPy, and pygame versions.

## Packaging and documentation

Update `setup.py` so install requirements no longer mention `py3ode`, declare Python 3.11 support, and constrain the tested dependency versions where reproducibility benefits from it. Expand the README with environment creation, editable installation, test, and headless demo commands.

## Success criteria

1. `import lcp_physics.physics` succeeds when `py3ode` is absent.
2. No runtime source file imports or references `ode`.
3. The core test suite passes under Python 3.11 and PyTorch 2.7.1.
4. A contact scene completes forward simulation without non-finite state.
5. A differentiable scene produces finite gradients.
6. Dependencies are installed in the existing `lcp_physics` Conda environment.
