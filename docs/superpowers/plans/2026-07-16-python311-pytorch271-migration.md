# Python 3.11 and PyTorch 2.7.1 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the core differentiable physics engine on Python 3.11 and PyTorch 2.7.1 without `py3ode`, with reproducible installation and regression tests.

**Architecture:** Replace ODE geometry objects with body-owned collision metadata and a conservative all-pairs bounding-circle broad phase. Preserve the existing differentiable narrow phase and LCP equations while migrating removed batched LU and custom-autograd APIs to `torch.linalg` and the static `Function.apply` protocol.

**Tech Stack:** Python 3.11, PyTorch 2.7.1 CPU, NumPy 1.26.4, pygame 2.6.1, pytest 8.x, setuptools.

---

### Task 1: Install and record the target environment

**Files:**
- Modify: `setup.py`
- Create: `requirements.txt`
- Test: environment import commands

- [ ] **Step 1: Install pinned runtime and test dependencies**

Run:

```bash
conda run -n lcp_physics python -m pip install \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n lcp_physics python -m pip install \
  numpy==1.26.4 pygame==2.6.1 pytest==8.4.1
```

Expected: every package installs successfully in `/home/mirrency/anaconda3/envs/lcp_physics`.

- [ ] **Step 2: Record the import baseline**

Run:

```bash
conda run -n lcp_physics python -c \
  "import torch, numpy, pygame; print(torch.__version__, numpy.__version__, pygame.version.ver)"
conda run -n lcp_physics python -c "import lcp_physics.physics"
```

Expected: dependency imports pass; package import fails with `ModuleNotFoundError: No module named 'ode'`, proving the first compatibility defect.

- [ ] **Step 3: Declare supported dependencies**

Create `requirements.txt`:

```text
torch==2.7.1
numpy==1.26.4
pygame==2.6.1
```

Update `setup.py` to remove `py3ode`, declare `python_requires='>=3.11,<3.12'`, use the three runtime requirements above, and expose a `test` extra containing `pytest==8.4.1`.

- [ ] **Step 4: Verify metadata and commit**

Run:

```bash
conda run -n lcp_physics python setup.py --name --version
git diff --check
```

Expected: `lcp_physics`, `0.1.0`, and no whitespace errors.

Commit: `build: target Python 3.11 and PyTorch 2.7.1`.

### Task 2: Remove py3ode and provide native broad-phase collision selection

**Files:**
- Modify: `lcp_physics/physics/bodies.py`
- Modify: `lcp_physics/physics/contacts.py`
- Modify: `lcp_physics/physics/world.py`
- Create: `tests/test_contacts.py`
- Modify: `tests/test_bodies.py`

- [ ] **Step 1: Write failing dependency and broad-phase tests**

Add tests that require bodies to own `no_contact` sets and conservative `bounding_radius` tensors, verify `add_no_contact()` is symmetric, and construct worlds containing near and far circles while a recording callback receives only the near candidate pair. Add a subprocess import test that blocks any import named `ode` and imports `lcp_physics.physics`.

Representative assertions:

```python
assert other in body.no_contact
assert body in other.no_contact
assert torch.equal(circle.bounding_radius, circle.rad)
assert recorder.pairs == [(0, 1)]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
conda run -n lcp_physics pytest tests/test_bodies.py tests/test_contacts.py -q
```

Expected: collection/import fails because `bodies.py`, `contacts.py`, and `world.py` import `ode`.

- [ ] **Step 3: Implement body-owned collision metadata**

Remove `ode` imports and `_create_geom()` methods. Initialize `self.no_contact = set()` in `Body`; make `add_no_contact()` add body references symmetrically. Define `bounding_radius` as the radius for circles and the maximum norm of local vertices for hulls/rectangles. `set_p()` updates only Torch state and rotated vertices.

- [ ] **Step 4: Implement pure Python/Torch broad phase**

Make contact callbacks accept `(world, body1_index, body2_index)`. Remove `OdeContactHandler`. In `World.find_contacts()`, iterate over each unique pair, skip exclusions, compare center distance against `radius1 + radius2 + eps`, then call `DiffContactHandler` for candidates. Preserve contact tuples as `((normal, p1, p2, penetration), i, j)`.

- [ ] **Step 5: Run focused and existing body/contact tests**

Run:

```bash
conda run -n lcp_physics pytest tests/test_bodies.py tests/test_contacts.py tests/test_hull.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

Run `rg -n '\bode\b|py3ode' lcp_physics setup.py requirements.txt` and expect no runtime dependency reference. Commit as `refactor: replace py3ode collision broad phase`.

### Task 3: Migrate the PDIPM solver to torch.linalg

**Files:**
- Modify: `lcp_physics/lcp/solvers/pdipm.py`
- Modify: `lcp_physics/lcp/util.py`
- Create: `tests/test_pdipm.py`

- [ ] **Step 1: Write failing LU compatibility tests**

Test batched matrix and vector right-hand sides against `torch.linalg.solve`:

```python
lu = pdipm.lu_factor(matrix)
actual_vector = pdipm.lu_solve(lu, vector)
actual_matrix = pdipm.lu_solve(lu, rhs_matrix)
assert torch.allclose(actual_vector, torch.linalg.solve(matrix, vector.unsqueeze(-1)).squeeze(-1))
assert torch.allclose(actual_matrix, torch.linalg.solve(matrix, rhs_matrix))
```

Also call `pre_factor_kkt`, `factor_kkt`, and `solve_kkt` on a small positive-definite system.

- [ ] **Step 2: Run tests and verify RED**

Run `conda run -n lcp_physics pytest tests/test_pdipm.py -q`.

Expected: failure because the compatibility helpers do not exist and legacy `btrifact` is unavailable.

- [ ] **Step 3: Add LU compatibility helpers and replace removed calls**

Implement:

```python
def lu_factor(matrix):
    return torch.linalg.lu_factor(matrix)

def lu_solve(factorization, rhs):
    lu, pivots = factorization
    is_vector = rhs.ndim == lu.ndim - 1
    solution = torch.linalg.lu_solve(
        lu, pivots, rhs.unsqueeze(-1) if is_vector else rhs)
    return solution.squeeze(-1) if is_vector else solution
```

Replace production `btrifact`/`btrisolve` calls and uint8 masks with modern factorization/solve calls and boolean masks. Keep the public PDIPM function signatures intact.

- [ ] **Step 4: Run focused solver tests and commit**

Run:

```bash
conda run -n lcp_physics pytest tests/test_pdipm.py -q
rg -n 'btrifact|btrisolve|uint8|\.byte\(\)' lcp_physics/lcp/solvers/pdipm.py lcp_physics/lcp/util.py
```

Expected: tests pass and no removed runtime API remains. Commit as `refactor: migrate PDIPM to torch.linalg`.

### Task 4: Migrate the differentiable LCP autograd boundary

**Files:**
- Modify: `lcp_physics/lcp/lcp.py`
- Modify: `lcp_physics/physics/engines.py`
- Create: `tests/test_lcp.py`

- [ ] **Step 1: Write failing forward and gradient tests**

Define a one-variable convex complementarity problem with `Q=[[1]]`, `p=[[-1]]`, `G=[[-1]]`, `h=[[0]]`, no equalities, and `F=[[0]]`. Assert the solution is finite and close to `1`; use `torch.autograd.gradcheck` on `p` and a smooth interior problem to validate the analytical backward path.

- [ ] **Step 2: Run tests and verify RED**

Run `conda run -n lcp_physics pytest tests/test_lcp.py -q`.

Expected: the legacy instance-style autograd function raises the PyTorch legacy-function error.

- [ ] **Step 3: Implement static autograd function and callable factory**

Use `@staticmethod forward(ctx, ...)`, `ctx.save_for_backward(...)`, `@staticmethod backward(ctx, grad_output)`, and `LCPFunction.apply`. Preserve configuration through a `lcp_function(...)` factory returning a callable closure. Update `PdipmEngine` to store and call that closure. Remove obsolete numerical backward code based on `Variable` and `collections.Iterable`.

- [ ] **Step 4: Run focused tests and commit**

Run:

```bash
conda run -n lcp_physics pytest tests/test_lcp.py tests/test_pdipm.py -q
```

Expected: forward and gradient tests pass. Commit as `refactor: modernize differentiable LCP autograd`.

### Task 5: Modernize demos, document usage, and perform integration verification

**Files:**
- Modify: `demos/grad_demo.py`
- Modify: `README.md`
- Modify: `tests/test_demos.py`
- Delete: `docs/superpowers/specs/2026-07-16-python311-pytorch271-migration-design.md`

- [ ] **Step 1: Add a short differentiable simulation regression test**

Build a two-circle scene with a learnable initial impulse, simulate a few steps, differentiate a final-position scalar, and assert the gradient exists and is finite. Keep runtime below several seconds.

- [ ] **Step 2: Run it and verify RED**

Run the new test alone and confirm it fails on remaining demo/integration incompatibility rather than a test typo.

- [ ] **Step 3: Modernize demo tensor usage**

Remove `Variable`, `.data[0]`, and `.grad.data`; use leaf tensors with `requires_grad=True`, `.item()`, `detach()`, and `torch.no_grad()` updates. Do not change the simulated scene or optimization objective.

- [ ] **Step 4: Document reproducible setup and remove the approved design document**

Expand README with Conda activation, dependency installation, editable install, tests, and `python demos/demo.py -nd`. Delete the design document as explicitly requested after its implementation requirements have been transferred into tests and documentation.

- [ ] **Step 5: Install the editable package and run full verification**

Run:

```bash
conda run -n lcp_physics python -m pip install -e . --no-deps
conda run -n lcp_physics pytest -q
SDL_VIDEODRIVER=dummy conda run -n lcp_physics python demos/fixed_joint_demo.py -nd
conda run -n lcp_physics python - <<'PY'
import numpy, pygame, torch
import lcp_physics.physics
print(torch.__version__, numpy.__version__, pygame.version.ver)
PY
rg -n '\bode\b|py3ode|btrifact|btrisolve|Variable|\.data\[0\]' \
  lcp_physics demos setup.py requirements.txt
git diff --check
```

Expected: all tests and smoke commands exit zero; versions report Python 3.11, Torch 2.7.1, NumPy 1.26.4, and pygame 2.6.1; forbidden legacy runtime patterns are absent.

- [ ] **Step 6: Commit**

Commit as `docs: document modern environment and verification`.
