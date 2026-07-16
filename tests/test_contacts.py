import subprocess
import sys
from pathlib import Path

import torch

from lcp_physics.physics.bodies import Circle, Rect
from lcp_physics.physics.world import World


class NoOpEngine:
    pass


class RecordingContactHandler:
    def __init__(self):
        self.calls = []

    def __call__(self, world, body1_index, body2_index):
        self.calls.append((world, body1_index, body2_index))


def test_physics_import_does_not_require_ode():
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockOde(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'ode':
            raise AssertionError('lcp_physics must not import ode')
        return None

sys.meta_path.insert(0, BlockOde())
import lcp_physics.physics
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_world_calls_contact_handler_only_for_bounding_circle_candidates():
    bodies = [
        Circle([0, 0], 1),
        Circle([2.05, 0], 1),
        Circle([5, 0], 1),
    ]

    world = World(
        bodies,
        engine=NoOpEngine,
        contact_callback=RecordingContactHandler,
        eps=0.1,
        strict_no_penetration=False,
    )

    assert [(i, j) for _, i, j in world.contact_callback.calls] == [(0, 1)]
    assert world.contact_callback.calls[0][0] is world


def test_world_skips_excluded_candidate_pairs():
    body1 = Circle([0, 0], 1)
    body2 = Circle([1.5, 0], 1)
    body1.add_no_contact(body2)

    world = World(
        [body1, body2],
        engine=NoOpEngine,
        contact_callback=RecordingContactHandler,
        strict_no_penetration=False,
    )

    assert world.contact_callback.calls == []


def test_diff_contact_handler_preserves_contact_tuple_indices():
    world = World(
        [Circle([0, 0], 1), Circle([1.5, 0], 1)],
        engine=NoOpEngine,
        strict_no_penetration=False,
    )

    assert len(world.contacts) == 1
    contact, body1_index, body2_index = world.contacts[0]
    normal, point1, point2, penetration = contact
    assert (body1_index, body2_index) == (0, 1)
    assert normal.shape == point1.shape == point2.shape == (2,)
    assert penetration.ndim == 0


def test_differentiable_narrow_phase_handles_circle_hull_and_hull_hull():
    circle_hull = World(
        [Circle([1.25, 0], 0.5), Rect([0, 0], [2, 2])],
        engine=NoOpEngine,
        strict_no_penetration=False,
    )
    hull_hull = World(
        [Rect([0, 0], [2, 2]), Rect([1.5, 0], [2, 2])],
        engine=NoOpEngine,
        strict_no_penetration=False,
    )

    assert circle_hull.contacts
    assert hull_hull.contacts
    assert all(contact[1:] == (0, 1) for contact in circle_hull.contacts)
    assert all(contact[1:] == (0, 1) for contact in hull_hull.contacts)


def test_coincident_circles_produce_finite_contact_data():
    world = World(
        [Circle([0, 0], 1), Circle([0, 0], 1)],
        engine=NoOpEngine,
        strict_no_penetration=False,
    )

    contact = world.contacts[0][0]
    assert all(torch.isfinite(value).all() for value in contact)
