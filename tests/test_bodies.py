import math
import unittest

import torch

from lcp_physics.physics.bodies import Circle, Rect, Hull
from lcp_physics.physics.forces import Gravity
from lcp_physics.physics.utils import Defaults


DTYPE = Defaults.DTYPE


class TestBodies(unittest.TestCase):
    def setUp(self):
        # Run without displaying
        self.screen = None

        # Run with display
        # import pygame
        # pygame.init()
        # width, height = 1000, 600
        # self.screen = pygame.display.set_mode((width, height), pygame.DOUBLEBUF)
        # self.screen.set_alpha(None)
        # pygame.display.set_caption('2D Engine')

    def testBody(self):
        pass

    def testCircle(self):
        c1 = Circle([0, 0], 1, vel=[0, 0])
        c2 = Circle([0, 0, 0], 1)
        c3 = Circle(torch.tensor([0, 0], dtype=DTYPE), torch.tensor(1, dtype=DTYPE), vel=torch.tensor([0, 0], dtype=DTYPE))
        c4 = Circle(torch.tensor([0, 0, 0], dtype=DTYPE), torch.tensor(1, dtype=DTYPE), vel=torch.tensor([1, 1, 1], dtype=DTYPE))
        c5 = Circle([0, 0, 0], 1, [0, 0, 0], mass=torch.tensor(1, dtype=DTYPE))

        c1.add_no_contact(c2)
        c2.add_force(Gravity())
        c2.apply_forces(1)
        c3.set_p(c3.p.new_tensor([1, 1, 1]))
        c4.move(0.1)

    def test_bodies_own_symmetric_no_contact_sets(self):
        c1 = Circle([0, 0], 1)
        c2 = Circle([2, 0], 1)

        self.assertEqual(c1.no_contact, set())
        self.assertEqual(c2.no_contact, set())

        c1.add_no_contact(c2)

        self.assertIn(c2, c1.no_contact)
        self.assertIn(c1, c2.no_contact)

    def test_circle_bounding_radius_is_a_tensor_equal_to_radius(self):
        radius = torch.tensor(2.5, dtype=DTYPE)
        circle = Circle([0, 0], radius)

        self.assertIsInstance(circle.bounding_radius, torch.Tensor)
        torch.testing.assert_close(circle.bounding_radius, radius)

    def test_hull_and_rect_bounding_radii_cover_local_vertices(self):
        hull = Hull([0, 0], [[3, 0], [0, 4], [0, 0]])
        rect = Rect([0, 0], [6, 8])

        for body in (hull, rect):
            expected = torch.stack([vertex.norm() for vertex in body.verts]).max()
            self.assertIsInstance(body.bounding_radius, torch.Tensor)
            torch.testing.assert_close(body.bounding_radius, expected)

    def test_set_p_updates_state_and_rotates_hull_vertices_by_delta(self):
        rect = Rect([0, 0], [2, 4])
        initial_verts = [vertex.clone() for vertex in rect.verts]
        new_p = rect.p.new_tensor([math.pi / 2, 3, 4])

        rect.set_p(new_p)

        torch.testing.assert_close(rect.p, new_p)
        torch.testing.assert_close(rect.rot, new_p[:1])
        torch.testing.assert_close(rect.pos, new_p[1:])
        expected_rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=DTYPE)
        for actual, initial in zip(rect.verts, initial_verts):
            torch.testing.assert_close(actual, expected_rotation.matmul(initial))

    def testHull(self):
        # test_hull.py
        pass

    def testRect(self):
        r1 = Rect([0, 0], [1, 1], vel=[0, 0])
        r2 = Rect([0, 0, 0], [1, 1])
        r3 = Rect(torch.tensor([0, 0], dtype=DTYPE), [1, 1], vel=torch.tensor([0, 0], dtype=DTYPE))
        r4 = Rect(torch.tensor([0, 0, 0], dtype=DTYPE), [1, 1], vel=torch.tensor([1, 1, 1], dtype=DTYPE))
        r5 = Rect([0, 0, 0], [1, 1], [0, 0, 0], mass=torch.tensor(1, dtype=DTYPE))

        r1.add_no_contact(r2)
        r2.add_force(Gravity())
        r2.apply_forces(1)
        r3.set_p(r3.p.new_tensor([1, 1, 1]))
        r4.move(0.1)


if __name__ == '__main__':
    unittest.main()
