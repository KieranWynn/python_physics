import unittest
import numpy as np
from core import object_primitives
from core import geometry
from core import structure

class PhysicsTest(unittest.TestCase):

    def test_relative_to(self):
        object_a = object_primitives.Particle(velocity=(1, 0, 0))
        object_b = object_primitives.Particle(velocity=(-2, 0, 0))

        self.assertTrue((object_a.velocity.relative_to(object_b) == np.array((-3, 0, 0))).all())

    def test_in_frame(self):
        inertial = object_primitives.PhysicsPrimitive()
        particle_a = object_primitives.Particle(velocity=(0, 1, 0))
        particle_b = object_primitives.Particle(velocity=(1, 0, 0))
        inertial.add(particle_a)
        particle_a.add(particle_b)

        self.assertTrue((particle_a.velocity.in_frame(inertial.frame) == np.array((1, 1, 0))).all())

    def test_in_parent_frame(self):
        node_a = object_primitives.PhysicsPrimitive()
        frame_a = geometry.ReferenceFrame(node=structure.Node)
        a = geometry.FramedPoint()


