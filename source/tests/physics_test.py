import unittest
import numpy as np
from core import object_primitives
import core.reference_frame as reference_frame

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
        frame_a = reference_frame.ReferenceFrame.get_base()
        frame_b = reference_frame.ReferenceFrame(position=(2, 0, 0))
        frame_a.add(frame_b)
        point_b = reference_frame.FramedPoint([1, 1, 0], frame=frame_b)
        point_a = point_b.in_parent_frame()
        self.assertEqual(point_a, reference_frame.FramedPoint((1, 1, 2), frame=frame_a))




