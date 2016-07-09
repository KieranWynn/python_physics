import core.object_primitives as primitives
from numpy import array as Vector
from numpy import linalg
import math

class Simulation(object):

    UPDATE_INTERVAL_MS = 10
    SPRING_CONSTANT = 6.0

    def __init__(self):
        self.elements = {
            1: primitives.RigidBody(
                position=(1, 0.0, 0.0),
                angular_velocity=(0, 0.2, 0.5)
            ),
            2: primitives.RigidBody(
                position=(0.0, 5.0, 0),
                angular_velocity=(0.0, 0.0, 2.5)
            ),
            3: primitives.RigidBody(
                position=(0.0, 0.0, 1),
                angular_velocity=(1.0, 1.5, 0.0)
            ),
            4: primitives.RigidBody(
                position=(4.0, 0.0, 0),
                angular_velocity=(0.0, 0.0, 5)
            ),
            5: primitives.RigidBody(
                position=(10.0, 0.0, 0),
                angular_velocity=(0.0, 0.5, -7.5)
            ),
            6: primitives.RigidBody(
                position=(3.0, 0.0, 0),
                angular_velocity=(2.0, 0.0, 0.0)
            ),
        }

    def update(self, inputs, dt):
        for element_a in self.elements.values():
            element_a.force = Vector([0.,0.,0.])
            for element_b in self.elements.values():
                if element_a is not element_b:
                    # Add a springlike force between all objects
                    separation_vector = element_a.position - element_b.position
                    norm = linalg.norm(separation_vector)
                    if norm > 0:
                        unit_v = separation_vector / norm
                        element_a.force -= (separation_vector - (unit_v * 10)) * self.SPRING_CONSTANT
            # Add an extra force to keep all objects centered (with some damping to avoid oscillation)
            element_a.force -= ( (element_a.position * self.SPRING_CONSTANT) + (element_a.velocity * 0.1) )
            element_a.update(dt)

