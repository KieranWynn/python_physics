from rigid_body import RigidBody
from numpy import array as Vector
import math

class Simulation(object):

    UPDATE_INTERVAL_MS = 10
    SPRING_CONSTANT = 3.0

    def __init__(self):
        self.elements = {
            1: RigidBody(
                position=(0.3, 0.0, 0.0),
                angular_velocity=(3, 0.0, 0.0)
            ),
            2: RigidBody(
                position=(0.0, 0.0, 0.5),
                angular_velocity=(0.0, 0.0, 0.5)
            ),
        }

    def update(self, inputs, dt):
        for element in self.elements.values():
            #element.force = -element.position*self.SPRING_CONSTANT
            element.update(dt)
