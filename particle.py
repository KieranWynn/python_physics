import numpy as np


class Particle(object):
    def __init__(
            self,
            mass=1.0,
            position=np.array([0., 0., 0.]),
            velocity=np.array([0., 0., 0.]),
            acceleration=np.array([0., 0., 0.])
    ):

        self.mass = mass

        # Translational physics
        self.force = np.array([0., 0., 0.])
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration

    @property
    def momentum(self):
        return self.mass * self.velocity

    @property
    def kinetic_energy(self):
        return self.mass * (self.velocity.dot(self.velocity))

    def update(self, time_step):
        # Uses Euler's Method to solve ODE's from initial values.
        if self.mass == 0.0:
            raise ZeroDivisionError("Mass must not be zero")
        self.acceleration = self.force / self.mass
        self.velocity = self.velocity + self.acceleration * time_step
        self.position = self.position + self.velocity * time_step
