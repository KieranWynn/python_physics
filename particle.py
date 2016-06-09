from numpy import array as Vector


class Particle(object):
    def __init__(
            self,
            mass=1.0,
            position=(0., 0., 0.),
            velocity=(0., 0., 0.),
            acceleration=(0., 0., 0.)
    ):

        self.mass = mass

        # Translational physics
        self.force = Vector([0., 0., 0.])
        self.position = Vector(position)
        self.velocity = Vector(velocity)
        self.acceleration = Vector(acceleration)

    @property
    def momentum(self):
        return self.mass * self.velocity

    @property
    def kinetic_energy(self):
        return self.mass * (self.velocity.dot(self.velocity)) / 2.0

    def apply_force(self, force):
        self.force = Vector(force)

    def add_force(self, force):
        self.force += Vector(force)

    def update(self, time_step):
        # Uses Euler's Method to solve ODE's from initial values.
        if self.mass == 0.0:
            raise ZeroDivisionError("Mass must not be zero")
        self.acceleration = self.force / self.mass
        self.velocity = self.velocity + self.acceleration * time_step
        self.position = self.position + self.velocity * time_step

    def serialise(self):
        return {
            'position': list(self.position),
            'orientation': [1.0, 0.0, 0.0, 0.0]
        }
