from numpy import array as Vector
from particle import Particle
from pyquaternion import Quaternion


class RigidBody(Particle):

    def __init__(
            self,
            mass=1.0,
            moment_of_inertia=(1., 1., 1.),
            position=(0., 0., 0.),
            orientation=(1.0, 0.0, 0.0, 0.0),
            velocity=(0., 0., 0.),
            angular_velocity=(0., 0., 0.),
            acceleration=(0., 0., 0.),
            angular_acceleration=(0., 0., 0.)
    ):
        """

        :param mass: Mass in kg
        :param moment_of_inertia: Mass moment of inertia (in body-fixed frame)
        :param position: Initial position
        :param orientation: Initial quaternion orientation
        :param velocity: Initial linear velocity
        :param angular_velocity: vector in body-fixed frame about which it is spinning
            (magnitude determines angular_rate)
        :param acceleration: Initial angular acceleration
        :param angular_acceleration: vector in body-fixed frame about which it is accelerating
            (magnitude determines angular_acceleration)
        :return:
        """
        super().__init__(mass, position, velocity, acceleration)

        self.moment_of_inertia = Vector(moment_of_inertia)

        # Rotational physics
        self.torque = Vector([0., 0., 0.])
        self.orientation = Quaternion(orientation)
        self.angular_velocity = Vector(angular_velocity)
        self.angular_acceleration = Vector(angular_acceleration)

    def apply_force_at_point(self, force, application_point=None):
        if application_point is not None:
            # apply torque
            pass
        super().apply_force(force)


    def add_force_at_point(self, force, application_point=None):
        if application_point is not None:
            # add torque
            pass
        super().add_force(force)


    def update(self, time_step):
        super().update(time_step)

        # Uses Euler's Method to solve ODE's from initial values.
        if not bool(self.moment_of_inertia.dot(self.moment_of_inertia)):
            raise ZeroDivisionError("Mass moment of inertia must not have zero magnitude")
        self.angular_acceleration = self.torque / self.moment_of_inertia
        self.angular_velocity = self.angular_velocity + self.angular_acceleration * time_step
        self.orientation.integrate(self.angular_velocity, time_step)

    def serialise(self):
        return {
            'position': list(self.position),
            'orientation': list(self.orientation)
        }