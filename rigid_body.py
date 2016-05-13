from particle import Particle
import numpy as np
from pyquaternion import Quaternion


class RigidBody(Particle):

    def __init__(
            self,
            mass=1.0,
            moment_of_inertia=np.array([1., 1., 1.]),
            position=np.array([0., 0., 0.]),
            orientation=Quaternion(),
            velocity=np.array([0., 0., 0.]),
            angular_velocity=np.array([0., 0., 0.]),
            acceleration=np.array([0., 0., 0.]),
            angular_acceleration=np.array([0., 0., 0.])
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

        self.moment_of_inertia = moment_of_inertia

        # Rotational physics
        self.torque = np.array([0., 0., 0.])
        self.orientation = orientation
        self.angular_velocity = angular_velocity
        self.angular_acceleration = angular_acceleration

    def update(self, time_step):
        super().update(time_step)

        # Uses Euler's Method to solve ODE's from initial values.
        if not bool(self.moment_of_inertia.dot(self.moment_of_inertia)):
            raise ZeroDivisionError("Mass moment of inertia must not have zero magnitude")
        self.angular_acceleration = self.torque / self.moment_of_inertia
        self.angular_velocity = self.angular_velocity + self.angular_acceleration * time_step
        self.orientation.integrate(self.angular_velocity, time_step)
