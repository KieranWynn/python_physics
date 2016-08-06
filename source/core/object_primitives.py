import core.reference_frame as reference_frame
import core.geometry as geometry


class PhysicsPrimitive(object):
    """
    Abstract physical object. Has no concept of mass, but is part of a hierarchical structure and has an
    internal reference frame in which to refer to children.
    """
    def __init__(self):
        super().__init__()
        self.frame = reference_frame.ReferenceFrame()

    @property
    def parent_frame(self):
        return self.frame.parent

    def update(self, time_step):
        raise NotImplementedError


class Particle(PhysicsPrimitive):
    """
    Basic physical object. Has a mass and is capable of linear (non-rotational) 3D motion.
    """
    def __init__(
            self,
            mass=1.0,
            position=(0., 0., 0.),
            velocity=(0., 0., 0.),
            acceleration=(0., 0., 0.),
    ):
        """
        :param mass: Mass in kg
        :param position: Initial position (relative to parent frame)
        :param velocity: Initial linear velocity (relative to parent frame)
        :param acceleration: Initial linear acceleration (relative to parent frame)
        """
        super().__init__()
        self.mass = mass
        self.force = geometry.FramedVector(frame=self.frame.parent)
        self.frame = reference_frame.ReferenceFrame(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
        )

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value):
        if value <= 0.0:
            raise ZeroDivisionError("An object's mass cannot be less than or equal to zero")
        self._mass = value

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, value):
        self._force = geometry.FramedVector.convert_to_frame(value, self.frame.parent)

    @property
    def position(self):
        return self.frame.position

    @position.setter
    def position(self, value):
        self.frame.position = value

    @property
    def velocity(self):
        return self.frame.velocity

    @velocity.setter
    def velocity(self, value):
        self.frame.velocity = value

    @property
    def acceleration(self):
        return self.frame.acceleration

    @acceleration.setter
    def acceleration(self, value):
        self.frame.acceleration = value

    @property
    def momentum(self):
        return self.mass * self.velocity

    @property
    def kinetic_energy(self):
        return self.mass * (self.velocity.dot(self.velocity)) / 2.0

    def apply_force(self, force):
        self.force = geometry.Vector(force)

    def add_force(self, force):
        self.force += geometry.Vector(force)

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

class RigidBody(Particle):
    """
    Basic physical object. Has a mass and rotational inertia and is capable of general 3D motion.
    """

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
        :param moment_of_inertia: Mass moment of inertia (relative to body frame)
        :param position: Initial position (relative to parent frame)
        :param orientation: Initial quaternion orientation (relative to parent frame)
        :param velocity: Initial linear velocity (relative to parent frame)
        :param angular_velocity: Initial angular velocity vector (relative to parent frame)
            (magnitude determines angular_rate)
        :param acceleration: Initial linear acceleration (relative to parent frame)
        :param angular_acceleration: Initial angular_acceleration vector (relative to parent frame)
            (magnitude determines angular_acceleration)
        """
        super().__init__(mass, position, velocity, acceleration)

        self.moment_of_inertia = moment_of_inertia

        # Rotational physics
        self.torque = geometry.FramedVector(frame=self.frame.parent)
        self.frame.orientation = orientation
        self.frame.angular_velocity = angular_velocity
        self.frame.angular_acceleration = angular_acceleration


    @property
    def moment_of_inertia(self):
        return self._moment_of_inertia

    @moment_of_inertia.setter
    def moment_of_inertia(self, value):
        self._moment_of_inertia = geometry.FramedVector.convert_to_frame(value, self.frame)
        if not bool(self.moment_of_inertia.dot(self.moment_of_inertia)):
            raise ZeroDivisionError("Mass moment of inertia cannot have zero magnitude")

    @property
    def torque(self):
        return self._torque

    @torque.setter
    def torque(self, value):
        self._torque = geometry.FramedVector.convert_to_frame(value, self.frame.parent)

    @property
    def orientation(self):
        return self.frame.orientation

    @orientation.setter
    def orientation(self, value):
        self.frame.orientation = value

    @property
    def angular_velocity(self):
        return self.frame.angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value):
        self.frame.angular_velocity = value

    @property
    def angular_acceleration(self):
        return self.frame.angular_acceleration

    @angular_acceleration.setter
    def angular_acceleration(self, value):
        self.frame.angular_acceleration = value

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
        self.angular_acceleration = (self.torque - (self.angular_velocity.cross(self.angular_velocity))) / self.moment_of_inertia
        self.angular_velocity = self.angular_velocity + self.angular_acceleration * time_step
        self.orientation.integrate(self.angular_velocity, time_step)

    def serialise(self):
        return {
            'position': list(self.position),
            'orientation': list(self.orientation)
        }