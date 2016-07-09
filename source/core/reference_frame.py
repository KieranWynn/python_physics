import core.geometry as geometry

class ReferenceFrameNode(object):
    """
    A ReferenceFrameNode object is a base for all physical objects and a means for adding structure to a physical model.

    This is the basic building block for a tree-based structure of objects and their attached or contained
    child objects.
    """
    _base = None  # Global single-instance base reference frame

    @classmethod
    def get_base(cls):
        if not cls._base:
            cls._base = cls()
            cls._base.parent = None
        return cls._base

    @property
    def base(self):
        return self.get_base()

    @property
    def parent(self):
        return self._parent

    def __init__(self):
        self.children = []
        self._parent = self.base

    def add(self, child_node):
        child_node._parent.children.remove(child_node)
        child_node._parent = self
        self.children.append(child_node)

    def move(self, to_parent):
        self._parent.children.remove(self)
        self._parent = to_parent

    def remove(self, child_node):
        self.children.remove(child_node)
        del(child_node)


class ReferenceFrame(ReferenceFrameNode):
    def __init__(
            self,
            position=(0., 0., 0.),
            orientation=(1.0, 0.0, 0.0, 0.0),
            velocity=(0., 0., 0.),
            angular_velocity=(0., 0., 0.),
            acceleration=(0., 0., 0.),
            angular_acceleration=(0., 0., 0.),
    ):
        super().__init__()
        self.position = position,
        self.orientation = orientation,
        self.velocity = velocity,
        self.angular_velocity = angular_velocity,
        self.acceleration = acceleration,
        self.angular_acceleration = angular_acceleration,

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = geometry.FramedPoint.convert_to_frame(value, self.parent)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = geometry.FramedQuaternion.convert_to_frame(value, self.parent)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = geometry.FramedVector.convert_to_frame(value, self.parent)

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value):
        self._angular_velocity = geometry.FramedVector.convert_to_frame(value, self.parent)

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value):
        self._acceleration = geometry.FramedVector.convert_to_frame(value, self.parent)

    @property
    def angular_acceleration(self):
        return self._angular_acceleration

    @angular_acceleration.setter
    def angular_acceleration(self, value):
        self._angular_acceleration = geometry.FramedVector.convert_to_frame(value, self.parent)

