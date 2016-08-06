import numpy as np
from pyquaternion import Quaternion

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
            cls._base = cls.__new__(cls)
            cls._base._parent = None
            cls._base.__init__(cls._base)
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
        self.base.children.append(self)

    def add(self, child_node):
        if child_node._parent:
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
        ''' :rtype: geometry.FramedPoint '''
        return self._position

    @position.setter
    def position(self, value):
        self._position = FramedPoint.convert_to_frame(value, self.parent)

    @property
    def orientation(self):
        ''' :rtype: geometry.FramedQuaternion '''
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = FramedQuaternion.convert_to_frame(value, self.parent)

    @property
    def velocity(self):
        ''' :rtype: geometry.FramedVector '''
        return self._velocity

    @velocity.setter
    def velocity(self, value):
        self._velocity = FramedVector.convert_to_frame(value, self.parent)

    @property
    def angular_velocity(self):
        ''' :rtype: geometry.FramedVector '''
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value):
        self._angular_velocity = FramedVector.convert_to_frame(value, self.parent)

    @property
    def acceleration(self):
        ''' :rtype: geometry.FramedVector '''
        return self._acceleration

    @acceleration.setter
    def acceleration(self, value):
        self._acceleration = FramedVector.convert_to_frame(value, self.parent)

    @property
    def angular_acceleration(self):
        ''' :rtype: geometry.FramedVector '''
        return self._angular_acceleration

    @angular_acceleration.setter
    def angular_acceleration(self, value):
        self._angular_acceleration = FramedVector.convert_to_frame(value, self.parent)

class FramedGeometry(object):
    def in_parent_frame(self):
        raise NotImplementedError

    def relative_to(self, node):
        raise NotImplementedError

    def in_frame(self, frame):
        raise NotImplementedError

    @classmethod
    def convert_to_frame(cls, instance, frame):
        try:
            if instance.frame is frame:
                return instance
            else:
                instance = instance.in_frame(frame)
        except AttributeError:
            pass
        return instance  # TODO cls(instance, frame=frame)


class FramedArray(np.ndarray):

    def __new__(cls, input_array, frame=None):
        """
        :param frame: Frame in which this geometry is defined (defaults to global base frame)
        :type frame: core.reference_frame.ReferenceFrame or None
        """
        obj = np.asarray(input_array).view(cls)
        obj.frame = frame or ReferenceFrame.get_base()
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.frame = getattr(obj, 'frame', None)

    def __array_wrap__(self, out_arr, context=None):
        # just call the parent
        return np.ndarray.__array_wrap__(self, out_arr, context)




class FramedPoint(FramedArray):

    def in_parent_frame(self):
        if self.frame is self.frame.base:
            # terminating condition
            return self
        else:
            # TODO Check this!
            # Align point vector to it's parent's axes by performing inverse rotation
            un_rotated = self.frame.orientation.inverse().rotate(self)
            # Add linear offset of frame from it's parent
            return self.__class__(un_rotated + self.frame.parent.position, frame=self.frame.parent)

    def relative_to(self, node):
        pass

    def in_frame(self, frame):
        pass


class FramedVector(FramedArray):

    def relative_to(self, node):
        pass

    def in_frame(self, frame):
        pass


class FramedQuaternion(Quaternion):
    def __init__(self, *args, **kwargs):
        self.frame = kwargs.pop('frame', None)
        super().__init__(*args, **kwargs)

    def relative_to(self, node):
        pass

    def in_frame(self, frame):
        pass
