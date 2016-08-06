import numpy as np
from pyquaternion import Quaternion

class FramedGeometry(object):
    def __init__(self, frame):
        """
        :param frame: Frame in which this geometry is defined (defaults to global base frame)
        :type frame: core.reference_frame.ReferenceFrame or None
        """
        self.frame = frame  # TODO: Find a way to default this to ReferenceFrame.get_base() without a circular import

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


class Vector(np.ndarray):
    def __init__(self, *args, **kwargs):
        np.ndarray.__init__(*args, **kwargs)


class FramedPoint(FramedGeometry, Vector):
    def __init__(self, *args, **kwargs):
        FramedGeometry.__init__(self, kwargs.pop('frame', None))
        Vector.__init__(self, *args, **kwargs)

    def in_parent_frame(self):
        if self.frame is self.frame.base:
            # terminating condition
            return
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


class FramedVector(Vector, FramedGeometry):
    def __init__(self, *args, **kwargs):
        FramedGeometry.__init__(self, kwargs.pop('frame', None))
        super().__init__(*args, **kwargs)

    def relative_to(self, node):
        pass

    def in_frame(self, frame):
        pass


class FramedQuaternion(FramedGeometry, Quaternion):
    def __init__(self, *args, **kwargs):
        FramedGeometry.__init__(self, kwargs.pop('frame', None))
        super().__init__(*args, **kwargs)

    def relative_to(self, node):
        pass

    def in_frame(self, frame):
        pass
