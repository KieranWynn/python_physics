import numpy as np
from pyquaternion import Quaternion

class Vector(np.array):
    pass

class FramedGeometry(object):
    def __init__(self, frame):
        self.frame = frame

    def in_parent_frame(self):
        if self.frame and self.frame.node:
            pass

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
        return cls(instance, frame=frame)


class FramedPoint(Vector, FramedGeometry):
    def __init__(self, *args, **kwargs):
        FramedGeometry.__init__(self, kwargs.pop('frame', None))
        super().__init__(*args, **kwargs)

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


class FramedQuaternion(Quaternion, FramedGeometry):
    def __init__(self, *args, **kwargs):
        FramedGeometry.__init__(self, kwargs.pop('frame', None))
        super().__init__(*args, **kwargs)

    def relative_to(self, node):
        pass

    def in_frame(self, frame):
        pass
