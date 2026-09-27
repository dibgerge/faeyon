class _NoValue:
    """A unique sentinel to represent empty."""
    __slots__ = ()

    @property
    def value(self):
        return self

    def __repr__(self):
        return "<NO_VALUE>"

    def __str__(self):
        return "<NO_VALUE>"


class _MappingKey(str):
    """ 
    This is a sentinel type to be used with Delayable objects to indicate a map packing.
    """
    pass