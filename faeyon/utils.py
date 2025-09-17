from collections import namedtuple


ImageSize = namedtuple("ImageSize", ["height", "width"])


def is_ipython() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def is_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError, ImportError):
        return False      # Probably standard Python interpreter


class Period:
    def __init__(self, value: int, unit: str):
        self.value = value
        self.unit = unit

    def __str__(self):
        return f"{self.value}{self.unit}"
    
    def __repr__(self):
        return f"Period({self.value}, {self.unit})"
