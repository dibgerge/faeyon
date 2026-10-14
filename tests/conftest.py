import pytest

from faeyon import faek


@pytest.fixture(autouse=True)
def _faek_on():
    """
    Importing faeyon no longer patches nn.Module automatically; most tests build
    expressions from modules, so enable faek for every test.

    Function-scoped (not session) so each test starts from a clean on-state —
    including under pytest-xdist, where workers do not share process state and a
    prior test's ``faek.off()`` must not leak.
    """
    faek.off()
    faek.on()
    yield
    faek.off()
