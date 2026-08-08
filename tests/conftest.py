import pytest

from faeyon import faek


@pytest.fixture(autouse=True, scope="session")
def _faek_on():
    """
    Importing faeyon no longer patches nn.Module automatically; the tests build
    expressions from modules everywhere, so enable faek for the whole session.
    Tests that exercise on/off/context-manager behavior manage the state
    themselves (and restore it).
    """
    faek.on()
    yield
    faek.off()
