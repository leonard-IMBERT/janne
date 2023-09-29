"""Some tooling for the test suite
"""

from collections.abc import Callable
import numpy
from numpy.typing import NDArray

class MockInternalRepresentation:
  """Mock internal representation. This is basically a wrapper around a numpy array here
  """
  def __init__(self, data: NDArray):
    if not isinstance(data, numpy.ndarray):
      raise RuntimeError("Got an error here")
    self._data = data

  def apply(self, f: Callable[[NDArray], NDArray]):
    return MockInternalRepresentation(f(self._data))

  def map(self, rhs: "MockInternalRepresentation", f: Callable[[NDArray, NDArray], NDArray]):
    return rhs.apply(lambda d2: f(self._data, d2))

  def get(self):
    return self._data

  def __add__(self, rhs: "MockInternalRepresentation") -> "MockInternalRepresentation":
    return self.map(rhs, lambda d1, d2: d1 + d2)

  def __repr__(self):
    return f"<MockInternalRepresentation : {repr(self._data)}>"
