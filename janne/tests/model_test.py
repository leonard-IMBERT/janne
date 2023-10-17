"""Module containing the tests related to the Model

There is no test for the moment (everything is implementation dependent) so it is just
some mock implementation of the Model
"""

from janne.interfaces import IModel
from .ann_test import MockInternalRepresentation

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from numpy.typing import NDArray


@dataclass
class MockModelConf:
  lr: float = 1e-3

class MockModel(IModel[MockInternalRepresentation]):
  """A mock model. It's predictions are pretty shitty and it cannot optimize itself
  """
  def __init__(self) -> None:
    self._config: Optional[MockModelConf] = None

  def initialize(self, config: MockModelConf) -> None:
    self._config = config

  def migrate(self, raw_data: List[NDArray]) -> MockInternalRepresentation:
    if self._config is None:
      raise RuntimeError("MockModel hasn't been initialized")
    return MockInternalRepresentation(np.array(raw_data))

  def unmigrate(self, internal_data: MockInternalRepresentation) -> NDArray:
    if self._config is None:
      raise RuntimeError("MockModel hasn't been initialized")
    return internal_data.get()

  def transform(self, raw_data: List[NDArray]) -> List[NDArray]:
    if self._config is None:
      raise RuntimeError("MockModel hasn't been initialized")
    return list(map(lambda rd: rd[0:10, :], raw_data))

  def predict(self, transformed_data: MockInternalRepresentation) -> MockInternalRepresentation:
    if self._config is None:
      raise RuntimeError("MockModel hasn't been initialized")
    return transformed_data.apply(lambda d: np.array([np.mean(d), np.std(d), np.min(d), np.max(d)]))

  def loss(self, prediction: MockInternalRepresentation, truth: List[NDArray]) -> MockInternalRepresentation:
    if self._config is None:
      raise RuntimeError("MockModel hasn't been initialized")
    i_truth = self.migrate(truth)
    return prediction.map(i_truth, lambda d1, d2: np.mean((d1 - d2) ** 2, axis=-1))

  def back_propagate(self, loss: MockInternalRepresentation) -> None:
    if self._config is None:
      raise RuntimeError("MockModel hasn't been initialized")
    pass
