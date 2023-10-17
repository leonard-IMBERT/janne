"""Module containing the tests related to the ANN

There is no test for the moment (everything is implementation dependent) so it is just
some mock implementation of the ANN
"""

from janne.interfaces import IAdversorial

from dataclasses import dataclass
from typing import Any, Optional, List

import numpy as np
from numpy.typing import NDArray

from .test_tools import MockInternalRepresentation


@dataclass
class MockAdversorialConf:
  shift: float
  lr: float = 1e-3

class MockAdversorial(IAdversorial[MockInternalRepresentation]):
  """Mock Adversorial Neural Network

  It's perturbation is just a shift on the last data

  It's adv loss is MSE truth vs pred

  It's reg loss is MSE data vs pert data

  It's backpropagation is just moving it's shift by -loss * lr
  """
  def __init__(self):
    self._config : Optional[MockAdversorialConf] = None
    self._shift : float = -1.
    self._lr : float = -1

  def initialize(self, config: Any) -> None:
    self._config = config
    self._shift = config.shift
    self._lr = config.lr

  def perturbate(self, internal_data: MockInternalRepresentation) -> MockInternalRepresentation:
    if self._config is None:
      raise RuntimeError("MockAdversorial hasn't been initialized")
    return internal_data.apply(lambda d: np.concatenate((d[...,:, :-1], d[...,:, -1:] + self._shift), axis=-1))

  def migrate(self, raw_data: List[NDArray]) -> MockInternalRepresentation:
    if self._config is None:
      raise RuntimeError("MockAdversorial hasn't been initialized")
    return MockInternalRepresentation(np.array(raw_data))

  def unmigrate(self, internal_data: MockInternalRepresentation) -> List[NDArray]:
    if self._config is None:
      raise RuntimeError("MockAdversorial hasn't been initialized")
    return list(internal_data.get())

  def adv_loss(self, truth: List[NDArray], prediction: List[NDArray], raw_data: List[NDArray],
               perturbated_data: MockInternalRepresentation) -> MockInternalRepresentation:
    if self._config is None:
      raise RuntimeError("MockAdversorial hasn't been initialized")
    i_truth = self.migrate(truth)
    i_pred = self.migrate(prediction)

    return i_truth.map(i_pred, lambda d1, d2: np.mean((d1 - d2) ** 2, axis=-1))

  def reg_loss(self, truth: List[NDArray], prediction: List[NDArray],
               raw_data: List[NDArray], perturbated_data: MockInternalRepresentation) -> MockInternalRepresentation:
    if self._config is None:
      raise RuntimeError("MockAdversorial hasn't been initialized")
    i_raw = self.migrate(raw_data)

    return i_raw.map(perturbated_data, lambda d1, d2: np.mean((d1 - d2) ** 2, axis=(-1, -2)))

  def back_propagate(self, loss: MockInternalRepresentation) -> None:
    if self._config is None:
      raise RuntimeError("MockAdversorial hasn't been initialized")

    self._shift += - self._lr * loss.get().item()

  def combine_losses(self, adv_loss: MockInternalRepresentation,
                     reg_loss: MockInternalRepresentation) -> MockInternalRepresentation:
    return adv_loss.map(reg_loss, lambda d1, d2: np.array(np.mean(d1) + np.mean(d2)))
