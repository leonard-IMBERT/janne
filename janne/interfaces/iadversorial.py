"""Description of the informal interface :class:`IAdversorial`

This interface represent an object that can "perturbate" and event
so that a :class:`IModel` does not recontruct correctly the
truth
"""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np

T = TypeVar("T")

class IAdversorial(ABC, Generic[T]):
  """Informal interface representing an adversorial algorithm

  This algorithm work in tandem with a :class:`IModel` and will
  try to fool it by `perturbating` the data.
  """

  @abstractmethod
  def __init__(self):
    pass

  @abstractmethod
  def initialize(self, config: Any) -> None:
    """Initialize the IAdversorial with the given configuration

    :param Any config: The configuration for the adversorial. Basic type is
     `any` but should be overwritten when implementing in concrete class
    """
    pass

  @abstractmethod
  def perturbate(self, raw_data: T) -> T:
    """Perturbate the raw data

    :param T raw_data: The raw data to perturbate
    :return: The perturbated data in a format that can be understood by
     the :class:`IModel`
    """
    pass

  @abstractmethod
  def loss(self, truth: np.ndarray, prediction: T,
           raw_data: np.ndarray, perturbated_data: T) -> T:
    """Compute the loss of the adversorial algorithm

    :param truth: The true prediction
    :param prediction: The prediction produced by the :class:`IModel`
    :param raw_data: The raw data
    :param perturbated_data: The data pertubated by this :class:`IAdversorial`
    :return: The computed loss
    """
    pass


  @abstractmethod
  def back_propagate(self, loss: T) -> None:
    """Backpropagate the loss to the IAdversorial

    :param T loss: The loss that should have been computed with the
     loss method
    """
    pass
