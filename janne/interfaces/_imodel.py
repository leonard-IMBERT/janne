"""Description of the informal interface :class:`IModel`.

This interface represent an object that can "infer" from a juno event a set of
observables.
"""
from abc import ABC, abstractmethod
from typing import Any, List, TypeVar, Generic

from numpy.typing import NDArray

T = TypeVar("T")
"""Generic type representing the internal data type of the IModel
"""

class IModel(ABC, Generic[T]):
  """Informal interface representing a model.

  This model MUST be able to take raw data as numpy array, but can use any
  type, represented as the generic type T, s intermediate representation.
  This type is usefull for neural network where the output of the transform
  is a representation of data on the GPU and you don't want to move them
  from the gpu memory between steps

  The methods will be used by the framework in the following order:
  transform -> predict -> loss
  """

  @abstractmethod
  def __init__(self):
    pass

  @abstractmethod
  def initialize(self, config: Any) -> None:
    """Initialize the IModel from the given configuration object

    :param Any config: The configuration for the model. Basic type is `any`
      but should be overwritten when implementing in concrete class
    """
    pass

  @abstractmethod
  def migrate(self, raw_data: List[NDArray]) -> T:
    """Migrate the data from numpy to the internal representation

    :param raw_data: The raw data to migrate
    :return: The raw_data in the representation of the internal computation
     library
    """
    pass

  @abstractmethod
  def unmigrate(self, internal_data: T) -> List[NDArray]:
    """Migrate the data from the internal representation to numpy

    :param internal_data: The raw data to migrate
    :return: The internal_data in as a NDArray
     library
    """
    pass

  @abstractmethod
  def transform(self, raw_data: List[NDArray]) -> List[NDArray]:
    """Transform raw data

    Transform raw data in a format that is fitted for prediction.The raw
    data should looks like:
    ``[[pmt_x, pmt_y, pmt_z, charge, time_of_first_hit], ...]``

    :param ndarray raw_data: A numpy array containing the raw data.

    """
    pass

  @abstractmethod
  def predict(self, transformed_data: T) -> T:
    """Predict a results from transformed data

    Predict from the transformed data. ``transformed_data`` should be the
    results from the `transform` method

    :param T transformed_data: Transformer data
    :return: The prediction corresponding to the transformed_data
    :rtype: T
    """
    pass

  @abstractmethod
  def loss(self, prediction: T, truth: List[NDArray]) -> T:
    """Compute the loss between the prediction and the truth

    :param T prediction: The prediction produced from the ``predict method``
    :param ndarray truth: The truth corresponding to the events that produced
     ``prediction``
    :return: The computed loss in a coherent data format
    :rtype: T
    """
    pass

  @abstractmethod
  def back_propagate(self, loss: T) -> None:
    """Backpropagate the loss to the IModel

    :param T loss: The loss that should have been computed with the loss method
    """
    pass
