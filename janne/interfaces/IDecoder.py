"""Description of the informal interface :class:`IDecoder`.

This interface represent an object that can "decode" pottentially an input
and return an event and the truth it's associated with if availabe
"""
from typing import Optional, Tuple, Any
from abc import ABC, abstractmethod

import numpy as np

class IDecoder(ABC):
  """Informal interface representing a decoder. It should not be instacied but
  instead be inherited by decoder implementation

  This interface ensure that the concrete implementation can be used as an
  iterator

  .. highlight:: python
  .. code-block:: python

    for (event, truth) in decoder:
      ...
  """

  @abstractmethod
  def __init__(self):
    pass

  @abstractmethod
  def initialize(self, config: Any) -> None:
    """Initialize the decoder

    The type of ``config`` is ``Any`` but should be overwritten when
    implementing the concrete class

    :param Any config: The configuration to initialize the IDecoder
    """
    pass

  def __iter__(self):
    return self

  @abstractmethod
  def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return the next element that have been decoded

    :return: A tuple containing the element and the corresponding truth if
     it exist
    """
    pass

  def next_event(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return the next event

    :return: A tuple containing the element and the corresponding truth if
     it exist

    """

    return next(self)
