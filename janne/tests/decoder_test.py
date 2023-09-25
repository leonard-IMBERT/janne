"""Module that contains test for the IDecoder interface
"""
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Any

import numpy as np
import pytest

from janne.interfaces import IDecoder

@dataclass
class StandardMockDecoderConfig:
  filename: str
  mode: int
  w_truth: bool
  offset: int = 0

def config_genrator(n_dec: int, w_truth: bool, offset=False):
  for i in range(n_dec):
    yield StandardMockDecoderConfig(f"/some/file/with.input.${i}.root", i % 2, w_truth, i if offset else 0)


MOCK_DECODER_SIZE=100

class StandardMockDecoder(IDecoder):
  """A standard mock decoder
  """
  def __init__(self, config: Union[StandardMockDecoderConfig, None] = None):
    self._config = None
    if config:
      self._config = config
      self.initialize(self._config)
    self._cur_event = 0

  def initialize(self, config: StandardMockDecoderConfig) -> None:
    self._config = config

  def __next__(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if self._config is None:
      raise RuntimeError("StandardMockDecoder is not initialized")

    if self._cur_event >= MOCK_DECODER_SIZE:
      raise StopIteration()

    self._cur_event += 1

    return (np.full((100, 5), self._cur_event + self._config.offset),
            np.full((4), self._cur_event) if self._config.w_truth else None)

  def config(self) -> Any:
    return self._config

class EmptyMockDecoder(IDecoder):
  pass


class TestDecoder():
  """Class containing the tests for the decoder
  """
  def test_non_initializd_should_fail(self):
    u_decoder = StandardMockDecoder()

    with pytest.raises(RuntimeError):
      _ = next(u_decoder)

  def test_init_with_truth(self):
    i_decoder = StandardMockDecoder()

    config = StandardMockDecoderConfig(
            filename="foo.root",
            mode=0,
            w_truth=True)

    i_decoder.initialize(config)

    evt, truth = next(i_decoder)

    assert evt is not None

    assert truth is not None

  def test_raise_when_stop_iteration(self):
    i_decoder = StandardMockDecoder()

    config = StandardMockDecoderConfig(
            filename="foo.root",
            mode=0,
            w_truth=True)

    i_decoder.initialize(config)

    with pytest.raises(StopIteration):
      for _ in range(MOCK_DECODER_SIZE * 2):
        next(i_decoder)


