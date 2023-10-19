"""Internal module implementing the Reader class that automatically paralellise
IDecoder
"""
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, Union

import random
import itertools

from multiprocessing import Process, Queue, Event as AtomicBoolean
from multiprocessing.synchronize import Event as TAtomicBoolean

import numpy as np
from numpy._typing import _Shape
from numpy.typing import NDArray


from .interfaces import IDecoder

Event = Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]
RawEvent = Tuple[_Shape, bytes, Optional[_Shape], Optional[bytes]]

def _decode_consumer(tdecoder: type[IDecoder], config: Any, event_queue: Queue,
                     stop_flag: TAtomicBoolean,
                     ):
  decoder = tdecoder()
  decoder.initialize(config)
  for event in decoder:
    if stop_flag.is_set():
      break
    event_queue.put((event[0].shape, event[0].tobytes(),
                     event[1].shape if event[1] is not None else None,
                     event[1].tobytes() if event[1] is not None else None))
  event_queue.put(None)

class WorkerState(Enum):
  UNINITIALIZED = 0
  INITIALIZED = 1
  CONSUMED = 2

class Worker():
  """A worker that will run a decoder in a different Process

  The worker is responsible for the managing of the ressources needed
  by the decoder

  :param tdecoders: The type or the :class:`IDecoder`
  :param conf: A configuration for the :class:`IDecoder`
  :param int buffer: The size of the buffer to use
  """
  def __init__(self, tdecoder: type[IDecoder], config: Any, buffer: int) -> None:
    if buffer <= 0:
      raise ValueError(f"Cannot initialize a buffer with a size of {buffer}")

    self._stop_flag: TAtomicBoolean = AtomicBoolean()
    self._stop_flag.clear()

    self._state = WorkerState.UNINITIALIZED

    self._worker: Optional[Process] = None
    self._tdecoder = tdecoder
    self._config = config

    self._buffer_size = buffer

    self.queue: Queue = Queue(maxsize=self._buffer_size)

  def start(self):
    self._worker = Process(target=_decode_consumer, args=(self._tdecoder, self._config, self.queue, self._stop_flag))
    self._worker.start()

    self._state = WorkerState.INITIALIZED

  def stop(self):
    if self._state == WorkerState.INITIALIZED:
      self._stop_flag.set()

      while self.queue.get() is not None:
        pass

      self.queue.close()

      if self._worker:
        self._worker.join()

      self._state = WorkerState.CONSUMED
    elif self._state == WorkerState.UNINITIALIZED:
      self.queue.close()

      if self._worker:
        self._worker.join()

      self._state = WorkerState.CONSUMED

  def reset(self):
    if self._state != WorkerState.CONSUMED:
      raise RuntimeError("Cannot reset workers while they are not fully consumed")

    self.queue = Queue(maxsize=self._buffer_size)
    self._stop_flag.clear()

    self._state = WorkerState.UNINITIALIZED

  def __next__(self) -> Event:
    if self._state == WorkerState.CONSUMED:
      raise StopIteration

    ret_val = self.queue.get()

    if ret_val is None:
      self.queue.close()
      if self._worker:
        self._worker.join()
      self._state = WorkerState.CONSUMED

      raise StopIteration

    dshape, data, tshape, truth = ret_val
    return (np.ndarray(shape=dshape, dtype=np.float64, buffer=data),
            np.ndarray(shape=tshape, dtype=np.float64, buffer=truth) if truth is not None else None)

  def get_state(self):
    return self._state

  def __del__(self):
    self.stop()

class Reader(Iterable[Event]):
  """A :class:`Reader` is responsible for managing and reading

  The :class:`Reader` reads the files using the specified :class:`IDecoder`
  and the configuration given. The :class:`Reader` support the :class:`Iterable`
  api.

  :param config_generator: A generator of configuration of :class:`IDecoder`
  :param decoders: The type or the repeatable list of type of
   :class:`IDecoder` to use
  :param int seed: The seed to initialize the random number generator
  :param int n_workers: The number of worker to use for the Reader.
   Must be superior or equal to 1
  """
  def __init__(self, config_generator: Union[List[Any],  Iterable[Any]],
               decoders: Union[List[type[IDecoder]],  type[IDecoder]],
               seed: int = 42,
               n_workers: int = 1,
               buffer: int = 1000):

    self._p_config_generator = config_generator
    self._p_decoders = decoders
    self._p_seed = seed
    self._p_n_workers = n_workers

    self._buffer_len = buffer

    # pylint: disable=line-too-long
    self._random = random.Random(seed)
    self._np_random = np.random.default_rng(seed)


    # Decoders and their threads
    self._n_workers = n_workers
    self._workers : List[Worker] = []

    decoders_it = itertools.cycle(decoders if isinstance(decoders, Iterable) else [decoders])

    # Initialize workers
    for (config, decoder) in zip(config_generator, decoders_it):
      self._workers.append(Worker(decoder, config, int(self._buffer_len/self._p_n_workers)))

    self._indexes = np.linspace(0, len(self._workers)-1, len(self._workers)).astype(int)


    # Start the threads
    self._check_workers()

  def _worker_mask(self, state: WorkerState) -> NDArray:
    return np.array(list(map(lambda w: w.get_state() == state, self._workers)))

  def __iter__(self):
    return self

  def __next__(self) -> Event:
    index = self._np_random.choice(self._indexes[self._worker_mask(WorkerState.INITIALIZED)])

    try:
      return next(self._workers[index])
    except StopIteration as exc:
      self._check_workers()

      if np.alltrue(self._worker_mask(WorkerState.CONSUMED)):
        raise StopIteration from exc
      else:
        return next(self)

  def _check_workers(self) -> bool:
    while np.sum(self._worker_mask(WorkerState.INITIALIZED)) < self._n_workers:
      if np.alltrue(np.invert(self._worker_mask(WorkerState.UNINITIALIZED))):
        return False
      index = self._np_random.choice(self._indexes[self._worker_mask(WorkerState.UNINITIALIZED)])
      self._workers[index].start()

    return True

  def close(self):
    """Close the reader, freeing all ressources"""
    for worker in self._workers:
      worker.stop()

  def __del__(self):
    self.close()

  def regenerate(self, seed: Union[int, None] = None):
    """Regenerate this reader using the same parameters except
    the seed if specified

    :param seed: The new seed to use. If None, use the same seed
    """

    self.close()

    self._random = random.Random(seed if seed is not None else self._p_seed)
    self._np_random = np.random.default_rng(seed if seed is not None else self._p_seed)

    for worker in self._workers:
      worker.reset()

    self._check_workers()

    return self
