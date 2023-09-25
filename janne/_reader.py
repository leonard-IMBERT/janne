"""Internal module implementing the Reader class that automatically paralellise
IDecoder
"""
import queue
from typing import Any, Iterable, List, Optional, Tuple, Union

import random
import itertools


from multiprocessing import Process, Queue, Event as AtomicBoolean
from multiprocessing.synchronize import Event as TAtomicBoolean

import numpy as np

from .interfaces import IDecoder

Event = Tuple[np.ndarray, Optional[np.ndarray]]

def _decode_consumer(tdecoder: type[IDecoder], config: Any, event_queue: Queue,
                     stop_flag: TAtomicBoolean,
                     ):
  decoder = tdecoder()
  decoder.initialize(config)
  for event in decoder:
    if stop_flag.is_set(): return
    event_queue.put(event)



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
               n_workers: int = 1):

    self._p_config_generator = config_generator
    self._p_decoders = decoders
    self._p_seed = seed
    self._p_n_workers = n_workers

    # pylint: disable=line-too-long
    self._random = random.Random(seed)
    self._np_random = np.random.default_rng(seed)

    # Event storage and fetching
    self._events : Queue = Queue(10)
    self._stop_flag: TAtomicBoolean = AtomicBoolean()
    self._stop_flag.clear()

    self._n_workers = n_workers

    # Decoders and their threads
    self._decoders : List[Tuple[Any, type[IDecoder]]] = []
    self._workers : List[Union[Process, None]] = []
    self._consumed_mask = []
    self._indexes = []

    decoders_it = itertools.cycle(decoders if isinstance(decoders, Iterable) else [decoders])

    # Initialize workers
    for (config, decoder) in zip(config_generator, decoders_it):
      self._decoders.append((config, decoder))
      self._workers.append(None)

    self._consumed_mask = np.full((len(self._decoders)), False)
    self._indexes = np.array(range(len(self._decoders)))

    # Start the threads
    while sum(map(lambda thread: thread is not None and thread.is_alive(), self._workers)) < self._n_workers:
      index = self._np_random.choice(self._indexes[np.invert(self._consumed_mask)])

      dec_config, dec_typ = self._decoders[index]
      self._workers[index] = Process(
          target=_decode_consumer,
          args=(dec_typ, dec_config, self._events, self._stop_flag)
          )
      self._workers[index].start()
      self._consumed_mask[index] = True

    # pylint: enable=line-too-long

  def __iter__(self):
    return self

  def __next__(self) -> Event:
    while True:
      self._check_workers()

    # If every thread is dead (never start or finished reading) and every
    # file have been consumed, raise Stop Iteration
      if (sum(map(lambda thread: thread is not None and thread.is_alive(), self._workers)) == 0
          and sum(np.invert(self._consumed_mask)) == 0
          and self._events.empty()):
        raise StopIteration

      try:
        return self._events.get(True, 1/100)
      except queue.Empty:
        continue

  def _check_workers(self) -> bool:
    while sum(map(lambda thread: thread is not None and thread.is_alive(), self._workers)) < self._n_workers:
      avail_files = self._indexes[np.invert(self._consumed_mask)]
      if len(avail_files) <= 0:
        return False
      index = self._np_random.choice(avail_files)


      dec_config, dec_typ = self._decoders[index]
      self._workers[index] = Process(
          target=_decode_consumer,
          args=(dec_typ, dec_config, self._events, self._stop_flag)
          )
      self._workers[index].start()
      self._consumed_mask[index] = True

      while sum(map(lambda _:  _ is not None and _.is_alive(), self._workers)) == 0:
        continue

    return True

  def __del__(self):
    self._stop_flag.set()


    while not self._events.empty():
      self._events.get()

    for worker in self._workers:
      if worker is not None and worker.is_alive():
        worker.join()
    self._events.close()
    self._events.join_thread()

  def regenerate(self, seed: Union[int, None] = None):
    """Regenerate this reader using the same parameters except
    the seed if specified

    :param seed: The new seed to use. If None, use the same seed
    """

    return Reader(list(map(lambda _: _[0] , self._decoders)),
                  self._p_decoders,
                  seed if seed is not None else self._p_seed,
                  self._p_n_workers)
