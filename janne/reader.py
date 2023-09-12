from typing import Any, Iterable, List, Optional, Tuple, Union

import random
import itertools
from threading import Thread, Condition
from queue import Queue

import numpy as np

from janne.interfaces.idecoder import IDecoder

Event = Tuple[np.ndarray, Optional[np.ndarray]]

def _decode_consumer(decoder: IDecoder, condition: Condition,
                     event_queue: Queue[Event]
                     ):
  with condition:
    for event in decoder:
      with condition:
        condition.wait()
        event_queue.put(event)
        condition.notify()



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
    # pylint: disable=line-too-long
    self._random = random.Random(seed)
    self._np_random = np.random.default_rng(seed)

    # Event storage and fetching
    self._need = Condition()
    self._events : Queue[Event] = Queue()

    self._n_workers = n_workers

    # Decoders and their threads
    self._decoders : List[IDecoder] = []
    self._workers : List[Thread] = []
    self._consumed_mask = []
    self._indexes = []

    decoders_it = itertools.cycle(decoders if isinstance(decoders, Iterable) else [decoders])

    # Initialize workers
    for (config, decoder) in zip(config_generator, decoders_it):
      dec = decoder()
      dec.initialize(config)
      self._decoders.append(dec)
      self._workers.append(Thread(target=_decode_consumer,
                                  args=(dec, self._need, self._events)))
      self._consumed_mask.append(False)
      self._indexes.append(0 if len(self._indexes) == 0 else self._indexes[-1] + 1)

    self._indexes = np.array(self._indexes)
    self._consumed_mask = np.array(self._consumed_mask)

    # Start the threads
    while sum(map(lambda thread: thread.is_alive(), self._workers)) < self._n_workers:
      index = self._np_random.choice(self._indexes[np.invert(self._consumed_mask)])

      self._workers[index].start()
      self._consumed_mask[index] = True

    # pylint: enable=line-too-long

  def __iter__(self):
    return self

  def __next__(self) -> Event:
    if not self._events.empty():
      return self._events.get()

    # If every thread is dead (never start or finished reading) and every
    # file have been consumed, raise Stop Iteration
    if (sum(map(lambda thread: thread.is_alive(), self._workers)) == 0
        and sum(np.invert(self._consumed_mask)) == 0):
      raise StopIteration


    # Notify then release lock
    with self._need:
      self._need.notify()
      self._need.wait()

    # Try to aquire the lock again

    return self._events.get()

  def _check_workers(self) -> bool:
    while sum(map(lambda thread: thread.is_alive(), self._workers)) < self._n_workers:
      avail_files = self._indexes[np.invert(self._consumed_mask)]
      if len(avail_files) <= 0:
        return False
      index = self._np_random.choice(avail_files)

      self._workers[index].start()
      self._consumed_mask[index] = True

    return True
