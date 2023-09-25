"""Provide training loop for the ANN
"""
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..interfaces import IAdversorial, IModel
from .._reader import Reader

@dataclass
class ANNTrainingLoopConfig:
  n_epochs: int
  batch_size: int
  batch_per_epoch: int

  n_cs_event: int


def _generate_readers(reader: Reader) -> Iterator[Reader]:
  yield reader

  while True:
    yield reader.regenerate()


def _get_events_from_repeated_reader(reader: Reader, batch_size: int) -> Iterator[Tuple[NDArray, NDArray]]:

  r_generator = _generate_readers(reader)
  try:
    curr_generator = next(r_generator)
  except StopIteration:
    return

  def _get_events(init_events: Union[List[NDArray], None] = None,
                  init_truths: Union[List[NDArray], None] = None) -> Tuple[NDArray, NDArray]:
    nonlocal r_generator
    nonlocal curr_generator

    events: List[NDArray] = init_events if init_events is not None else []
    truths: List[NDArray] = init_truths if init_truths is not None else []

    for event, truth in curr_generator:
      if truth is None:
        raise RuntimeError("CS Reader gave event without truth, cannot"
                           " train without truth")
      events.append(event)
      truths.append(truth)

      # If batch full
      if len(events) == batch_size:
        break

    # If reader ended but not enough events
    if len(events) < batch_size:
      curr_generator = next(r_generator)

      #continuer filling
      return _get_events(events, truths)

    #Cs batch is full
    return (np.array(events),
            np.array(truths))
  while True:
    yield _get_events()


def ann_training_loop(reader: Reader, cs_reader: Reader,
                      ann: IAdversorial, reco: IModel,
                      config : ANNTrainingLoopConfig) -> None:

  epoch = 0
  n_batches = 0

  event_batch_generator = _get_events_from_repeated_reader(reader, config.batch_size)
  cs_batch_generator = _get_events_from_repeated_reader(cs_reader, config.n_cs_event)

  while epoch < config.n_epochs:

    # Contruct batches
    b_events, b_truths = next(event_batch_generator)

    # Run the ann
    ann_b_events = ann.migrate(b_events)
    ann_perturbated = ann.perturbate(ann_b_events)
    np_perturbated = ann.unmigrate(ann_perturbated)

    # Run the reco
    model_b_p_events = reco.migrate(np_perturbated)
    model_prediction = reco.migrate(reco.predict(model_b_p_events))

    adv_loss = ann.adv_loss(
            truth= b_truths,
            prediction= model_prediction,
            raw_data= b_events,
            perturbated_data= ann_perturbated)

    # Construct cs_batches
    cs_events, cs_truths = next(cs_batch_generator)

    # Run the ann on cs
    ann_cs_events = ann.migrate(cs_events)
    ann_cs_perturbated = ann.perturbate(ann_cs_events)
    np_cs_perturbated = ann.unmigrate(ann_cs_perturbated)

    # Run the reco on cs
    model_cs_p_event = reco.migrate(np_cs_perturbated)
    model_cs_prediction = reco.migrate(reco.predict(model_cs_p_event))

    reg_loss = ann.reg_loss(
            truth= cs_truths,
            prediction= model_cs_prediction,
            raw_data= cs_events,
            perturbated_data= ann_perturbated)

    ann.back_propagate(adv_loss + reg_loss)

    n_batches = (n_batches + 1) % config.batch_per_epoch

    if n_batches == 0: epoch += 1

