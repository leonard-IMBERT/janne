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


def _get_events_from_repeated_reader(reader: Reader, batch_size: int) -> Iterator[Tuple[NDArray, NDArray]]:

  def _get_events(init_events: Union[List[NDArray], None] = None,
                  init_truths: Union[List[NDArray], None] = None) -> Tuple[NDArray, NDArray]:

    events: List[NDArray] = init_events if init_events is not None else []
    truths: List[NDArray] = init_truths if init_truths is not None else []

    for event, truth in reader:
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
      reader.regenerate()

      #continuer filling
      return _get_events(events, truths)

    #Cs batch is full
    return (np.array(events),
            np.array(truths))
  while True:
    yield _get_events()


def ann_training_loop(reader: Reader, cs_reader: Reader,
                      ann: IAdversorial, reco: IModel,
                      config : ANNTrainingLoopConfig, verbose=False) -> None:

  epoch = 0
  n_batches = 0

  event_batch_generator = _get_events_from_repeated_reader(reader, config.batch_size)
  cs_batch_generator = _get_events_from_repeated_reader(cs_reader, config.n_cs_event)

  if verbose:
    print("\nStarting training the ANN")
  while epoch < config.n_epochs:

    # Contruct batches
    b_events, b_truths = next(event_batch_generator)

    # Run the ann
    ann_b_events = ann.migrate(b_events)
    ann_perturbated = ann.perturbate(ann_b_events)
    np_perturbated = ann.unmigrate(ann_perturbated)

    # Run the reco
    model_b_p_events = reco.migrate(np_perturbated)
    model_prediction = reco.predict(model_b_p_events)

    adv_loss = ann.adv_loss(
            truth= b_truths,
            prediction= reco.unmigrate(model_prediction),
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
    model_cs_prediction = reco.predict(model_cs_p_event)

    reg_loss = ann.reg_loss(
            truth= cs_truths,
            prediction= reco.unmigrate(model_cs_prediction),
            raw_data= cs_events,
            perturbated_data= ann_cs_perturbated)

    ann.back_propagate(ann.combine_losses(adv_loss, reg_loss))

    n_batches = (n_batches + 1) % config.batch_per_epoch

    if verbose:
      print(f"\33[2K\rEpoch {epoch+1}/{config.n_epochs} : {n_batches}/{config.batch_per_epoch}", end="")

    if n_batches == 0:
      epoch += 1

  reader.close()
  cs_reader.close()

