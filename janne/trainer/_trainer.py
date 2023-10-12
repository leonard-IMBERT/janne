"""Provide training loop for the ANN
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterator, Protocol, Tuple, TypeVar, Union

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


def _get_events_from_repeated_reader(reader: Reader) -> Iterator[Tuple[NDArray, NDArray]]:

  def _get_events() -> Tuple[NDArray, NDArray]:

    try:
      evt, truth = next(reader)
      if truth is None:
        raise RuntimeError("Cannot train without truth")

      return (evt, truth)
    except StopIteration:
      reader.regenerate()
      return _get_events()

  while True:
    yield _get_events()

Sa = TypeVar("Sa", bound="SupportsAdd")
class SupportsAdd(Protocol):
  @abstractmethod
  def __add__(self, other: Sa) -> Sa:
    pass

Ta = TypeVar("Ta", bound=SupportsAdd)
Tm = TypeVar("Tm", bound=SupportsAdd)
def ann_training_loop(reader: Reader, cs_reader: Reader,
                      ann: IAdversorial[Ta], reco: IModel[Tm],
                      config : ANNTrainingLoopConfig, verbose=False) -> None:

  epoch = 0
  n_batches = 0

  event_batch_generator = _get_events_from_repeated_reader(reader)
  cs_batch_generator = _get_events_from_repeated_reader(cs_reader)

  if verbose:
    print("\nStarting training the ANN")
  while epoch < config.n_epochs:


    adv_loss: Union[Ta, None] = None

    # Run the ann
    for _ in range(config.batch_size):
      # Contruct batches
      events, truths = next(event_batch_generator)

      ann_events = ann.migrate(events)
      ann_perturbated = ann.perturbate(ann_events)
      np_perturbated = ann.unmigrate(ann_perturbated)

      # Run the reco
      model_p_events = reco.migrate(np_perturbated)
      model_prediction = reco.predict(model_p_events)

      e_adv_loss = ann.adv_loss(
              truth= truths,
              prediction= reco.unmigrate(model_prediction),
              raw_data= events,
              perturbated_data= ann_perturbated)

      adv_loss = e_adv_loss if adv_loss is None else adv_loss + e_adv_loss

    reg_loss: Union[Ta, None] = None
    for _ in range(config.n_cs_event):
      # Construct cs_batches
      cs_events, cs_truths = next(cs_batch_generator)

      # Run the ann on cs
      ann_cs_events = ann.migrate(cs_events)
      ann_cs_perturbated = ann.perturbate(ann_cs_events)
      np_cs_perturbated = ann.unmigrate(ann_cs_perturbated)

      # Run the reco on cs
      model_cs_p_event = reco.migrate(np_cs_perturbated)
      model_cs_prediction = reco.predict(model_cs_p_event)

      e_reg_loss = ann.reg_loss(
              truth= cs_truths,
              prediction= reco.unmigrate(model_cs_prediction),
              raw_data= cs_events,
              perturbated_data= ann_cs_perturbated)

      reg_loss = e_reg_loss if reg_loss is None else reg_loss + e_reg_loss


    if reg_loss is None or adv_loss is None:
      raise RuntimeError("Seems that the batch size is 0, cannot run with batch size of 0")
    ann.back_propagate(ann.combine_losses(adv_loss, reg_loss))

    n_batches = (n_batches + 1) % config.batch_per_epoch

    if verbose:
      print(f"\33[2K\rEpoch {epoch+1}/{config.n_epochs} : {n_batches}/{config.batch_per_epoch}", end="")

    if n_batches == 0:
      epoch += 1

  reader.close()
  cs_reader.close()


@dataclass
class ModelTrainingLoopConfig:
  n_epochs: int
  batch_size: int
  batch_per_epoch: int

  validation_n_batch: int

def model_training_loop(reader: Reader, reco: IModel[Tm],
                        config: ModelTrainingLoopConfig, verbose=False) -> None:

  event_batch_generator = _get_events_from_repeated_reader(reader)

  validation_events = []
  validation_truth = []
  for _ in range(config.validation_n_batch):
    evt, truth = next(event_batch_generator)
    validation_events.append(evt)
    validation_truth.append(truth)

  validation_events = np.concatenate(validation_events, axis=0)
  validation_truth = np.concatenate(validation_truth, axis=0)

  n_batches = 0

  if verbose:
    print("\nStarting training the Model")

  for epoch in range(config.n_epochs):

    for n_batches in range(config.batch_per_epoch):

      loss: Union[Tm, None] = None
      for _ in range(config.batch_size):

        evt, truth = next(event_batch_generator)

        t_evt = reco.transform(reco.migrate(evt))
        model_pred = reco.predict(t_evt)

        e_loss = reco.loss(model_pred, truth)

        loss = e_loss if loss is None else loss + e_loss

      if loss is None:
        raise RuntimeError("Seems that the batch size is 0, cannot run with batch size of 0")
      reco.back_propagate(loss)

      if verbose:
        print(f"\33[2K\rEpoch {epoch+1}/{config.n_epochs} : {n_batches}/{config.batch_per_epoch}", end="")

  reader.close()
