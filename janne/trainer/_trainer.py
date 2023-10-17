"""Provide training loop for the ANN
"""
from abc import abstractmethod
from dataclasses import dataclass
from typing import (Callable, Generic, Iterator, Optional,
                    Protocol, Tuple, TypeVar, OrderedDict,
                    List)

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

def _accumulate_event(it: Iterator[Tuple[NDArray, NDArray]], batch_size: int) -> Tuple[List[NDArray], List[NDArray]]:
  if batch_size <= 0:
    raise ValueError("Batch size must be positive")

  evts: List[NDArray] = []
  truths: List[NDArray] = []

  while len(evts) < batch_size:
    evt, truth = next(it)
    evts.append(evt)
    truths.append(truth)

  return evts, truths



Sa = TypeVar("Sa", bound="SupportsAdd")
class SupportsAdd(Protocol):
  @abstractmethod
  def __add__(self, other: Sa) -> Sa:
    pass

T = TypeVar("T")
@dataclass
class MonitoringVars(Generic[T]):
  variables: OrderedDict[str, Callable[[List[T], List[T], List[T]], float]]

  callback: Callable[[str, float], None]



Ta = TypeVar("Ta", bound=SupportsAdd)
Tm = TypeVar("Tm", bound=SupportsAdd)
def ann_training_loop(reader: Reader, cs_reader: Reader,
                      ann: IAdversorial[Ta], reco: IModel[Tm],
                      config : ANNTrainingLoopConfig, verbose=False,
                      cs_monitoring: Optional[MonitoringVars[NDArray[np.float64]]] = None,
                      evt_monitoring: Optional[MonitoringVars[NDArray[np.float64]]] = None) -> None:

  epoch = 0
  n_batches = 0

  event_batch_generator = _get_events_from_repeated_reader(reader)
  cs_batch_generator = _get_events_from_repeated_reader(cs_reader)

  if verbose:
    print("\nStarting training the ANN")
  while epoch < config.n_epochs:

    # Run the ann

    events, truths = _accumulate_event(event_batch_generator, config.batch_size)

    ann_events = ann.migrate(events)
    ann_perturbated = ann.perturbate(ann_events)
    np_perturbated = ann.unmigrate(ann_perturbated)

    # Run the reco
    model_t_events = reco.transform(np_perturbated)
    model_p_events = reco.migrate(model_t_events)
    model_prediction = reco.predict(model_p_events)

    adv_loss = ann.adv_loss(
            truth= truths,
            prediction= reco.unmigrate(model_prediction),
            raw_data= events,
            perturbated_data= ann_perturbated)

    cs_events, cs_truths = _accumulate_event(cs_batch_generator, config.n_cs_event)

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



    if cs_monitoring:
      for name, fun in cs_monitoring.variables.items():
        cs_monitoring.callback(name, fun(ann.unmigrate(reg_loss), cs_truths, reco.unmigrate(model_cs_prediction)))

    if evt_monitoring:
      for name, fun in evt_monitoring.variables.items():
        evt_monitoring.callback(name, fun(ann.unmigrate(reg_loss), truths, reco.unmigrate(model_prediction)))


    if reg_loss is None or adv_loss is None:
      raise RuntimeError("Seems that the batch size is 0, cannot run with batch size of 0")
    ann.back_propagate(ann.combine_losses(adv_loss, reg_loss))

    n_batches = (n_batches + 1) % config.batch_per_epoch

    if verbose:
      print(f"\33[2K\rEpoch {epoch+1}/{config.n_epochs} : {n_batches}/{config.batch_per_epoch}", end="")

    if n_batches == 0:
      epoch += 1

@dataclass
class ModelTrainingLoopConfig:
  n_epochs: int
  batch_size: int
  batch_per_epoch: int

  validation_n_batch: int

def model_training_loop(reader: Reader, reco: IModel[Tm],
                        config: ModelTrainingLoopConfig, verbose=False,
                        model_monitoring: Optional[MonitoringVars[NDArray]] = None) -> None:

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

      evt, truth = _accumulate_event(event_batch_generator, config.batch_size)

      t_evt = reco.transform(evt)
      i_evt = reco.migrate(t_evt)
      model_pred = reco.predict(i_evt)

      loss = reco.loss(model_pred, truth)


      if loss is None:
        raise RuntimeError("Seems that the batch size is 0, cannot run with batch size of 0")
      reco.back_propagate(loss)

      if model_monitoring:
        for name, fun in model_monitoring.variables.items():
          model_monitoring.callback(name, fun(reco.unmigrate(loss), truth, reco.unmigrate(model_pred)))

      if verbose:
        print(f"\33[2K\rEpoch {epoch+1}/{config.n_epochs} : {n_batches}/{config.batch_per_epoch}", end="")
