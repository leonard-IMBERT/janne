"""Module containing test for the training loops
"""

import warnings
from .ann_test import MockAdversorial, MockAdversorialConf
from .model_test import MockModel, MockModelConf
from .decoder_test import StandardMockDecoder, StandardMockDecoderConfig

from janne import Reader
from janne.trainer import ANNTrainingLoopConfig, ann_training_loop

class TestTrainingLoop():
  """Class containing the tests for the ANN training loop
  """
  def test_training_loop(self):
    ann = MockAdversorial()
    ann_conf = MockAdversorialConf(shift=2, lr=1e-4)
    ann.initialize(ann_conf)

    model = MockModel()
    model_conf = MockModelConf()
    model.initialize(model_conf)

    def decoder_conf_gen(n_gen: int, offset=0):
      for i in range(n_gen):
        yield StandardMockDecoderConfig(filename=f"{i}", mode=0, w_truth=True, offset=offset)

    reader = Reader(decoder_conf_gen(10), StandardMockDecoder, n_workers=8, seed=0, buffer=10000)
    reader_cs = Reader(decoder_conf_gen(10, 2), StandardMockDecoder, n_workers=8, seed=0, buffer=10000)

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      ann_training_loop(reader, reader_cs, ann, model,
                        ANNTrainingLoopConfig(n_epochs=5, batch_size=256, batch_per_epoch=300, n_cs_event=100),
                        verbose=True)

    assert True

    reader.close()
    reader_cs.close()

