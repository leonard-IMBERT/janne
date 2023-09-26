"""Module containing the test for the Reader class
"""


from janne import Reader
from janne.tests.decoder_test import (StandardMockDecoder,
                           config_genrator,
                           MOCK_DECODER_SIZE)
import multiprocessing
import numpy as np

class TestReader:
  """Class containing the test for the Reader class
  """
  def test_reader_initialisation_and_destruction(self):
    reader = Reader(config_genrator(5, True), StandardMockDecoder, n_workers=3, seed=0)

    assert True

    reader.close()

    assert True

    del reader

    assert True

  def test_reader_get_event(self):
    n_decoder = 5
    n_worker = 1
    reader = Reader(config_genrator(n_decoder, True), StandardMockDecoder, n_workers=n_worker, seed=0)

    assert reader is not None

    n_evt = 0
    for event, truth in reader:
      assert truth is not None
      assert event is not None

      n_evt += 1

    assert n_evt == MOCK_DECODER_SIZE * n_decoder

    reader.close()

  def test_number_workers(self):
    n_decoder = 5
    n_worker = 3
    _ = Reader(config_genrator(n_decoder, True), StandardMockDecoder, n_workers=n_worker, seed=0)

    assert len(multiprocessing.active_children()) == n_worker

    _.close()

  def test_regeneration(self):
    n_decoder = 10
    n_worker = 1

    reader1 = Reader(config_genrator(n_decoder, True, offset=True), StandardMockDecoder, n_workers=n_worker, seed=0)

    f_r1_ev, f_r1_tru = next(reader1)

    reader2 = reader1.regenerate()

    f_r2_ev, f_r2_tru = next(reader2)

    assert np.array_equal(f_r1_ev, f_r2_ev)
    assert f_r2_tru is not None
    assert f_r1_tru is not None
    assert np.array_equal(f_r1_tru, f_r2_tru)

    for _ in reader1:
      pass

    reader3 = reader1.regenerate()

    f_r3_ev, f_r3_tru = next(reader3)

    assert np.array_equal(f_r1_ev, f_r3_ev)
    assert f_r3_tru is not None
    assert f_r1_tru is not None
    assert np.array_equal(f_r1_tru, f_r3_tru)

    reader1.close()
    reader2.close()
    reader3.close()

