# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for DQN components."""

# pylint: disable=g-bad-import-order

import collections
import math
from unittest import mock

from dqn_zoo import parts
from dqn_zoo import test_utils
from absl.testing import absltest


class LinearScheduleTest(absltest.TestCase):

  def test_descent(self):
    """Checks basic linear decay schedule."""
    schedule = parts.LinearSchedule(
        begin_t=5, decay_steps=7, begin_value=1.0, end_value=0.3)
    for step in range(20):
      val = schedule(step)
      if step <= 5:
        self.assertEqual(1.0, val)
      elif step >= 12:
        self.assertEqual(0.3, val)
      else:
        self.assertAlmostEqual(1.0 - ((step - 5) / 7) * 0.7, val)

  def test_ascent(self):
    """Checks basic linear ascent schedule."""
    schedule = parts.LinearSchedule(
        begin_t=5, end_t=12, begin_value=-0.4, end_value=0.4)
    for step in range(20):
      val = schedule(step)
      if step <= 5:
        self.assertEqual(-0.4, val)
      elif step >= 12:
        self.assertEqual(0.4, val)
      else:
        self.assertAlmostEqual(-0.4 + ((step - 5) / 7) * 0.8, val)

  def test_constant(self):
    """Checks constant schedule."""
    schedule = parts.LinearSchedule(
        begin_t=5, decay_steps=7, begin_value=0.5, end_value=0.5)
    for step in range(20):
      val = schedule(step)
      self.assertAlmostEqual(0.5, val)

  def test_error_wrong_end_args(self):
    """Checks error in case none or both of end_t, decay_steps are given."""
    with self.assertRaisesRegex(ValueError, 'Exactly one of'):
      _ = parts.LinearSchedule(begin_value=0.0, end_value=1.0, begin_t=5)
    with self.assertRaisesRegex(ValueError, 'Exactly one of'):
      _ = parts.LinearSchedule(
          begin_value=0.0, end_value=1.0, begin_t=5, end_t=12, decay_steps=7)


class RunLoopTest(absltest.TestCase):

  def test_basic(self):
    """Tests sequence of agent and environment interactions in typical usage."""
    tape = []
    agent = test_utils.DummyAgent(tape)
    environment = test_utils.DummyEnvironment(tape, episode_length=4)

    episode_index = 0
    t = 0  # steps = t + 1
    max_steps = 14
    loop_outputs = parts.run_loop(
        agent, environment, max_steps_per_episode=100, yield_before_reset=True)

    for unused_env, timestep_t, unused_agent, unused_a_t in loop_outputs:
      tape.append((episode_index, t, timestep_t is None))

      if timestep_t is None:
        tape.append('Episode begin')
        continue

      if timestep_t.last():
        tape.append('Episode end')
        episode_index += 1

      if t + 1 >= max_steps:
        tape.append('Maximum number of steps reached')
        break

      t += 1

    expected_tape = [
        (0, 0, True),
        'Episode begin',
        'Agent reset',
        'Environment reset',
        'Agent step',
        (0, 0, False),
        'Environment step (0)',
        'Agent step',
        (0, 1, False),
        'Environment step (0)',
        'Agent step',
        (0, 2, False),
        'Environment step (0)',
        'Agent step',
        (0, 3, False),
        'Environment step (0)',
        'Agent step',
        (0, 4, False),
        'Episode end',
        (1, 5, True),
        'Episode begin',
        'Agent reset',
        'Environment reset',
        'Agent step',
        (1, 5, False),
        'Environment step (0)',
        'Agent step',
        (1, 6, False),
        'Environment step (0)',
        'Agent step',
        (1, 7, False),
        'Environment step (0)',
        'Agent step',
        (1, 8, False),
        'Environment step (0)',
        'Agent step',
        (1, 9, False),
        'Episode end',
        (2, 10, True),
        'Episode begin',
        'Agent reset',
        'Environment reset',
        'Agent step',
        (2, 10, False),
        'Environment step (0)',
        'Agent step',
        (2, 11, False),
        'Environment step (0)',
        'Agent step',
        (2, 12, False),
        'Environment step (0)',
        'Agent step',
        (2, 13, False),
        'Maximum number of steps reached',
    ]
    self.assertEqual(expected_tape, tape)


class CsvWriterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_open = mock.patch.object(__builtins__, 'open').start()
    self.fake_file = mock.Mock()
    self.mock_open.return_value.__enter__.return_value = self.fake_file  # pytype: disable=attribute-error  # py39-upgrade

    mock.patch('os.path.exists').start().return_value = True

  def tearDown(self):
    super().tearDown()
    mock.patch.stopall()

  def test_file_writes(self):
    """Tests that file is opened and written correctly."""
    writer = parts.CsvWriter('test.csv')
    self.mock_open.assert_not_called()
    self.fake_file.write.assert_not_called()
    writer.write(collections.OrderedDict([('a', 1), ('b', 2)]))
    self.mock_open.assert_called_once_with('test.csv', mock.ANY)
    self.assertSequenceEqual(
        [mock.call('a,b\r\n'), mock.call('1,2\r\n')],
        self.fake_file.write.call_args_list)
    writer.write(collections.OrderedDict([('a', 3), ('b', 4)]))
    self.assertSequenceEqual(
        [mock.call('test.csv', mock.ANY),
         mock.call('test.csv', mock.ANY)], self.mock_open.call_args_list)
    self.assertSequenceEqual(
        [mock.call('a,b\r\n'),
         mock.call('1,2\r\n'),
         mock.call('3,4\r\n')], self.fake_file.write.call_args_list)

  def test_deserialize_after_header(self):
    """Tests that no header is written unnecessarily after deserialization."""
    writer1 = parts.CsvWriter('test.csv')
    writer1.write(collections.OrderedDict([('a', 1), ('b', 2)]))
    self.assertSequenceEqual(
        [mock.call('a,b\r\n'), mock.call('1,2\r\n')],
        self.fake_file.write.call_args_list)
    writer2 = parts.CsvWriter('test.csv')
    writer2.set_state(writer1.get_state())
    writer2.write(collections.OrderedDict([('a', 3), ('b', 4)]))
    self.assertSequenceEqual(
        [mock.call('a,b\r\n'),
         mock.call('1,2\r\n'),
         mock.call('3,4\r\n')], self.fake_file.write.call_args_list)

  def test_deserialize_before_header(self):
    """Tests that header is written after deserialization if not written yet."""
    writer1 = parts.CsvWriter('test.csv')
    self.fake_file.write.assert_not_called()
    writer2 = parts.CsvWriter('test.csv')
    writer2.set_state(writer1.get_state())
    writer2.write(collections.OrderedDict([('a', 1), ('b', 2)]))
    self.assertSequenceEqual(
        [mock.call('a,b\r\n'), mock.call('1,2\r\n')],
        self.fake_file.write.call_args_list)

  def test_error_new_keys(self):
    """Tests that an error is thrown when an unexpected key occurs."""
    writer = parts.CsvWriter('test.csv')
    writer.write(collections.OrderedDict([('a', 1), ('b', 2)]))
    with self.assertRaisesRegex(ValueError, 'fields not in fieldnames'):
      writer.write(collections.OrderedDict([('a', 3), ('b', 4), ('c', 5)]))

  def test_missing_keys(self):
    """Tests that when a key is missing, an empty value is used."""
    writer = parts.CsvWriter('test.csv')
    writer.write(collections.OrderedDict([('a', 1), ('b', 2), ('c', 3)]))
    writer.write(collections.OrderedDict([('a', 4), ('c', 6)]))
    self.assertSequenceEqual(
        [mock.call('a,b,c\r\n'),
         mock.call('1,2,3\r\n'),
         mock.call('4,,6\r\n')], self.fake_file.write.call_args_list)

  def test_insertion_order_of_fields_preserved(self):
    """Tests that when a key is missing, an empty value is used."""
    writer = parts.CsvWriter('test.csv')
    writer.write(collections.OrderedDict([('c', 3), ('a', 1), ('b', 2)]))
    writer.write(collections.OrderedDict([('b', 5), ('c', 6), ('a', 4)]))
    self.assertSequenceEqual([
        mock.call('c,a,b\r\n'),
        mock.call('3,1,2\r\n'),
        mock.call('6,4,5\r\n')
    ], self.fake_file.write.call_args_list)

  def test_create_dir(self):
    """Tests that a csv file dir is created if it doesn't exist yet."""
    with mock.patch('os.path.exists') as fake_exists, \
         mock.patch('os.makedirs') as fake_makedirs:
      fake_exists.return_value = False
      dirname = '/some/sub/dir'
      _ = parts.CsvWriter(dirname + '/test.csv')
      fake_exists.assert_called_once_with(dirname)
      fake_makedirs.assert_called_once_with(dirname)


class AgentWithStatistics(parts.Agent):

  def __init__(self, statistics):
    self._statistics = statistics

  def step(self, timestep):
    return parts.Action(0)

  def reset(self) -> None:
    pass

  def get_state(self):
    return {}

  def set_state(self, state):
    pass

  @property
  def statistics(self):
    return self._statistics

  @statistics.setter
  def statistics(self, value):
    self._statistics = value


class UnbiasedExponentialWeightedAverageAgentTrackerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    sample_statistics = dict(a=math.nan, b=0)
    self.agent = AgentWithStatistics(sample_statistics)
    self.tracker = parts.UnbiasedExponentialWeightedAverageAgentTracker(
        step_size=0.1, initial_agent=self.agent)

  def test_average_equals_input_on_first_step(self):
    statistics = {'a': 1, 'b': 2}
    self.agent.statistics = statistics
    self.tracker.step(None, None, self.agent, None)
    self.assertEqual(statistics, self.tracker.get())

  def test_trace_strictly_increases_from_0_to_1(self):
    self.assertEqual(0, self.tracker.trace)

    for i in range(100):
      prev_trace = self.tracker.trace
      self.agent.statistics = {'a': i, 'b': 2 * i}
      self.tracker.step(None, None, self.agent, None)
      self.assertGreater(self.tracker.trace, prev_trace)
      self.assertLess(self.tracker.trace, 1)

    self.assertAlmostEqual(1, self.tracker.trace, places=4)


if __name__ == '__main__':
  absltest.main()
