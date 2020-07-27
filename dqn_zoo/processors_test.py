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
"""Tests for processors."""

# pylint: disable=g-bad-import-order

import collections
import hashlib
import typing

import dm_env
from dm_env import test_utils
import numpy as np

from dqn_zoo import gym_atari
from dqn_zoo import processors
from absl.testing import absltest
from absl.testing import parameterized

F = dm_env.StepType.FIRST
M = dm_env.StepType.MID
L = dm_env.StepType.LAST


class FixedPaddedBufferTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.buffer = processors.FixedPaddedBuffer(length=4, initial_index=-1)

  def test_basic(self):
    self.assertEqual([None, None, None, 1], self.buffer(1))
    self.assertEqual([2, None, None, None], self.buffer(2))
    self.assertEqual([2, 3, None, None], self.buffer(3))
    self.assertEqual([2, 3, 4, None], self.buffer(4))
    self.assertEqual([2, 3, 4, 5], self.buffer(5))
    self.assertEqual([6, None, None, None], self.buffer(6))

  def test_reset(self):
    for i in range(3):
      self.buffer(i)
    self.buffer.reset()
    self.assertEqual([None, None, None, -1], self.buffer(-1))


def make_timesteps_from_step_types(step_types):

  def make_timestep(step_type):
    return dm_env.TimeStep(
        step_type=step_type,
        observation=0,
        reward=None if step_type == dm_env.StepType.FIRST else 0,
        discount=None if step_type == dm_env.StepType.FIRST else 0)

  return [make_timestep(st) for st in step_types]


class TimestepBufferConditionTest(parameterized.TestCase):

  def test_basic(self):
    step_types_and_expected = [
        ([None, None, None, F], True),
        ([M, None, None, None], False),
        ([M, M, None, None], False),
        ([M, M, M, None], False),
        ([M, M, M, M], True),
        ([M, None, None, None], False),
        ([M, M, None, None], False),
        ([M, M, M, None], False),
        ([M, M, M, M], True),
        ([M, None, None, None], False),
        ([M, L, None, None], True),
    ]

    processor = processors.TimestepBufferCondition(period=4)
    for step_types, expected in step_types_and_expected:
      timesteps = make_timesteps_from_step_types(step_types)
      self.assertEqual(expected, processor(timesteps))

  @parameterized.parameters(
      # Can't have F & L occur in same sequence.
      [
          [[None, None, F], [F, None, None], [F, M, None], [F, M, L]],
      ],
      # Can't have two F's occur in same sequence.
      [
          [[None, None, F], [F, None, None], [F, M, None], [F, M, F]],
      ],
  )
  def test_errors_with_multiple_first_or_last(self, step_types_list):
    processor = processors.TimestepBufferCondition(period=3)
    for step_types in step_types_list[:-1]:
      timesteps = make_timesteps_from_step_types(step_types)
      _ = processor(timesteps)
    last_timesteps = make_timesteps_from_step_types(step_types_list[-1])
    with self.assertRaisesRegex(RuntimeError, 'at most one FIRST or LAST'):
      _ = processor(last_timesteps)

  def test_errors_if_no_reset_after_last(self):
    step_types_list = [
        ([None, None, None, F]),
        ([M, None, None, None]),
        ([M, L, None, None]),
        ([M, L, F, None]),
    ]
    processor = processors.TimestepBufferCondition(period=3)
    for step_types in step_types_list[:-1]:
      timesteps = make_timesteps_from_step_types(step_types)
      _ = processor(timesteps)
    last_timesteps = make_timesteps_from_step_types(step_types_list[-1])
    with self.assertRaisesRegex(RuntimeError, 'Should have reset'):
      _ = processor(last_timesteps)


def make_timestep_from_step_type_string(step_type_str, observation):
  if step_type_str == 'f':
    return dm_env.restart(observation=observation)
  elif step_type_str == 'm':
    return dm_env.transition(reward=0, observation=observation)
  elif step_type_str == 'l':
    return dm_env.termination(reward=0, observation=observation)
  else:
    raise ValueError('Unknown step type string %s.' % step_type_str)


class ActionRepeatsTest(absltest.TestCase):
  """Tests action repeats can be implemented."""

  def setUp(self):
    super().setUp()
    num_repeats = 4
    self.processor = processors.Sequential(
        processors.FixedPaddedBuffer(length=num_repeats, initial_index=-1),
        processors.ConditionallySubsample(
            processors.TimestepBufferCondition(period=num_repeats)),
        processors.Maybe(
            processors.Sequential(
                processors.none_to_zero_pad,
                processors.named_tuple_sequence_stack,
            ),),
    )

  def test_basic(self):
    sequence = [
        ('f', '0001'),
        ('m', None),
        ('m', None),
        ('m', None),
        ('m', '2345'),
        ('m', None),
        ('l', '6700'),
        ('f', '0008'),
        ('m', None),
    ]

    prev_timestep = None
    for i, (step_type_str, expected_obs_str) in enumerate(sequence, start=1):
      if prev_timestep and prev_timestep.last():
        self.processor.reset()
      timestep = make_timestep_from_step_type_string(step_type_str, i)
      processed = self.processor(timestep)
      if processed is None:
        obs_str = None
      else:
        obs_str = ''.join(str(o) for o in processed.observation)
      self.assertEqual(expected_obs_str, obs_str)
      prev_timestep = timestep

  def test_exception_raised_if_reset_not_called_between_last_and_first(self):
    sequence = list('fmmmmmlfm')
    with self.assertRaisesRegex(RuntimeError, 'reset'):
      for i, step_type_str in enumerate(sequence, start=1):
        timestep = make_timestep_from_step_type_string(step_type_str, i)
        self.processor(timestep)


Pair = collections.namedtuple('Pair', ['a', 'b'])


class ApplyToNamedTupleFieldTest(absltest.TestCase):

  def test_basic_usage(self):
    pair = Pair(a=1, b=2)
    processor = processors.ApplyToNamedTupleField('a', lambda x: x + 10)
    self.assertEqual(Pair(a=11, b=2), processor(pair))


class ZeroDiscountOnLifeLossTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_loss', 'fmmmmm', 'n11111', '333333', 'fmmmmm', 'n11111'),
      ('one_loss', 'fmmmmm', 'n11111', '333222', 'fmmmmm', 'n11011'),
      ('two_losses', 'fmmmmm', 'n11111', '332211', 'fmmmmm', 'n10101'),
      ('episode_end', 'fmmlfm', '1110n1', '333355', 'fmmlfm', '1110n1'),
      ('episode_end_same', 'fmmlfm', '1110n1', '333333', 'fmmlfm', '1110n1'),
  )
  def test_basic(
      self,
      input_step_types_str,
      input_discounts_str,
      input_lives_str,
      expected_step_types_str,
      expected_discounts_str,
  ):
    processor = processors.ZeroDiscountOnLifeLoss()
    step_type_map = {'f': F, 'm': M, 'l': L}

    for timestep_part in zip(
        input_step_types_str,
        input_discounts_str,
        input_lives_str,
        expected_step_types_str,
        expected_discounts_str,
    ):
      (
          input_step_type,
          input_discount,
          input_lives,
          expected_step_type,
          expected_discount,
      ) = timestep_part
      input_timestep = dm_env.TimeStep(
          step_type=step_type_map[input_step_type],
          reward=8,
          discount=None if input_discount == 'n' else float(input_discount),
          observation=(9, int(input_lives)),
      )
      output_timestep = processor(input_timestep)

      self.assertEqual(step_type_map[expected_step_type],
                       output_timestep.step_type)
      self.assertEqual(
          None if expected_discount == 'n' else float(expected_discount),
          output_timestep.discount)


class ReduceStepTypeTest(parameterized.TestCase):

  @parameterized.parameters(
      ([0, 0, 0, F], F),
      ([M, M, M, L], L),
      ([M, M, L, 0], L),
      ([M, M, M, M], M),
  )
  def test_valid_cases(self, step_types, expected_step_type):
    self.assertEqual(
        expected_step_type,
        processors.reduce_step_type(np.asarray(step_types), debug=True))

  @parameterized.parameters(
      ([0, 0, 0, M],),
      ([0, 0, 0, L],),
      ([M, 0, 0, 0],),
      ([L, 0, 0, M],),
      ([M, L, F, M],),
  )
  def test_invalid_cases(self, step_types):
    with self.assertRaises(ValueError):
      processors.reduce_step_type(np.asarray(step_types), debug=True)


class AggregateRewardsTest(parameterized.TestCase):

  @parameterized.parameters(
      ([None], None),
      ([0, None], None),
      ([0, 0, None], None),
      ([0, 0, 0, None], None),
      ([0], 0),
      ([1], 1),
      ([1, 2], 3),
      ([1, 2, 3], 6),
      ([1, -2, 3], 2),
  )
  def test_basic(self, rewards, expected):
    self.assertEqual(expected,
                     processors.aggregate_rewards(rewards, debug=True))

  @parameterized.parameters(
      ([1., None],),
      ([0., 1., None],),
      ([1., 0., None],),
  )
  def test_error_raised_in_debug_with_none_and_no_zero_padding(self, rewards):
    with self.assertRaisesRegex(ValueError, 'None.*FIRST'):
      processors.aggregate_rewards(rewards, debug=True)


class AggregateDiscountsTest(parameterized.TestCase):

  @parameterized.parameters(
      ([None], None),
      ([0, None], None),
      ([0, 0, None], None),
      ([0, 0, 0, None], None),
      ([0], 0),
      ([1], 1),
      ([1, 1], 1),
      ([1, 1, 1], 1),
      ([1, 1, 0], 0),
  )
  def test_basic(self, discounts, expected):
    self.assertEqual(expected,
                     processors.aggregate_discounts(discounts, debug=True))

  @parameterized.parameters(
      ([1., None],),
      ([0., 1., None],),
      ([1., 0., None],),
  )
  def test_error_raised_in_debug_with_none_and_no_zero_padding(self, discounts):
    with self.assertRaisesRegex(ValueError, 'None.*FIRST'):
      processors.aggregate_discounts(discounts, debug=True)


class ClipRewardTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, 0),
      (1, 1),
      (-1, -1),
      (-2.5, -2),
      (2.5, 2),
      (None, None),
  )
  def test_basic(self, reward, expected):
    self.assertEqual(expected, processors.clip_reward(2)(reward))


class AgentWithPreprocessing:
  """Agent that does standard Atari preprocessing.

  Returns actions `0, 1, ..., num_actions, 0, 1, ...` unless the processor
  returns `None` in which case the agent repeats the previous action.
  """

  def __init__(self, num_actions):
    self._processor = processors.atari()
    self._num_actions = num_actions
    self._action = None

  def reset(self):
    processors.reset(self._processor)
    self._action = None

  def step(self, timestep):
    processed_timestep = self._processor(timestep)

    # Repeat previous action if processed timestep is None.
    if processed_timestep is None:
      return self._action

    # This block would normally contain the forward pass through the network.
    if self._action is None:
      self._action = 0
    else:
      self._action = (self._action + 1) % self._num_actions

    return self._action


class AtariTest(absltest.TestCase):

  def test_can_use_in_an_agent(self):
    """Example of using Atari processor on the agent side."""
    env = gym_atari.GymAtari('pong', seed=1)
    action_spec = env.action_spec()
    agent = AgentWithPreprocessing(num_actions=action_spec.num_values)

    agent.reset()
    timestep = env.reset()

    actions = []
    for _ in range(20):
      action = agent.step(timestep)
      timestep = env.step(action)
      assert not timestep.last()
      actions.append(action)

    self.assertEqual(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], actions)

  def test_default_on_fixed_input(self):
    """End-to-end test on fixed input.

    This is to test (mainly observation) processors do not change due to updates
    in underlying library functions.
    """
    # Create environment just for the observation spec.
    env = gym_atari.GymAtari('pong', seed=1)
    rgb_spec, unused_lives_spec = env.observation_spec()
    random_state = np.random.RandomState(seed=1)

    # Generate timesteps with fixed data to feed into processor.
    def generate_rgb_obs():
      return random_state.randint(
          0, 256, size=rgb_spec.shape, dtype=rgb_spec.dtype)

    step_types = [F, M, M, M, M]
    rewards = [None, 0.5, 0.2, 0, 0.1]
    discounts = [None, 0.9, 0.9, 0.9, 0.9]
    rgb_obs = [generate_rgb_obs() for _ in range(len(step_types))]
    lives_obs = [3, 3, 3, 3, 3]

    timesteps = []
    for i in range(len(step_types)):
      timesteps.append(
          dm_env.TimeStep(
              step_type=step_types[i],
              reward=rewards[i],
              discount=discounts[i],
              observation=(rgb_obs[i], lives_obs[i])))

    def hash_array(array):
      return hashlib.sha256(array).hexdigest()

    # Make sure generated observation data is fixed and the random number
    # generator has not changed from underneath us, causing the test to fail.
    hash_rgb_obs = [hash_array(obs) for obs in rgb_obs]
    expected_hashes = [
        '250557b2184381fc2ec541fc313127050098fce825a6e98a728c2993874db300',
        'db8054ca287971a0e1264bfbc5642233085f1b27efbca9082a29f5be8a24c552',
        '7016e737a257fcdb77e5f23daf96d94f9820bd7361766ca7b1401ec90984ef71',
        '356dfcf0c6eaa4e2b5e80f4611375c0131435cc22e6a413b573818d7d084e9b2',
        '73078bedd438422ad1c3dda6718aa1b54f6163f571d2c26ed714c515a6372159',
    ]
    assert hash_rgb_obs == expected_hashes, (hash_rgb_obs, expected_hashes)

    # Run timesteps through processor.
    processor = processors.atari()
    for timestep in timesteps:
      processed = processor(timestep)

    # Assert the returned timestep is not None, and tell pytype.
    self.assertIsNotNone(processed)
    processed = typing.cast(dm_env.TimeStep, processed)

    # Compare with expected timestep, just the hash for the observation.
    self.assertEqual(dm_env.StepType.MID, processed.step_type)
    self.assertAlmostEqual(0.5 + 0.2 + 0. + 0.1, processed.reward)
    self.assertAlmostEqual(0.9**4 * 0.99, processed.discount)
    processed_obs_hash = hash_array(processed.observation.flatten())

    # Note the algorithm used for image resizing can have a noticeable impact on
    # learning performance. This test helps ensure changes to image processing
    # are intentional.
    self.assertEqual(
        '0d158a8f45aa09aa6fad0354d2eb1fc0e3f57add88e772f3b71f54819d8200aa',
        processed_obs_hash)


class AtariEnvironmentWrapperTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('grayscaling', True, 'grayscale', (84, 84, 4)),
      ('no_grayscaling', False, 'RGB', (84, 84, 3, 4)),
  )
  def test_atari_grayscaling_observation_spec(self, grayscaling, expected_name,
                                              expected_shape):
    env = gym_atari.GymAtari('pong', seed=1)
    env = processors.AtariEnvironmentWrapper(
        environment=env, grayscaling=grayscaling)
    spec = env.observation_spec()
    self.assertEqual(spec.shape, expected_shape)
    self.assertEqual(spec.name, expected_name)

  @parameterized.named_parameters(
      ('grayscaling', True, (84, 84, 4)),
      ('no_grayscaling', False, (84, 84, 3, 4)),
  )
  def test_atari_grayscaling_observation_shape(self, grayscaling,
                                               expected_shape):
    env = gym_atari.GymAtari('pong', seed=1)
    env = processors.AtariEnvironmentWrapper(
        environment=env, grayscaling=grayscaling)

    timestep = env.reset()
    for _ in range(10):
      assert not timestep.step_type.last()
      self.assertEqual(timestep.observation.shape, expected_shape)
      timestep = env.step(0)


class AtariEnvironmentWrapperInterfaceTest(test_utils.EnvironmentTestMixin,
                                           absltest.TestCase):

  def make_object_under_test(self):
    env = gym_atari.GymAtari('pong', seed=1)
    return processors.AtariEnvironmentWrapper(environment=env)


if __name__ == '__main__':
  absltest.main()
