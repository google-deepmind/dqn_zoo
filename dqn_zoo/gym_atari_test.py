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
"""Unit tests for gym_atari."""

# pylint: disable=g-bad-import-order

import itertools

import dm_env.test_utils
import numpy as np

from dqn_zoo import gym_atari
from dqn_zoo import test_utils
from absl.testing import absltest
from absl.testing import parameterized


def make_gym_atari_env(game, seed=1):
  env = gym_atari.GymAtari(game, seed=seed)
  env = gym_atari.RandomNoopsEnvironmentWrapper(
      env, min_noop_steps=1, max_noop_steps=30, seed=seed)
  return env


class GymAtariEnvironmentTest(dm_env.test_utils.EnvironmentTestMixin,
                              absltest.TestCase):
  """Sanity checks compliance with `dm_env.Environment` interface contract."""

  def make_object_under_test(self):
    return make_gym_atari_env('pong', seed=1)


def random_action(random_state, action_spec):
  return random_state.randint(
      action_spec.minimum, action_spec.maximum + 1, dtype=action_spec.dtype)


def timestep_generator(seed):
  random_state = np.random.RandomState(seed=seed)
  env = make_gym_atari_env('pong', seed=seed)
  yield env.reset()
  while True:
    action = random_action(random_state, env.action_spec())
    yield env.step(action)


class GymAtariTest(absltest.TestCase):
  """Sanity checks expected properties of Gym Atari."""

  def test_seed_range(self):
    for seed in (0, 1, 2**32 - 1):
      gym_atari.GymAtari('pong', seed=seed)

  def test_can_call_close(self):
    gym_atari.GymAtari('pong', seed=1).close()

  def test_determinism(self):
    num_timesteps = 1000

    # Check using same seed produces the same timesteps.
    same_timesteps = zip(timestep_generator(seed=1), timestep_generator(seed=1))
    for ts1, ts2 in itertools.islice(same_timesteps, num_timesteps):
      self.assertEqual(ts1.step_type, ts2.step_type)
      self.assertEqual(ts1.reward, ts2.reward)
      self.assertEqual(ts1.discount, ts2.discount)
      self.assertEqual(ts1.observation[1], ts2.observation[1])
      np.testing.assert_array_equal(ts1.observation[0], ts2.observation[0])

    # Sanity check different seeds produces different timesteps.
    diff_timesteps = zip(timestep_generator(seed=2), timestep_generator(seed=3))
    same = True
    for ts1, ts2 in itertools.islice(diff_timesteps, num_timesteps):
      same = same and (ts1.step_type == ts2.step_type)
      same = same and (ts1.reward == ts2.reward)
      same = same and (ts1.discount == ts2.discount)
      same = same and (ts1.observation[1] == ts2.observation[1])
      same = same and np.array_equal(ts1.observation[0], ts2.observation[0])
    assert not same


class RandomNoopsEnvironmentWrapperTest(parameterized.TestCase):

  @parameterized.parameters((0, 5), (2, 5), (0, 0), (3, 3))
  def test_basic(self, min_noop_steps, max_noop_steps):
    noop_action = 3
    tape = []
    environment = test_utils.DummyEnvironment(tape, episode_length=10)
    wrapped_environment = gym_atari.RandomNoopsEnvironmentWrapper(
        environment,
        min_noop_steps=min_noop_steps,
        max_noop_steps=max_noop_steps,
        noop_action=noop_action,
        seed=42)

    # Make sure noops are applied appropriate number of times (in min/max range
    # and not always the same number), with correct action.
    num_noop_steps = set()
    for i in range(20):
      # Switch between different ways of starting a new episode.
      if i % 4 == 0:
        tape.clear()
        wrapped_environment.reset()
        num_steps = len(tape)
        expected_tape = (['Environment reset'] +
                         ['Environment step (%s)' % noop_action] *
                         (num_steps - 1))
      else:
        timestep = wrapped_environment.reset()
        while not timestep.last():
          timestep = wrapped_environment.step(0)
        tape.clear()
        wrapped_environment.step(noop_action)
        num_steps = len(tape)
        expected_tape = (['Environment step (%s)' % noop_action] * num_steps)

      self.assertEqual(expected_tape, tape)
      # +1 because of the extra initial reset() / step().
      self.assertBetween(num_steps, min_noop_steps + 1, max_noop_steps + 1)
      num_noop_steps.add(num_steps)

      # Do some regular steps & check pass-through of actions.
      wrapped_environment.step(6)
      wrapped_environment.step(7)
      self.assertLen(tape, num_steps + 2)
      self.assertEqual(['Environment step (6)', 'Environment step (7)'],
                       tape[-2:])

    # Check it's not always the same number of random noop steps.
    if max_noop_steps > min_noop_steps:
      self.assertGreater(len(num_noop_steps), 1)

  def test_specs(self):
    environment = test_utils.DummyEnvironment([], episode_length=10)
    wrapped_environment = gym_atari.RandomNoopsEnvironmentWrapper(
        environment, max_noop_steps=5)
    self.assertEqual(environment.observation_spec(),
                     wrapped_environment.observation_spec())
    self.assertEqual(environment.action_spec(),
                     wrapped_environment.action_spec())
    self.assertEqual(environment.reward_spec(),
                     wrapped_environment.reward_spec())
    self.assertEqual(environment.discount_spec(),
                     wrapped_environment.discount_spec())

  def test_determinism(self):

    def num_noops_sequence(seed, num_episodes):
      tape = []
      environment = test_utils.DummyEnvironment(tape, episode_length=10)
      wrapped_environment = gym_atari.RandomNoopsEnvironmentWrapper(
          environment, max_noop_steps=8, seed=seed)
      seq = []
      for _ in range(num_episodes):
        tape.clear()
        wrapped_environment.reset()
        seq.append(len(tape))
      return seq

    sequence_1 = num_noops_sequence(seed=123, num_episodes=20)
    sequence_2 = num_noops_sequence(seed=123, num_episodes=20)
    sequence_3 = num_noops_sequence(seed=124, num_episodes=20)
    self.assertEqual(sequence_1, sequence_2)
    self.assertNotEqual(sequence_1, sequence_3)

  def test_episode_end_during_noop_steps(self):
    environment = test_utils.DummyEnvironment([], episode_length=5)
    wrapped_environment = gym_atari.RandomNoopsEnvironmentWrapper(
        environment, min_noop_steps=10, max_noop_steps=20)
    with self.assertRaisesRegex(RuntimeError, 'Episode ended'):
      wrapped_environment.reset()


if __name__ == '__main__':
  absltest.main()
