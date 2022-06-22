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
"""Tests for transition replay components."""

# pylint: disable=g-bad-import-order

import collections
import copy
import itertools
from typing import Any, Mapping, Sequence, Text

import chex
import dm_env
import numpy as np

from dqn_zoo import replay as replay_lib
from absl.testing import absltest
from absl.testing import parameterized

Pair = collections.namedtuple('Pair', ['a', 'b'])
ReplayStructure = collections.namedtuple('ReplayStructure', ['value'])

## Tests for regular (unprioritized) transition replay.


class UniformDistributionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    random_state = np.random.RandomState(seed=1)
    self.dist = replay_lib.UniformDistribution(random_state)

  def test_add_and_sample_one(self):
    self.dist.add([2])
    self.assertEqual(2, self.dist.sample(1))
    self.assertTrue(*self.dist.check_valid())

  def test_adding_existing_id_raises_an_error(self):
    self.dist.add([2, 5])
    with self.assertRaisesRegex(IndexError, 'Cannot add ID'):
      self.dist.add([6, 5])

  def test_remove(self):
    self.dist.add([2, 5, 7])
    self.dist.remove([5])
    self.assertNotIn(5, self.dist.sample(100))
    self.assertTrue(*self.dist.check_valid())

  def test_remove_final_id(self):
    # IDs are removed by swapping with the final one in a list and popping, so
    # check this works when the ID being removed is the last one.
    self.dist.add([2, 5, 7])
    self.dist.remove([7])
    self.assertEqual(2, self.dist.size)
    self.assertNotIn(7, self.dist.sample(100))
    self.assertTrue(*self.dist.check_valid())

  def test_removing_nonexistent_id_raises_an_error(self):
    self.dist.add([2, 5])
    with self.assertRaisesRegex(IndexError, 'Cannot remove ID'):
      self.dist.remove([7])

  def test_size(self):
    self.dist.add([2, 5, 3])
    self.assertEqual(3, self.dist.size)
    self.dist.remove([5])
    self.assertEqual(2, self.dist.size)

  def test_get_state_and_set_state(self):
    self.dist.add([2, 5, 3, 8])
    self.dist.remove([5])
    self.assertTrue(*self.dist.check_valid())

    state = copy.deepcopy(self.dist.get_state())

    new_dist = replay_lib.UniformDistribution(np.random.RandomState(seed=1))
    new_dist.set_state(state)

    self.assertTrue(new_dist.check_valid())
    self.assertEqual(new_dist.size, self.dist.size)

    new_state = new_dist.get_state()
    chex.assert_trees_all_close(state, new_state)


class TransitionReplayTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.capacity = 10
    self.replay = replay_lib.TransitionReplay(
        capacity=self.capacity,
        structure=Pair(a=None, b=None),
        random_state=np.random.RandomState(1))
    self.items = [
        Pair(a=1, b=2),
        Pair(a=11, b=22),
        Pair(a=111, b=222),
        Pair(a=1111, b=2222),
    ]
    for item in self.items:
      self.replay.add(item)

  def test_size(self):
    self.assertLen(self.items, self.replay.size)

  def test_capacity(self):
    self.assertEqual(self.capacity, self.replay.capacity)

  def test_sample(self):
    num_samples = 2
    samples = self.replay.sample(num_samples)
    self.assertEqual((num_samples,), samples.a.shape)

  def test_invariants(self):
    capacity = 10
    num_samples = 3
    replay = replay_lib.TransitionReplay(
        capacity=capacity,
        structure=Pair(a=None, b=None),
        random_state=np.random.RandomState(1))

    for i in range(31):
      replay.add(Pair(a=i, b=2 * i))
      self.assertLessEqual(replay.size, capacity)
      if i > 2:
        self.assertEqual(replay.sample(num_samples).a.shape, (num_samples,))

      ids = list(replay.ids())
      self.assertLen(ids, min(i + 1, capacity))
      self.assertEqual(ids, sorted(ids))
      self.assertEqual(ids[0] + len(ids) - 1, ids[-1])

  def test_get_state_and_set_state(self):
    replay = replay_lib.TransitionReplay(
        capacity=self.capacity,
        structure=Pair(a=None, b=None),
        random_state=np.random.RandomState(1))

    for i in range(100):
      replay.add(Pair(a=i, b=i))
    replay.sample(10)
    self.assertTrue(*replay.check_valid())

    state = copy.deepcopy(replay.get_state())

    new_replay = replay_lib.TransitionReplay(
        capacity=self.capacity,
        structure=Pair(a=None, b=None),
        random_state=np.random.RandomState(1))
    new_replay.set_state(state)

    self.assertTrue(*new_replay.check_valid())
    self.assertEqual(new_replay.size, replay.size)

    new_state = new_replay.get_state()
    chex.assert_trees_all_close(state, new_state)


class NStepTransitionAccumulatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.n = 3
    self.accumulator = replay_lib.NStepTransitionAccumulator(self.n)

    self.num_timesteps = 10
    self.step_types = ([dm_env.StepType.FIRST] + [dm_env.StepType.MID] *
                       (self.num_timesteps - 1))
    self.states = list(range(self.num_timesteps))
    self.discounts = np.linspace(0.9, 1., self.num_timesteps, endpoint=False)
    self.rewards = np.linspace(-5, 5, self.num_timesteps, endpoint=False)
    self.actions = [i % 4 for i in range(self.num_timesteps)]

    self.accumulator_output = []
    for i in range(self.num_timesteps):
      timestep = dm_env.TimeStep(
          step_type=self.step_types[i],
          observation=self.states[i],
          discount=self.discounts[i],
          reward=self.rewards[i])
      self.accumulator_output.append(
          list(self.accumulator.step(timestep, self.actions[i])))

  def test_no_transitions_returned_for_first_n_steps(self):
    self.assertEqual([[]] * self.n, self.accumulator_output[:self.n])
    self.assertNotEqual([], self.accumulator_output[self.n])

  def test_states_accumulation(self):
    actual_s_tm1 = [
        tr.s_tm1 for tr in itertools.chain(*self.accumulator_output)
    ]
    actual_s_t = [tr.s_t for tr in itertools.chain(*self.accumulator_output)]

    expected_s_tm1 = self.states[:-self.n]
    expected_s_t = self.states[self.n:]

    np.testing.assert_array_equal(expected_s_tm1, actual_s_tm1)
    np.testing.assert_array_equal(expected_s_t, actual_s_t)

  def test_discount_accumulation(self):
    expected = []
    for i in range(len(self.discounts) - self.n):
      # Offset by 1 since first discount is unused.
      expected.append(np.prod(self.discounts[i + 1:i + 1 + self.n]))

    actual = [tr.discount_t for tr in itertools.chain(*self.accumulator_output)]

    np.testing.assert_allclose(expected, actual)

  def test_reward_accumulation(self):
    expected = []
    for i in range(len(self.discounts) - self.n):
      # Offset by 1 since first discount and reward is unused.
      discounts = np.concatenate([[1.],
                                  self.discounts[i + 1:i + 1 + self.n - 1]])
      cumulative_discounts = np.cumprod(discounts)
      rewards = self.rewards[i + 1:i + 1 + self.n]
      expected.append(np.sum(cumulative_discounts * rewards))

    actual = [tr.r_t for tr in itertools.chain(*self.accumulator_output)]

    np.testing.assert_allclose(expected, actual)

  def test_correct_action_is_stored_in_transition(self):
    expected = self.actions[:-self.n]
    actual = [tr.a_tm1 for tr in itertools.chain(*self.accumulator_output)]
    np.testing.assert_array_equal(expected, actual)

  def test_reset(self):
    self.accumulator.reset()
    transitions = self.accumulator.step(
        timestep_t=dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            observation=-1,
            discount=1.,
            reward=3),
        a_t=1)
    self.assertEqual([], list(transitions))

  def test_consistent_with_transition_accumulator(self):
    n_step_transition_accumulator = replay_lib.NStepTransitionAccumulator(1)
    transition_accumulator = replay_lib.TransitionAccumulator()

    # Add the same timesteps to both accumulators.
    for i in range(self.num_timesteps):
      timestep = dm_env.TimeStep(
          step_type=self.step_types[i],
          observation=self.states[i],
          discount=self.discounts[i],
          reward=self.rewards[i])
      transitions = list(transition_accumulator.step(timestep, self.actions[i]))
      n_step_transitions = list(
          n_step_transition_accumulator.step(timestep, self.actions[i]))
      self.assertEqual(transitions, n_step_transitions)

  def test_all_remaining_transitions_yielded_when_timestep_is_last(self):
    f = dm_env.StepType.FIRST
    m = dm_env.StepType.MID
    l = dm_env.StepType.LAST

    n = 3
    accumulator = replay_lib.NStepTransitionAccumulator(n)
    step_types = [f, m, m, m, m, m, l, f, m, m, m, m, f, m]
    num_timesteps = len(step_types)
    states = list(range(num_timesteps))
    discounts = np.arange(1, num_timesteps + 1) / num_timesteps
    rewards = np.ones(num_timesteps)
    actions = list(range(num_timesteps, 0, -1))

    accumulator_output = []
    for i in range(num_timesteps):
      timestep = dm_env.TimeStep(
          step_type=step_types[i],
          observation=states[i],
          discount=discounts[i],
          reward=rewards[i])
      accumulator_output.append(list(accumulator.step(timestep, actions[i])))

    output_lengths = [len(output) for output in accumulator_output]
    expected_output_lengths = [0, 0, 0, 1, 1, 1, n, 0, 0, 0, 1, 1, 0, 0]
    self.assertEqual(expected_output_lengths, output_lengths)

    # Get transitions yielded at the end of an episode.
    end_index = expected_output_lengths.index(n)
    episode_end_transitions = accumulator_output[end_index]

    # Check the start and end states are correct.
    # Normal n-step transition
    self.assertEqual(episode_end_transitions[0].s_t, end_index)
    self.assertEqual(episode_end_transitions[0].s_tm1, end_index - n)
    # (n - 1)-step transition.
    self.assertEqual(episode_end_transitions[1].s_t, end_index)
    self.assertEqual(episode_end_transitions[1].s_tm1, end_index - (n - 1))
    # (n - 2)-step transition.
    self.assertEqual(episode_end_transitions[2].s_t, end_index)
    self.assertEqual(episode_end_transitions[2].s_tm1, end_index - (n - 2))

  def test_transitions_returned_if_episode_length_less_than_n(self):
    f = dm_env.StepType.FIRST
    m = dm_env.StepType.MID
    l = dm_env.StepType.LAST

    n = 4
    accumulator = replay_lib.NStepTransitionAccumulator(n)
    step_types = [f, m, l]
    num_timesteps = len(step_types)
    states = list(range(num_timesteps))
    discounts = np.ones(num_timesteps)
    rewards = np.ones(num_timesteps)
    actions = np.ones(num_timesteps)

    accumulator_output = []
    for i in range(num_timesteps):
      timestep = dm_env.TimeStep(
          step_type=step_types[i],
          observation=states[i],
          discount=discounts[i],
          reward=rewards[i])
      accumulator_output.append(list(accumulator.step(timestep, actions[i])))
    output_lengths = [len(output) for output in accumulator_output]
    output_states = [
        [(tr.s_tm1, tr.s_t) for tr in output] for output in accumulator_output
    ]

    # Expect a 1-step transition and a 2-step transition after LAST timestep.
    expected_output_lengths = [0, 0, 2]
    expected_output_states = [[], [], [(0, 2), (1, 2)]]
    self.assertEqual(expected_output_lengths, output_lengths)
    self.assertEqual(expected_output_states, output_states)


## Tests for prioritized replay.


def add(replay, values, priorities=None):
  priorities = [0.] * len(values) if priorities is None else priorities
  for v, p in zip(values, priorities):
    replay.add(ReplayStructure(value=v), priority=p)


def get(replay, ids):
  return [x.value for x in replay.get(ids)]


def make_replay(
    capacity=10,
    structure=None,
    priority_exponent=0.8,
    importance_sampling_exponent=lambda t: 0.6,
    uniform_sample_probability=0.1,
    normalize_weights=True,
    seed=None,
):
  return replay_lib.PrioritizedTransitionReplay(
      capacity=capacity,
      structure=structure or ReplayStructure(value=None),
      priority_exponent=priority_exponent,
      importance_sampling_exponent=importance_sampling_exponent,
      uniform_sample_probability=uniform_sample_probability,
      normalize_weights=normalize_weights,
      random_state=np.random.RandomState(seed),
  )


def sample_replay_bin_count(replay, num, sample_size):
  all_values = []
  for _ in range(num):
    samples, unused_indices, unused_weights = replay.sample(size=sample_size)
    all_values.append(samples.value)
  return np.bincount(np.array(all_values).flatten())


def make_distribution(min_capacity=0,
                      max_capacity=None,
                      priority_exponent=0.8,
                      uniform_sample_probability=0.1,
                      seed=1):
  random_state = np.random.RandomState(seed)
  return replay_lib.PrioritizedDistribution(
      max_capacity=max_capacity,
      min_capacity=min_capacity,
      priority_exponent=priority_exponent,
      uniform_sample_probability=uniform_sample_probability,
      random_state=random_state)


def sample_distribution_bin_count(distribution, num, sample_size):
  all_values = []
  for _ in range(num):
    indices, unused_probabilities = distribution.sample(size=sample_size)
    all_values.extend(indices)
  counter = collections.Counter(all_values)
  sampled_indices = sorted(counter.keys())
  counts = np.array([counter[idx] for idx in sampled_indices])
  return sampled_indices, counts


class PrioritizedDistributionTest(absltest.TestCase):

  def test_adding_ids_that_already_exist_raises_an_exception(self):
    dist = make_distribution()
    dist.add_priorities(ids=[1, 2], priorities=[0.1, 0.2])
    with self.assertRaisesRegex(IndexError, 'already exists'):
      dist.add_priorities(ids=[2], priorities=[0.2])

  def test_size_is_correct_with_max_capacity(self):
    capacity = 7
    dist = make_distribution(min_capacity=capacity, max_capacity=capacity)
    self.assertTrue(*dist.check_valid())
    self.assertEqual(0, dist.size)

    # Setting 3 new priorities counts.
    dist.add_priorities(ids=[2, 3, 5], priorities=[0.2, 0.3, 0.5])
    self.assertTrue(*dist.check_valid())
    self.assertEqual(3, dist.size)

    # Overwriting existing priority does not increase size.
    dist.update_priorities(ids=[3], priorities=[1.])
    self.assertTrue(*dist.check_valid())
    self.assertEqual(3, dist.size)

    # Add priority for a new index increases size, even if priority is 0.
    dist.add_priorities(ids=[4], priorities=[0.])
    self.assertTrue(*dist.check_valid())
    self.assertEqual(4, dist.size)

    # Add priorities up to capacity.
    dist.add_priorities(ids=[7, 1, 0], priorities=[2., 3., 4.])
    self.assertTrue(*dist.check_valid())
    self.assertEqual(7, dist.size)

    # Add priorities beyond capacity.
    with self.assertRaisesRegex(ValueError, 'max capacity would be exceeded'):
      dist.add_priorities(ids=[9, 8], priorities=[2., 3.])
    self.assertTrue(*dist.check_valid())
    self.assertEqual(7, dist.size)
    self.assertSequenceEqual(list(dist.ids()), [2, 3, 5, 4, 7, 1, 0])

  def test_capacity_does_not_grow_unless_needed(self):
    dist = make_distribution(min_capacity=4, max_capacity=None)

    dist.add_priorities(ids=[0], priorities=[0.])
    self.assertEqual(1, dist.size)
    self.assertEqual(4, dist.capacity)

    dist.add_priorities(ids=[1, 2], priorities=[1., 2.])
    self.assertEqual(3, dist.size)
    self.assertEqual(4, dist.capacity)

    dist.add_priorities(ids=[3], priorities=[3.])
    self.assertEqual(4, dist.size)
    self.assertEqual(4, dist.capacity)

    dist.add_priorities(ids=[4, 5, 6, 7, 8], priorities=[4., 5., 6., 7., 8.])
    self.assertEqual(9, dist.size)
    self.assertEqual(9, dist.capacity)

  def test_capacity_grows_automatically_as_ids_are_added(self):
    dist = make_distribution(min_capacity=0, max_capacity=None)
    self.assertEqual(0, dist.capacity)
    self.assertTrue(*dist.check_valid())

    id_iter = itertools.count()

    def add_priorities(num_ids):
      ids = list(itertools.islice(id_iter, num_ids))
      priorities = [float(i) for i in ids]
      dist.add_priorities(ids, priorities)

    # Add zero IDs, a bit contrived, but should not raise an error.
    add_priorities(num_ids=0)
    self.assertEqual(0, dist.capacity)
    self.assertTrue(*dist.check_valid())

    # Add one ID.
    add_priorities(num_ids=1)
    self.assertEqual(0 + 1, dist.capacity)
    self.assertTrue(*dist.check_valid())

    # Add another ID.
    add_priorities(num_ids=1)
    self.assertEqual(1 + 1, dist.capacity)
    self.assertTrue(*dist.check_valid())

    # Add another ID, capacity grows to 4 as capacity doubles.
    add_priorities(num_ids=1)
    self.assertEqual(2 * 2, dist.capacity)
    self.assertEqual(3, dist.size)
    self.assertTrue(*dist.check_valid())

    # Add 6 IDs, capacity grows to 4 + 5 as doubling is not sufficient.
    add_priorities(num_ids=6)
    self.assertEqual(4 + 5, dist.capacity)
    self.assertEqual(9, dist.size)
    self.assertTrue(*dist.check_valid())

  def test_min_capacity_is_respected(self):
    min_capacity = 3
    dist = make_distribution(min_capacity=min_capacity)

    self.assertEqual(min_capacity, dist.capacity)
    self.assertEqual(0, dist.size)

  def test_capacity_correct_after_increasing_capacity(self):
    min_capacity = 4
    dist = make_distribution(min_capacity=min_capacity)
    self.assertEqual(min_capacity, dist.capacity)
    self.assertEqual(0, dist.size)

    new_capacity = 7
    dist.ensure_capacity(new_capacity)

    self.assertEqual(new_capacity, dist.capacity)
    self.assertEqual(0, dist.size)
    self.assertTrue(*dist.check_valid())

  def test_increasing_capacity_beyond_max_capacity_raises_an_error(self):
    dist = make_distribution(max_capacity=7)

    dist.ensure_capacity(3)

    with self.assertRaisesRegex(ValueError, 'cannot exceed max_capacity'):
      dist.ensure_capacity(9)

  def test_setting_capacity_lower_than_current_capacity_does_nothing(self):
    min_capacity = 4
    dist = make_distribution(min_capacity=min_capacity)
    self.assertEqual(min_capacity, dist.capacity)

    dist.ensure_capacity(2)

    # Capacity should remain the same.
    self.assertEqual(min_capacity, dist.capacity)
    self.assertTrue(*dist.check_valid())

  def test_changing_capacity_does_not_alter_existing_ids(self):
    ids = [2, 3, 5]
    priorities = [0.2, 0.3, 0.5]
    dist = make_distribution(min_capacity=len(ids), priority_exponent=1.)
    dist.add_priorities(ids, priorities)

    dist.ensure_capacity(10)

    same_ids = sorted(dist.ids())
    same_priorities = list(dist.get_exponentiated_priorities(same_ids))
    self.assertSequenceEqual(ids, same_ids)
    self.assertSequenceEqual(priorities, same_priorities)
    self.assertTrue(*dist.check_valid())

  def test_new_size_greater_than_2x_capacity_with_max_capacity_set(self):
    ids = [2, 3, 5]
    priorities = [0.2, 0.3, 0.5]
    dist = make_distribution(min_capacity=len(ids), max_capacity=100)
    dist.add_priorities(ids, priorities)

    # Add more IDs and priorities, beyond 2x current capacity.
    dist.add_priorities([6, 7, 8, 9], [0.6, 0.7, 0.8, 0.9])
    self.assertTrue(*dist.check_valid())

    self.assertEqual(7, dist.capacity)

  def test_get_state_and_set_state(self):
    ids = [2, 3, 5]
    priorities = [0.2, 0.3, 0.5]

    dist = make_distribution(priority_exponent=1., min_capacity=9)
    dist.add_priorities(ids, priorities)
    self.assertTrue(*dist.check_valid())

    state = copy.deepcopy(dist.get_state())

    new_dist = make_distribution(priority_exponent=1., min_capacity=9)
    new_dist.set_state(state)

    self.assertTrue(*new_dist.check_valid())
    self.assertEqual(new_dist.size, dist.size)

    new_state = new_dist.get_state()
    chex.assert_trees_all_close(state, new_state)

    new_priorities = new_dist.get_exponentiated_priorities(ids)

    # Equal to raw priorities since priority exponent is 1.
    np.testing.assert_array_equal(new_priorities, priorities)

  def test_priorities_can_be_set_again(self):
    priority_exponent = 0.45
    dist = make_distribution(priority_exponent=priority_exponent)
    ids = [2, 3, 5]
    priorities = [0.2, 0.3, 0.5]
    dist.add_priorities(ids, priorities)
    orig_priorities = dist.get_exponentiated_priorities(ids)
    dist.update_priorities([3], [1.3])
    new_priorities = dist.get_exponentiated_priorities(ids)
    self.assertNotAlmostEqual(orig_priorities[1], new_priorities[1])
    self.assertAlmostEqual(1.3**priority_exponent, new_priorities[1])

  def test_add_priorities_with_empty_args(self):
    dist = make_distribution()
    dist.add_priorities([], [])
    self.assertTrue(*dist.check_valid())

  def test_priorities_can_be_removed(self):
    dist = make_distribution()
    ids = [2, 3, 5, 7]
    priorities = [0.2, 0.3, 0.5, 0.7]
    dist.add_priorities(ids, priorities)
    self.assertEqual(4, dist.size)
    dist.remove_priorities([3, 7])
    self.assertEqual(2, dist.size)
    self.assertTrue(*dist.check_valid())

  def test_remove_priorities_with_empty_args(self):
    dist = make_distribution()
    ids = [2, 3, 5]
    priorities = [0.2, 0.3, 0.5]
    dist.add_priorities(ids, priorities)
    dist.remove_priorities([])
    self.assertTrue(*dist.check_valid())

  def test_update_priorities_with_empty_args(self):
    dist = make_distribution()
    ids = [2, 3, 5]
    priorities = [0.2, 0.3, 0.5]
    dist.add_priorities(ids, priorities)
    dist.update_priorities([], [])
    self.assertTrue(*dist.check_valid())

  def test_all_zero_priorities_results_in_uniform_sampling(self):
    dist = make_distribution()
    dist.add_priorities(ids=[2, 3, 5], priorities=[0., 0., 0.])
    for _ in range(10):
      unused_ids, probabilities = dist.sample(size=2)
      np.testing.assert_allclose(probabilities, 1. / 3.)

  def test_sample_distribution(self):
    priority_exponent = 0.8
    uniform_sample_probability = 0.1
    dist = make_distribution(
        priority_exponent=priority_exponent,
        uniform_sample_probability=uniform_sample_probability)

    # Set priorities, update one.
    ids = [2, 3, 5]
    initial_priorities = np.array([1., 0., 3.], dtype=np.float64)
    dist.add_priorities(ids=ids, priorities=initial_priorities)
    final_priorities = np.array([1., 4., 3.], dtype=np.float64)
    dist.update_priorities([ids[1]], [final_priorities[1]])

    usp = uniform_sample_probability
    expected_raw_sample_dist = final_priorities**priority_exponent
    expected_raw_sample_dist /= expected_raw_sample_dist.sum()
    expected_sample_dist = ((1 - usp) * expected_raw_sample_dist +
                            usp * 1 / len(final_priorities))

    sampled_ids, counts = sample_distribution_bin_count(
        dist, num=50_000, sample_size=2)
    self.assertEqual(ids, sampled_ids)
    sample_dist = counts / counts.sum()

    np.testing.assert_allclose(sample_dist, expected_sample_dist, rtol=1e-2)

  def test_update_priorities_raises_an_error_if_id_not_present(self):
    dist = make_distribution()
    dist.add_priorities(ids=[2, 3, 5], priorities=[1., 2., 3.])

    with self.assertRaises(IndexError):
      dist.update_priorities(ids=[4], priorities=[0.])

    with self.assertRaises(IndexError):
      dist.update_priorities(ids=[1], priorities=[1.])

    with self.assertRaises(IndexError):
      dist.update_priorities(ids=[0], priorities=[2.])

  def test_priorities_can_be_updated(self):
    dist = make_distribution(priority_exponent=1.)
    ids = [2, 3, 5]
    dist.add_priorities(ids=ids, priorities=[1., 2., 3.])
    dist.update_priorities(ids=[3, 5], priorities=[4., 6.])
    updated_priorities = dist.get_exponentiated_priorities(ids)
    np.testing.assert_allclose(updated_priorities, [1, 4, 6])

  def test_removing_ids_results_in_only_remaining_ids_being_sampled(self):
    usp = 0.1
    dist = make_distribution(
        priority_exponent=1., uniform_sample_probability=usp)
    ids = [2, 3, 5]
    dist.add_priorities(ids=ids, priorities=[1., 100., 9.])

    dist.remove_priorities([3])
    ids, probabilities = dist.sample(1000)
    unique_probs = list(sorted(set(probabilities)))

    self.assertSetEqual({2, 5}, set(ids))
    self.assertLen(unique_probs, 2)
    self.assertAlmostEqual(usp * 0.5 + (1 - usp) * 0.1, unique_probs[0])
    self.assertAlmostEqual(usp * 0.5 + (1 - usp) * 0.9, unique_probs[1])

  def test_removing_last_id_is_valid(self):
    # This tests internal logic for ID removal where the final ID is "special".
    dist = make_distribution()
    ids = [2, 3, 5]
    dist.add_priorities(ids=ids, priorities=[1., 100., 9.])

    dist.remove_priorities([5])
    self.assertTrue(*dist.check_valid())


class PrioritizedTransitionReplayTest(absltest.TestCase):

  def test_empty_replay_properties_are_correct(self):
    capacity = 7
    replay = make_replay(capacity=capacity)
    self.assertEqual(0, replay.size)
    self.assertEqual(capacity, replay.capacity)

  def test_add(self):
    replay = make_replay()
    add(replay, [10])
    add(replay, [11])
    self.assertListEqual([10], get(replay, [0]))
    self.assertListEqual([11], get(replay, [1]))

  def test_only_latest_elements_are_kept(self):
    capacity = 5
    replay = make_replay(capacity=capacity)
    num_items = 7
    assert num_items > capacity
    add(replay, list(range(num_items)))
    self.assertTrue(*replay.check_valid())

    values = get(replay, list(range(num_items - capacity, num_items)))
    expected_values = list(range(num_items - capacity, num_items))
    self.assertCountEqual(expected_values, values)

  def test_sample_returns_batch(self):
    replay = make_replay()
    add(replay, [1, 2, 3])
    sample_size = 2
    samples, unused_ids, unused_weights = replay.sample(sample_size)
    chex.assert_shape(samples.value, (sample_size,))

  def test_get_state_and_set_state(self):
    orig_replay = make_replay(priority_exponent=1.)
    add(orig_replay, values=[11, 22, 33], priorities=[1., 2., 3.])
    state = orig_replay.get_state()
    new_replay = make_replay()
    new_replay.set_state(state)
    self.assertEqual(orig_replay.size, new_replay.size)

  def test_sample_distribution(self):
    priority_exponent = 0.8
    uniform_sample_probability = 0.1

    replay = make_replay(
        capacity=3,
        priority_exponent=priority_exponent,
        uniform_sample_probability=uniform_sample_probability,
        seed=1)

    priorities = np.array([3., 2., 0., 4.], dtype=np.float64)
    add(replay,
        values=list(range(len(priorities))),
        priorities=list(priorities))

    pe, usp = priority_exponent, uniform_sample_probability
    expected_dist = np.zeros_like(priorities)
    active_priorities = priorities[-replay.size:].copy()
    exp_priorities = active_priorities**pe
    prioritized_probs = exp_priorities / exp_priorities.sum()
    uniform_prob = 1. / replay.size
    expected_dist[-replay.size:] = ((1. - usp) * prioritized_probs +
                                    usp * uniform_prob)

    counts = sample_replay_bin_count(replay, num=10000, sample_size=2)
    dist = counts / counts.sum()
    np.testing.assert_allclose(dist, expected_dist, rtol=0.1)


class SumTreeTest(parameterized.TestCase):

  def test_can_create_empty(self):
    sum_tree = replay_lib.SumTree()
    self.assertTrue(*sum_tree.check_valid())
    self.assertEqual(0, sum_tree.size)
    self.assertTrue(np.isnan(sum_tree.root()))

  def test_size_is_correct(self):
    sum_tree = replay_lib.SumTree()
    self.assertEqual(0, sum_tree.size)
    size = 3
    sum_tree.resize(size)
    self.assertEqual(size, sum_tree.size)

  def test_resize_returns_zero_values_initially(self):
    sum_tree = replay_lib.SumTree()
    size = 3
    sum_tree.resize(size)
    for i in range(size):
      self.assertEqual(0, sum_tree.get([i]))

  def test_resize_to_1(self):
    sum_tree = replay_lib.SumTree()
    sum_tree.resize(1)
    self.assertTrue(*sum_tree.check_valid())
    self.assertEqual(0, sum_tree.root())

  def test_resize_to_0(self):
    sum_tree = replay_lib.SumTree()
    sum_tree.resize(0)
    self.assertTrue(*sum_tree.check_valid())
    self.assertTrue(np.isnan(sum_tree.root()))

  def test_set_all(self):
    sum_tree = replay_lib.SumTree()
    values = [4., 5., 3.]
    sum_tree.set_all(values)
    self.assertLen(values, sum_tree.size)
    for i in range(len(values)):
      np.testing.assert_array_almost_equal([values[i]], sum_tree.get([i]))
    self.assertTrue(*sum_tree.check_valid())

  def test_capacity_greater_or_equal_to_size_and_power_of_2(self):
    sum_tree = replay_lib.SumTree()
    sum_tree.set_all([4., 5., 3., 2.])
    self.assertEqual(4, sum_tree.capacity)

    sum_tree = replay_lib.SumTree()
    sum_tree.set_all([4., 5., 3., 2., 9])
    self.assertEqual(8, sum_tree.capacity)

  def test_values_returns_values(self):
    sum_tree = replay_lib.SumTree()
    values = [4., 5., 3.]
    sum_tree.set_all(values)
    np.testing.assert_allclose(values, sum_tree.values)

  def test_resize_preserves_values_and_zeros_the_rest_when_growing(self):
    sum_tree = replay_lib.SumTree()
    values = [4., 5., 3.]
    sum_tree.set_all(values)
    new_size = len(values) + 5
    sum_tree.resize(new_size)
    for i in range(len(values)):
      np.testing.assert_array_almost_equal([values[i]], sum_tree.get([i]))
    for i in range(len(values), new_size):
      np.testing.assert_array_almost_equal([0.], sum_tree.get([i]))
    self.assertTrue(*sum_tree.check_valid())

  def test_resizes_preserves_values_when_shrinking(self):
    sum_tree = replay_lib.SumTree()
    values = [4., 5., 3., 8., 2.]
    sum_tree.set_all(values)
    new_size = len(values) - 2
    sum_tree.resize(new_size)
    for i in range(new_size):
      np.testing.assert_array_almost_equal([values[i]], sum_tree.get([i]))
    self.assertTrue(*sum_tree.check_valid())

  def test_resizing_to_size_between_current_size_and_capacity(self):
    sum_tree = replay_lib.SumTree()
    values = [4., 5., 3., 8., 2.]
    sum_tree.set_all(values)
    new_size = 7
    assert sum_tree.size < new_size < sum_tree.capacity
    sum_tree.resize(new_size)
    np.testing.assert_allclose(values + [0., 0.], sum_tree.values)
    self.assertTrue(*sum_tree.check_valid())

  def test_exception_raised_when_index_out_of_bounds_in_get(self):
    sum_tree = replay_lib.SumTree()
    size = 3
    sum_tree.resize(size)
    for i in [-1, size]:
      with self.assertRaises(IndexError):
        sum_tree.get([i])

  def test_get_with_multiple_indexes(self):
    sum_tree = replay_lib.SumTree()
    values = [4., 5., 3., 9.]
    sum_tree.set_all(values)
    selected = sum_tree.get([1, 3])
    np.testing.assert_allclose([values[1], values[3]], selected)

  def test_set_single(self):
    sum_tree = replay_lib.SumTree()
    values = [4, 5, 3, 9]
    sum_tree.set_all(values)
    sum_tree.set([2], [99])
    np.testing.assert_allclose([4, 5, 99, 9], sum_tree.values)

  def test_set_multiple(self):
    sum_tree = replay_lib.SumTree()
    values = [4, 5, 3, 9]
    sum_tree.set_all(values)
    sum_tree.set([2, 0], [99, 88])
    np.testing.assert_allclose([88, 5, 99, 9], sum_tree.values)

  @parameterized.parameters(
      (0, 0.),
      (0, 3. - 0.1),
      (1, 3.),
      (1, 4. - 0.1),
      (2, 4.),
      (2, 6. - 0.1),
      (3, 6.),
      (3, 11. - 0.1),
  )
  def test_query_typical(self, expected_index, target):
    sum_tree = replay_lib.SumTree()
    values = [3., 1., 2., 5.]
    sum_tree.set_all(values)
    self.assertEqual([expected_index], sum_tree.query([target]))

  def test_query_raises_exception_if_target_out_of_range(self):
    sum_tree = replay_lib.SumTree()
    values = [3., 1., 2., 5.]
    sum_tree.set_all(values)

    with self.assertRaises(ValueError):
      sum_tree.query([-1.])

    with self.assertRaises(ValueError):
      sum_tree.query([sum(values)])

    with self.assertRaises(ValueError):
      sum_tree.query([sum(values) + 1.])

    with self.assertRaises(ValueError):
      sum_tree.query([sum_tree.root()])

  def test_query_multiple(self):
    sum_tree = replay_lib.SumTree()
    values = [3., 1., 2., 5.]
    sum_tree.set_all(values)
    np.testing.assert_array_equal([0, 1, 2], sum_tree.query([2.9, 3., 4]))

  @parameterized.parameters(
      (t,)
      for t in [0, 0.1, 0.9, 1, 1.1, 3.9, 4, 4.1, 5.9, 6, 6.1, 8.9, 8.999999])
  def test_query_never_returns_an_index_with_zero_index(self, target):
    sum_tree = replay_lib.SumTree()
    values = np.array([0, 1, 0, 0, 3, 0, 2, 0, 3, 0], dtype=np.float64)
    zero_indices = (values == 0).nonzero()[0]
    sum_tree.set_all(values)
    self.assertNotIn(sum_tree.query([target])[0], zero_indices)

  def test_root_returns_sum(self):
    sum_tree = replay_lib.SumTree()
    values = [3., 1., 2., 5.]
    sum_tree.set_all(values)
    self.assertAlmostEqual(sum(values), sum_tree.root())

  def test_set_cannot_add_negative_nan_or_inf_values(self):
    sum_tree = replay_lib.SumTree()
    sum_tree.set_all([0, 1, 2])

    with self.assertRaises(ValueError):
      sum_tree.set([1], [-1])

    with self.assertRaises(ValueError):
      sum_tree.set([1], [np.nan])

    with self.assertRaises(ValueError):
      sum_tree.set([1], [np.inf])

  def test_set_all_cannot_add_negative_nan_or_inf_values(self):

    with self.assertRaises(ValueError):
      replay_lib.SumTree().set_all([1, -1])

    with self.assertRaises(ValueError):
      replay_lib.SumTree().set_all([1, np.nan])

    with self.assertRaises(ValueError):
      replay_lib.SumTree().set_all([1, np.inf])

  def test_set_updates_total_sum(self):
    sum_tree = replay_lib.SumTree()
    values = [4, 5, 3, 9]
    sum_tree.set_all(values)
    sum_tree.set([1], [2])
    self.assertAlmostEqual(sum(values) - 5 + 2, sum_tree.root())
    self.assertTrue(*sum_tree.check_valid())

  def test_get_state_and_set_state(self):
    sum_tree = replay_lib.SumTree()
    values = [4, 5, 3, 9]
    sum_tree.set_all(values)
    self.assertTrue(*sum_tree.check_valid())

    state = copy.deepcopy(sum_tree.get_state())

    new_sum_tree = replay_lib.SumTree()
    new_sum_tree.set_state(state)

    self.assertTrue(*new_sum_tree.check_valid())
    self.assertEqual(sum_tree.size, new_sum_tree.size)

    new_state = new_sum_tree.get_state()
    chex.assert_trees_all_close(state, new_state)

    np.testing.assert_allclose(new_sum_tree.values, sum_tree.values)
    self.assertEqual(sum_tree.capacity, new_sum_tree.capacity)


class NaiveSumTree:
  """Same as `SumTree`, but less efficient with a simpler implementation."""

  def __init__(self):
    self._values = np.zeros(0, np.float64)

  def resize(self, size: int) -> None:
    """Resizes tree, truncating or expanding with zeros as needed."""
    # Usually there shouldn't be references to self._values, but to prevent
    # certain Coverage test errors, we pass refcheck=False.
    self._values.resize(size, refcheck=False)

  def get(self, indices: Sequence[int]) -> Sequence[float]:
    """Gets values corresponding to given indices."""
    indices = np.asarray(indices)
    if not ((0 <= indices) & (indices < self.size)).all():
      raise IndexError('Index out range expect 0 <= index < %s' % self.size)
    return self._values[indices]

  def set(self, indices: Sequence[int], values: Sequence[float]):
    """Sets values at the given indices."""
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0.).any():
      raise ValueError('value must be finite positive numbers.')
    self._values[indices] = values

  def set_all(self, values: Sequence[float]) -> None:
    """Sets many values all at once, also setting size of the sum tree."""
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0.).any():
      raise ValueError('Values must be finite positive numbers.')
    self._values = values

  def query(self, targets: Sequence[float]) -> Sequence[int]:
    """Finds smallest index such that `target <` cumulative sum up to index."""
    return [self._query_single(t) for t in targets]

  def _query_single(self, target: float) -> int:
    """Queries a single target, see `SumTree.query` for more documentation."""
    if not 0. <= target < self.root():
      raise ValueError('Require 0 <= target < total sum.')
    acc_sum = 0.
    for i in range(self.size):
      acc_sum += self.values[i]
      if target < acc_sum:
        return i
    raise RuntimeError('Should not reach here as target < total sum.')

  def root(self) -> float:
    return self._values.sum() if self.size > 0 else np.nan

  @property
  def values(self) -> np.ndarray:
    return self._values

  @property
  def size(self) -> int:
    return len(self._values)

  @property
  def capacity(self) -> int:
    return len(self._values)

  def get_state(self) -> Mapping[Text, Any]:
    return {
        'values': self._values,
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    self._values = state['values']


def random_operations(sum_tree, seed):
  random_state = np.random.RandomState(seed)
  random_values = lambda m: np.abs(random_state.standard_cauchy(m))
  random_indices = lambda m: random_state.randint(sum_tree.size, size=m)
  random_targets = lambda m: random_state.uniform(0, sum_tree.root(), size=m)
  random_size = lambda: random_state.randint(10, 40)

  for _ in range(20):
    sum_tree.resize(random_size())
    yield
    sum_tree.set(random_indices(3), random_values(3))
    yield
    yield sum_tree.query(random_targets(4))
    sum_tree.set_all(random_values(random_size()))
    sum_tree.set(random_indices(4), random_values(4))
    yield sum_tree.query(random_targets(3))
    sum_tree.set_state(sum_tree.get_state())
    yield


class NaiveSumTreeEquivalenceTest(parameterized.TestCase):
  """Tests equivalence with naive implementation.

  Has better coverage but harder to debug failures.
  """

  @parameterized.parameters([(i,) for i in list(range(10))])
  def test_with_random_data(self, seed):
    actual_sum_tree = replay_lib.SumTree()
    naive_sum_tree = NaiveSumTree()

    # Randomly perform operations, periodically stopping to compare.
    operation_iterator = zip(
        random_operations(actual_sum_tree, seed),
        random_operations(naive_sum_tree, seed))
    for actual_value, naive_value in operation_iterator:
      if actual_value is not None and naive_value is not None:
        np.testing.assert_allclose(actual_value, naive_value)
      self.assertTrue(*actual_sum_tree.check_valid())
      self.assertAlmostEqual(naive_sum_tree.root(), actual_sum_tree.root())
      np.testing.assert_allclose(naive_sum_tree.values, actual_sum_tree.values)


if __name__ == '__main__':
  absltest.main()
