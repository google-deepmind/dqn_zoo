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
"""Replay components for DQN-type agents."""

# pylint: disable=g-bad-import-order

import collections
import typing
from typing import Any, Callable, Generic, Iterable, List, Mapping, Optional, Sequence, Text, Tuple, TypeVar

import dm_env
import numpy as np
import snappy

from dqn_zoo import parts

CompressedArray = Tuple[bytes, Tuple, np.dtype]

# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


class Transition(typing.NamedTuple):
  s_tm1: Optional[np.ndarray]
  a_tm1: Optional[parts.Action]
  r_t: Optional[float]
  discount_t: Optional[float]
  s_t: Optional[np.ndarray]


class TransitionReplay(Generic[ReplayStructure]):
  """Uniform replay, with circular buffer storage for flat named tuples."""

  def __init__(self,
               capacity: int,
               structure: ReplayStructure,
               random_state: np.random.RandomState,
               encoder: Optional[Callable[[ReplayStructure], Any]] = None,
               decoder: Optional[Callable[[Any], ReplayStructure]] = None):
    self._capacity = capacity
    self._structure = structure
    self._random_state = random_state
    self._encoder = encoder or (lambda s: s)
    self._decoder = decoder or (lambda s: s)

    self._storage = [None] * capacity
    self._num_added = 0

  def add(self, item: ReplayStructure) -> None:
    """Adds single item to replay."""
    self._storage[self._num_added % self._capacity] = self._encoder(item)
    self._num_added += 1

  def get(self, indices: Sequence[int]) -> List[ReplayStructure]:
    """Retrieves items by indices."""
    return [self._decoder(self._storage[i]) for i in indices]

  def sample(self, size: int) -> ReplayStructure:
    """Samples batch of items from replay uniformly, with replacement."""
    indices = self._random_state.choice(self.size, size=size, replace=True)
    samples = self.get(indices)
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    return type(self._structure)(*stacked)  # pytype: disable=not-callable

  @property
  def size(self) -> int:
    """Number of items currently contained in replay."""
    return min(self._num_added, self._capacity)

  @property
  def capacity(self) -> int:
    """Total capacity of replay (max number of items stored at any one time)."""
    return self._capacity

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves replay state as a dictionary (e.g. for serialization)."""
    return {
        'storage': self._storage,
        'num_added': self._num_added,
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets replay state from a (potentially de-serialized) dictionary."""
    self._storage = state['storage']
    self._num_added = state['num_added']


def _power(base, exponent) -> np.ndarray:
  """Same as usual power except `0 ** 0` is zero."""
  # By default 0 ** 0 is 1 but we never want indices with priority zero to be
  # sampled, even if the priority exponent is zero.
  base = np.asarray(base)
  return np.where(base == 0., 0., base**exponent)


def importance_sampling_weights(
    probabilities: np.ndarray,
    uniform_probability: float,
    exponent: float,
    normalize: bool,
) -> np.ndarray:
  """Calculates importance sampling weights from given sampling probabilities.

  Args:
    probabilities: Array of sampling probabilities for a subset of items. Since
      this is a subset the probabilites will typically not sum to `1`.
    uniform_probability: Probability of sampling an item if uniformly sampling.
    exponent: Scalar that controls the amount of importance sampling correction
      in the weights. Where `1` corrects fully and `0` is no correction
      (resulting weights are all `1`).
    normalize: Whether to scale all weights so that the maximum weight is `1`.
      Can be enabled for stability since weights will only scale down.

  Returns:
    Importance sampling weights that can be used to scale the loss. These have
    the same shape as `probabilities`.
  """
  if not 0. <= exponent <= 1.:
    raise ValueError('Require 0 <= exponent <= 1.')
  if not 0. <= uniform_probability <= 1.:
    raise ValueError('Expected 0 <= uniform_probability <= 1.')

  weights = (uniform_probability / probabilities)**exponent
  if normalize:
    weights /= np.max(weights)
  if not np.isfinite(weights).all():
    raise ValueError('Weights are not finite: %s.' % weights)
  return weights


class SumTree:
  """A binary tree where non-leaf nodes are the sum of child nodes.

  Leaf nodes contain non-negative floats and are set externally. Non-leaf nodes
  are the sum of their children. This data structure allows O(log n) updates and
  O(log n) queries of which index corresponds to a given sum. The main use
  case is sampling from a multinomial distribution with many probabilities
  which are updated a few at a time.
  """

  def __init__(self):
    """Initializes an empty `SumTree`."""
    # When there are n values, the storage array will have size 2 * n. The first
    # n elements are non-leaf nodes (ignoring the very first element), with
    # index 1 corresponding to the root node. The next n elements are leaf nodes
    # that contain values. A non-leaf node with index i has children at
    # locations 2 * i, 2 * i + 1.
    self._size = 0
    self._storage = np.zeros(0, dtype=np.float64)
    self._first_leaf = 0

  def resize(self, size: int) -> None:
    """Resizes tree, truncating or expanding with zeros as needed."""
    self._initialize(size, values=None)

  def get(self, indices: Sequence[int]) -> np.ndarray:
    """Gets values corresponding to given indices."""
    indices = np.asarray(indices)
    if not ((0 <= indices) & (indices < self.size)).all():
      raise IndexError('index out of range, expect 0 <= index < %s' % self.size)
    return self.values[indices]

  def set(self, indices: Sequence[int], values: Sequence[float]) -> None:
    """Sets values at the given indices."""
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0.).any():
      raise ValueError('value must be finite and positive.')
    self.values[indices] = values
    storage = self._storage
    for idx in np.asarray(indices) + self._first_leaf:
      parent = idx // 2
      while parent > 0:
        # At this point the subtree with root parent is consistent.
        storage[parent] = storage[2 * parent] + storage[2 * parent + 1]
        parent //= 2

  def set_all(self, values: Sequence[float]) -> None:
    """Sets many values all at once, also setting size of the sum tree."""
    values = np.asarray(values)
    if not np.isfinite(values).all() or (values < 0.).any():
      raise ValueError('Values must be finite positive numbers.')
    self._initialize(len(values), values)

  def query(self, targets: Sequence[float]) -> Sequence[int]:
    """Finds smallest indices where `target <` cumulative value sum up to index.

    Args:
      targets: The target sums.

    Returns:
      For each target, the smallest index such that target is strictly less than
      the cumulative sum of values up to and including that index.

    Raises:
      ValueError: if `target >` sum of all values or `target < 0` for any
        of the given targets.
    """
    return [self._query_single(t) for t in targets]

  def root(self) -> float:
    """Returns sum of values."""
    return self._storage[1] if self.size > 0 else np.nan

  @property
  def values(self) -> np.ndarray:
    """View of array containing all (leaf) values in the sum tree."""
    return self._storage[self._first_leaf:self._first_leaf + self.size]

  @property
  def size(self) -> int:
    """Number of (leaf) values in the sum tree."""
    return self._size

  @property
  def capacity(self) -> int:
    """Current sum tree capacity (exceeding it will trigger resizing)."""
    return self._first_leaf

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves sum tree state as a dictionary (e.g. for serialization)."""
    return {
        'size': self._size,
        'storage': self._storage,
        'first_leaf': self._first_leaf,
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets sum tree state from a (potentially de-serialized) dictionary."""
    self._size = state['size']
    self._storage = state['storage']
    self._first_leaf = state['first_leaf']

  def check_valid(self) -> None:
    """Checks internal consistency."""
    self._assert(len(self._storage) == 2 * self._first_leaf)
    self._assert(0 <= self.size <= self.capacity)
    self._assert(len(self.values) == self.size)

    storage = self._storage
    for i in range(1, self._first_leaf):
      self._assert(storage[i] == storage[2 * i] + storage[2 * i + 1])

  def _assert(self, condition, message='SumTree is internally inconsistent.'):
    """Raises `RuntimeError` with given message if condition is not met."""
    if not condition:
      raise RuntimeError(message)

  def _initialize(self, size: int, values: Optional[Sequence[float]]) -> None:
    """Resizes storage and sets new values if supplied."""
    assert size >= 0
    assert values is None or len(values) == size

    if size < self.size:  # Keep storage and values, zero out extra values.
      if values is None:
        new_values = self.values[:size]  # Truncate existing values.
      else:
        new_values = values
      self._size = size
      self._set_values(new_values)
      # self._first_leaf remains the same.
    elif size <= self.capacity:  # Reuse same storage, but size increases.
      self._size = size
      if values is not None:
        self._set_values(values)
      # self._first_leaf remains the same.
      # New activated leaf nodes are already zero and sum nodes already correct.
    else:  # Allocate new storage.
      new_capacity = 1
      while new_capacity < size:
        new_capacity *= 2
      new_storage = np.empty((2 * new_capacity,), dtype=np.float64)
      if values is None:
        new_values = self.values
      else:
        new_values = values
      self._storage = new_storage
      self._first_leaf = new_capacity
      self._size = size
      self._set_values(new_values)

  def _set_values(self, values: Sequence[float]) -> None:
    """Sets values assuming storage has enough capacity and update sums."""
    # Note every part of the storage is set here.
    assert len(values) <= self.capacity
    storage = self._storage
    storage[self._first_leaf:self._first_leaf + len(values)] = values
    storage[self._first_leaf + len(values):] = 0
    for i in range(self._first_leaf - 1, 0, -1):
      storage[i] = storage[2 * i] + storage[2 * i + 1]
    storage[0] = 0.  # Unused.

  def _query_single(self, target: float) -> int:
    """Queries a single target, see query for more detailed documentation."""
    if not 0. <= target < self.root():
      raise ValueError('Require 0 <= target < total sum.')

    storage = self._storage
    idx = 1  # Root node.
    while idx < self._first_leaf:
      # At this point we always have target < storage[idx].
      assert target < storage[idx]
      left_idx = 2 * idx
      right_idx = left_idx + 1
      left_sum = storage[left_idx]
      if target < left_sum:
        idx = left_idx
      else:
        idx = right_idx
        target -= left_sum

    assert idx < 2 * self.capacity
    return idx - self._first_leaf


class PrioritizedDistribution:
  """Distribution for weighted sampling."""

  def __init__(
      self,
      capacity: int,
      priority_exponent: float,
      uniform_sample_probability: float,
      random_state: np.random.RandomState,
  ):
    if priority_exponent < 0.:
      raise ValueError('Require priority_exponent >= 0.')
    self._priority_exponent = priority_exponent
    if not 0. <= uniform_sample_probability <= 1.:
      raise ValueError('Require 0 <= uniform_sample_probability <= 1.')
    self._uniform_sample_probability = uniform_sample_probability
    self._sum_tree = SumTree()
    self._sum_tree.resize(capacity)
    self._random_state = random_state
    self._active_indices = []  # For uniform sampling.
    self._active_indices_mask = np.zeros(capacity, dtype=np.bool)

  def set_priorities(self, indices: Sequence[int],
                     priorities: Sequence[float]) -> None:
    """Sets priorities for indices, whether or not all indices already exist."""
    for idx in indices:
      if not self._active_indices_mask[idx]:
        self._active_indices.append(idx)
        self._active_indices_mask[idx] = True
    self._sum_tree.set(indices, _power(priorities, self._priority_exponent))

  def update_priorities(self, indices: Sequence[int],
                        priorities: Sequence[float]) -> None:
    """Updates priorities for existing indices."""
    for idx in indices:
      if not self._active_indices_mask[idx]:
        raise IndexError('Index %s cannot be updated as it is inactive.' % idx)
    self._sum_tree.set(indices, _power(priorities, self._priority_exponent))

  def sample(self, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Returns sample of indices with corresponding probabilities."""
    uniform_indices = [
        self._active_indices[i] for i in self._random_state.randint(
            len(self._active_indices), size=size)
    ]

    if self._sum_tree.root() == 0.:
      prioritized_indices = uniform_indices
    else:
      targets = self._random_state.uniform(size=size) * self._sum_tree.root()
      prioritized_indices = np.asarray(self._sum_tree.query(targets))

    usp = self._uniform_sample_probability
    indices = np.where(
        self._random_state.uniform(size=size) < usp, uniform_indices,
        prioritized_indices)

    uniform_prob = np.asarray(1. / self.size)  # np.asarray is for pytype.
    priorities = self._sum_tree.get(indices)

    if self._sum_tree.root() == 0.:
      prioritized_probs = np.full_like(priorities, fill_value=uniform_prob)
    else:
      prioritized_probs = priorities / self._sum_tree.root()

    sample_probs = (1. - usp) * prioritized_probs + usp * uniform_prob
    return indices, sample_probs

  def get_exponentiated_priorities(self,
                                   indices: Sequence[int]) -> Sequence[float]:
    """Returns priority ** priority_exponent for the given indices."""
    return self._sum_tree.get(indices)

  @property
  def size(self) -> int:
    """Number of elements currently tracked by distribution."""
    return len(self._active_indices)

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves distribution state as a dictionary (e.g. for serialization)."""
    return {
        'sum_tree': self._sum_tree.get_state(),
        'active_indices': self._active_indices,
        'active_indices_mask': self._active_indices_mask,
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets distribution state from a (potentially de-serialized) dictionary."""
    self._sum_tree.set_state(state['sum_tree'])
    self._active_indices = state['active_indices']
    self._active_indices_mask = state['active_indices_mask']


class PrioritizedTransitionReplay(Generic[ReplayStructure]):
  """Prioritized replay, with circular buffer storage for flat named tuples.

  This is the proportional variant as described in
  http://arxiv.org/abs/1511.05952.
  """

  def __init__(
      self,
      capacity: int,
      structure: ReplayStructure,
      priority_exponent: float,
      importance_sampling_exponent: Callable[[int], float],
      uniform_sample_probability: float,
      normalize_weights: bool,
      random_state: np.random.RandomState,
      encoder: Optional[Callable[[ReplayStructure], Any]] = None,
      decoder: Optional[Callable[[Any], ReplayStructure]] = None,
  ):
    self._capacity = capacity
    self._structure = structure
    self._random_state = random_state
    self._encoder = encoder or (lambda s: s)
    self._decoder = decoder or (lambda s: s)
    self._distribution = PrioritizedDistribution(
        capacity=capacity,
        priority_exponent=priority_exponent,
        uniform_sample_probability=uniform_sample_probability,
        random_state=random_state)
    self._importance_sampling_exponent = importance_sampling_exponent
    self._normalize_weights = normalize_weights
    self._storage = [None] * capacity
    self._t = 0

  def add(self, item: ReplayStructure, priority: float) -> None:
    """Adds a single item with a given priority to the replay buffer."""
    index = self._t % self._capacity
    self._distribution.set_priorities([index], [priority])
    self._storage[index] = self._encoder(item)
    self._t += 1

  def get(self, indices: Sequence[int]) -> List[ReplayStructure]:
    """Retrieves transitions by indices."""
    return [self._decoder(self._storage[i]) for i in indices]

  def sample(
      self,
      size: int,
  ) -> Tuple[ReplayStructure, np.ndarray, np.ndarray]:
    """Samples a batch of transitions."""
    indices, probabilities = self._distribution.sample(size)
    weights = importance_sampling_weights(
        probabilities,
        uniform_probability=1. / self.size,
        exponent=self.importance_sampling_exponent,
        normalize=self._normalize_weights)
    samples = self.get(indices)
    transposed = zip(*samples)
    stacked = [np.stack(xs, axis=0) for xs in transposed]
    # pytype: disable=not-callable
    return type(self._structure)(*stacked), indices, weights
    # pytype: enable=not-callable

  def update_priorities(self, indices: Sequence[int],
                        priorities: Sequence[float]) -> None:
    """Updates indices with given priorities."""
    priorities = np.asarray(priorities)
    self._distribution.update_priorities(indices, priorities)

  @property
  def size(self) -> int:
    """Number of elements currently contained in replay."""
    return min(self._t, self._capacity)

  @property
  def capacity(self) -> int:
    """Total capacity of replay (maximum number of items that can be stored)."""
    return self._capacity

  @property
  def importance_sampling_exponent(self):
    """Importance sampling exponent at current step."""
    return self._importance_sampling_exponent(self._t)

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves replay state as a dictionary (e.g. for serialization)."""
    return {
        't': self._t,
        'storage': self._storage,
        'distribution': self._distribution.get_state(),
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets replay state from a (potentially de-serialized) dictionary."""
    self._t = state['t']
    self._storage = state['storage']
    self._distribution.set_state(state['distribution'])


class TransitionAccumulator:
  """Accumulates timesteps to form transitions."""

  def __init__(self):
    self.reset()

  def step(self, timestep_t: dm_env.TimeStep,
           a_t: parts.Action) -> Iterable[Transition]:
    """Accumulates timestep and resulting action, maybe yield a transition."""
    if timestep_t.first():
      self.reset()

    if self._timestep_tm1 is None:
      if not timestep_t.first():
        raise ValueError('Expected FIRST timestep, got %s.' % str(timestep_t))
      self._timestep_tm1 = timestep_t
      self._a_tm1 = a_t
      return  # Empty iterable.
    else:
      transition = Transition(
          s_tm1=self._timestep_tm1.observation,
          a_tm1=self._a_tm1,
          r_t=timestep_t.reward,
          discount_t=timestep_t.discount,
          s_t=timestep_t.observation,
      )
      self._timestep_tm1 = timestep_t
      self._a_tm1 = a_t
      yield transition

  def reset(self) -> None:
    """Resets the accumulator. Following timestep is expected to be `FIRST`."""
    self._timestep_tm1 = None
    self._a_tm1 = None


def _build_n_step_transition(transitions):
  """Builds a single n-step transition from n 1-step transitions."""
  r_t = 0.
  discount_t = 1.
  for transition in transitions:
    r_t += discount_t * transition.r_t
    discount_t *= transition.discount_t

  # n-step transition, letting s_tm1 = s_tmn, and a_tm1 = a_tmn.
  return Transition(
      s_tm1=transitions[0].s_tm1,
      a_tm1=transitions[0].a_tm1,
      r_t=r_t,
      discount_t=discount_t,
      s_t=transitions[-1].s_t,
  )


class NStepTransitionAccumulator:
  """Accumulates timesteps to form n-step transitions.

  Let `t` be the index of a timestep within an episode and `T` be the index of
  the final timestep within an episode. Then given the step type of the timestep
  passed into `step()` the accumulator will:
  *   `FIRST`: yield nothing.
  *   `MID`: if `t < n`, yield nothing, else yield one n-step transition
      `s_{t - n} -> s_t`.
  *   `LAST`: yield all transitions that end at `s_t = s_T` from up to n steps
      away, specifically `s_{T - min(n, T)} -> s_T, ..., s_{T - 1} -> s_T`.
      These are `min(n, T)`-step, ..., `1`-step transitions.
  """

  def __init__(self, n):
    self._transitions = collections.deque(maxlen=n)  # Store 1-step transitions.
    self.reset()

  def step(self, timestep_t: dm_env.TimeStep,
           a_t: parts.Action) -> Iterable[Transition]:
    """Accumulates timestep and resulting action, yields transitions."""
    if timestep_t.first():
      self.reset()

    # There are no transitions on the first timestep.
    if self._timestep_tm1 is None:
      assert self._a_tm1 is None
      if not timestep_t.first():
        raise ValueError('Expected FIRST timestep, got %s.' % str(timestep_t))
      self._timestep_tm1 = timestep_t
      self._a_tm1 = a_t
      return  # Empty iterator.

    self._transitions.append(
        Transition(
            s_tm1=self._timestep_tm1.observation,
            a_tm1=self._a_tm1,
            r_t=timestep_t.reward,
            discount_t=timestep_t.discount,
            s_t=timestep_t.observation,
        ))

    self._timestep_tm1 = timestep_t
    self._a_tm1 = a_t

    if timestep_t.last():
      # Yield any remaining n, n-1, ..., 1-step transitions at episode end.
      while self._transitions:
        yield _build_n_step_transition(self._transitions)
        self._transitions.popleft()
    else:
      # Wait for n transitions before yielding anything.
      if len(self._transitions) < self._transitions.maxlen:
        return  # Empty iterator.

      assert len(self._transitions) == self._transitions.maxlen

      # This is the typical case, yield a single n-step transition.
      yield _build_n_step_transition(self._transitions)

  def reset(self) -> None:
    """Resets the accumulator. Following timestep is expected to be FIRST."""
    self._transitions.clear()
    self._timestep_tm1 = None
    self._a_tm1 = None


def compress_array(array: np.ndarray) -> CompressedArray:
  """Compresses a numpy array with snappy."""
  return snappy.compress(array), array.shape, array.dtype


def uncompress_array(compressed: CompressedArray) -> np.ndarray:
  """Uncompresses a numpy array with snappy given its shape and dtype."""
  compressed_array, shape, dtype = compressed
  byte_string = snappy.uncompress(compressed_array)
  return np.frombuffer(byte_string, dtype=dtype).reshape(shape)
