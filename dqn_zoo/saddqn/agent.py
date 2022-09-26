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
"""SAD-DQN agent class."""

# pylint: disable=g-bad-import-order

from typing import Any, Callable, Mapping, Text

from absl import logging
import chex
import distrax
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib

# Batch variant of sad_q_learning with fixed tau input across batch.


Array = chex.Array
Numeric = chex.Numeric

def sad_q_learning(
    dist_q_tm1: Array,
    a_tm1: Numeric,
    r_t: Numeric,
    discount_t: Numeric,
    dist_q_t_selector: Array,
    dist_q_t: Array,
    dist_q_target_tm1: Array,
    mixture_ratio: Numeric,
    stop_target_gradients: bool = True,
) -> Numeric:
  """Implements Q-learning for avar-valued Q distributions.

  See "xxxx" by
  Achab et al. (https://arxiv.org/abs/xxxx).

  Args:
    dist_q_tm1: Q distribution at time t-1.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    dist_q_t_selector: Q distribution at time t for selecting greedy action in
      target policy. This is separate from dist_q_t as in Double Q-Learning, but
      can be computed with the target network and a separate set of samples.
    dist_q_t: target Q distribution at time t.
    stop_target_gradients: bool indicating whether or not to apply stop gradient
      to targets.

  Returns:
    SAD-Q-learning temporal difference error.
  """
  chex.assert_rank([
      dist_q_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t, dist_q_target_tm1, mixture_ratio
  ], [2, 0, 0, 0, 2, 2, 2, 0])
  chex.assert_type([
      dist_q_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t, dist_q_target_tm1, mixture_ratio
  ], [float, int, float, float, float, float, float, float])

  # Only update the taken actions.
  dist_qa_tm1 = dist_q_tm1[:, a_tm1]
  dist_qa_target_tm1 = dist_q_target_tm1[:, a_tm1]

  # Select target action according to greedy policy w.r.t. dist_q_t_selector.
  q_t_selector = jnp.mean(dist_q_t_selector, axis=0)
  a_t = jnp.argmax(q_t_selector)
  dist_qa_t = dist_q_t[:, a_t]

  # ADDED BY MASTANE
  target_tm1 = r_t + discount_t * jnp.mean(dist_qa_t)
  num_avars = dist_qa_tm1.shape[-1]
  # take argsort on atoms, then reorder atoms and probabilities
  probas = ( (1.0 - mixture_ratio) / jnp.float32( num_avars ) ) * jnp.ones_like( dist_qa_target_tm1 , dtype='float32')
  probas = jnp.append(probas, mixture_ratio)
  atoms_target_tm1 = jnp.append(dist_qa_target_tm1, target_tm1)
  sigma = jnp.argsort( atoms_target_tm1 )
  atoms_target_tm1 = atoms_target_tm1[sigma]
  probas = probas[sigma]
  # avar intervals
  i_window = jnp.arange( 1, num_avars + 1 ) / jnp.float32( num_avars )  # avar integration segments
  j_right = jnp.cumsum(probas)  # cumulative probabilities of the N+1 atoms
  j_left = j_right - probas
  i_window = jnp.expand_dims( i_window, axis=1 )
  j_right = jnp.expand_dims( j_right, axis=0 )
  j_left = jnp.expand_dims( j_left, axis=0 )
  # compute avars
  minij = jnp.minimum( i_window, j_right )
  maxij = jnp.maximum( i_window - 1.0/ jnp.float32( num_avars ) , j_left )
  lengths_inter = jnp.maximum( 0.0, minij - maxij )  # matrix of lengths of intersections of intervals [(i-1)/N, i/N] with [(j-1)/(N+1), j/(N+1)]
  dist_target = jnp.float32( num_avars ) * jnp.dot(lengths_inter, atoms_target_tm1)

  # Compute target, do not backpropagate into it.
  dist_target = jax.lax.select(stop_target_gradients,
                               jax.lax.stop_gradient(dist_target), dist_target)

  return dist_target - dist_qa_tm1

_batch_sad_q_learning = jax.vmap(sad_q_learning)

class SadDqn(parts.Agent):
  """Atomic Distributional Deep Q-Network agent."""

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: parts.Network,
      avars: jnp.ndarray,
      optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      grad_error_bound: float,
      rng_key: parts.PRNGKey,
      mixture_ratio: float,
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(network_rng_key,
                                       sample_network_input[None, ...])
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.
    self._statistics = {'state_value': np.nan}

    self._mixture_ratio = mixture_ratio

    # Define jitted loss, update, and policy functions here instead of as
    # class methods, to emphasize that these are meant to be pure functions
    # and should not access the agent object's state via `self`.

    def loss_fn(online_params, target_params, transitions, rng_key, mixture_ratio):
      """Calculates loss given network parameters and transitions."""
      # Compute Q value distributions.
      _, online_key, target_key = jax.random.split(rng_key, 3)
      dist_q_tm1 = network.apply(online_params, online_key,
                                 transitions.s_tm1).q_dist
      dist_q_target_t = network.apply(target_params, target_key,
                                      transitions.s_t).q_dist
      dist_q_target_tm1 = network.apply(target_params, target_key,
                                      transitions.s_tm1).q_dist
      td_errors = _batch_sad_q_learning(
          dist_q_tm1,
          transitions.a_tm1,
          transitions.r_t,
          transitions.discount_t,
          dist_q_target_t,  # No double Q-learning here.
          dist_q_target_t,
          dist_q_target_tm1,  # ADDED BY MASTANE: target dist for mixture update
          mixture_ratio * jnp.ones_like(transitions.r_t, dtype='float32'),  # mixture update parameter
      )
      td_errors = rlax.clip_gradient(td_errors, -grad_error_bound,
                                     grad_error_bound)
      losses = rlax.l2_loss(td_errors)
      #chex.assert_shape(losses, (self._batch_size,))
      loss = jnp.mean(losses, axis=-1)
      chex.assert_shape(loss, (self._batch_size,))
      return jnp.mean(loss)

    def update(rng_key, opt_state, online_params, target_params, transitions,mixture_ratio):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)
      d_loss_d_params = jax.grad(loss_fn)(online_params, target_params,
                                          transitions, update_key,mixture_ratio)
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params

    self._update = jax.jit(update)

    def select_action(rng_key, network_params, s_t, exploration_epsilon):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      a_t = distrax.EpsilonGreedy(q_t,
                                  exploration_epsilon).sample(seed=policy_key)
      v_t = jnp.max(q_t, axis=-1)
      return rng_key, a_t, v_t

    self._select_action = jax.jit(select_action)

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      action = self._action
    else:
      action = self._action = self._act(timestep)

      for transition in self._transition_accumulator.step(timestep, action):
        self._replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep) -> parts.Action:
    """Selects action given timestep, according to epsilon-greedy policy."""
    s_t = timestep.observation
    self._rng_key, a_t, v_t = self._select_action(self._rng_key,
                                                  self._online_params, s_t,
                                                  self.exploration_epsilon)
    a_t, v_t = jax.device_get((a_t, v_t))
    self._statistics['state_value'] = v_t
    return parts.Action(a_t)

  def _learn(self) -> None:
    """Samples a batch of transitions from replay and learns from it."""
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    transitions = self._replay.sample(self._batch_size)
    self._rng_key, self._opt_state, self._online_params = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions,
        self._mixture_ratio
    )

  @property
  def online_params(self) -> parts.NetworkParams:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def statistics(self) -> Mapping[Text, float]:
    """Returns current agent statistics as a dictionary."""
    # Check for DeviceArrays in values as this can be very slow.
    assert all(
        not isinstance(x, jnp.DeviceArray) for x in self._statistics.values())
    return self._statistics

  @property
  def exploration_epsilon(self) -> float:
    """Returns epsilon value currently used by (eps-greedy) behavior policy."""
    return self._exploration_epsilon(self._frame_t)

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state': self._opt_state,
        'online_params': self._online_params,
        'target_params': self._target_params,
        'replay': self._replay.get_state(),
    }
    return state

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = jax.device_put(state['opt_state'])
    self._online_params = jax.device_put(state['online_params'])
    self._target_params = jax.device_put(state['target_params'])
    self._replay.set_state(state['replay'])
