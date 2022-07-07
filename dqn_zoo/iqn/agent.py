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
"""IQN agent classes."""

# pylint: disable=g-bad-import-order

from typing import Any, Callable, Mapping, Text, Tuple

from absl import logging
import chex
import distrax
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax

from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib

# Batch variant of quantile_q_learning.
_batch_quantile_q_learning = jax.vmap(
    rlax.quantile_q_learning, in_axes=(0, 0, 0, 0, 0, 0, 0, None))

IqnInputs = networks.IqnInputs


def _sample_tau(
    rng_key: parts.PRNGKey,
    shape: Tuple[int, ...],
) -> Tuple[jnp.ndarray]:
  """Samples tau values uniformly between 0 and 1."""
  return jax.random.uniform(rng_key, shape=shape)


class IqnEpsilonGreedyActor(parts.Agent):
  """Agent that acts with a given set of IQN-network parameters and epsilon.

  Network parameters are set on the actor. The actor can be checkpointed for
  determinism.
  """

  def __init__(
      self,
      preprocessor: processors.Processor,
      network: parts.Network,
      exploration_epsilon: float,
      tau_samples: int,
      rng_key: parts.PRNGKey,
  ):
    self._preprocessor = preprocessor
    self._rng_key = rng_key
    self._action = None
    self.network_params = None  # Nest of arrays (haiku.Params), set externally.

    def select_action(rng_key, network_params, s_t):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, tau_key, apply_key, policy_key = jax.random.split(rng_key, 4)
      tau_t = _sample_tau(tau_key, (1, tau_samples))
      q_t = network.apply(network_params, apply_key,
                          IqnInputs(s_t[None, ...], tau_t)).q_values[0]
      a_t = distrax.EpsilonGreedy(q_t,
                                  exploration_epsilon).sample(seed=policy_key)
      return rng_key, a_t

    self._select_action = jax.jit(select_action)

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given a timestep."""
    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      return self._action

    s_t = timestep.observation
    self._rng_key, a_t = self._select_action(self._rng_key, self.network_params,
                                             s_t)
    self._action = parts.Action(jax.device_get(a_t))
    return self._action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    processors.reset(self._preprocessor)
    self._action = None

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    # State contains network params to make agent easy to run from a checkpoint.
    return {
        'rng_key': self._rng_key,
        'network_params': self.network_params,
    }

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self.network_params = state['network_params']

  @property
  def statistics(self) -> Mapping[Text, float]:
    return {}


class Iqn(parts.Agent):
  """Implicit Quantile Network agent."""

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: IqnInputs,
      network: parts.Network,
      optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      huber_param: float,
      tau_samples_policy: int,
      tau_samples_s_tm1: int,
      tau_samples_s_t: int,
      rng_key: parts.PRNGKey,
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
    self._online_params = network.init(
        network_rng_key,
        jax.tree_map(lambda x: x[None, ...], sample_network_input))
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.
    self._statistics = {'state_value': np.nan}

    # Define jitted loss, update, and policy functions here instead of as
    # class methods, to emphasize that these are meant to be pure functions
    # and should not access the agent object's state via `self`.

    def loss_fn(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      # Sample tau values for q_tm1, q_t_selector, q_t.
      batch_size = self._batch_size
      rng_key, *sample_keys = jax.random.split(rng_key, 4)
      tau_tm1 = _sample_tau(sample_keys[0], (batch_size, tau_samples_s_tm1))
      tau_t_selector = _sample_tau(sample_keys[1],
                                   (batch_size, tau_samples_policy))
      tau_t = _sample_tau(sample_keys[2], (batch_size, tau_samples_s_t))

      # Compute Q value distributions.
      _, *apply_keys = jax.random.split(rng_key, 4)
      dist_q_tm1 = network.apply(online_params, apply_keys[0],
                                 IqnInputs(transitions.s_tm1, tau_tm1)).q_dist
      dist_q_t_selector = network.apply(
          target_params, apply_keys[1],
          IqnInputs(transitions.s_t, tau_t_selector)).q_dist
      dist_q_target_t = network.apply(target_params, apply_keys[2],
                                      IqnInputs(transitions.s_t, tau_t)).q_dist
      losses = _batch_quantile_q_learning(
          dist_q_tm1,
          tau_tm1,
          transitions.a_tm1,
          transitions.r_t,
          transitions.discount_t,
          dist_q_t_selector,
          dist_q_target_t,
          huber_param,
      )
      chex.assert_shape(losses, (self._batch_size,))
      loss = jnp.mean(losses)
      return loss

    def update(rng_key, opt_state, online_params, target_params, transitions):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)
      d_loss_d_params = jax.grad(loss_fn)(online_params, target_params,
                                          transitions, update_key)
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params

    self._update = jax.jit(update)

    def select_action(rng_key, network_params, s_t, exploration_epsilon):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, sample_key, apply_key, policy_key = jax.random.split(rng_key, 4)
      tau_t = _sample_tau(sample_key, (1, tau_samples_policy))
      q_t = network.apply(network_params, apply_key,
                          IqnInputs(s_t[None, ...], tau_t)).q_values[0]
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
    )

  @property
  def online_params(self) -> parts.NetworkParams:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def statistics(self) -> Mapping[Text, float]:
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
