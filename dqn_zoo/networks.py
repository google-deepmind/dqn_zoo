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
"""DQN agent network components and implementation."""

# pylint: disable=g-bad-import-order

import typing
from typing import Any, Callable, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Network = hk.Transformed
Params = hk.Params
NetworkFn = Callable[..., Any]


class QNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray


class IqnInputs(typing.NamedTuple):
  state: jnp.ndarray
  taus: jnp.ndarray


class IqnOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_dist: jnp.ndarray


class QRNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_dist: jnp.ndarray


class C51NetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_logits: jnp.ndarray


def _dqn_default_initializer(
    num_input_units: int) -> hk.initializers.Initializer:
  """Default initialization scheme inherited from past implementations of DQN.

  This scheme was historically used to initialize all weights and biases
  in convolutional and linear layers of DQN-type agents' networks.
  It initializes each weight as an independent uniform sample from [`-c`, `c`],
  where `c = 1 / np.sqrt(num_input_units)`, and `num_input_units` is the number
  of input units affecting a single output unit in the given layer, i.e. the
  total number of inputs in the case of linear (dense) layers, and
  `num_input_channels * kernel_width * kernel_height` in the case of
  convolutional layers.

  Args:
    num_input_units: number of input units to a single output unit of the layer.

  Returns:
    Haiku weight initializer.
  """
  max_val = np.sqrt(1 / num_input_units)
  return hk.initializers.RandomUniform(-max_val, max_val)


def conv(
    num_features: int,
    kernel_shape: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> NetworkFn:
  """Convolutional layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing conv layer with DQN's legacy initialization."""
    num_input_units = inputs.shape[-1] * kernel_shape[0] * kernel_shape[1]
    initializer = _dqn_default_initializer(num_input_units)
    layer = hk.Conv2D(
        num_features,
        kernel_shape=kernel_shape,
        stride=stride,
        w_init=initializer,
        b_init=initializer,
        padding='VALID')
    return layer(inputs)

  return net_fn


def linear(num_outputs: int, with_bias=True) -> NetworkFn:
  """Linear layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing linear layer with DQN's legacy initialization."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    layer = hk.Linear(
        num_outputs,
        with_bias=with_bias,
        w_init=initializer,
        b_init=initializer)
    return layer(inputs)

  return net_fn


def linear_with_shared_bias(num_outputs: int) -> NetworkFn:
  """Linear layer with single shared bias instead of one bias per output."""

  def layer_fn(inputs):
    """Function representing a linear layer with single shared bias."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    bias_free_linear = hk.Linear(
        num_outputs, with_bias=False, w_init=initializer)
    linear_output = bias_free_linear(inputs)
    bias = hk.get_parameter('b', [1], inputs.dtype, init=initializer)
    bias = jnp.broadcast_to(bias, linear_output.shape)
    return linear_output + bias

  return layer_fn


def noisy_linear(num_outputs: int,
                 weight_init_stddev: float,
                 with_bias: bool = True) -> NetworkFn:
  """Linear layer with weight randomization http://arxiv.org/abs/1706.10295."""

  def make_noise_sqrt(rng, shape):
    noise = jax.random.truncated_normal(rng, lower=-2., upper=2., shape=shape)
    return jax.lax.stop_gradient(jnp.sign(noise) * jnp.sqrt(jnp.abs(noise)))

  def net_fn(inputs):
    """Function representing a linear layer with learned noise distribution."""
    num_inputs = inputs.shape[-1]
    mu_initializer = _dqn_default_initializer(num_inputs)
    mu_layer = hk.Linear(
        num_outputs,
        name='mu',
        with_bias=with_bias,
        w_init=mu_initializer,
        b_init=mu_initializer)
    sigma_initializer = hk.initializers.Constant(  #
        weight_init_stddev / jnp.sqrt(num_inputs))
    sigma_layer = hk.Linear(
        num_outputs,
        name='sigma',
        with_bias=True,
        w_init=sigma_initializer,
        b_init=sigma_initializer)

    # Broadcast noise over batch dimension.
    input_noise_sqrt = make_noise_sqrt(hk.next_rng_key(), [1, num_inputs])
    output_noise_sqrt = make_noise_sqrt(hk.next_rng_key(), [1, num_outputs])

    # Factorized Gaussian noise.
    mu = mu_layer(inputs)
    noisy_inputs = input_noise_sqrt * inputs
    sigma = sigma_layer(noisy_inputs) * output_noise_sqrt
    return mu + sigma

  return net_fn


def dqn_torso() -> NetworkFn:
  """DQN convolutional torso.

  Includes scaling from [`0`, `255`] (`uint8`) to [`0`, `1`] (`float32`)`.

  Returns:
    Network function that `haiku.transform` can be called on.
  """

  def net_fn(inputs):
    """Function representing convolutional torso for a DQN Q-network."""
    network = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        conv(32, kernel_shape=(8, 8), stride=(4, 4)),
        jax.nn.relu,
        conv(64, kernel_shape=(4, 4), stride=(2, 2)),
        jax.nn.relu,
        conv(64, kernel_shape=(3, 3), stride=(1, 1)),
        jax.nn.relu,
        hk.Flatten(),
    ])
    return network(inputs)

  return net_fn


def dqn_value_head(num_actions: int, shared_bias: bool = False) -> NetworkFn:
  """Regular DQN Q-value head with single hidden layer."""

  last_layer = linear_with_shared_bias if shared_bias else linear

  def net_fn(inputs):
    """Function representing value head for a DQN Q-network."""
    network = hk.Sequential([
        linear(512),
        jax.nn.relu,
        last_layer(num_actions),
    ])
    return network(inputs)

  return net_fn


def rainbow_atari_network(
    num_actions: int,
    support: jnp.ndarray,
    noisy_weight_init: float,
) -> NetworkFn:
  """Rainbow network, expects `uint8` input."""

  if support.ndim != 1:
    raise ValueError('support should be 1D.')
  num_atoms = len(support)
  support = support[None, None, :]

  def net_fn(inputs):
    """Function representing Rainbow Q-network."""
    inputs = dqn_torso()(inputs)

    # Advantage head.
    advantage = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
    advantage = jax.nn.relu(advantage)
    advantage = noisy_linear(
        num_actions * num_atoms, noisy_weight_init, with_bias=False)(
            advantage)
    advantage = jnp.reshape(advantage, (-1, num_actions, num_atoms))

    # Value head.
    value = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
    value = jax.nn.relu(value)
    value = noisy_linear(num_atoms, noisy_weight_init, with_bias=False)(value)
    value = jnp.reshape(value, (-1, 1, num_atoms))

    # Q-distribution and values.
    q_logits = value + advantage - jnp.mean(advantage, axis=-2, keepdims=True)
    assert q_logits.shape[1:] == (num_actions, num_atoms)
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * support, axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

  return net_fn


def iqn_atari_network(num_actions: int, latent_dim: int) -> NetworkFn:
  """IQN network, expects `uint8` input."""

  def net_fn(iqn_inputs):
    """Function representing IQN-DQN Q-network."""
    state = iqn_inputs.state  # batch x state_shape
    taus = iqn_inputs.taus  # batch x samples
    # Apply DQN convnet to embed state.
    state_embedding = dqn_torso()(state)
    state_dim = state_embedding.shape[-1]
    # Embed taus with cosine embedding + linear layer.
    # cos(pi * i * tau) for i = 1,...,latents for each batch_element x sample.
    # Broadcast everything to batch x samples x latent_dim.
    pi_multiples = jnp.arange(1, latent_dim + 1, dtype=jnp.float32) * jnp.pi
    tau_embedding = jnp.cos(pi_multiples[None, None, :] * taus[:, :, None])
    # Map tau embedding onto state_dim via linear layer.
    embedding_layer = linear(state_dim)
    tau_embedding = hk.BatchApply(embedding_layer)(tau_embedding)
    tau_embedding = jax.nn.relu(tau_embedding)
    # Reshape/broadcast both embeddings to batch x num_samples x state_dim
    # and multiply together, before applying value head.
    head_input = tau_embedding * state_embedding[:, None, :]
    value_head = dqn_value_head(num_actions)
    q_dist = hk.BatchApply(value_head)(head_input)
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return IqnOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn


def qr_atari_network(num_actions: int, quantiles: jnp.ndarray) -> NetworkFn:
  """QR-DQN network, expects `uint8` input."""

  if quantiles.ndim != 1:
    raise ValueError('quantiles has to be 1D.')
  num_quantiles = len(quantiles)

  def net_fn(inputs):
    """Function representing QR-DQN Q-network."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_quantiles * num_actions),
    ])
    network_output = network(inputs)
    q_dist = jnp.reshape(network_output, (-1, num_quantiles, num_actions))
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return QRNetworkOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn


def c51_atari_network(num_actions: int, support: jnp.ndarray) -> NetworkFn:
  """C51 network, expects `uint8` input."""

  if support.ndim != 1:
    raise ValueError('support has to be 1D.')
  num_atoms = len(support)

  def net_fn(inputs):
    """Function representing C51 Q-network."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions * num_atoms),
    ])
    network_output = network(inputs)
    q_logits = jnp.reshape(network_output, (-1, num_actions, num_atoms))
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * support[None, None, :], axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

  return net_fn


def double_dqn_atari_network(num_actions: int) -> NetworkFn:
  """DQN network with shared bias in final layer, expects `uint8` input."""

  def net_fn(inputs):
    """Function representing DQN Q-network with shared bias output layer."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions, shared_bias=True),
    ])
    return QNetworkOutputs(q_values=network(inputs))

  return net_fn


def dqn_atari_network(num_actions: int) -> NetworkFn:
  """DQN network, expects `uint8` input."""

  def net_fn(inputs):
    """Function representing DQN Q-network."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions),
    ])
    return QNetworkOutputs(q_values=network(inputs))

  return net_fn
