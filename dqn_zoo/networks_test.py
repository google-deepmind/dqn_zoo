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
"""Tests for networks."""

# pylint: disable=g-bad-import-order

import haiku as hk
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np
import tree

from dqn_zoo import networks
from absl.testing import absltest


def _sample_input(input_shape):
  return jnp.zeros((1,) + input_shape, dtype=jnp.float32)


class SimpleLayersTest(absltest.TestCase):

  def test_linear(self):
    layer = hk.transform(networks.linear(4))
    params = layer.init(jax.random.PRNGKey(1), _sample_input((3,)))
    self.assertCountEqual(['linear'], params)
    lin_params = params['linear']
    self.assertCountEqual(['w', 'b'], lin_params)
    self.assertEqual((3, 4), lin_params['w'].shape)
    self.assertEqual((4,), lin_params['b'].shape)

  def test_conv(self):
    layer = hk.transform(networks.conv(4, (3, 3), 2))
    params = layer.init(jax.random.PRNGKey(1), _sample_input((7, 7, 3)))
    self.assertCountEqual(['conv2_d'], params)
    conv_params = params['conv2_d']
    self.assertCountEqual(['w', 'b'], conv_params)
    self.assertEqual((3, 3, 3, 4), conv_params['w'].shape)
    self.assertEqual((4,), conv_params['b'].shape)


class LinearWithSharedBiasTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng_key = jax.random.PRNGKey(1)
    self.init_rng_key, self.apply_rng_key = jax.random.split(rng_key)
    self.input_shape = (4,)
    self.output_shape = (3,)
    self.weights_shape = (self.input_shape[0], self.output_shape[0])
    network_fn = networks.linear_with_shared_bias(self.output_shape[0])
    self.network = hk.transform(network_fn)

  def test_bias_parameter_shape(self):
    params = self.network.init(self.init_rng_key,
                               _sample_input(self.input_shape))
    self.assertLen(tree.flatten(params), 2)

    def check_params(path, param):
      if path[-1] == 'b':
        self.assertNotEqual(self.output_shape, param.shape)
        self.assertEqual((1,), param.shape)
      elif path[-1] == 'w':
        self.assertEqual(self.weights_shape, param.shape)
      else:
        self.fail('Unexpected parameter %s.' % path)

    tree.map_structure_with_path(check_params, params)

  def test_output_shares_bias(self):
    bias = 1.23
    params = self.network.init(self.init_rng_key,
                               _sample_input(self.input_shape))

    def replace_params(path, param):
      if path[-1] == 'b':
        return jnp.ones_like(param) * bias
      else:
        return jnp.zeros_like(param)

    params = tree.map_structure_with_path(replace_params, params)
    output = self.network.apply(params, self.apply_rng_key,
                                jnp.zeros((1,) + self.input_shape))
    self.assertEqual((1,) + self.output_shape, output.shape)
    np.testing.assert_allclose([bias] * self.output_shape[0], list(output[0]))


class NoisyLinearTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rng_key = jax.random.PRNGKey(1)
    self.init_rng_key, self.apply_rng_key = jax.random.split(rng_key)
    self.input_shape = (4,)
    self.output_shape = (3,)
    self.network_fn = networks.noisy_linear(self.output_shape[0], 0.1)
    self.network = hk.transform(self.network_fn)
    self.params = self.network.init(self.init_rng_key,
                                    _sample_input(self.input_shape))
    self.inputs = jnp.zeros((2,) + self.input_shape)

  def test_basic(self):
    self.network.apply(self.params, self.apply_rng_key, self.inputs)

  def test_error_raised_if_rng_is_not_passed_in(self):
    with self.assertRaisesRegex(ValueError, 'must be called with an RNG'):
      self.network.apply(self.params, self.inputs)

  def test_error_raised_if_transformed_without_rng_1(self):
    network = hk.without_apply_rng(hk.transform(self.network_fn))
    with self.assertRaisesRegex(ValueError, 'PRNGKey'):
      network.apply(self.params, self.inputs)

  def test_error_raised_if_transformed_without_rng_2(self):
    network = hk.without_apply_rng(hk.transform(self.network_fn))
    with self.assertRaisesRegex(TypeError, 'positional argument'):
      network.apply(self.params, self.apply_rng_key, self.inputs)

  def test_same_rng_produces_same_outputs(self):
    outputs_1 = self.network.apply(self.params, self.apply_rng_key, self.inputs)
    outputs_2 = self.network.apply(self.params, self.apply_rng_key, self.inputs)
    np.testing.assert_allclose(outputs_1, outputs_2)

  def test_different_rngs_produce_different_outputs(self):
    rng_1, rng_2 = jax.random.split(jax.random.PRNGKey(1))
    outputs_1 = self.network.apply(self.params, rng_1, self.inputs)
    outputs_2 = self.network.apply(self.params, rng_2, self.inputs)
    self.assertFalse(np.allclose(outputs_1, outputs_2))

  def test_number_of_params_with_bias_correct(self):
    net_fn = networks.noisy_linear(self.output_shape[0], 0.1, with_bias=True)
    network = hk.transform(net_fn)
    params = network.init(self.init_rng_key, _sample_input(self.input_shape))
    self.assertCountEqual(['mu', 'sigma'], params)
    self.assertCountEqual(['b', 'w'], params['mu'])
    self.assertCountEqual(['b', 'w'], params['sigma'])

  def test_number_of_params_without_bias_correct(self):
    net_fn = networks.noisy_linear(self.output_shape[0], 0.1, with_bias=False)
    network = hk.transform(net_fn)
    params = network.init(self.init_rng_key, _sample_input(self.input_shape))
    self.assertCountEqual(['mu', 'sigma'], params)
    self.assertCountEqual(['w'], params['mu'])
    self.assertCountEqual(['b', 'w'], params['sigma'])

  def test_sigma_params_are_constant(self):
    self.assertCountEqual(['mu', 'sigma'], self.params)
    sigma_params = self.params['sigma']
    sigma_w_values = np.unique(sigma_params['w'])
    sigma_b_values = np.unique(sigma_params['b'])
    self.assertLen(sigma_w_values, 1)
    self.assertLen(sigma_b_values, 1)
    value = 0.1 / np.sqrt(self.input_shape[0])
    self.assertAlmostEqual(value, sigma_w_values)
    self.assertAlmostEqual(value, sigma_b_values)


if __name__ == '__main__':
  config.update('jax_numpy_rank_promotion', 'raise')
  absltest.main()
