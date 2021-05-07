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
"""Common functions and classes for testing."""

# pylint: disable=g-bad-import-order

from absl import flags
import dm_env
from dm_env import specs

from dqn_zoo import parts

FLAGS = flags.FLAGS


class DummyAgent(parts.Agent):
  """Agent that returns a dummy action.

  Records whether it took a step or reset on a tape.
  """

  def __init__(self, tape):
    self._tape = tape

  def reset(self):
    self._tape.append('Agent reset')

  def step(self, timestep):
    del timestep
    self._tape.append('Agent step')
    return 0

  def get_state(self):
    return {}

  def set_state(self, state):
    del state

  @property
  def statistics(self):
    return {}


class DummyEnvironment(dm_env.Environment):
  """Environment that ignores actions and generates dummy timesteps.

  Records whether it took a step or reset on a tape.
  """

  def __init__(self, tape, episode_length):
    self._tape = tape
    self._episode_length = episode_length

  def reset(self):
    self._t = 0
    self._tape.append('Environment reset')
    step_type = dm_env.StepType.FIRST
    return dm_env.TimeStep(
        step_type=step_type, reward=0., discount=0., observation=1.)

  def step(self, action):
    self._tape.append('Environment step (%s)' % action)
    self._t += 1
    if self._t == 0:
      step_type = dm_env.StepType.FIRST
    elif self._t == self._episode_length:
      step_type = dm_env.StepType.LAST
      self._t = -1
    else:
      step_type = dm_env.StepType.MID

    discount = 0. if step_type == dm_env.StepType.LAST else 1.
    return dm_env.TimeStep(
        step_type=step_type, reward=2., discount=discount, observation=1.)

  def action_spec(self):
    return specs.Array(shape=(), dtype=int)

  def observation_spec(self):
    return specs.Array(shape=(), dtype=float)
