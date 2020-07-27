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
"""Tests for atari_data."""

# pylint: disable=g-bad-import-order

import math

from dqn_zoo import atari_data
from absl.testing import absltest


class AtariDataTest(absltest.TestCase):

  def test_num_games(self):
    self.assertLen(atari_data.ATARI_GAMES, 57)

  def test_monotonic_scores(self):
    # Test that for each game a higher raw score implies a higher normalized
    # score, which implicitly tests that
    #  a) all game data is present
    #  b) human score > random score for each game.
    for game in atari_data.ATARI_GAMES:
      low_score = atari_data.get_human_normalized_score(game, 10.)
      high_score = atari_data.get_human_normalized_score(game, 1000.)
      self.assertGreater(high_score, low_score)

  def test_returns_nan_for_unknown_games(self):
    score = atari_data.get_human_normalized_score('unknown_game', 10.)
    self.assertTrue(math.isnan(score))


if __name__ == '__main__':
  absltest.main()
