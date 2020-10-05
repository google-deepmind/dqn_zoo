#!/bin/bash

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

# This script does the following:
# 1. Clones the DQN Zoo repository.
# 2. Builds a Docker image with all necessary dependencies and runs unit tests.
# 3. Starts a short run of DQN on Pong in a GPU-accelerated container.

# Before running:
# * Install docker-ce version 19.03 or later for the --gpus options.
# * Install nvidia-docker.
# * Enable sudoless docker.
# * Verify: `docker run --gpus all --rm nvidia/cuda:10.1-base nvidia-smi`.

# To remove all containers run:
# `docker rm -vf $(docker ps -a -q)`

# To remove all images run:
# `docker rmi -f $(docker images -a -q)`

set -u -e  # Check for uninitialized variables and exit if any command fails.

WORK_DIR=$(mktemp -d -t dqn_zoo_"$(date +"%Y%m%d_%H%M%S_XXXXXX")")
echo "Working directory: $WORK_DIR"

function clean_up() {
  echo "Removing $WORK_DIR"
  rm -rf "$WORK_DIR"
}

# Clean up on exit.
trap clean_up INT EXIT TERM

echo "Clone DQN Zoo repository"
git clone https://github.com/deepmind/dqn_zoo.git "$WORK_DIR"
find "$WORK_DIR"

echo "Remove container if it exists"
docker rm dqn_zoo_dqn || true

echo "Remove image if it exists"
docker rmi dqn_zoo:latest || true

echo "Build image with tag 'dqn_zoo:latest' and run tests"
docker build -t dqn_zoo:latest "$WORK_DIR"

echo "Run DQN on GPU in a container named dqn_zoo_dqn"
docker run --gpus all --name dqn_zoo_dqn dqn_zoo:latest \
    -m dqn_zoo.dqn.run_atari \
    --jax_platform_name=gpu \
    --environment_name=pong \
    --replay_capacity=1000 \
    --target_network_update_period=40 \
    --num_iterations=10 \
    --num_train_frames=1000 \
    --num_eval_frames=500

# Note $WORK_DIR will be removed on exit.
echo "Finished"
