# Parent image.
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update

# Install dependencies for snappy.
RUN apt-get install -y libsnappy-dev=1.1.7-1

# Install git to clone RLax repository.
RUN apt-get install -y git=1:2.17.1-1ubuntu0.7

# Install Python 3, pip, setuptools.
RUN apt-get install -y python3=3.6.7-1~18.04 python3-pip python3-setuptools
RUN pip3 install --upgrade pip==20.1.1 setuptools==47.3.1

WORKDIR /workspace

# Copy requirements file specifying pinned dependencies.
COPY ./docker_requirements.txt /workspace/

# Install Python dependencies.
RUN pip3 install -r docker_requirements.txt

# List Python dependencies.
RUN pip3 freeze

# Copy over source code from build context.
COPY ./dqn_zoo /workspace/dqn_zoo

# List files in current directory.
RUN find .

# Run tests on CPU.
ARG JAX_PLATFORM_NAME=cpu
RUN python3 -m dqn_zoo.gym_atari_test
RUN python3 -m dqn_zoo.networks_test
RUN python3 -m dqn_zoo.parts_test
RUN python3 -m dqn_zoo.replay_test
RUN python3 -m dqn_zoo.c51.run_atari_test
RUN python3 -m dqn_zoo.double_q.run_atari_test
RUN python3 -m dqn_zoo.dqn.run_atari_test
RUN python3 -m dqn_zoo.iqn.run_atari_test
RUN python3 -m dqn_zoo.prioritized.run_atari_test
RUN python3 -m dqn_zoo.qrdqn.run_atari_test
RUN python3 -m dqn_zoo.rainbow.run_atari_test

# Allow running container as an executable. E.g. to run DQN:
# docker run --gpus all --name dqn_zoo_dqn dqn_zoo:latest \
#     -m dqn_zoo.dqn.run_atari --environment_name=pong
ENTRYPOINT ["python3"]

