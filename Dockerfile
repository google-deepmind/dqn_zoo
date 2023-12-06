# Parent image.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update

# Avoid interaction when installing tzdata.
ARG DEBIAN_FRONTEND=noninteractive

# Install Python 3.9.
RUN apt-get install -y python3.9=3.9.5-3ubuntu0~20.04.1

# Install pip.
RUN apt-get install -y python3-pip
RUN python3.9 -m pip install --upgrade pip==22.1.2

# Install wget and unrar to download and extract ROMs.
RUN apt-get install -y wget unrar

WORKDIR /workspace

# Copy requirements file specifying pinned dependencies.
COPY ./docker_requirements.txt /workspace/

# Install Python dependencies.
RUN python3.9 -m pip install -r docker_requirements.txt

# List Python dependencies.
RUN python3.9 -m pip freeze

# Copy over source code from build context.
COPY ./dqn_zoo /workspace/dqn_zoo

# Download, extract and import Atari ROMs. The ROMs are available from
# http://www.atarimania.com for free but make sure the respective license covers
# your particular use case.
RUN mkdir atari_roms
RUN wget http://www.atarimania.com/roms/Roms.rar --directory-prefix=atari_roms/
RUN unrar x -y atari_roms/Roms.rar atari_roms
RUN python3.9 -m atari_py.import_roms atari_roms/ROMS
RUN rm -rf atari_roms

# List files in current directory.
RUN find .

# Run tests on CPU.
ARG JAX_PLATFORM_NAME=cpu
RUN python3.9 -m dqn_zoo.gym_atari_test
RUN python3.9 -m dqn_zoo.networks_test
RUN python3.9 -m dqn_zoo.parts_test
RUN python3.9 -m dqn_zoo.replay_test
RUN python3.9 -m dqn_zoo.c51.run_atari_test
RUN python3.9 -m dqn_zoo.double_q.run_atari_test
RUN python3.9 -m dqn_zoo.dqn.run_atari_test
RUN python3.9 -m dqn_zoo.iqn.run_atari_test
RUN python3.9 -m dqn_zoo.prioritized.run_atari_test
RUN python3.9 -m dqn_zoo.qrdqn.run_atari_test
RUN python3.9 -m dqn_zoo.rainbow.run_atari_test

# Allow running container as an executable. E.g. to run DQN:
# docker run --gpus all --name dqn_zoo_dqn dqn_zoo:latest \
#     -m dqn_zoo.dqn.run_atari --environment_name=pong
ENTRYPOINT ["python3.9"]
