FROM ubuntu:bionic

# inspired from https://sourcery.ai/blog/python-docker/

RUN apt-get update \
    && apt-get install -y less \
    && apt-get install -y wget dialog apt-utils

# create user account, and create user home dir
RUN useradd -ms /bin/bash pi_runner
ENV DEBIAN_FRONTEND="noninteractive"

WORKDIR /home/pi_runner

# Install rbdl-org dependencies

RUN apt-get -qq install --no-install-recommends -y build-essential cmake && \
    apt-get -qq install --no-install-recommends -y git-core libeigen3-dev && \
    apt-get -qq install --no-install-recommends -y cmake-curses-gui lua5.1 liblua5.1-0-dev

RUN apt-get -qq install --no-install-recommends -y cython3 && \
    apt-get -qq install --no-install-recommends -y python3-numpy && \
    apt-get -qq install --no-install-recommends -y python3-matplotlib && \
    apt-get -qq install --no-install-recommends -y python3-scipy &&\
    apt-get -qq install --no-install-recommends -y python3-pybind11

RUN apt-get -qq install --no-install-recommends libboost-all-dev
RUN apt-get -qq install --no-install-recommends -y git
RUN git clone https://gitlab.com/flxalr/rbdl_balance.git /home/pi_runner/rbdl-orb

RUN mkdir -p /home/pi_runner/rbdl-orb-build
RUN cd /home/pi_runner/rbdl-orb-build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D RBDL_BUILD_ADDON_LUAMODEL=ON \
    -D RBDL_BUILD_ADDON_URDFREADER=ON -D RBDL_BUILD_ADDON_BALANCE=ON \
    -D RBDL_BUILD_PYTHON_WRAPPER=ON ../rbdl-orb
RUN cd /home/pi_runner/rbdl-orb-build && \
    make

RUN apt-get -qq install --no-install-recommends python3-pip
RUN apt-get -qq install --no-install-recommends -y python3-setuptools

RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install pyyaml
RUN pip3 install scipy
RUN pip3 install Shapely
RUN pip3 install csaps
RUN pip3 install matplotlib
RUN pip3 install sophuspy

COPY *.py /home/pi_runner/
# set the user as owner of the copied files.
RUN chown -R pi_runner:pi_runner /home/pi_runner/

USER pi_runner
WORKDIR /home/pi_runner
ENV PYTHONPATH=${PYTHONPATH}:/home/pi_runner/rbdl-orb-build/python
