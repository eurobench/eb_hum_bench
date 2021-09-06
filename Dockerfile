FROM ubuntu:bionic

# inspired from https://sourcery.ai/blog/python-docker/

RUN apt-get update \
    && apt-get install -y less \
    && apt-get install -y wget dialog apt-utils \
    && apt-get -y install locales

# Set the locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

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
    apt-get -qq install --no-install-recommends -y python3-scipy && \
    apt-get -qq install --no-install-recommends -y python3-pytest

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

# Install pybind from source because of cmake error in ubuntu 18.04
RUN git clone https://github.com/pybind/pybind11.git /home/pi_runner/pybind11

RUN mkdir -p /home/pi_runner/pybind11/build
RUN cd /home/pi_runner/pybind11/build && \cmake ..
RUN cd /home/pi_runner/pybind11/build && \
    make DESTDIR=/home/pi_runner/pybind11/build install

RUN apt-get -qq install --no-install-recommends python3-pip
RUN apt-get -qq install --no-install-recommends -y python3-setuptools

RUN pip3 install --upgrade pip
#RUN apt-get -qq install --no-install-recommends -y python3-wheel
#RUN pip3 install numpy
#RUN pip3 install pandas
#RUN pip3 install pyyaml
#RUN pip3 install scipy
#RUN pip3 install Shapely
#RUN pip3 install csaps
#RUN pip3 install matplotlib
RUN export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/home/pi_runner/pybind11/build/usr/local/share/cmake/pybind11 && \
    pip3 install sophuspy

COPY src /home/pi_runner/src
# set the user as owner of the copied files.
RUN chown -R pi_runner:pi_runner /home/pi_runner/src

WORKDIR /home/pi_runner
RUN pip3 install --ignore-installed -e src/
#USER pi_runner
ENV PYTHONPATH=${PYTHONPATH}:/home/pi_runner/rbdl-orb-build/python
ENV export PYTHONPATH=${PYTHONPATH}:/home/pi_runner/.local/bin
