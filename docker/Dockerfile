# Ubuntu 20.04 + ROS Noetic: the only platform x_evaluate / x_events have been tested on.
FROM ros:noetic-ros-base-focal

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget unzip build-essential cmake autoconf automake libtool \
      python3-catkin-tools python3-vcstool python3-pip python3-osrf-pycommon \
      ros-noetic-cmake-modules ros-noetic-cv-bridge ros-noetic-rosbag ros-noetic-tf2-msgs \
      ros-noetic-geometry-msgs ros-noetic-sensor-msgs \
      libopencv-dev libyaml-cpp-dev libboost-all-dev libeigen3-dev \
      libgoogle-glog-dev libgflags-dev libsuitesparse-dev libatlas-base-dev libblas-dev liblapack-dev \
      libsdl2-2.0-0 libgl1 libopengl0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# PyYAML comes from apt (python3-yaml, needed by ROS) and pip cannot uninstall it, so install the newer one alongside
RUN python3 -m pip install --no-cache-dir --upgrade "pip<24.1" \
    && python3 -m pip install --no-cache-dir --ignore-installed "pyyaml>=5.4.1" \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# x_events' HASTE tracker needs Eigen >= 3.4 (Ubuntu 20.04 ships 3.3.7): install 3.4 headers over the system ones
RUN wget -qO- https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz | tar xz -C /tmp \
    && cmake -S /tmp/eigen-3.4.0 -B /tmp/eigen-build -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_TESTING=OFF \
    && cmake --install /tmp/eigen-build \
    && rm -rf /tmp/eigen-3.4.0 /tmp/eigen-build

RUN groupadd -o -g ${GROUP_ID} dev \
    && useradd -m -o -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash dev \
    && mkdir -p /catkin_ws/src && chown -R dev:dev /catkin_ws
USER dev
WORKDIR /catkin_ws

# Fetch and pre-build the X library and its catkin dependencies (ceres is compiled from source, takes a while).
# x_evaluate itself is bind-mounted at runtime into /catkin_ws/src/x_evaluate, see docker/run.sh.
COPY --chown=dev:dev dependencies.yaml /tmp/dependencies.yaml
RUN vcs import --shallow src < /tmp/dependencies.yaml \
    && find src/rpg_dvs_ros -mindepth 1 -maxdepth 1 -type d ! -name dvs_msgs -exec touch {}/CATKIN_IGNORE \; \
    && catkin init \
    && catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && catkin build --no-status x dvs_msgs gflags_catkin glog_catkin easy_profiler_catkin

# evo calls mpl.use(<its plot_backend setting>) on import, ignoring MPLBACKEND, and defaults to TkAgg when DISPLAY is set
RUN evo_config set plot_backend Agg \
    && git config --global --add safe.directory '*' \
    && echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc \
    && echo '[ -f /catkin_ws/devel/setup.bash ] && source /catkin_ws/devel/setup.bash' >> ~/.bashrc

# Rerun for the 3D views of the runs. 0.22.0 is the last version that runs on Python 3.8 (0.22.1 declares 3.8 but uses
# list[float] annotations); its recordings open in the 0.22.1 viewer on the host. Installed after the catkin build so
# that adding it does not recompile ceres.
USER root
RUN python3 -m pip install --no-cache-dir rerun-sdk==0.22.0
USER dev

COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
