FROM ubuntu:20.04

## Install basic packages and useful utilities
## ===========================================
ENV DEBIAN_FRONTEND noninteractive
ENV MPLBACKEND=Agg
#ARG PYCHARM_SOURCE="https://download.jetbrains.com/python/pycharm-community-2020.1.2.tar.gz"
ARG ANACONDA_SOURCE="https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh"
ARG DOCKUSER_UID=4283
ARG DOCKUSER_GID=4283

#ENV CUDA_VERSION 11.2
#ENV CUDA_PKG_VERSION 11-2=$CUDA_VERSION-1
ENV CUDA_PKG 11-2
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,video,utility,graphics
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
#ENV NCCL_VERSION 2.5.6
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV CUDNN_VERSION 8

RUN apt update -y && \
    apt install -y gnupg2 curl software-properties-common ca-certificates \
        apt-utils wget bzip2 locales sudo sshfs rsync ssh bc zsh unzip mc \
        nfs-common man openssh-server htop tree pv udev pkg-config \
        graphviz desktop-file-utils usbutils p7zip-full p7zip-rar \
        systemd systemd-sysv apt-transport-https nautilus firefox \
        && \
    apt dist-upgrade -y && \
# Cmake
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
        2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg \
        >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \

# Cuda
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" && \

    apt update -y && \

    apt install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG cuda-compat-$CUDA_PKG \
        cuda-libraries-$CUDA_PKG cuda-nvtx-$CUDA_PKG \
        libcublas-$CUDA_PKG libnccl2 libnpp-$CUDA_PKG \
        cuda-nvml-dev-$CUDA_PKG cuda-command-line-tools-$CUDA_PKG \
        cuda-libraries-dev-$CUDA_PKG cuda-minimal-build-$CUDA_PKG \
        cuda-nvprof-$CUDA_PKG libnpp-dev-$CUDA_PKG \
        libcusparse-$CUDA_PKG libcusparse-dev-$CUDA_PKG \
        libnccl-dev libcublas-dev-$CUDA_PKG \
        libcudnn$CUDNN_VERSION libcudnn$CUDNN_VERSION-dev \
        libcutensor-dev libcutensor1 libcusolver10 \
        libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers7 \
        libnvinfer-plugin7 libnvinfer7 && \
    ln -s /usr/local/cuda-11.2 /usr/local/cuda && \
    PATH="/usr/local/cuda/bin:${PATH}" && \
    echo PATH="${PATH}" > /etc/environment && \

    apt install -y --no-install-recommends \
        build-essential cmake gcc g++ g++-multilib gcc-multilib \
        gfortran subversion nasm cmake-qt-gui libsm6 mercurial \
        nano vim emacs git tig tmux tzdata silversearcher-ag ctags cscope \
        jed libxrender1 lmodern netcat pandoc vlc \
        ffmpeg faac faad x264 x265 webcam guvcview mesa-utils v4l-utils qv4l2 \
        fonts-liberation default-jdk texlive-fonts-recommended neovim \
        texlive-latex-base texlive-latex-extra texlive-xetex \
        texlive-fonts-extra \
        libglib2.0-0 libblas-dev liblapack-dev libfreetype6-dev libpng-dev \
        libxext-dev libusb-1.0-0-dev libglew-dev libgirepository1.0-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev \
        libjpeg-dev libpng-dev libtiff-dev qt5-default libvtk6-dev \
        zlib1g-dev libwebp-dev libtiff5-dev libopenexr-dev libgdal-dev \
        libcups2-dev dumb-init libcanberra-gtk-module \
        libgl1-mesa-dev libglew-dev libegl1-mesa-dev \
        libwayland-dev libxkbcommon-dev wayland-protocols \
        libavcodec-dev libavutil-dev libavformat-dev libswscale-dev \
        libavdevice-dev libdc1394-22-dev libraw1394-dev libjpeg-dev \
        libpng-dev libtiff5-dev libopenexr-dev doxygen libopencv-dev \
        python3 python3-dev python3-pip python3-tk python3-opencv \
        python3-distutils-extra python3-xlib python3-apt \
        libqt5opengl5-dev libxmu-dev libboost-numpy-dev libboost-python-dev \
        && \
## Python 2.7
## python-xlib python python-dev python-pip python-qt4 python-setuptools
## python-requests python-opencv python-distutils-extra python-cairo
## python3-apt
## Slam2
# libeigen3-dev libpython2.7-dev ros-cmake-modules \
    ln -s /usr/lib/x86_64-linux-gnu/cmake/boost_numpy-1.71.0/boost_numpy-config.cmake \
          /usr/lib/x86_64-linux-gnu/cmake/boost_numpy-1.71.0/boost_numpy-py35-config.cmake && \

## Set locale and update prompt
## ============================
    LANG=en_US.UTF-8 && \
    LANGUAGE="en_US:en" && \
    LC_ALL=en_US.UTF-8 && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    echo 'export PS1="Docker-"$PS1' >> /etc/skel/.bashrc && \
    systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target && \

## Set Enviroments
#unlink /usr/bin/python 2>/dev/null && \
#unlink /usr/bin/python3 2>/dev/null && \
ln -sv /usr/bin/python3 /usr/bin/python && \
ln -sv /usr/bin/pip3 /usr/bin/pip && \
#echo "alias python=python3" >> /etc/bash.bashrc && \
#echo "alias pip=pip3" >> /etc/bash.bashrc && \
#alias python=python3 && \
#alias pip=pip3 && \
# Fix sudo -i problems with ubuntu 20.04
echo "Set disable_coredump false" > /etc/sudo.conf && \

## Install pycharm
## ===============
    add-apt-repository -y ppa:viktor-krivak/pycharm && \
    apt install -y pycharm && \
#    mkdir /opt/pycharm && \
#    cd /opt/pycharm && \
#    curl -L $PYCHARM_SOURCE -o installer.tgz && \
#    tar --strip-components=1 -xzf installer.tgz && \
#    rm installer.tgz && \
#    ln -s /opt/pycharm/bin/pycharm.sh /usr/local/bin/pycharm && \
#    umake ide pycharm /opt/pycharm && \
#    cp /opt/pycharm/bin/pycharm.sh /usr/local/bin/pycharm && \

## VSCode
## ======
    cd /tmp && \
    curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg && \
    install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/ && \
    echo "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" > /etc/apt/sources.list.d/vscode.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends code && \
    apt-get clean && \
    rm microsoft.gpg && \
    rm -rf /var/lib/apt/lists/* && \

## Pangolin
#    cd /root && \
#    git clone https://github.com/stevenlovegrove/Pangolin.git && \
#    cd Pangolin && \
#    mkdir build && \
#    cd build && \
#    cmake .. && \
#    cmake --build . && \
#    make && \
#    make install && \
#    cd ../.. && \
#    rm -rf /root/Pangolin && \
#    rm -rf /root/.cmake && \

## DBoW2
#    cd /root && \
#    git clone https://github.com/dorian3d/DBoW2.git && \
#    cd DBoW2 && \
#    mkdir build && \
#    cd build && \
#    cmake .. && \
#    make install && \
#    cd ../.. && \
#    rm -rf /root/DBoW2 && \
#    rm -rf /root/.cmake && \

## ROS
#    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
#    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
#    apt-get update && \
#    apt-get install -y ros-melodic-desktop-full ros-melodic-ros-base ros-cmake-modules && \
#    rosdep init && \
#    rosdep update && \

## Phidgets drivers
## ================
    cd /tmp && \
    wget https://www.phidgets.com/downloads/phidget22/libraries/linux/libphidget22.tar.gz && \
    tar xf libphidget22.tar.gz && \
    rm libphidget22.tar.gz && \
    cd libphidget22-* && \
    ./configure && \
    make -j40 && \
    make install && \
    cp plat/linux/udev/99-libphidget22.rules /etc/udev/rules.d && \
    cd /root && \
    rm -rf /tmp/libphidget22-*/* && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/tmp/* && \
    rm -rf /tmp/* && \
    rm -rf /var/cache/debconf/*.old && \
    rm -rf /var/log/* && \

## SSH server
## ==========
    mkdir /var/run/sshd && \
    sed 's/^#\?PasswordAuthentication .*$/PasswordAuthentication yes/g' -i /etc/ssh/sshd_config && \
    sed 's/^Port .*$/Port 9022/g' -i /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \

## Install Spark
## ===============
#    cd /tmp && \
#    wget -q http://mirrors.ukfast.co.uk/sites/ftp.apache.org/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
#    echo "E8B7F9E1DEC868282CADCAD81599038A22F48FB597D44AF1B13FCC76B7DACD2A1CAF431F95E394E1227066087E3CE6C2137C4ABAF60C60076B78F959074FF2AD *spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | sha512sum -c - && \
#    tar xzf spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /usr/local --owner root --group root --no-same-owner && \
#    rm spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
#    cd /usr/local && ln -s spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} spark

## Setup app folder
## ================
    mkdir /app && \
    chmod 777 /app

## Setup python environment
## ========================
RUN cd /root && \
#    curl -O tftp://132.68.58.130/tensorflow-2.5.0-cp38-cp38-linux_x86_64.whl && \
## Update all Python packages
#    python3 -m pip --no-color --no-cache-dir install -U --force-reinstall `python3 -m pip list --outdated | awk 'NR>2 {print $1}'` && \
    python3 -m pip --no-color --no-cache-dir install -U --force-reinstall \
        certifi idna Mako MarkupSafe numpy~=1.19.2 python-xlib requests six \
        ssh-import-id urllib3 wheel~=0.35 \
        && \
    python3 -m pip --no-color --no-cache-dir install -U --ignore-installed pyqt5 && \
    PATH=/usr/local/cuda/bin:$PATH && \
    python3 -m pip install --no-color --no-cache-dir -U \
        pyopengl pybind11 pynvim cython setuptools testresources \
        virtualenv ipython scipy matplotlib seaborn plotly \
        bokeh ggplot altair pandas protobuf ipdb flake8 \
        sympy nose sphinx tqdm opencv-contrib-python scikit-image \
        scikit-learn imageio grpcio~=1.34.0 \
        jupyter jupyterthemes jupyter_contrib_nbextensions jupyterlab \
        jupyterhub ipywidgets keras pillow toposort \
        pydarknetserver RISE cub kazam future \
        voxelmorph simpleitk neurite nibabel && \
    python3 -m pip install --no-color --no-cache-dir -U \
        tensorboard pycuda darknet tensorboardX \
        tf-nightly-gpu torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio \
        -f https://download.pytorch.org/whl/torch_stable.html \
        && \
#        /root/tensorflow-2.5.0-cp38-cp38-linux_x86_64.whl \
#    rm /root/tensorflow-2.5.0-cp38-cp38-linux_x86_64.whl && \
    rm -rf /root/.cache/pip/* && \

## Setup OpenCV version for Python 3
    python3 -c "import cv2 as cv" && \

## Import matplotlib the first time to build the font cache.
## =========================================================
    python3 -c "import matplotlib.pyplot" && \
    cp -r /root/.cache /etc/skel/ && \

## Setup Jupyter
## =============
    jupyter nbextension enable --py widgetsnbextension && \
    jupyter contrib nbextension install --system && \
    jupyter nbextensions_configurator enable && \
    jupyter serverextension enable --py jupyterlab --system && \
    jupyter-nbextension install rise --py --sys-prefix --system && \
    cp -r /root/.jupyter /etc/skel/ && \

## Create virtual environment
## ==========================
    cd /app/ && \
    virtualenv --system-site-packages dockvenv && \
    grep -rlnw --null /usr/local/bin/ -e '#!/usr/bin/python3' | xargs -0r cp -t /app/dockvenv/bin/ && \
    sed -i "s/#"'!'"\/usr\/bin\/python3/#"'!'"\/usr\/bin\/env python/g" /app/dockvenv/bin/* && \
    mv /app/dockvenv /root/ && \
    ln -sfT /root/dockvenv /app/dockvenv && \
    cp -rp /root/dockvenv /etc/skel/ && \

# Cleanup
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/debconf/*.old && \
    rm -rf /var/log/* && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/* && \
    echo PATH="${PATH}" > /etc/environment

## Install anaconda3
## =================
### Removed
#RUN wget $ANACONDA_SOURCE -O ~/anaconda.sh && \
#    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh && \
#    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#    PATH="$PATH:/opt/conda/bin" && \
#    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/skel/.bashrc && \
##    echo "conda activate base" >> /etc/skel/.bashrc && \
#    conda update conda -y && \
##    conda install opencv && \
##    conda install -c menpo \
##    conda install \
#      PyTorch matplotlib six rise \
#      virtualenv ipython numpy scipy matplotlib seaborn plotly \
#      bokeh pandas pyyaml protobuf flake8 cython \
#      sympy nose sphinx tqdm scikit-image jupyter jupyterlab \
#      scikit-learn imageio torchvision tensorflow-gpu tensorboard \
#      jupyterhub ipywidgets keras pillow tensorflow \
#      altair pyopengl pybind11 cython setuptools \
#      jupyter jupyterlab \
#      RISE tensorboard future && \
#      && \
# Missing packages
# ggplot ipdb opencv-contrib-python toposort pycuda pynvim testresources
# jupyterthemes jupyter_contrib_nbextensions darknet pydarknetserver cub kazam
#    /opt/conda/bin/pip install --no-color --no-cache-dir -U \
#        pip pycuda pyopengl pybind11 pynvim cython setuptools testresources \
#        virtualenv ipython scipy matplotlib seaborn plotly \
#        bokeh ggplot altair pandas pyyaml protobuf ipdb flake8 \
#        sympy nose sphinx tqdm opencv-contrib-python scikit-image \
#        scikit-learn imageio torchvision tensorboardX \
#        jupyter jupyterthemes jupyter_contrib_nbextensions jupyterlab \
#        jupyterhub ipywidgets keras pillow toposort torch \
#        darknet pydarknetserver RISE tensorboard cub future && \
##    conda install -c pytorch pytorch && \
#    cd /root && \
#    curl -O tftp://132.68.58.130/tensorflow-2.5.0-cp38-cp38-linux_x86_64.whl && \
#    /opt/conda/bin/pip install --no-color --no-cache-dir -U \
#        /root/tensorflow-2.5.0-cp38-cp38-linux_x86_64.whl \
#        && \
#    rm /root/tensorflow-2.5.0-cp38-cp38-linux_x86_64.whl && \

#    conda update --all -y && \
#    anaconda-navigator --reset && \
#    conda clean -a -y && \

## Create dockuser user
## ====================
RUN groupadd -g $DOCKUSER_GID dockuser && \
    useradd --system --create-home --home /home/dockuser --shell /bin/bash -G sudo -g dockuser -u $DOCKUSER_UID dockuser && \
    mkdir /tmp/runtime-dockuser && \
    chown dockuser:dockuser /tmp/runtime-dockuser && \
    echo "dockuser ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers && \

## Copy scripts
## ============
    mkdir /app/bin && \
    chmod a=u -R /app/bin && \
    touch /etc/skel/.sudo_as_admin_successful && \

#    cd /tmp && \
#    curl -O tftp://132.68.58.130/docker/entrypoint.sh && \
#    curl -O tftp://132.68.58.130/docker/default_notebook.sh && \
#    curl -O tftp://132.68.58.130/docker/default_jupyterlab.sh && \
#    curl -O tftp://132.68.58.130/docker/run_server.sh && \
#    cp entrypoint.sh /app/bin/run && \
#    cp default_notebook.sh /app/bin/default_notebook && \
#    cp default_jupyterlab.sh /app/bin/default_jupyterlab && \
#    cp run_server.sh /app/bin/run_server && \
#    chmod 755 /app/bin/run && \
#    chmod 755 /app/bin/default_notebook && \
#    chmod 755 /app/bin/default_jupyterlab && \
#    chmod 755 /app/bin/run_server && \

# Fix missing /opt/conda/etc/profile.d/conda.sh
    mkdir /opt/conda/ && \
    mkdir /opt/conda/etc/ && \
    mkdir /opt/conda/etc/profile.d/ && \
    echo "#!/bin/bash" > /opt/conda/etc/profile.d/conda.sh && \
    chmod 755 /opt/conda/etc/profile.d/conda.sh && \

## Cleanup
## ============
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/debconf/*.old && \
    rm -rf /var/log/* && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/* && \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64 && \
    export PATH=/app/bin:/app/dockvenv/bin:$PATH && \
    echo "export PATH=$PATH" > /etc/environment

COPY ./resources/entrypoint.sh /app/bin/run
COPY ./resources/default_notebook.sh /app/bin/default_notebook
COPY ./resources/default_jupyterlab.sh /app/bin/default_jupyterlab
COPY ./resources/run_server.sh /app/bin/run_server

WORKDIR /root
ENTRYPOINT ["/usr/bin/dumb-init", "--", "/app/bin/run"]
#COPY Dockerfile /root
