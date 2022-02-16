# FROM nvcr.io/nvidia/pytorch:21.12-py3
FROM nvcr.io/nvidia/pytorch:20.12-py3
# FROM nvcr.io/nvidia/pytorch:19.12-py3
# FROM nvcr.io/nvidia/pytorch:18.12.1-py3

LABEL maintainer="kostonerx@gmail.com"

ARG TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    cmake \
    build-essential \
    libgl1-mesa-dev \
    freeglut3-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN conda update -n base -c defaults conda && conda init

RUN conda install -y -q \
    click \
    imageio \
    joblib \
    matplotlib \
    Pillow \
    plotly \
    pybind11

RUN pip install \
    gtimer \
    gym \
    moviepy \
    opencv-python \
    pyquaternion \
    Shapely \
    sk-video

RUN conda clean -a -y -q

ENV PYFLEXROOT=/workspace/softgym/PyFlex
ENV PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
ENV LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

WORKDIR /workspace
