FROM continuumio/anaconda3

# Apt installations
RUN apt update && \
    apt install -y sudo git wget curl build-essential

# Python Libraries Installations
RUN pip install --upgrade pip && \
    pip install \
    numpy \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image \
    seaborn \
    jupyterlab \
    black \
    flake8

WORKDIR /workspace
