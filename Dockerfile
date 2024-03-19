# How to build this image:
# docker build -t silleellie/laikallm:latest  https://github.com/silleellie/laikallm.git#main
# How to run it:
# docker run -t -d silleellie/laikallm

# ---- Base pytorch ----
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /LaikaLLM

# upgrade pip and install app dependencies
# we don't copy and install requirements since we are starting from pytorch image,
# no need to reinstall torch
RUN pip install -U pip

RUN pip install --no-cache-dir --upgrade  \
    transformers[torch]~=4.33.1 \
    wandb~=0.15.2 \
    pandas~=2.1.2 \
    requests \
    numpy~=1.24.3 \
    tqdm \
    datasets~=2.14.6 \
    pygit2 \
    pyyaml \
    cytoolz \
    yaspin

# copy src folder to docker image and relevant files
COPY . /LaikaLLM/

# separate install for gdown and after file copy since it should
# be executed at each docker build,
RUN pip install --upgrade gdown>5.0.0

ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:16:8
