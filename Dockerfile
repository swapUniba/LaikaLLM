# How to build this image:
# docker build -t silleellie/laikallm:latest -t silleellie/laikallm:<TAG>  https://github.com/silleellie/laikallm.git#<TAG>
# How to run it:
# docker run -t -d silleellie/laikallm:latest

# ---- Base pytorch ----
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /LaikaLLM

# copy only requirements so to cache layers
COPY requirements.txt /LaikaLLM/requirements.txt

# upgrade pip
RUN pip install -U pip

# since we have as base image "pytorch" we can avoid installing again it,
# so we start installing requirements from the 4th line onwards
RUN sed -n '4,$p' <requirements.txt >requirements-docker.txt

# install app dependencies. We are ok in installing each time all the dependencies
# upon docker build (if the source code changes) to avoid listing again all requirements here
# since requirements install is relatively lightweight
RUN pip install --no-cache-dir -U -r requirements-docker.txt && rm requirements-docker.txt

# copy src folder to docker image and relevant files
COPY . /LaikaLLM/

# set environment variables for reproducibility
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:16:8
