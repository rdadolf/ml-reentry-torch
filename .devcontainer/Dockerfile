FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        clang \
        lldb \
        lld \
        clang-format \
        clang-tidy \
        make \
        cmake \
        ninja-build \
        git \
        curl \
        ssh \
        vim \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

ENV CC=clang
ENV CXX=clang++

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

# Rename the base image's 'ubuntu' user (UID 1000) to 'devel'
RUN usermod -l devel -d /home/devel -m ubuntu \
    && groupmod -n devel ubuntu

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"
RUN python -m venv /opt/venv && chown -R devel:devel /opt/venv

WORKDIR /x/workspace
RUN chown devel:devel /x/workspace
COPY --chown=devel:devel pyproject.toml ./
USER devel
RUN uv sync --extra dev
