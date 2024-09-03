# syntax=docker/dockerfile:1.2

FROM torch2.2.0-cuda12.1-ubuntu22.04 as build


# Build Final ----------------------------------------------------------------------------------------

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS base
ARG CATEGORY=rag
ARG DEVICE=cuda
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

# Step 1: Install poetry and ffmpeg for unstructured
RUN apt update && apt install -y \
    curl \
    ffmpeg \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt remove -y curl \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry config virtualenvs.create false

# Step 3: Copy Source Code
WORKDIR /app
COPY src/gai/rag src/gai/rag
COPY pyproject.toml poetry.lock ./

# Step 4: Install from wheel
RUN poetry build -f wheel
RUN pip install dist/*.whl
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader averaged_perceptron_tagger_eng

# Step 5: Startup
RUN echo '{"app_dir":"/root/.gai"}' > /root/.gairc
VOLUME /root/.gai
ENV MODEL_PATH="/root/.gai/models"
ENV CATEGORY=${CATEGORY}
WORKDIR /workspaces/${PROJECT_NAME}/src/gai/rag/server/api


CMD ["bash","-c","python main.py"]