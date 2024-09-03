# syntax=docker/dockerfile:1.2
FROM python:3.10-bullseye

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ARG CATEGORY=ttt
ARG DEVICE=cuda

# devcontainer Options
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
COPY library-scripts/*.sh library-scripts/*.env /tmp/library-scripts/

#Step 1: Install deps
RUN --mount=type=cache,target=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache,sharing=locked \
    apt-get update \
    && apt-get upgrade --assume-yes \
    && apt-get install --assume-yes --no-install-recommends \
        wget \
        curl \
        # ffmpeg for unstructured
        ffmpeg \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt remove -y curl \
    && apt autoremove -y \
    && apt clean \    
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${PATH}"
RUN poetry config virtualenvs.create false

# Step 3: Install project deps
WORKDIR /src
COPY ../pyproject.toml ../poetry.lock /src/
RUN --mount=type=cache,target=/root/.cache/pip \
    poetry export --output=requirements.txt \
    && pip install  --disable-pip-version-check --no-deps --requirement=/app/requirements.txt --only-binary :all:

# # Step 4: Install from wheel
# RUN poetry build -f wheel
# RUN pip install dist/*.whl
# RUN python -m nltk.downloader punkt
# RUN python -m nltk.downloader punkt_tab
# RUN python -m nltk.downloader averaged_perceptron_tagger_eng

# # Step 5: Startup
# RUN echo '{"app_dir":"/app/.gai"}' > /root/.gairc
# VOLUME /app/.gai
# ENV MODEL_PATH="/app/.gai/models"
# ENV CATEGORY=${CATEGORY}
# WORKDIR /app/src/gai/rag/server/api
# CMD ["bash","-c","python main.py"]