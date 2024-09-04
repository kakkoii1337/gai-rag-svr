FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ARG CATEGORY=rag
ARG DEVICE=cuda

ENV HOME_PATH="/root"
ENV PROJECT_NAME="gai-rag-svr"
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

# Step 1: Install poetry and ffmpeg for unstructured
RUN apt update && apt install -y \
    ffmpeg \
    && apt remove -y curl \
    && apt autoremove -y \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Copy Source Code
WORKDIR /workspaces/${PROJECT_NAME}
COPY src/gai/rag src/gai/rag
COPY pyproject.toml ./
RUN rm src/gai/rag/server/api/gai.yml

# Step 4: Install project
RUN pip install -e .
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader averaged_perceptron_tagger_eng

# Step 5: Startup
RUN echo '{"app_dir":"/root/.gai"}' > /root/.gairc
VOLUME /root/.gai
ENV MODEL_PATH="/root/.gai/models"
ENV CATEGORY=${CATEGORY}
WORKDIR /workspaces/${PROJECT_NAME}/src/gai/rag/server/api

# Install debugpy and start
RUN echo 0
RUN pip install debugpy
COPY startup.sh .
CMD ["bash","startup.sh"]