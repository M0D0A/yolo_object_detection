FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04


# python
RUN apt-get update && apt-get install -y \
   build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libxml2-dev \
    libxslt1-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

RUN pyenv install 3.11.9 \
    && pyenv global 3.11.9


# requirements
WORKDIR /project/yolo_object_detection
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
RUN pip install -r requirements.txt


# file structure
COPY . .
RUN mkdir datasets
RUN mkdir models

CMD ["python", "main.py"]
