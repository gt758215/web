FROM nvidia/cuda:9.0-devel-ubuntu16.04

ARG BRANCH
ENV BRANCH ${BRANCH:-master}

RUN apt-get update && apt-get install -y \
    curl \
    openjdk-8-jdk \
    python-dev \
    python-pip \
    python-tk \
    vim \
    git \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libcudnn7-dev=7.0.5.15-1+cuda9.0 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone -b ${BRANCH} https://github.com/myelintek/web.git /build/ && \
    pip install --upgrade pip==9.0.3 && pip install numpy && pip install -e /build/

RUN python -m digits.download_data cifar10 /cifar10 && \
    python -m digits.download_data mnist /mnist

RUN pip install tensorflow-gpu

EXPOSE 2500
ENV DIGITS_JOBS_DIR=/data

ADD run.sh /root/.bashrc

CMD ["/usr/local/bin/digits-devserver"]
