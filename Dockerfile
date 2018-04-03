FROM nvidia/cuda:9.0-devel-ubuntu16.04

ARG BRANCH
ENV BRANCH ${BRANCH:-master}

RUN apt-get update && apt-get install -y \
    curl \
    openjdk-8-jdk \
    protobuf-compiler \
    python-dev \
    python-pip \
    python-tk \
    vim \
    git \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libcudnn7-dev=7.0.5.15-1+cuda9.0 \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN git clone -b ${BRANCH} https://github.com/myelintek/web.git /build/ && \
    pip install --upgrade pip && pip install numpy && pip install -e /build/

RUN pip install tensorflow-gpu

RUN git clone https://github.com/myelintek/tensorpack.git /tensorpack && \
    pip install --upgrade opencv-python setuptools && pip install -e /tensorpack

EXPOSE 2500
ENV DIGITS_JOBS_DIR=/data
ENV TENSORPACK_DATASET=${DIGITS_JOBS_DIR}/tensorpack_data

ADD run.sh /root/.bashrc

CMD ["/usr/local/bin/digits-devserver"]
