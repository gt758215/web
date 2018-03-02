#!/bin/bash

xcode-select --install
brew install jpeg-turbo

ROOT_DIR=$(dirs -l)
INSTALL_DIR="ROOT_DIR"/"caffe"

# get source
git clone "https://github.com/myelintek/caffe.git" "$INSTALL_DIR" --depth 1


# configure project
cp $(ROOT_DIR)/scripts/Makefile-mac.config $(INSTALL_DIR)/Makefile.config
cd $(INSTALL_DIR)
make all

mkdir -p "${INSTALL_DIR}/build"
cd "${INSTALL_DIR}/build"
cmake .. -DCPU_ONLY=On -DBLAS=Open

# build
make --jobs="$(nproc)"

# mark cache
WEEK=$(date +%Y-%W)
echo "$WEEK" > "${INSTALL_DIR}/cache-version.txt"

