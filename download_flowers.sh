#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "usage download_flowers.sh [data dir]"
  exit
fi

# Create the output and temporary directories.
DATA_DIR="${1%/}"
SCRATCH_DIR="${DATA_DIR}/raw-data/"
mkdir -p "${DATA_DIR}"
mkdir -p "${SCRATCH_DIR}"
WORK_DIR="$0.runfiles/inception/inception"

# Download the flowers data.
DATA_URL="http://download.tensorflow.org/example_images/flower_photos.tgz"
CURRENT_DIR=$(pwd)
cd "${DATA_DIR}"
TARBALL="flower_photos.tgz"
if [ ! -f ${TARBALL} ]; then
  echo "Downloading flower data set."
  wget -O ${TARBALL} "${DATA_URL}"
else
  echo "Skipping download of flower data."
fi

# Note the locations of the train and validation data.
TRAIN_DIRECTORY="${SCRATCH_DIR}train/"
VALIDATION_DIRECTORY="${SCRATCH_DIR}validation/"

# Expands the data into the flower_photos/ directory and rename it as the
# train directory.
tar xf flower_photos.tgz
rm -rf "${TRAIN_DIRECTORY}" "${VALIDATION_DIRECTORY}"
mv flower_photos "${TRAIN_DIRECTORY}"

# Generate a list of 5 labels: daisy, dandelion, roses, sunflowers, tulips
LABELS_FILE="${SCRATCH_DIR}/labels.txt"
ls -1 "${TRAIN_DIRECTORY}" | grep -v 'LICENSE' | sed 's/\///' | sort > "${LABELS_FILE}"

# Generate the validation data set.
while read LABEL; do
  VALIDATION_DIR_FOR_LABEL="${VALIDATION_DIRECTORY}${LABEL}"
  TRAIN_DIR_FOR_LABEL="${TRAIN_DIRECTORY}${LABEL}"

  # Move the first randomly selected 100 images to the validation set.
  mkdir -p "${VALIDATION_DIR_FOR_LABEL}"
  VALIDATION_IMAGES=$(ls -1 "${TRAIN_DIR_FOR_LABEL}" | shuf | head -100)
  for IMAGE in ${VALIDATION_IMAGES}; do
    mv -f "${TRAIN_DIRECTORY}${LABEL}/${IMAGE}" "${VALIDATION_DIR_FOR_LABEL}"
  done
done < "${LABELS_FILE}"
