#!/bin/bash

if [ ! -d 'digits' ]; then
  echo "Can't find directory 'digits'!"
  exit 1
fi

mkdir -p digits/jobs
cd digits/jobs
curl -L https://www.dropbox.com/s/kj9t5631fpoz5yv/mnist.tar.gz | tar -zx
