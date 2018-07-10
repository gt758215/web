# Copyright (c) 2018. ITRI.  All rights reserved.
from __future__ import absolute_import

from .datasetreader import DataReader

import tensorflow as tf
import PIL
from digits.log import logger

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

class TFRecordReader(DataReader):
    """
    Reads a database
    """

    def __init__(self, location):
        """
        Arguments:
        location -- where is the database
        """
        tf_file_path = '%s/train-*' % location
        self.tf_shard_files = tf.gfile.Glob(tf_file_path)
        self.total_entries = 0
        for shard_path in self.tf_shard_files:
            for r in tf.python_io.tf_record_iterator(shard_path):
                self.total_entries += 1

    def parsed_entries(self):
        """
        Generator returning all entries in the TFRecords
        """
        example = tf.train.Example()

        for shard_path in self.tf_shard_files:
            for r in tf.python_io.tf_record_iterator(shard_path):
                example.ParseFromString(r)
                label = int(example.features.feature['image/class/label']
                            .int64_list
                            .value[0])
                img_string = (example.features.feature['image/encoded'].bytes_list.value[0])

                s = StringIO()
                s.write(img_string)
                s.seek(0)
                img = PIL.Image.open(s)

                data = {
                    "label": label,
                    "img": img,
                }
                yield data
