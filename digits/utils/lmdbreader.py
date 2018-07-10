# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .datasetreader import DataReader
import lmdb
from digits.tools.tf import caffe_tf_pb2
import PIL.Image
# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
class DbReader(DataReader):
    """
    Reads a database
    """

    def __init__(self, location):
        """
        Arguments:
        location -- where is the database
        """
        self._db = lmdb.open(
            location,
            map_size=1024**3,  # 1MB
            readonly=True,
            lock=False)

        with self._db.begin() as txn:
            self.total_entries = txn.stat()['entries']

        self.txn = self._db.begin()

    def entries(self):
        """
        Generator returning all entries in the DB
        """
        with self._db.begin() as txn:
            cursor = txn.cursor()
            for item in cursor:
                yield item

    def parsed_entries(self):
        with self._db.begin() as txn:
            cursor = txn.cursor()
            for item in cursor:
                key, value = item

                datum = caffe_tf_pb2.Datum()
                datum.ParseFromString(value)
                if datum.encoded:
                    s = StringIO()
                    s.write(datum.data)
                    s.seek(0)
                    img = PIL.Image.open(s)
                else:
                    import caffe.io
                    arr = caffe.io.datum_to_array(datum)
                    # CHW -> HWC
                    arr = arr.transpose((1, 2, 0))
                    if arr.shape[2] == 1:
                        # HWC -> HW
                        arr = arr[:, :, 0]
                    elif arr.shape[2] == 3:
                        # BGR -> RGB
                        # XXX see issue #59
                        arr = arr[:, :, [2, 1, 0]]
                    img = PIL.Image.fromarray(arr)

                data = {
                    "label": datum.label,
                    "img": img,
                }
                yield data

    def entry(self, key):
        """Return single entry"""
        return self.txn.get(key)
