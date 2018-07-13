# Copyright (c) 2018, ITRI.  All rights reserved.
from __future__ import absolute_import


class DataReader(object):
    """
    inteface of  read dataset data in records
    """

    def entries(self):
        """
        Generator returning all entries in the DB
        """
        raise NotImplementedError

    def parsed_entries(self):
        """
        Generator returning all entries in the DB
        """
        raise NotImplementedError

    def entry(self, key):
        """Return single entry"""
        raise NotImplementedError

