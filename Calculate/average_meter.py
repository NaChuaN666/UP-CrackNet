"""
Average meters
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class AverageMeter(object):
    """
    Class to be an average meter
    """

    def __init__(self):
        self.current_value = 0
        self.average_value = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """
        Reset average meter.
        """
        self.current_value = 0
        self.average_value = 0
        self.sum = 0
        self.count = 0

    def update(self, current_value, increment=1):
        """
        Update average meter by given current value and number of increment.
        """
        self.current_value = current_value
        self.sum += current_value * increment
        self.count += increment
        self.average_value = self.sum / self.count

    @property
    def val(self):
        """
        Return average value of the average meter
        """
        return self.average_value
