from enum import Enum


class AggregationMode(Enum):
    OVER_FLOW = 1
    OVER_CLASSIFIER = 2

    def __str__(self):
        return self.name

    @property
    def counter(self):
        return self.value


DEFAULT_AGGREGATION_MODE = AggregationMode.OVER_FLOW
