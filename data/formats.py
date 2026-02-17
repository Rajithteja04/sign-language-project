from dataclasses import dataclass
from typing import List

@dataclass
class SequenceExample:
    features: list  # list of per-frame feature vectors
    label: str      # gloss or sentence

@dataclass
class DatasetSplit:
    train: List[SequenceExample]
    val: List[SequenceExample]
    test: List[SequenceExample]
