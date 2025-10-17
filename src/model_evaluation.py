from dataclasses import dataclass
from typing import Self

@dataclass
class ModelMetrics:
    accuracy: float
    f1: float


    def compare_accuracies(self, comparision_metrics: Self) -> bool:
        return self.accuaracy > comparision_metrics.accuracy


class ModelEvaluator:
    def __init__(self):
        