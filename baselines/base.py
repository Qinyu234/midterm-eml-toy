"""Unified fit result for benchmark comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FitResult:
    method: str
    target_id: str
    in_rmse: float
    ood_rmse: float
    n_params: int
    complexity: float
    steps_to_threshold: Optional[int]
    total_steps: int
    compute: float
    tes: float
    ood_ratio: float = field(default=0.0)
    meta: Dict[str, Any] = field(default_factory=dict)
    rmse_curve: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.in_rmse > 0 and self.ood_ratio == 0.0:
            self.ood_ratio = self.ood_rmse / self.in_rmse
