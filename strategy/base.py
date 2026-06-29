"""Generation strategy stubs (DESIGN §0). / 生成策略接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from codec.node import decode_node, is_evaluable


class GenerationStrategy(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_nodes(self, budget: int) -> List[int]: ...


class G0Random(GenerationStrategy):
    @property
    def name(self) -> str:
        return 'G0'

    def generate_nodes(self, budget: int) -> List[int]:
        import random
        pool = [b for b in range(256) if is_evaluable(b)]
        return random.sample(pool, min(budget, len(pool)))


class G1BeamSearch(GenerationStrategy):
    @property
    def name(self) -> str:
        return 'G1'

    def generate_nodes(self, budget: int) -> List[int]:
        return sorted(
            (b for b in range(256) if is_evaluable(b)),
            key=lambda b: decode_node(b).n_eml,
        )[:budget]


class G2Adaptive(GenerationStrategy):
    @property
    def name(self) -> str:
        return 'G2'

    def generate_nodes(self, budget: int) -> List[int]:
        return [b for b in range(256) if is_evaluable(b)][:budget]
