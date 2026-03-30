"""Action selection: one interface, multiple strategies.

Invariants:
- Strategy is chosen from engine state (mood), not by a separate meta-agent
- Commitment is a reading, not a decision
- mood = (temperature, kappa_dot) influences scoring continuously
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from substrate.state import Mood


@dataclass
class Candidate:
    """A candidate action with associated metadata."""
    action: Any  # the action payload (format depends on environment)
    score: float = 0.0
    commitment: float = 0.0  # how much coherence this would commit


class Strategy(ABC):
    @abstractmethod
    def select(self, candidates: list[Candidate], mood: Mood) -> Candidate:
        ...


class GreedyStrategy(Strategy):
    """Pick highest-scoring candidate."""
    def select(self, candidates: list[Candidate], mood: Mood) -> Candidate:
        return max(candidates, key=lambda c: c.score)


class MetropolisStrategy(Strategy):
    """Metropolis-Hastings: accept worse candidates with probability exp(-dE/T)."""
    def select(self, candidates: list[Candidate], mood: Mood) -> Candidate:
        if not candidates:
            raise ValueError("No candidates")
        # Sort by score descending
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        best = ranked[0]
        T = max(mood.temperature, 0.01)
        for cand in ranked:
            delta = best.score - cand.score
            if delta <= 0 or np.random.random() < np.exp(-delta / T):
                return cand
        return best


class RepairStrategy(Strategy):
    """Retreat to highest-commitment candidate (most coherent with current state)."""
    def select(self, candidates: list[Candidate], mood: Mood) -> Candidate:
        return max(candidates, key=lambda c: c.commitment)


class ActionSelector:
    """Unified action selector. Strategy chosen continuously from mood.

    - Low T, positive kappa_dot → Greedy (exploiting)
    - High T, low kappa_dot → Metropolis (exploring)
    - Negative kappa_dot → Repair (retreating)
    """

    def __init__(self):
        self._greedy = GreedyStrategy()
        self._metropolis = MetropolisStrategy()
        self._repair = RepairStrategy()

    def select_action(self, candidates: list[Candidate], mood: Mood) -> Candidate:
        if not candidates:
            raise ValueError("No candidates to select from")

        strategy = self._pick_strategy(mood)
        return strategy.select(candidates, mood)

    def _pick_strategy(self, mood: Mood) -> Strategy:
        """Continuous regime selection from mood."""
        if mood.kappa_dot < -0.05:
            return self._repair
        if mood.temperature > 2.0:
            return self._metropolis
        # Blend: higher T → more likely Metropolis
        if mood.temperature > 0.5 and mood.kappa_dot < 0.01:
            return self._metropolis
        return self._greedy

    @property
    def strategy_name(self) -> str:
        """For logging."""
        return "ActionSelector"
