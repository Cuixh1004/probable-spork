"""
Budget-constrained multi-armed bandit for coupon allocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class BanditArmState:
    """
    Tracks posterior parameters for an arm under Beta distribution.
    """

    successes: float = 1.0
    failures: float = 1.0
    cost_mean: float = 1.0
    cost_var: float = 1.0

    def sample_roi(self) -> float:
        reward = np.random.beta(self.successes, self.failures)
        cost = max(np.random.normal(self.cost_mean, np.sqrt(self.cost_var)), 1e-6)
        return reward / cost


@dataclass
class BudgetedThompsonSampling:
    """
    Selects coupon arms while respecting a per-round budget.
    """

    arms: Dict[str, BanditArmState] = field(default_factory=dict)
    round_budget: float = 1.0

    def register_arm(self, arm_id: str, initial_cost: float) -> None:
        if arm_id in self.arms:
            return
        self.arms[arm_id] = BanditArmState(cost_mean=initial_cost, cost_var=initial_cost ** 2)

    def select(self, costs: Dict[str, float]) -> List[str]:
        """
        Return a list of arm ids to serve this round under budget constraint.
        """

        sampled_roi = {
            arm_id: state.sample_roi() for arm_id, state in self.arms.items()
        }
        sorted_arms = sorted(
            sampled_roi.items(), key=lambda item: item[1], reverse=True
        )

        allocated: List[str] = []
        spend = 0.0
        for arm_id, _ in sorted_arms:
            cost = costs.get(arm_id, self.arms[arm_id].cost_mean)
            if spend + cost <= self.round_budget:
                allocated.append(arm_id)
                spend += cost
        return allocated

    def update(
        self,
        arm_id: str,
        reward: float,
        cost: float,
        step_size: float = 0.1,
    ) -> None:
        """
        Update posterior parameters with observed reward and cost.
        """

        state = self.arms[arm_id]
        state.successes += reward
        state.failures += 1.0 - reward

        # Exponential moving average for cost parameters
        state.cost_mean = (1 - step_size) * state.cost_mean + step_size * cost
        state.cost_var = max(
            (1 - step_size) * state.cost_var + step_size * (cost - state.cost_mean) ** 2,
            1e-6,
        )
