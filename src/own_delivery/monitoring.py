"""
Monitoring utilities for coupon campaign performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


def redemption_metrics(events: pd.DataFrame) -> Dict[str, float]:
    """
    Compute basic KPIs from redemption logs.
    """

    total_exposed = len(events)
    redeemed = events["redeemed"].sum()
    redemption_rate = redeemed / total_exposed if total_exposed else 0.0
    avg_subsidy = events.loc[events["redeemed"], "subsidy_cost"].mean()
    avg_margin = events.loc[events["redeemed"], "order_margin"].mean()

    return {
        "exposures": float(total_exposed),
        "redemption_rate": float(redemption_rate),
        "avg_subsidy": float(avg_subsidy if not np.isnan(avg_subsidy) else 0.0),
        "avg_margin": float(avg_margin if not np.isnan(avg_margin) else 0.0),
    }


def incremental_roi(events: pd.DataFrame) -> float:
    """
    Estimate ROI = (incremental_margin - subsidy) / subsidy.
    """

    incremental_margin = events["incremental_margin"].sum()
    subsidy = events["subsidy_cost"].sum()
    if subsidy == 0:
        return float("inf") if incremental_margin > 0 else 0.0
    return float((incremental_margin - subsidy) / subsidy)


@dataclass
class BayesianGuardrail:
    """
    Monitors redemption rate vs baseline using Beta posterior.
    """

    alpha_prior: float = 1.0
    beta_prior: float = 1.0

    def posterior_probability(
        self, successes: int, trials: int, baseline_rate: float
    ) -> float:
        import scipy.stats

        alpha_post = self.alpha_prior + successes
        beta_post = self.beta_prior + trials - successes
        posterior = scipy.stats.beta(alpha_post, beta_post)
        prob = posterior.cdf(baseline_rate)
        return float(prob)
