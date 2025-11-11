"""
Touchpoint assignment optimizer using Hungarian algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class TouchpointProblem:
    user_ids: Sequence[str]
    touchpoints: Sequence[str]
    priority_matrix: np.ndarray  # shape (n_users, n_touchpoints)


def assign_touchpoints(problem: TouchpointProblem) -> Dict[str, str]:
    """
    Solve maximum weight matching by converting to cost minimisation.
    """

    max_priority = np.max(problem.priority_matrix)
    cost_matrix = max_priority - problem.priority_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    allocation: Dict[str, str] = {}
    for r, c in zip(row_ind, col_ind):
        allocation[problem.user_ids[r]] = problem.touchpoints[c]
    return allocation
