"""
Linear programming allocator for batch coupon assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linprog


@dataclass
class AllocationProblem:
    """
    Encapsulates the inputs required for the linear program.
    """

    user_ids: Sequence[str]
    coupon_ids: Sequence[str]
    expected_profit: np.ndarray  # shape (n_users, n_coupons)
    subsidy_cost: np.ndarray  # shape (n_users, n_coupons)
    budget: float
    exposure_cap: float = 1.0


class LinearAllocator:
    """
    Solves for the assignment of coupons that maximises uplift profit.
    """

    def solve(self, problem: AllocationProblem) -> np.ndarray:
        n_users = len(problem.user_ids)
        n_coupons = len(problem.coupon_ids)
        n_variables = n_users * n_coupons

        c = -problem.expected_profit.flatten()

        # Budget constraint
        A = problem.subsidy_cost.reshape(1, n_variables)
        b = np.array([problem.budget])

        # Exposure cap: each user receives at most `exposure_cap` coupons
        A_user = np.zeros((n_users, n_variables))
        for u in range(n_users):
            for c_idx in range(n_coupons):
                col = u * n_coupons + c_idx
                A_user[u, col] = 1.0
        A = np.vstack([A, A_user])
        b = np.concatenate([b, np.full(n_users, problem.exposure_cap)])

        bounds = [(0.0, 1.0) for _ in range(n_variables)]

        result = linprog(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Linear programming failed: {result.message}")

        assignment = result.x.reshape(n_users, n_coupons)
        return assignment
