"""
End-to-end demo wiring together the own_delivery components.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import (
    bandit,
    clv,
    feature_engineering,
    linear_allocator,
    price_propensity,
    state_machine,
    touchpoint_assignment,
    uplift,
)


def synthetic_order_history(num_users: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    user_ids = [f"user_{i}" for i in range(num_users)]
    records = []
    for user in user_ids:
        orders = rng.integers(1, 10)
        for _ in range(orders):
            order_value = rng.normal(80, 20)
            records.append(
                {
                    "order_id": rng.integers(1e6),
                    "user_id": user,
                    "order_ts": pd.Timestamp("2025-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 90))),
                    "order_value": max(order_value, 20),
                    "gross_margin": max(order_value * rng.uniform(0.15, 0.35), 5),
                    "discount_cost": max(order_value * rng.uniform(0.05, 0.20), 0),
                }
            )
    return pd.DataFrame(records)


def build_models():
    orders = synthetic_order_history()
    base_features = feature_engineering.default_user_features(orders)
    target_clv = (
        base_features["orders_90d"] * base_features["avg_order_value"] * 0.3
        - base_features["discount_cost_90d"]
    )

    clv_model = clv.CLVModel(
        feature_names=[
            "orders_90d",
            "avg_order_value",
            "gross_margin_rate",
            "discount_cost_90d",
            "recency_days",
        ]
    )
    clv_model.fit(base_features, target_clv)
    clv_scores = clv_model.predict(base_features)

    promo_df = base_features.copy()
    promo_df["channel"] = "app"
    promo_df["user_segment"] = pd.cut(
        base_features["orders_90d"], bins=[0, 2, 5, 10], labels=["low", "mid", "high"]
    )
    promo_df["redeemed"] = np.random.binomial(1, 0.3, size=len(promo_df))

    propensity_model = price_propensity.PromoPropensityModel(
        numeric_features=["orders_30d", "orders_90d", "avg_order_value"],
        categorical_features=["channel", "user_segment"],
    )
    propensity_model.fit(promo_df, target_col="redeemed")
    propensity = propensity_model.predict(promo_df)

    features = base_features.assign(
        clv_score=clv_scores,
        promo_propensity=propensity,
    )
    segments = feature_engineering.encode_segments(features)
    features["segment"] = segments

    return features, clv_model, propensity_model


def run_bandit():
    bandit_policy = bandit.BudgetedThompsonSampling(round_budget=100.0)
    bandit_policy.register_arm("coupon_small", initial_cost=5.0)
    bandit_policy.register_arm("coupon_large", initial_cost=20.0)

    costs = {"coupon_small": 5.0, "coupon_large": 20.0}
    chosen = bandit_policy.select(costs)
    for arm in chosen:
        # Simulated outcomes
        reward = np.random.binomial(1, 0.5 if arm == "coupon_large" else 0.3)
        cost = costs[arm]
        bandit_policy.update(arm, reward=reward, cost=cost)
    return bandit_policy


def run_allocator(features: pd.DataFrame):
    coupons = ["coupon_small", "coupon_large"]
    expected_profit = np.random.uniform(1, 5, size=(len(features), len(coupons)))
    subsidy = np.array([[5.0, 20.0]])
    subsidy = np.repeat(subsidy, repeats=len(features), axis=0)

    problem = linear_allocator.AllocationProblem(
        user_ids=list(features.index),
        coupon_ids=coupons,
        expected_profit=expected_profit,
        subsidy_cost=subsidy,
        budget=1000.0,
        exposure_cap=1.0,
    )
    allocator = linear_allocator.LinearAllocator()
    assignment = allocator.solve(problem)
    return assignment


def run_state_machine():
    config = state_machine.InflationConfig(
        base_value=10.0, share_increment=5.0, interaction_increment=2.0, cap=30.0
    )
    sm = state_machine.CouponInflationStateMachine(config)
    user_id = "user_1"
    sm.register_interaction(user_id)
    sm.register_share(user_id)
    sm.register_share(user_id)
    sm.register_share(user_id)
    increased_value = sm.register_share(user_id)
    final_value = sm.register_order(user_id)
    return increased_value, final_value


def run_touchpoint_assignment(features: pd.DataFrame):
    users = list(features.index[:3])
    touchpoints = ["banner", "push", "coupon_center"]
    priority = np.random.uniform(0, 1, size=(len(users), len(touchpoints)))
    problem = touchpoint_assignment.TouchpointProblem(
        user_ids=users, touchpoints=touchpoints, priority_matrix=priority
    )
    return touchpoint_assignment.assign_touchpoints(problem)


def main():
    features, clv_model, propensity_model = build_models()
    bandit_policy = run_bandit()
    assignment = run_allocator(features.head(10))
    increased_value, final_value = run_state_machine()
    touchpoint_alloc = run_touchpoint_assignment(features)

    print("Segments sample:", features.head(3)[["clv_score", "promo_propensity", "segment"]])
    print("Bandit arms:", bandit_policy.arms)
    print("Allocator assignment shape:", assignment.shape)
    print("Inflation values:", increased_value, final_value)
    print("Touchpoint allocation:", touchpoint_alloc)


if __name__ == "__main__":
    main()
