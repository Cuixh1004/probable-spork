"""
End-to-end demo wiring together the own_delivery components
with event logging and monitoring.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from . import (
    bandit,
    clv,
    events,
    feature_engineering,
    linear_allocator,
    monitoring,
    price_propensity,
    state_machine,
)


def synthetic_order_history(num_users: int = 1000) -> pd.DataFrame:
    """
    Create a fake orders table. Each row is a historical order for a user.
    """
    rng = np.random.default_rng(seed=42)
    user_ids = [f"user_{i}" for i in range(num_users)]
    records = []
    for user in user_ids:
        orders = rng.integers(1, 10)
        for _ in range(orders):
            order_value = rng.normal(80, 20)
            order_value = max(order_value, 20)
            gross_margin = max(order_value * rng.uniform(0.15, 0.35), 5)
            discount_cost = max(order_value * rng.uniform(0.05, 0.20), 0)

            records.append(
                {
                    "order_id": int(rng.integers(1e9)),
                    "user_id": user,
                    "order_ts": pd.Timestamp("2025-01-01")
                    + pd.Timedelta(days=int(rng.integers(0, 90))),
                    "order_value": float(order_value),
                    "gross_margin": float(gross_margin),
                    "discount_cost": float(discount_cost),
                }
            )
    return pd.DataFrame(records)


def build_models():
    """
    Build:
      - base user features from order history
      - a simple CLV model
      - a simple promo propensity model

    Returns:
      features: DataFrame with per-user features + clv_score + promo_propensity + segment
      clv_model: fitted CLVModel instance
      propensity_model: fitted PromoPropensityModel instance
    """
    orders = synthetic_order_history()
    base_features = feature_engineering.default_user_features(orders)

    # Simple CLV proxy for training target
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
    clv_r2 = clv_model.fit(base_features, target_clv)
    clv_scores = clv_model.predict(base_features)

    # Build training data for propensity model
    promo_df = base_features.copy()
    promo_df["channel"] = "app"
    promo_df["user_segment"] = pd.cut(
        base_features["orders_90d"],
        bins=[0, 2, 5, 10],
        labels=["low", "mid", "high"],
    )
    # Fake historical redemption label for training
    promo_df["redeemed"] = np.random.binomial(1, 0.3, size=len(promo_df))

    propensity_model = price_propensity.PromoPropensityModel(
        numeric_features=["orders_30d", "orders_90d", "avg_order_value"],
        categorical_features=["channel", "user_segment"],
    )
    auc = propensity_model.fit(promo_df, target_col="redeemed")
    propensity = propensity_model.predict(promo_df)

    # Final per-user feature view
    features = base_features.assign(
        clv_score=clv_scores,
        promo_propensity=propensity,
    )
    segments = feature_engineering.encode_segments(features)
    features["segment"] = segments

    print(f"[build_models] CLV R^2 on hold-out: {clv_r2:.3f}")
    print(f"[build_models] Propensity ROC-AUC (train): {auc:.3f}")

    return features, clv_model, propensity_model


def run_allocator_with_logging(
    features: pd.DataFrame,
    logger: events.EventLogger,
    rng: np.random.Generator,
) -> None:
    """
    Use the linear allocator to assign coupons to a subset of users,
    simulate a simple outcome with the state machine, and log everything.
    """
    # Use a small subset for the demo
    subset = features.head(50).copy()
    user_ids = list(subset.index)

    coupons = ["coupon_small", "coupon_large"]

    # For the demo, expected profit is random but could be:
    #   expected_profit ~ promo_propensity * margin - subsidy
    expected_profit = np.random.uniform(1, 5, size=(len(subset), len(coupons)))
    subsidy = np.array([[5.0, 20.0]])
    subsidy = np.repeat(subsidy, repeats=len(subset), axis=0)

    problem = linear_allocator.AllocationProblem(
        user_ids=user_ids,
        coupon_ids=coupons,
        expected_profit=expected_profit,
        subsidy_cost=subsidy,
        budget=500.0,
        exposure_cap=1.0,
    )
    allocator = linear_allocator.LinearAllocator()
    assignment = allocator.solve(problem)

    # Setup coupon inflation logic for the experiment
    infl_config = state_machine.InflationConfig(
        base_value=10.0,
        share_increment=5.0,
        interaction_increment=2.0,
        cap=30.0,
    )
    infl_sm = state_machine.CouponInflationStateMachine(infl_config)

    # For each user, pick the coupon with highest assigned weight
    for user_idx, user_id in enumerate(user_ids):
        user_weights = assignment[user_idx]  # array of length len(coupons)
        if user_weights.max() <= 0:
            # This user received nothing in the optimal solution
            continue
        coupon_idx = int(user_weights.argmax())
        coupon_id = coupons[coupon_idx]

        # Log exposure
        logger.log_exposure(
            user_id=user_id,
            coupon_id=coupon_id,
            policy="allocator",
            ts=datetime.utcnow(),
        )

        # Simulate some engagement so the coupon inflates
        # For demo, random number of interactions and shares
        n_interactions = int(rng.integers(0, 4))
        n_shares = int(rng.integers(0, 5))
        for _ in range(n_interactions):
            infl_sm.register_interaction(user_id)
        for _ in range(n_shares):
            infl_sm.register_share(user_id)

        # Simulate whether the user redeems, using their promo_propensity
        p_redeem = float(subset.loc[user_id, "promo_propensity"])
        redeemed = bool(rng.random() < p_redeem)

        if redeemed:
            # State machine returns the coupon value at conversion
            subsidy_cost = infl_sm.register_order(user_id)
            # Approximate order margin using historical margin rate and AOV
            avg_value = float(subset.loc[user_id, "avg_order_value"])
            margin_rate = float(subset.loc[user_id, "gross_margin_rate"])
            order_margin = avg_value * margin_rate
            # Simple incremental margin: "extra" margin assumed equal to order_margin
            incremental_margin = order_margin
        else:
            subsidy_cost = 0.0
            order_margin = 0.0
            incremental_margin = 0.0

        logger.log_outcome(
            user_id=user_id,
            coupon_id=coupon_id,
            policy="allocator",
            redeemed=redeemed,
            subsidy_cost=subsidy_cost,
            order_margin=order_margin,
            incremental_margin=incremental_margin,
            ts=datetime.utcnow(),
        )


def run_bandit_demo(logger: events.EventLogger, rng: np.random.Generator) -> None:
    """
    Small separate demo of the bandit policy that also logs events.
    Here we treat each 'round' as a pseudo-user just to illustrate logging.
    """
    bandit_policy = bandit.BudgetedThompsonSampling(round_budget=30.0)
    costs = {"coupon_small": 5.0, "coupon_large": 20.0}
    bandit_policy.register_arm("coupon_small", initial_cost=5.0)
    bandit_policy.register_arm("coupon_large", initial_cost=20.0)

    for round_idx in range(10):
        pseudo_user_id = f"bandit_round_{round_idx}"
        chosen_arms = bandit_policy.select(costs)

        for arm_id in chosen_arms:
            # Log exposure
            logger.log_exposure(
                user_id=pseudo_user_id,
                coupon_id=arm_id,
                policy="bandit",
                ts=datetime.utcnow(),
            )

            # Simulate a simple outcome: large coupon performs better on average
            base_p = 0.3 if arm_id == "coupon_small" else 0.5
            redeemed = bool(rng.random() < base_p)
            subsidy_cost = float(costs[arm_id])
            order_margin = float(rng.normal(15.0 if arm_id == "coupon_small" else 25.0, 5.0))
            incremental_margin = order_margin if redeemed else 0.0

            bandit_policy.update(
                arm_id=arm_id,
                reward=1.0 if redeemed else 0.0,
                cost=subsidy_cost,
            )

            logger.log_outcome(
                user_id=pseudo_user_id,
                coupon_id=arm_id,
                policy="bandit",
                redeemed=redeemed,
                subsidy_cost=subsidy_cost if redeemed else 0.0,
                order_margin=order_margin if redeemed else 0.0,
                incremental_margin=incremental_margin,
                ts=datetime.utcnow(),
            )


def main():
    rng = np.random.default_rng(seed=123)
    features, clv_model, propensity_model = build_models()

    # Show a small sample of the final user features table
    print("[main] Sample of enriched user features:")
    print(features.head(3)[["clv_score", "promo_propensity", "segment"]])

    logger = events.EventLogger()

    # 1) Run allocator + state machine and log events
    run_allocator_with_logging(features, logger, rng)

    # 2) Run a small bandit demo and log events
    run_bandit_demo(logger, rng)

    # Build the events DataFrame
    events_df = logger.to_dataframe()
    print(f"[main] Logged {len(events_df)} events.")
    print("[main] Events sample:")
    print(events_df.head())

    # For monitoring, we care about outcome rows (where redeemed/subsidy/margin are filled)
    outcome_events = events_df[events_df["event_type"] == "outcome"].copy()
    if not outcome_events.empty:
        kpis = monitoring.redemption_metrics(outcome_events)
        roi = monitoring.incremental_roi(outcome_events)
        print("[main] Redemption metrics:", kpis)
        print(f"[main] Incremental ROI: {roi:.3f}")
    else:
        print("[main] No outcome events logged; monitoring skipped.")


if __name__ == "__main__":
    main()
