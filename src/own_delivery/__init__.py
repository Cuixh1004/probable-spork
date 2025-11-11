"""
own_delivery
============

A reference implementation of the modelling components used to drive
large-coupon distribution strategies for an own-channel food delivery
business.

The package groups together data structures, feature engineering helpers,
predictive models, decision policies, and monitoring utilities that mirror
the conceptual designs discussed with the user.
"""

from . import (
    bandit,
    churn_sequence_model,
    clv,
    data_models,
    feature_engineering,
    linear_allocator,
    monitoring,
    price_propensity,
    rl_policy,
    state_machine,
    touchpoint_assignment,
    uplift,
)

__all__ = [
    "bandit",
    "churn_sequence_model",
    "clv",
    "data_models",
    "feature_engineering",
    "linear_allocator",
    "monitoring",
    "price_propensity",
    "rl_policy",
    "state_machine",
    "touchpoint_assignment",
    "uplift",
]
