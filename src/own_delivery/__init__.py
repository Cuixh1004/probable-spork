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
    clv,
    feature_engineering,
    linear_allocator,
    monitoring,
    price_propensity,
    state_machine,
)

__all__ = [
    "bandit",
    "clv",
    "feature_engineering",
    "linear_allocator",
    "monitoring",
    "price_propensity",
    "state_machine",
]
