"""
Core data models used across the own_delivery package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class UserProfile:
    """
    Stores the key attributes about a user that flow into modelling.
    """

    user_id: str
    total_orders_90d: int
    average_order_value: float
    gross_margin_rate: float
    discount_cost_90d: float
    last_order_ts: Optional[datetime] = None
    segment: Optional[str] = None
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class CouponTemplate:
    """
    Represents one coupon family (arm) with budget and business rules.
    """

    template_id: str
    face_value: float
    min_spend: float
    max_redemptions: int
    budget: float
    eligible_merchants: Optional[List[str]] = None
    time_windows: Optional[List[str]] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class CouponExposure:
    """
    Captures the serving decision made for a user at a time.
    """

    user_id: str
    template_id: str
    exposure_ts: datetime
    predicted_redemption_prob: float
    expected_margin_uplift: float
    channel_touchpoint: str


@dataclass
class RedemptionOutcome:
    """
    Logs the observed outcome after a coupon exposure.
    """

    user_id: str
    template_id: str
    redeemed: bool
    order_margin: float
    subsidy_cost: float
    observed_ts: datetime
