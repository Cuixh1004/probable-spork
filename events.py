"""
Event logging utilities for own_delivery experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class EventLogger:
    """
    Collects exposure and outcome events into a single DataFrame.

    Each record is a flat dict; use `to_dataframe()` at the end of a run.
    """

    records: List[Dict[str, Any]] = field(default_factory=list)

    def log_exposure(
        self,
        user_id: str,
        coupon_id: str,
        policy: str,
        channel: str = "app",
        ts: Optional[datetime] = None,
    ) -> None:
        """
        Log that a user was exposed to a coupon.
        """
        self.records.append(
            {
                "event_type": "exposure",
                "event_ts": ts or datetime.utcnow(),
                "user_id": user_id,
                "coupon_id": coupon_id,
                "policy": policy,
                "channel": channel,
                # outcome fields left blank for exposure rows
                "redeemed": None,
                "subsidy_cost": None,
                "order_margin": None,
                "incremental_margin": None,
            }
        )

    def log_outcome(
        self,
        user_id: str,
        coupon_id: str,
        policy: str,
        redeemed: bool,
        subsidy_cost: float,
        order_margin: float,
        incremental_margin: float,
        ts: Optional[datetime] = None,
        channel: str = "app",
    ) -> None:
        """
        Log the observed outcome for a user-coupon pair.
        """
        self.records.append(
            {
                "event_type": "outcome",
                "event_ts": ts or datetime.utcnow(),
                "user_id": user_id,
                "coupon_id": coupon_id,
                "policy": policy,
                "channel": channel,
                "redeemed": bool(redeemed),
                "subsidy_cost": float(subsidy_cost),
                "order_margin": float(order_margin),
                "incremental_margin": float(incremental_margin),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all logged events into a single DataFrame.
        """
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "event_type",
                    "event_ts",
                    "user_id",
                    "coupon_id",
                    "policy",
                    "channel",
                    "redeemed",
                    "subsidy_cost",
                    "order_margin",
                    "incremental_margin",
                ]
            )
        return pd.DataFrame(self.records)
