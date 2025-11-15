"""
Feature engineering helpers for own_delivery models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureSpec:
    """
    Defines a single feature transformation.
    """

    name: str
    transform: Callable[[pd.DataFrame], pd.Series]
    description: str


def default_user_features(order_history: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard RFM-style features for each user.

    Parameters
    ----------
    order_history:
        DataFrame with columns
        ['user_id', 'order_ts', 'order_value', 'gross_margin', 'discount_cost'].
        `order_ts` must be convertible to datetime.

    Returns
    -------
    DataFrame indexed by user_id, containing:
        - orders_30d, orders_90d
        - avg_order_value
        - gross_margin_rate
        - discount_cost_90d
        - recency_days
    """

    df = order_history.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"])
    snapshot = df["order_ts"].max() + pd.Timedelta(days=1)

    def orders_in_window(days: int) -> pd.Series:
        mask = df["order_ts"] >= snapshot - pd.Timedelta(days=days)
        return df[mask].groupby("user_id")["order_id"].count()

    orders_30d = orders_in_window(30)
    orders_90d = orders_in_window(90)
    avg_order_value = df.groupby("user_id")["order_value"].mean()
    gross_margin_rate = (
        df.groupby("user_id")[["gross_margin", "order_value"]]
        .sum()
        .eval("gross_margin / order_value")
        .fillna(0.0)
    )
    discount_cost_90d = (
        df[df["order_ts"] >= snapshot - pd.Timedelta(days=90)]
        .groupby("user_id")["discount_cost"]
        .sum()
    )
    last_order_ts = df.groupby("user_id")["order_ts"].max()
    recency_days = (snapshot - last_order_ts).dt.days

    features = pd.DataFrame(
        {
            "orders_30d": orders_30d,
            "orders_90d": orders_90d,
            "avg_order_value": avg_order_value,
            "gross_margin_rate": gross_margin_rate,
            "discount_cost_90d": discount_cost_90d,
            "recency_days": recency_days,
        }
    ).fillna(0.0)

    return features


def merge_features(
    base_features: pd.DataFrame, extra_specs: Iterable[FeatureSpec]
) -> pd.DataFrame:
    """
    Apply additional feature transformations and merge with the base table.
    """

    feature_frames: List[pd.Series] = []
    for spec in extra_specs:
        feature = spec.transform(base_features)
        feature.name = spec.name
        feature_frames.append(feature)
    if not feature_frames:
        return base_features
    additional = pd.concat(feature_frames, axis=1)
    return base_features.join(additional, how="left").fillna(0.0)


def encode_segments(
    features: pd.DataFrame,
    value_bins: Tuple[float, float] = (0.33, 0.66),
    sensitivity_threshold: float = 0.5,
) -> pd.Series:
    """
    Create human-readable user segments using CLV and price sensitivity proxy.

    Parameters
    ----------
    features : DataFrame
        Must include `clv_score` and `promo_propensity`.
    value_bins : tuple
        Quantile cut-points for low/medium/high CLV.
    sensitivity_threshold : float
        Threshold above which user is considered price sensitive.
    """

    clv = features["clv_score"]
    quantiles = clv.quantile(value_bins).to_numpy()
    bins = [-np.inf, *quantiles, np.inf]
    labels = ["low_value", "mid_value", "high_value"]
    value_segment = pd.cut(clv, bins=bins, labels=labels, include_lowest=True)

    sensitivity = np.where(
        features["promo_propensity"] >= sensitivity_threshold,
        "price_sensitive",
        "full_price_friendly",
    )
    segment = value_segment.astype(str) + "_" + sensitivity
    return segment
