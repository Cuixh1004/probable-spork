"""
Two-model uplift estimation for coupon evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TwoModelUplift:
    """
    Trains separate outcome models for treatment and control to estimate uplift.
    """

    feature_cols: Tuple[str, ...]
    treatment_col: str = "treated"
    outcome_col: str = "ordered"

    treatment_model: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    control_model: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )

    def fit(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Fit the two models and return ROC-AUC for treatment and control groups.
        """

        X = df.loc[:, self.feature_cols]
        treatment_mask = df[self.treatment_col] == 1

        X_treat = X[treatment_mask]
        y_treat = df.loc[treatment_mask, self.outcome_col]

        X_control = X[~treatment_mask]
        y_control = df.loc[~treatment_mask, self.outcome_col]

        self.treatment_model.fit(X_treat, y_treat)
        self.control_model.fit(X_control, y_control)

        auc_treat = roc_auc_score(y_treat, self.treatment_model.predict_proba(X_treat)[:, 1])
        auc_control = roc_auc_score(
            y_control, self.control_model.predict_proba(X_control)[:, 1]
        )
        return float(auc_treat), float(auc_control)

    def predict_uplift(self, df: pd.DataFrame) -> pd.Series:
        """
        Estimate uplift per row (treatment probability minus control probability).
        """

        X = df.loc[:, self.feature_cols]
        treat_prob = self.treatment_model.predict_proba(X)[:, 1]
        control_prob = self.control_model.predict_proba(X)[:, 1]
        uplift = treat_prob - control_prob
        return pd.Series(uplift, index=df.index, name="uplift")

    def expected_profit(
        self,
        df: pd.DataFrame,
        margin_col: str,
        subsidy_col: str,
    ) -> pd.Series:
        """
        Convert uplift into expected profit per exposure.
        """

        uplift_prob = self.predict_uplift(df)
        margin = df[margin_col]
        subsidy = df[subsidy_col]
        profit = uplift_prob * margin - subsidy
        return profit.rename("expected_profit")
