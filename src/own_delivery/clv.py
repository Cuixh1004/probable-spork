"""
Customer lifetime value modelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class CLVModel:
    """
    Simple linear-regression based CLV estimator.
    """

    feature_names: Sequence[str]
    model: LinearRegression = LinearRegression()
    scaler: StandardScaler = StandardScaler()

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> float:
        """
        Train the CLV regression model and return R^2 on the hold-out set.
        """

        X = features[self.feature_names].to_numpy()
        y = target.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

        X_test_scaled = self.scaler.transform(X_test)
        r2 = self.model.score(X_test_scaled, y_test)
        return float(r2)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Estimate CLV for the provided features.
        """

        X = features[self.feature_names].to_numpy()
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return pd.Series(preds, index=features.index, name="clv_score")

    def explain_coefficients(self) -> pd.DataFrame:
        """
        Return coefficient weights for interpretation.
        """

        coefs = pd.Series(self.model.coef_, index=self.feature_names)
        return coefs.sort_values(ascending=False).to_frame("coefficient")
