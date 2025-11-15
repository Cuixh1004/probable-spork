"""
Price sensitivity / promo propensity modelling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PromoPropensityModel:
    """
    Logistic regression estimating probability a user redeems a coupon template.
    """

    numeric_features: Sequence[str]
    categorical_features: Sequence[str]
    penalty: str = "l2"
    C: float = 1.0

    pipeline: Optional[Pipeline] = None

    def fit(self, df: pd.DataFrame, target_col: str = "redeemed") -> float:
        """
        Train the model and return ROC-AUC on the training data for reference.
        """

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), list(self.numeric_features)),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(self.categorical_features),
                ),
            ]
        )

        classifier = LogisticRegression(
            penalty=self.penalty, C=self.C, solver="lbfgs", max_iter=1000
        )

        self.pipeline = Pipeline(
            steps=[("preprocess", preprocessor), ("clf", classifier)]
        )
        X = df[self.numeric_features + list(self.categorical_features)]
        y = df[target_col]
        self.pipeline.fit(X, y)
        pred = self.pipeline.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, pred))

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Output redemption probabilities for new data.
        """

        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = df[self.numeric_features + list(self.categorical_features)]
        prob = self.pipeline.predict_proba(X)[:, 1]
        return pd.Series(prob, index=df.index, name="promo_propensity")
