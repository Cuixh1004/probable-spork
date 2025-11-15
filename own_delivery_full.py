"""
Unified script containing all core own_delivery components for convenient copying.

This script mirrors the modular layout of the package in a single file:
    * feature engineering utilities
    * CLV regression model wrapper
    * promo propensity model
    * budget-aware Thompson sampling bandit
    * linear programming allocator
    * monitoring helpers
    * coupon inflation state machine
    * demonstration harness stitching the pieces together

Running the script will execute the same demo showcased in the package module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog, linear_sum_assignment
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    """Defines a single feature transformation."""

    name: str
    transform: Callable[[pd.DataFrame], pd.Series]
    description: str


def default_user_features(order_history: pd.DataFrame) -> pd.DataFrame:
    """
    Compute standard RFM-style features for each user.
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
    """Apply additional feature transformations and merge with the base table."""

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


# ---------------------------------------------------------------------------
# CLV modelling
# ---------------------------------------------------------------------------


@dataclass
class CLVModel:
    """Simple linear-regression based CLV estimator."""

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
        """Train the CLV regression model and return R^2 on the hold-out set."""

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
        """Estimate CLV for the provided features."""

        X = features[self.feature_names].to_numpy()
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return pd.Series(preds, index=features.index, name="clv_score")

    def explain_coefficients(self) -> pd.DataFrame:
        """Return coefficient weights for interpretation."""

        coefs = pd.Series(self.model.coef_, index=self.feature_names)
        return coefs.sort_values(ascending=False).to_frame("coefficient")


# ---------------------------------------------------------------------------
# Price propensity modelling
# ---------------------------------------------------------------------------


@dataclass
class PromoPropensityModel:
    """Logistic regression estimating coupon redemption probability."""

    numeric_features: Sequence[str]
    categorical_features: Sequence[str]
    penalty: str = "l2"
    C: float = 1.0

    pipeline: Optional[Pipeline] = None

    def fit(self, df: pd.DataFrame, target_col: str = "redeemed") -> float:
        """Train the model and return ROC-AUC on the training data."""

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
        """Output redemption probabilities for new data."""

        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = df[self.numeric_features + list(self.categorical_features)]
        prob = self.pipeline.predict_proba(X)[:, 1]
        return pd.Series(prob, index=df.index, name="promo_propensity")


# ---------------------------------------------------------------------------
# Bandit policy
# ---------------------------------------------------------------------------


@dataclass
class BanditArmState:
    """Tracks posterior parameters for an arm under Beta distribution."""

    successes: float = 1.0
    failures: float = 1.0
    cost_mean: float = 1.0
    cost_var: float = 1.0

    def sample_roi(self) -> float:
        reward = np.random.beta(self.successes, self.failures)
        cost = max(np.random.normal(self.cost_mean, np.sqrt(self.cost_var)), 1e-6)
        return reward / cost


@dataclass
class BudgetedThompsonSampling:
    """Selects coupon arms while respecting a per-round budget."""

    arms: Dict[str, BanditArmState] = field(default_factory=dict)
    round_budget: float = 1.0

    def register_arm(self, arm_id: str, initial_cost: float) -> None:
        if arm_id in self.arms:
            return
        self.arms[arm_id] = BanditArmState(
            cost_mean=initial_cost, cost_var=initial_cost ** 2
        )

    def select(self, costs: Dict[str, float]) -> List[str]:
        """Return a list of arm ids to serve this round under budget constraint."""

        sampled_roi = {
            arm_id: state.sample_roi() for arm_id, state in self.arms.items()
        }
        sorted_arms = sorted(
            sampled_roi.items(), key=lambda item: item[1], reverse=True
        )

        allocated: List[str] = []
        spend = 0.0
        for arm_id, _ in sorted_arms:
            cost = costs.get(arm_id, self.arms[arm_id].cost_mean)
            if spend + cost <= self.round_budget:
                allocated.append(arm_id)
                spend += cost
        return allocated

    def update(
        self,
        arm_id: str,
        reward: float,
        cost: float,
        step_size: float = 0.1,
    ) -> None:
        """Update posterior parameters with observed reward and cost."""

        state = self.arms[arm_id]
        state.successes += reward
        state.failures += 1.0 - reward

        state.cost_mean = (1 - step_size) * state.cost_mean + step_size * cost
        state.cost_var = max(
            (1 - step_size) * state.cost_var
            + step_size * (cost - state.cost_mean) ** 2,
            1e-6,
        )


# ---------------------------------------------------------------------------
# Linear allocator
# ---------------------------------------------------------------------------


@dataclass
class AllocationProblem:
    """Encapsulates inputs required for the linear program."""

    user_ids: Sequence[str]
    coupon_ids: Sequence[str]
    expected_profit: np.ndarray  # shape (n_users, n_coupons)
    subsidy_cost: np.ndarray  # shape (n_users, n_coupons)
    budget: float
    exposure_cap: float = 1.0


class LinearAllocator:
    """Solves for the assignment of coupons that maximises uplift profit."""

    def solve(self, problem: AllocationProblem) -> np.ndarray:
        n_users = len(problem.user_ids)
        n_coupons = len(problem.coupon_ids)
        n_variables = n_users * n_coupons

        c = -problem.expected_profit.flatten()

        A = problem.subsidy_cost.reshape(1, n_variables)
        b = np.array([problem.budget])

        A_user = np.zeros((n_users, n_variables))
        for u in range(n_users):
            for c_idx in range(n_coupons):
                col = u * n_coupons + c_idx
                A_user[u, col] = 1.0
        A = np.vstack([A, A_user])
        b = np.concatenate([b, np.full(n_users, problem.exposure_cap)])

        bounds = [(0.0, 1.0) for _ in range(n_variables)]

        result = linprog(
            c=c,
            A_ub=A,
            b_ub=b,
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            raise RuntimeError(f"Linear programming failed: {result.message}")

        assignment = result.x.reshape(n_users, n_coupons)
        return assignment


# ---------------------------------------------------------------------------
# Monitoring utilities
# ---------------------------------------------------------------------------


def redemption_metrics(events: pd.DataFrame) -> Dict[str, float]:
    """Compute basic KPIs from redemption logs."""

    total_exposed = len(events)
    redeemed = events["redeemed"].sum()
    redemption_rate = redeemed / total_exposed if total_exposed else 0.0
    avg_subsidy = events.loc[events["redeemed"], "subsidy_cost"].mean()
    avg_margin = events.loc[events["redeemed"], "order_margin"].mean()

    return {
        "exposures": float(total_exposed),
        "redemption_rate": float(redemption_rate),
        "avg_subsidy": float(avg_subsidy if not np.isnan(avg_subsidy) else 0.0),
        "avg_margin": float(avg_margin if not np.isnan(avg_margin) else 0.0),
    }


def incremental_roi(events: pd.DataFrame) -> float:
    """Estimate ROI = (incremental_margin - subsidy) / subsidy."""

    incremental_margin = events["incremental_margin"].sum()
    subsidy = events["subsidy_cost"].sum()
    if subsidy == 0:
        return float("inf") if incremental_margin > 0 else 0.0
    return float((incremental_margin - subsidy) / subsidy)


@dataclass
class BayesianGuardrail:
    """Monitors redemption rate vs baseline using Beta posterior."""

    alpha_prior: float = 1.0
    beta_prior: float = 1.0

    def posterior_probability(
        self, successes: int, trials: int, baseline_rate: float
    ) -> float:
        import scipy.stats

        alpha_post = self.alpha_prior + successes
        beta_post = self.beta_prior + trials - successes
        posterior = scipy.stats.beta(alpha_post, beta_post)
        prob = posterior.cdf(baseline_rate)
        return float(prob)


# ---------------------------------------------------------------------------
# Coupon inflation state machine
# ---------------------------------------------------------------------------


class CouponState(Enum):
    SEED = auto()
    ENGAGED = auto()
    SHARING = auto()
    CONVERSION = auto()
    COOLDOWN = auto()


@dataclass
class InflationConfig:
    base_value: float
    share_increment: float
    interaction_increment: float
    cap: float
    share_threshold: int = 3


@dataclass
class UserCouponState:
    state: CouponState = CouponState.SEED
    current_value: float = 0.0
    shares: int = 0
    interactions: int = 0


class CouponInflationStateMachine:
    """Determines how coupon value increases with user actions."""

    def __init__(self, config: InflationConfig) -> None:
        self.config = config
        self.user_states: Dict[str, UserCouponState] = {}

    def _ensure_state(self, user_id: str) -> UserCouponState:
        if user_id not in self.user_states:
            self.user_states[user_id] = UserCouponState(
                state=CouponState.SEED, current_value=self.config.base_value
            )
        return self.user_states[user_id]

    def register_interaction(self, user_id: str) -> float:
        user_state = self._ensure_state(user_id)
        user_state.interactions += 1
        if user_state.state == CouponState.SEED:
            user_state.state = CouponState.ENGAGED
        user_state.current_value = min(
            user_state.current_value + self.config.interaction_increment,
            self.config.cap,
        )
        return user_state.current_value

    def register_share(self, user_id: str) -> float:
        user_state = self._ensure_state(user_id)
        user_state.shares += 1
        if user_state.shares >= self.config.share_threshold:
            user_state.state = CouponState.SHARING
            user_state.current_value = min(
                user_state.current_value + self.config.share_increment,
                self.config.cap,
            )
        return user_state.current_value

    def register_order(self, user_id: str) -> float:
        user_state = self._ensure_state(user_id)
        user_state.state = CouponState.CONVERSION
        value_at_conversion = user_state.current_value
        user_state.current_value = self.config.base_value
        user_state.shares = 0
        user_state.interactions = 0
        user_state.state = CouponState.COOLDOWN
        return value_at_conversion

    def reset(self, user_id: str) -> None:
        if user_id in self.user_states:
            self.user_states[user_id] = UserCouponState(
                state=CouponState.SEED, current_value=self.config.base_value
            )


# ---------------------------------------------------------------------------
# Reinforcement learning policy (re-created for completeness)
# ---------------------------------------------------------------------------


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Basic replay buffer for DQN training."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0

    def push(self, exp: Experience) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.position] = exp
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class CouponServingPolicy:
    """Deep Q-learning policy that chooses which coupon action to take."""

    state_dim: int
    action_dim: int
    gamma: float = 0.95
    learning_rate: float = 1e-3
    batch_size: int = 64
    device: str = "cpu"
    update_target_every: int = 100

    q_net: Optional[QNetwork] = None
    target_q_net: Optional[QNetwork] = None
    optimiser: Optional[Adam] = None

    def __post_init__(self) -> None:
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimiser = Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer()
        self.step_count = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(
            np.stack([b.state for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [b.action for b in batch], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [b.reward for b in batch], dtype=torch.float32, device=self.device
        )
        next_states = torch.tensor(
            np.stack([b.next_state for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [b.done for b in batch], dtype=torch.float32, device=self.device
        )

        q_values = self.q_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_q_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def store_experience(self, exp: Experience) -> None:
        self.buffer.push(exp)


# ---------------------------------------------------------------------------
# Sequence churn model
# ---------------------------------------------------------------------------


class EventSequenceDataset(Dataset):
    """Torch dataset wrapping padded event sequences for each user."""

    def __init__(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.sequences[idx], self.lengths[idx], self.labels[idx]


class GRUChurnModel(nn.Module):
    """GRU network predicting churn probability from event sequences."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        logits = self.classifier(hidden[-1])
        return logits.squeeze(-1)


@dataclass
class ChurnEarlyWarning:
    """Wraps training/inference for the GRU churn model."""

    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 1
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    device: str = "cpu"

    model: Optional[GRUChurnModel] = None

    def fit(
        self,
        train_sequences: np.ndarray,
        train_lengths: np.ndarray,
        train_labels: np.ndarray,
    ) -> None:
        dataset = EventSequenceDataset(train_sequences, train_lengths, train_labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = GRUChurnModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_len, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_len = batch_len.to(self.device)
                batch_y = batch_y.to(self.device)

                optimiser.zero_grad()
                preds = self.model(batch_x, batch_len)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(batch_y)

            avg_loss = epoch_loss / len(dataset)
            print(
                f"[ChurnEarlyWarning] epoch={epoch+1}/{self.epochs} loss={avg_loss:.4f}"
            )

    def predict_proba(
        self,
        sequences: np.ndarray,
        lengths: np.ndarray,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        dataset = EventSequenceDataset(sequences, lengths, np.zeros(len(lengths)))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for batch_x, batch_len, _ in loader:
                batch_x = batch_x.to(self.device)
                batch_len = batch_len.to(self.device)
                prob = self.model(batch_x, batch_len).cpu().numpy()
                preds.append(prob)
        return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Touchpoint assignment
# ---------------------------------------------------------------------------


@dataclass
class TouchpointProblem:
    user_ids: Sequence[str]
    touchpoints: Sequence[str]
    priority_matrix: np.ndarray  # shape (n_users, n_touchpoints)


def assign_touchpoints(problem: TouchpointProblem) -> Dict[str, str]:
    """Solve maximum weight matching by converting to cost minimisation."""

    max_priority = np.max(problem.priority_matrix)
    cost_matrix = max_priority - problem.priority_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    allocation: Dict[str, str] = {}
    for r, c in zip(row_ind, col_ind):
        allocation[problem.user_ids[r]] = problem.touchpoints[c]
    return allocation


# ---------------------------------------------------------------------------
# Demo utilities
# ---------------------------------------------------------------------------


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
    base_features = default_user_features(orders)
    target_clv = (
        base_features["orders_90d"] * base_features["avg_order_value"] * 0.3
        - base_features["discount_cost_90d"]
    )

    clv_model = CLVModel(
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

    propensity_model = PromoPropensityModel(
        numeric_features=["orders_30d", "orders_90d", "avg_order_value"],
        categorical_features=["channel", "user_segment"],
    )
    propensity_model.fit(promo_df, target_col="redeemed")
    propensity = propensity_model.predict(promo_df)

    features = base_features.assign(
        clv_score=clv_scores,
        promo_propensity=propensity,
    )
    segments = encode_segments(features)
    features["segment"] = segments

    return features, clv_model, propensity_model


def run_bandit():
    bandit_policy = BudgetedThompsonSampling(round_budget=100.0)
    bandit_policy.register_arm("coupon_small", initial_cost=5.0)
    bandit_policy.register_arm("coupon_large", initial_cost=20.0)

    costs = {"coupon_small": 5.0, "coupon_large": 20.0}
    chosen = bandit_policy.select(costs)
    for arm in chosen:
        reward = np.random.binomial(1, 0.5 if arm == "coupon_large" else 0.3)
        cost = costs[arm]
        bandit_policy.update(arm, reward=reward, cost=cost)
    return bandit_policy


def run_allocator(features: pd.DataFrame):
    coupons = ["coupon_small", "coupon_large"]
    expected_profit = np.random.uniform(1, 5, size=(len(features), len(coupons)))
    subsidy = np.array([[5.0, 20.0]])
    subsidy = np.repeat(subsidy, repeats=len(features), axis=0)

    problem = AllocationProblem(
        user_ids=list(features.index),
        coupon_ids=coupons,
        expected_profit=expected_profit,
        subsidy_cost=subsidy,
        budget=1000.0,
        exposure_cap=1.0,
    )
    allocator = LinearAllocator()
    assignment = allocator.solve(problem)
    return assignment


def run_state_machine():
    config = InflationConfig(
        base_value=10.0, share_increment=5.0, interaction_increment=2.0, cap=30.0
    )
    sm = CouponInflationStateMachine(config)
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
    problem = TouchpointProblem(
        user_ids=users, touchpoints=touchpoints, priority_matrix=priority
    )
    return assign_touchpoints(problem)


def main() -> None:
    features, clv_model, propensity_model = build_models()
    bandit_policy = run_bandit()
    assignment = run_allocator(features.head(10))
    increased_value, final_value = run_state_machine()
    touchpoint_alloc = run_touchpoint_assignment(features)

    print(
        "Segments sample:",
        features.head(3)[["clv_score", "promo_propensity", "segment"]],
    )
    print("Bandit arms:", bandit_policy.arms)
    print("Allocator assignment shape:", assignment.shape)
    print("Inflation values:", increased_value, final_value)
    print("Touchpoint allocation:", touchpoint_alloc)


if __name__ == "__main__":
    main()
