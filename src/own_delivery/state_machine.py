"""
State machine representing coupon inflation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional


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
    """
    Determines how coupon value increases with user actions.
    """

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
