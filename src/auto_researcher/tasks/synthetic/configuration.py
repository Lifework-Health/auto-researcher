"""Strict task-owned configuration for the offline synthetic landscape."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SyntheticConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_family: Literal["linear", "tree", "neural"]
    complexity: int = Field(ge=1, le=10)
    learning_rate: float = Field(gt=0.0, le=1.0)
