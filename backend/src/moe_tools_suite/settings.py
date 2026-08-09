from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MOE_TOOLS_",
        env_file=".env",
        extra="ignore",
    )

    mode: Literal["mock", "vllm"] = "mock"
    data_dir: Path = Path("data")
    frontend_dist: Path = Path("frontend/dist")
    vllm_base_url: str = "http://127.0.0.1:8000"
    auth_token: str | None = Field(default=None, repr=False)
