from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
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
    vllm_command: Path = Path("/usr/local/bin/runpod-serve")
    vllm_startup_timeout_seconds: float = Field(default=1800, ge=1, le=7200)
    vllm_shutdown_timeout_seconds: float = Field(default=30, ge=1, le=300)
    vllm_poll_interval_seconds: float = Field(default=2, ge=0.1, le=30)
    require_auth: bool = False
    auth_token: SecretStr | None = Field(default=None, repr=False)
    cookie_secure: bool = False
    session_ttl_hours: int = Field(default=24, ge=1, le=168)

    @model_validator(mode="after")
    def validate_security(self) -> "Settings":
        if self.require_auth and self.auth_token is None:
            raise ValueError("MOE_TOOLS_AUTH_TOKEN is required when auth is enabled")
        if self.auth_token is not None:
            token = self.auth_token.get_secret_value()
            if len(token) < 24:
                raise ValueError("MOE_TOOLS_AUTH_TOKEN must be at least 24 characters")
            if "RUNPOD_SECRET_" in token or token.startswith("REPLACE"):
                raise ValueError(
                    "MOE_TOOLS_AUTH_TOKEN contains an unresolved placeholder"
                )
        return self
