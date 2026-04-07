"""
Configuration management for StockSquad.
Loads settings from environment variables with validation.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Azure OpenAI
    azure_openai_endpoint: str = Field(
        description="Azure OpenAI endpoint URL"
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key (optional - using DefaultAzureCredential)"
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4o",
        description="Azure OpenAI deployment name for chat"
    )
    azure_openai_embedding_deployment_name: str = Field(
        default="text-embedding-ada-002",
        description="Azure OpenAI deployment name for embeddings"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version"
    )

    # xAI / Grok API
    xai_api_key: Optional[str] = Field(
        default=None,
        description="xAI API key for Grok models"
    )
    grok_model: str = Field(
        default="grok-4-1-fast-reasoning",
        description="Grok model name for social media analysis"
    )

    # Optional APIs
    alpha_vantage_api_key: Optional[str] = Field(
        default=None,
        description="Alpha Vantage API key (optional)"
    )

    # ChromaDB
    chroma_db_path: Path = Field(
        default=Path("./chroma_db"),
        description="Path to ChromaDB storage"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # Authorization (Permit.io)
    permit_io_api_key: Optional[str] = Field(
        default=None,
        description="Permit.io API key for authorization"
    )
    permit_io_pdp_url: str = Field(
        default="https://cloudpdp.api.permit.io",
        description="Permit.io Policy Decision Point URL"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure ChromaDB directory exists
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create the global settings instance.

    Returns:
        Settings: The application settings

    Raises:
        ValueError: If required environment variables are missing
    """
    global settings
    if settings is None:
        try:
            settings = Settings()
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration. Please ensure .env file exists "
                f"and contains all required variables. Error: {e}"
            )
    return settings


def reload_settings() -> Settings:
    """
    Force reload of settings from environment.

    Returns:
        Settings: The reloaded settings
    """
    global settings
    settings = None
    return get_settings()
