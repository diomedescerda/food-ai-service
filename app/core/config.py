from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    api_title: str = "FoodAI AI Service"
    api_version: str = "0.1.0"
    api_environment: str = "development"


settings = Settings()