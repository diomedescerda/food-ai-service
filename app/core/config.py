from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="FOOD_AI_", extra="ignore"
    )

    api_title: str = "FoodAI Service"
    api_version: str = "0.1.0"
    api_environment: str = "development"

    # Puerto del dev server (run_dev.py). El contenedor usa 8000 interno.
    service_port: int = 8010

    # === Modelo de detección ===
    model_path: str = "weights/yolo11n.pt"
    model_version: str = "food-detector-v1"
    confidence_threshold: float = 0.35
    image_size: int = 640
    device: str = "cpu"
    max_detections: int = 20

    # === Modelo de segmentación ===
    seg_model_path: str = "weights/yolo11n-seg.pt"
    seg_model_version: str = "food-segmenter-v1"

    # === Clasificador (sin inferencia adicional: usa la clase del detector) ===
    classifier_version: str = "detector-based-v1"

    # Directorio para guardar imágenes de debug (bbox + labels). Vacío = off.
    debug_images_dir: str = ""


settings = Settings()