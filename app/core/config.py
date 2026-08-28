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

    # === Clasificador (sin modelo propio; usa la clase del detector) ===
    classifier_version: str = "detector-based-v1"

    # === Clasificador zero-shot (CLIP) — A/B: detector_based | zero_shot ===
    classifier_type: str = "detector_based"
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cpu"
    clip_threshold: float = 0.22
    clip_prompt_template: str = "a photo of {food}"
    clip_crop_padding: float = 0.0

    # === Estimador de porción ===
    # method: "basic" | "advanced" (depth). Con "advanced" y sin escala física
    # el resultado es relativo (gramos nulos) — ver docs/portion-estimation.md.
    portion_method: str = "basic"

    # === Modelo de profundidad (solo si portion_method=advanced) ===
    depth_enabled: bool = True
    depth_model_path: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_model_version: str = "depth-anything-v2-small"
    depth_device: str = "cpu"

    # Directorio para guardar imágenes de debug (bbox + labels). Vacío = off.
    debug_images_dir: str = ""


settings = Settings()