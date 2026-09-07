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

    # === Modelo de detecciÃ³n ===
    model_path: str = "weights/yolo11n.pt"
    model_version: str = "food-detector-v1"
    confidence_threshold: float = 0.35
    image_size: int = 640
    device: str = "cpu"
    max_detections: int = 20

    # === Detector de regiones (FASE 11): yolo | dino | hybrid ===
    detector_type: str = "yolo"
    dino_model: str = "IDEA-Research/grounding-dino-tiny"
    dino_prompt: str = "food on a plate"
    dino_threshold: float = 0.15

    # === Modelo de segmentaciÃ³n ===
    seg_model_path: str = "weights/yolo11n-seg.pt"
    seg_model_version: str = "food-segmenter-v1"

    # === Clasificador (sin modelo propio; usa la clase del detector) ===
    classifier_version: str = "detector-based-v1"

    # === Clasificador zero-shot (CLIP) â€” A/B: detector_based | zero_shot ===
    classifier_type: str = "detector_based"
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cpu"
    clip_threshold: float = 0.20
    clip_prompt_template: str = "a photo of {food}"

    # Plantillas extra separadas por '|' â†’ ensemble por media de scores
    # (FASE 14: padding 0.10 + ensemble mejorÃ³ top-1 0.648 â†’ 0.685).
    clip_prompt_ensemble: str = ""
    clip_crop_padding: float = 0.10

    # === Estimador de porciÃ³n ===
    # method: "basic" | "advanced" (depth). Con "advanced" y sin escala fÃ­sica
    # el resultado es relativo (gramos nulos) â€” ver docs/portion-estimation.md.
    portion_method: str = "basic"

    # === Modelo de profundidad (solo si portion_method=advanced) ===
    depth_enabled: bool = True
    depth_model_path: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_model_version: str = "depth-anything-v2-small"
    depth_device: str = "cpu"

    # Directorio para guardar imÃ¡genes de debug (bbox + labels). VacÃ­o = off.
    debug_images_dir: str = ""

    # === Retrieval masivo (FASE 42): catÃ¡logo 1.451 FNDDS ===
    # enabled=true reemplazarÃ­a el legacy (NO activado). shadow=true ejecuta
    # el pipeline completo sin tocar la respuesta (telemetrÃ­a).
    retrieval_enabled: bool = False
    retrieval_shadow_enabled: bool = False
    retrieval_top_k: int = 50

    # === Nutrición (FASE 52): lookup local precomputado ===
    # enabled=true usaría la nutrición mapeada (NO activado). shadow=true
    # registra la telemetría de nutrición sin cambiar la respuesta.
    nutrition_enabled: bool = False
    nutrition_shadow_enabled: bool = False

    # === Confianza/fallback (FASE 53): política de decisión ===
    # confidence_enabled=false: la decisión nunca rechaza por confianza
    # (el mecanismo queda preparado, sin activar).
    confidence_enabled: bool = False
    min_visual_confidence: float = 0.20


settings = Settings()
