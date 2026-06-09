from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # ── LLM Backend ───────────────────────────────────────────────────────────
    llm_backend: str = Field(default="mock", description="mock | ollama | vllm")

    # ── vLLM ──────────────────────────────────────────────────────────────────
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "llama3.1:8b"

    # ── Public APIs ───────────────────────────────────────────────────────────
    news_api_key: str = ""
    sec_user_agent: str = "VendorRiskIntel/1.0 dev@example.com"

    # ── Pipeline Limits ───────────────────────────────────────────────────────
    max_entities: int = 80
    max_depth: int = 3
    max_children_per_node: int = 7
    cache_ttl_hours: int = 24

    # ── Paths ─────────────────────────────────────────────────────────────────
    output_dir: Path = ROOT_DIR / "data" / "outputs"
    cache_dir: Path = ROOT_DIR / "data" / "cache"
    synthetic_data_path: Path = ROOT_DIR / "data" / "synthetic" / "vendor_registry.json"

    # ── Risk Scoring Weights ──────────────────────────────────────────────────
    risk_weights_path: Path = ROOT_DIR / "config" / "risk_weights.yaml"

    class Config:
        env_file = ROOT_DIR / ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
