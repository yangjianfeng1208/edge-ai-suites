from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from fastapi import FastAPI

from api.routes import create_router
from services.grading_task_pipeline import _load_detail_config

logger = logging.getLogger("grading.service")


@dataclass
class GradingServiceConfig:
    enabled: bool
    host_addr: str
    port: int
    provider: dict[str, Any]
    language: str


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return raw


def _resolve_root_config(path_value: str | None) -> Path:
    if path_value:
        return Path(path_value).resolve()

    # components/grading/grading_service.py -> smart-classroom/config.yaml
    return (Path(__file__).resolve().parents[2] / "config.yaml").resolve()


def load_grading_config(config_path: Path) -> GradingServiceConfig:
    raw = _load_yaml(config_path)
    grading = raw.get("grading", {})
    if not isinstance(grading, dict):
        raise ValueError("`grading` section is missing or invalid in config.yaml")

    provider = grading.get("provider", {})
    if not isinstance(provider, dict):
        provider = {}

    return GradingServiceConfig(
        enabled=bool(grading.get("enabled", False)),
        host_addr=str(grading.get("host_addr", "127.0.0.1")),
        port=int(grading.get("port", 9012)),
        provider=provider,
        language=str(grading.get("language", "en")),
    )


def create_app(cfg: GradingServiceConfig) -> FastAPI:
    app = FastAPI(
        title="Smart Classroom Grading Service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.include_router(create_router(cfg.language))

    @app.on_event("startup")
    async def _on_startup() -> None:
        logger.info("Grading service framework started")
        logger.info("language=%s", cfg.language)
        logger.info("providers=%s", cfg.provider)

        # Validate detail config (pipeline/ocr/detection_service/question_parsing)
        # early so a missing/broken component config surfaces at startup.
        try:
            detail = _load_detail_config()
            logger.info("detail config sections=%s", sorted(detail.keys()))
        except Exception as exc:
            logger.error("failed to load grading detail config: %s", exc)

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        logger.info("Grading service framework stopped")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grading service framework")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to root config.yaml (default: smart-classroom/config.yaml)",
    )
    parser.add_argument(
        "--ignore-enabled",
        action="store_true",
        help="Start service even when grading.enabled is false",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print resolved grading settings without starting server",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    args = parse_args()
    config_path = _resolve_root_config(args.config)

    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    cfg = load_grading_config(config_path)
    logger.info("Loaded grading config from %s", config_path)
    logger.info(
        "Resolved grading config: enabled=%s host=%s port=%s language=%s",
        cfg.enabled,
        cfg.host_addr,
        cfg.port,
        cfg.language,
    )

    if args.dry_run:
        logger.info("Dry run successful")
        return 0

    if not cfg.enabled and not args.ignore_enabled:
        logger.warning(
            "grading.enabled=false in config; service not started. "
            "Use --ignore-enabled to force start."
        )
        return 0

    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host_addr, port=cfg.port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
