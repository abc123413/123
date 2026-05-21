import os
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import get_settings
from app.utils.logger import setup_logger


def create_app() -> FastAPI:
    """Create and initialize the FastAPI application."""

    settings = get_settings()
    os.makedirs(settings.data_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.temp_dir, exist_ok=True)
    os.makedirs(settings.log_dir, exist_ok=True)

    setup_logger()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        debug=settings.app_debug,
        description="Enterprise RAG retrieval-augmented QA system",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def access_log_middleware(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Unhandled request error | method={} path={} client={} duration_ms={:.2f}",
                request.method,
                request.url.path,
                request.client.host if request.client else "unknown",
                duration_ms,
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        logger.bind(access_log=True).info(
            "{} {} status={} client={} duration_ms={:.2f}",
            request.method,
            request.url.path,
            response.status_code,
            request.client.host if request.client else "unknown",
            duration_ms,
        )
        return response

    app.include_router(router, prefix=settings.api_prefix)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("Application started: {}:{}{}", settings.app_host, settings.app_port, settings.api_prefix)

    return app


app = create_app()
