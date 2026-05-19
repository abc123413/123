import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import get_settings
from app.utils.logger import setup_logger


def create_app() -> FastAPI:
    """创建并初始化 FastAPI 应用。"""

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
        description="企业级 RAG 检索增强生成系统",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix=settings.api_prefix)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info("应用启动成功: {}:{}{}", settings.app_host, settings.app_port, settings.api_prefix)

    return app


app = create_app()
