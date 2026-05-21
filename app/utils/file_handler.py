import hashlib
import os
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings


class FileHandler:
    """File save/delete helpers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        os.makedirs(self.settings.upload_dir, exist_ok=True)
        os.makedirs(self.settings.temp_dir, exist_ok=True)

    def validate_upload_file(self, upload_file: UploadFile) -> None:
        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix not in self.settings.allowed_extensions:
            raise ValueError(f"Unsupported file type: {suffix}")

    async def save_upload_file(self, upload_file: UploadFile) -> tuple[str, str]:
        self.validate_upload_file(upload_file)
        document_id = str(uuid.uuid4())
        suffix = Path(upload_file.filename or "").suffix.lower()
        target_path = os.path.join(self.settings.upload_dir, f"{document_id}{suffix}")

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        return document_id, target_path

    def build_title(self, filename: str, custom_title: str | None = None) -> str:
        if custom_title and custom_title.strip():
            return custom_title.strip()
        return Path(filename).stem.strip() or filename

    def calculate_file_hash(self, file_path: str) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def delete_file(self, file_path: str) -> None:
        if os.path.exists(file_path):
            os.remove(file_path)
