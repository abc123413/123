import os
import shutil
import uuid
import hashlib
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings


class FileHandler:
    """处理上传文件的保存、校验与删除。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        os.makedirs(self.settings.upload_dir, exist_ok=True)
        os.makedirs(self.settings.temp_dir, exist_ok=True)

    def validate_upload_file(self, upload_file: UploadFile) -> None:
        """校验上传文件扩展名。"""

        suffix = Path(upload_file.filename or "").suffix.lower()
        if suffix not in self.settings.allowed_extensions:
            raise ValueError(f"不支持的文件类型: {suffix}")

    async def save_upload_file(self, upload_file: UploadFile) -> tuple[str, str]:
        """保存上传文件，返回文件 ID 和保存路径。"""

        self.validate_upload_file(upload_file)
        document_id = str(uuid.uuid4())
        suffix = Path(upload_file.filename or "").suffix.lower()
        target_path = os.path.join(self.settings.upload_dir, f"{document_id}{suffix}")

        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        return document_id, target_path

    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件 SHA256，用于文档级去重。"""

        digest = hashlib.sha256()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def delete_file(self, file_path: str) -> None:
        """删除本地文件。"""

        if os.path.exists(file_path):
            os.remove(file_path)
