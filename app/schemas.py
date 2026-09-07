"""Bounded, in-memory request types shared by all providers."""

import base64
import binascii
from pathlib import PurePath
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

Provider = Literal["Gemini", "GPT", "Claude"]
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 12 * 1024 * 1024
MAX_BODY_BYTES = 18 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".py", ".js", ".jsx",
    ".ts", ".tsx", ".html", ".css", ".scss", ".xml", ".yaml", ".yml", ".sql",
    ".sh", ".ps1", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".rb",
    ".php", ".swift", ".kt", ".r", ".toml", ".ini", ".log",
}
TEXT_TYPES = {
    "application/json", "application/x-ndjson", "application/javascript",
    "application/xml", "application/yaml", "application/x-yaml",
}


class Attachment(BaseModel):
    name: str = Field(min_length=1, max_length=180)
    mime_type: str = Field(max_length=100)
    data: str = Field(min_length=1, max_length=4 * ((MAX_FILE_BYTES + 2) // 3))
    _bytes: bytes = PrivateAttr(default=b"")
    _text: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_file(self):
        if any(ord(char) < 32 for char in self.name) or any(char in self.name for char in "\\/"):
            raise ValueError("Attachment names must be filenames without path separators.")
        try:
            self._bytes = base64.b64decode(self.data, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("Attachment data must be valid base64.") from None
        if not self._bytes or len(self._bytes) > MAX_FILE_BYTES:
            raise ValueError("Each attachment must contain 1 byte to 5 MiB.")
        mime = self.mime_type.lower().split(";", 1)[0].strip()
        suffix = PurePath(self.name).suffix.lower()
        if mime == "image/jpg":
            mime = "image/jpeg"
        if mime in IMAGE_TYPES:
            valid = {
                "image/png": self._bytes.startswith(b"\x89PNG\r\n\x1a\n"),
                "image/jpeg": self._bytes.startswith(b"\xff\xd8\xff"),
                "image/webp": self._bytes.startswith(b"RIFF") and self._bytes[8:12] == b"WEBP",
            }[mime]
            if not valid:
                raise ValueError("Image contents do not match the declared file type.")
        elif mime == "application/pdf" or (suffix == ".pdf" and mime in {"", "application/octet-stream"}):
            mime = "application/pdf"
            if not self._bytes.startswith(b"%PDF-"):
                raise ValueError("The attached PDF does not have a valid PDF signature.")
        elif suffix in TEXT_EXTENSIONS and (
            mime.startswith("text/") or mime in TEXT_TYPES or mime in {"", "application/octet-stream"}
        ):
            try:
                self._text = self._bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise ValueError("Text and code files must use UTF-8 encoding.") from None
            if any(ord(char) < 32 and char not in "\t\n\r" for char in self._text):
                raise ValueError("The attachment contains binary data; use a supported file format.")
            if len(self._text) > MAX_TEXT_CHARS:
                raise ValueError("Text attachments must contain at most 200,000 characters.")
            mime = "text/plain"
        else:
            raise ValueError("Unsupported file. Use PDF, PNG, JPEG, WEBP, or a UTF-8 text/code file. Export Word to PDF and spreadsheets to CSV.")
        self.mime_type = mime
        return self

    def text_content(self) -> str | None:
        if self._text is None:
            return None
        return f"Attached file: {self.name}\n<file_content>\n{self._text}\n</file_content>"


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(default="", max_length=100_000)
    attachments: list[Attachment] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def content_required(self):
        if not self.text.strip() and not self.attachments:
            raise ValueError("A history message must contain text or an attachment.")
        if self.role == "assistant" and self.attachments:
            raise ValueError("History attachments must belong to user messages.")
        return self


class ChatRequest(BaseModel):
    provider: Provider
    model: str = Field(min_length=1, max_length=100)
    prompt: str = Field(default="", max_length=100_000)
    privacy_mode: bool = True
    history: list[HistoryMessage] = Field(default_factory=list, max_length=40)
    attachments: list[Attachment] = Field(default_factory=list, max_length=4)
    instructions: str = Field(default="", max_length=10_000)
    mode: Literal["chat", "image"] = "chat"

    @model_validator(mode="before")
    @classmethod
    def discard_private_context(cls, values):
        # Discard before validation, so accidental private context is never processed.
        if isinstance(values, dict) and values.get("privacy_mode", True) not in (False, "false", "False", 0, "0"):
            values = {**values, "history": [], "instructions": ""}
        return values

    @model_validator(mode="after")
    def validate_limits(self):
        if not self.prompt.strip() and not self.attachments:
            raise ValueError("Write a message or attach a file first.")
        files = self.attachments + [file for message in self.history for file in message.attachments]
        if sum(len(file._bytes) for file in files) > MAX_ATTACHMENT_BYTES:
            raise ValueError("Attachments across this request exceed 12 MiB. Remove files or start a new chat.")
        text_size = len(self.prompt) + len(self.instructions) + sum(len(message.text) for message in self.history)
        text_size += sum(len(file._text or "") for file in files)
        if text_size > MAX_TEXT_CHARS:
            raise ValueError("The conversation and text files exceed 200,000 characters. Shorten the context or start a new chat.")
        return self

    def messages(self) -> list[HistoryMessage]:
        history = [] if self.privacy_mode else self.history
        return [*history, HistoryMessage(role="user", text=self.prompt or "Please review the attached files.", attachments=self.attachments)]


class GeneratedImage(BaseModel):
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    data: str


class ChatResponse(BaseModel):
    text: str = ""
    images: list[GeneratedImage] = Field(default_factory=list)
    provider: Provider
    model: str
    warnings: list[str] = Field(default_factory=list)
