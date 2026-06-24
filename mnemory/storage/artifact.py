"""Artifact store for detailed content (slow memory tier).

Supports two backends:
- S3 (MinIO compatible) for production/Kubernetes deployments
- Local filesystem for development and simple setups

Artifacts are stored as objects keyed by:
  {user_id}/{artifact_id}/{filename}

Note: memory_id was removed from the storage path to support M:N artifact
linking (one artifact referenced by multiple memories). This is a breaking
change from the previous path format {user_id}/{memory_id}/{artifact_id}/{filename}.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from mnemory.config import ArtifactConfig

logger = logging.getLogger(__name__)

# Pattern for validating path components (user_id, memory_id, artifact_id, filename).
# Allows alphanumeric, hyphens, underscores, dots, colons, at signs, and forward
# slashes (for project:<name> style IDs and email-based user_ids) but rejects
# path traversal sequences.
_SAFE_PATH_COMPONENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/@-]*$")


def _validate_path_component(value: str, name: str) -> None:
    """Validate a path component to prevent path traversal attacks."""
    if not value:
        raise ValueError(f"{name} must not be empty")
    if len(value) > 256:
        raise ValueError(f"{name} too long (max 256 chars)")
    if ".." in value:
        raise ValueError(f"{name} must not contain '..'")
    if not _SAFE_PATH_COMPONENT.match(value):
        raise ValueError(
            f"{name} contains invalid characters: {value!r}. "
            "Only alphanumeric, hyphens, underscores, dots, colons, "
            "at signs, and forward slashes are allowed."
        )


class ArtifactMetadata:
    """Metadata for a stored artifact."""

    def __init__(
        self,
        artifact_id: str,
        filename: str,
        content_type: str,
        size: int,
        created_at: str,
    ):
        self.artifact_id = artifact_id
        self.filename = filename
        self.content_type = content_type
        self.size = size
        self.created_at = created_at

    def to_dict(self) -> dict:
        return {
            "id": self.artifact_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArtifactMetadata:
        return cls(
            artifact_id=data["id"],
            filename=data["filename"],
            content_type=data["content_type"],
            size=data["size"],
            created_at=data["created_at"],
        )


class ArtifactBackend(Protocol):
    """Protocol for artifact storage backends.

    Storage path: {user_id}/{artifact_id}/{filename}
    memory_id is NOT part of the storage path — artifacts are independent
    entities referenced by one or more memories via metadata.
    """

    def save(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> None: ...

    def load(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> bytes: ...

    def delete(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> None: ...

    def delete_artifact(self, user_id: str, artifact_id: str) -> None: ...

    def delete_all_for_user(self, user_id: str) -> None: ...

    def exists(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> bool: ...


class S3Backend:
    """S3/MinIO artifact storage backend."""

    def __init__(self, config: ArtifactConfig):
        import boto3
        from botocore.config import Config as BotoConfig

        kwargs: dict[str, Any] = {
            "config": BotoConfig(signature_version="s3v4"),
        }
        if config.s3_endpoint:
            kwargs["endpoint_url"] = config.s3_endpoint
        if config.s3_access_key and config.s3_secret_key:
            kwargs["aws_access_key_id"] = config.s3_access_key
            kwargs["aws_secret_access_key"] = config.s3_secret_key
        if config.s3_region:
            kwargs["region_name"] = config.s3_region

        self._client = boto3.client("s3", **kwargs)
        self._bucket = config.s3_bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create the bucket if it doesn't exist."""
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except Exception:
            logger.info("Creating S3 bucket: %s", self._bucket)
            try:
                self._client.create_bucket(Bucket=self._bucket)
            except Exception as e:
                # Bucket may already exist (race condition) or we lack permissions
                logger.warning("Could not create bucket %s: %s", self._bucket, e)

    def _key(self, user_id: str, artifact_id: str, filename: str) -> str:
        for val, name in [
            (user_id, "user_id"),
            (artifact_id, "artifact_id"),
            (filename, "filename"),
        ]:
            _validate_path_component(val, name)
        return f"{user_id}/{artifact_id}/{filename}"

    def save(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> None:
        key = self._key(user_id, artifact_id, filename)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
        )

    def load(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> bytes:
        key = self._key(user_id, artifact_id, filename)
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def delete(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> None:
        key = self._key(user_id, artifact_id, filename)
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def delete_artifact(self, user_id: str, artifact_id: str) -> None:
        """Delete all files for a specific artifact."""
        prefix = f"{user_id}/{artifact_id}/"
        continuation_token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**kwargs)
            objects = response.get("Contents", [])
            if objects:
                self._client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

    def delete_all_for_user(self, user_id: str) -> None:
        """Delete all artifacts for a user."""
        _validate_path_component(user_id, "user_id")
        prefix = f"{user_id}/"
        continuation_token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._client.list_objects_v2(**kwargs)
            objects = response.get("Contents", [])
            if objects:
                self._client.delete_objects(
                    Bucket=self._bucket,
                    Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
                )
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")

    def exists(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> bool:
        key = self._key(user_id, artifact_id, filename)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False


class FilesystemBackend:
    """Local filesystem artifact storage backend."""

    def __init__(self, config: ArtifactConfig):
        self._base_path = Path(config.filesystem_path)
        self._base_path.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: str, artifact_id: str, filename: str) -> Path:
        for val, name in [
            (user_id, "user_id"),
            (artifact_id, "artifact_id"),
            (filename, "filename"),
        ]:
            _validate_path_component(val, name)
        path = (self._base_path / user_id / artifact_id / filename).resolve()
        if not path.is_relative_to(self._base_path.resolve()):
            raise ValueError(
                "Invalid path components: resolved path escapes base directory"
            )
        return path

    def save(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> None:
        path = self._path(user_id, artifact_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        # Store content_type in a sidecar metadata file
        meta_path = path.parent / ".metadata.json"
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        meta[filename] = {"content_type": content_type, "size": len(content)}
        meta_path.write_text(json.dumps(meta))

    def load(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> bytes:
        path = self._path(user_id, artifact_id, filename)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return path.read_bytes()

    def delete(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> None:
        path = self._path(user_id, artifact_id, filename)
        if path.exists():
            path.unlink()
        # Clean up empty parent directories
        artifact_dir = path.parent
        if artifact_dir.exists() and not any(
            f for f in artifact_dir.iterdir() if f.name != ".metadata.json"
        ):
            shutil.rmtree(artifact_dir)

    def delete_artifact(self, user_id: str, artifact_id: str) -> None:
        """Delete all files for a specific artifact."""
        _validate_path_component(user_id, "user_id")
        _validate_path_component(artifact_id, "artifact_id")
        artifact_dir = (self._base_path / user_id / artifact_id).resolve()
        if not artifact_dir.is_relative_to(self._base_path.resolve()):
            raise ValueError(
                "Invalid path components: resolved path escapes base directory"
            )
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

    def delete_all_for_user(self, user_id: str) -> None:
        """Delete all artifacts for a user."""
        _validate_path_component(user_id, "user_id")
        user_dir = (self._base_path / user_id).resolve()
        if not user_dir.is_relative_to(self._base_path.resolve()):
            raise ValueError(
                "Invalid path components: resolved path escapes base directory"
            )
        if user_dir.exists():
            shutil.rmtree(user_dir)

    def exists(
        self,
        user_id: str,
        artifact_id: str,
        filename: str,
    ) -> bool:
        return self._path(user_id, artifact_id, filename).exists()


class ArtifactStore:
    """High-level artifact store with metadata management.

    Artifacts are binary objects attached to fast memories. Each memory
    can have zero or more artifacts. Artifact metadata (id, filename,
    content_type, size) is stored in the fast memory's metadata in the
    vector store. The actual content is stored in S3 or the filesystem.
    """

    def __init__(self, config: ArtifactConfig, max_artifact_size: int = 10_485_760):
        if config.backend == "s3":
            self._backend: ArtifactBackend = S3Backend(config)
        elif config.backend == "filesystem":
            self._backend = FilesystemBackend(config)
        else:
            raise ValueError(f"Unsupported artifact backend: {config.backend}")

        self._max_size = max_artifact_size

    def save(
        self,
        user_id: str,
        content: str | bytes,
        filename: str = "note.md",
        content_type: str = "text/markdown",
        artifact_id: str | None = None,
    ) -> ArtifactMetadata:
        """Save an artifact.

        Args:
            user_id: Owner of the artifact.
            content: Text content (str) or binary content (bytes).
                     If str and content_type starts with "text/", encoded as UTF-8.
                     If str and content_type is binary, decoded from base64.
            filename: Name for the artifact file.
            content_type: MIME type of the content.
            artifact_id: Optional pre-generated artifact ID. If None, a new
                         UUID-based ID is generated.

        Returns:
            ArtifactMetadata with the artifact details.
        """
        # Convert content to bytes
        if isinstance(content, str):
            if content_type.startswith("text/"):
                content_bytes = content.encode("utf-8")
            else:
                # Binary content passed as base64 string
                content_bytes = base64.b64decode(content)
        else:
            content_bytes = content

        if len(content_bytes) > self._max_size:
            raise ValueError(
                f"Artifact too large: {len(content_bytes)} bytes "
                f"(max {self._max_size}). Summarize the content or split "
                "into multiple artifacts."
            )

        if artifact_id is None:
            artifact_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()

        self._backend.save(
            user_id=user_id,
            artifact_id=artifact_id,
            filename=filename,
            content=content_bytes,
            content_type=content_type,
        )

        return ArtifactMetadata(
            artifact_id=artifact_id,
            filename=filename,
            content_type=content_type,
            size=len(content_bytes),
            created_at=now,
        )

    @staticmethod
    def _resolve_meta(artifacts_meta: list[dict], artifact_id: str) -> tuple[str, str]:
        """Look up filename and content_type from artifact metadata list.

        Returns:
            Tuple of (filename, content_type).

        Raises:
            ValueError: If artifact_id is not found in the metadata list.
        """
        for a in artifacts_meta:
            if a.get("id") == artifact_id:
                return a["filename"], a.get("content_type", "text/plain")
        raise ValueError(f"Artifact {artifact_id} not found in metadata")

    def load(
        self,
        user_id: str,
        artifact_id: str,
        artifacts_meta: list[dict],
        offset: int = 0,
        limit: int = 5000,
    ) -> dict:
        """Load artifact content with pagination (text) or full content (binary).

        Text artifacts support pagination via offset/limit (character-based).
        Binary artifacts always return the full content as a single base64
        blob — offset and limit are ignored for binary content.

        Args:
            user_id: Owner of the artifact.
            artifact_id: ID of the artifact to load.
            artifacts_meta: List of artifact metadata dicts from a referencing
                memory. Used to look up filename and content_type.
            offset: Character offset for text content (ignored for binary).
            limit: Max characters to return for text (ignored for binary).

        Returns:
            Dict with content, total_size, offset, is_text, and has_more.
        """
        filename, content_type = self._resolve_meta(artifacts_meta, artifact_id)
        raw = self._backend.load(
            user_id=user_id,
            artifact_id=artifact_id,
            filename=filename,
        )

        is_text = content_type.startswith("text/")
        total_size = len(raw)

        if is_text:
            text = raw.decode("utf-8", errors="replace")
            chunk = text[offset : offset + limit]
            return {
                "content": chunk,
                "total_size": len(text),
                "offset": offset,
                "is_text": True,
                "has_more": offset + limit < len(text),
                "content_type": content_type,
                "filename": filename,
            }
        else:
            # Binary: return full content as a single base64 blob.
            # No pagination — offset/limit are ignored for binary content.
            return {
                "content": base64.b64encode(raw).decode("ascii"),
                "total_size": total_size,
                "offset": 0,
                "is_text": False,
                "has_more": False,
                "content_type": content_type,
                "filename": filename,
            }

    def load_raw(
        self,
        user_id: str,
        artifact_id: str,
        artifacts_meta: list[dict],
    ) -> tuple[bytes, str, str]:
        """Load raw artifact bytes without encoding.

        Returns the raw bytes, content_type, and filename — suitable for
        streaming directly to HTTP responses with the correct Content-Type.

        Args:
            user_id: Owner of the artifact.
            artifact_id: ID of the artifact to load.
            artifacts_meta: List of artifact metadata dicts from a referencing
                memory. Used to look up filename and content_type.

        Returns:
            Tuple of (raw_bytes, content_type, filename).
        """
        filename, content_type = self._resolve_meta(artifacts_meta, artifact_id)
        raw = self._backend.load(
            user_id=user_id,
            artifact_id=artifact_id,
            filename=filename,
        )
        return raw, content_type, filename

    def delete(
        self,
        user_id: str,
        artifact_id: str,
        artifacts_meta: list[dict],
    ) -> None:
        """Delete a specific artifact by ID.

        Uses artifacts_meta to look up the filename for the backend delete.
        """
        meta = None
        for a in artifacts_meta:
            if a.get("id") == artifact_id:
                meta = a
                break
        if meta is None:
            raise ValueError(f"Artifact {artifact_id} not found in metadata")

        self._backend.delete(
            user_id=user_id,
            artifact_id=artifact_id,
            filename=meta["filename"],
        )

    def delete_by_id(self, user_id: str, artifact_id: str) -> None:
        """Delete an artifact by ID without needing metadata.

        Uses the backend's delete_artifact method which removes all files
        under the artifact's directory. Used by garbage collection when
        we know the artifact_id but may not have metadata.
        """
        self._backend.delete_artifact(user_id=user_id, artifact_id=artifact_id)

    def delete_all_for_user(self, user_id: str) -> None:
        """Delete all artifacts for a user."""
        self._backend.delete_all_for_user(user_id=user_id)
