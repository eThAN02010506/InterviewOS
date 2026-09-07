"""Bounded helpers for reading multipart uploads.

``UploadFile`` is backed by a spooled file, but calling ``read()`` without a
size still copies the complete payload into application memory.  Public upload
routes use this helper so an oversized file is rejected after at most
``max_bytes + 1`` bytes have been materialized by the application.
"""

from __future__ import annotations

from fastapi import HTTPException, UploadFile

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


async def read_upload_bounded(
    upload: UploadFile,
    *,
    max_bytes: int,
    too_large_detail: str,
    chunk_bytes: int = UPLOAD_READ_CHUNK_BYTES,
) -> bytes:
    """Read an upload in bounded chunks and raise HTTP 413 at ``limit + 1``.

    Reading one byte beyond the accepted limit distinguishes an exactly-full
    valid upload from an oversized one without consuming the remainder.
    """

    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")

    remaining = max_bytes + 1
    chunks: list[bytes] = []
    total = 0
    while remaining > 0:
        chunk = await upload.read(min(chunk_bytes, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=too_large_detail)
        remaining = max_bytes + 1 - total
    return b"".join(chunks)


__all__ = ["UPLOAD_READ_CHUNK_BYTES", "read_upload_bounded"]
