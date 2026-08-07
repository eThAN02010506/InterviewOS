"""Database storage operations."""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import delete, insert, inspect, select
from sqlalchemy.ext.asyncio import AsyncConnection, async_sessionmaker, create_async_engine

from interview_os.database.schema import (
    LEGACY_OWNER,
    AuthToken,
    Base,
    EvidenceRecord,
    InterviewSession,
    User,
)

logger = logging.getLogger(__name__)

_TOKEN_TTL = timedelta(days=30)
_PBKDF2_ITERATIONS = 600_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt_bytes = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt_bytes, _PBKDF2_ITERATIONS
    )
    return f"{salt_bytes.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split("$", 1)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, TypeError):
        return False
    candidate = hash_password(password, salt)
    return secrets.compare_digest(candidate, stored)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def _ensure_schema(conn: AsyncConnection) -> None:
    """Idempotent hand-rolled migration for databases that predate accounts.

    Runs after ``create_all`` so fresh DBs get the new tables/columns from the
    ORM and only legacy DBs need the ALTER + backfill. Every step is guarded by
    an existence check so re-runs are no-ops.
    """

    def _column_names(sync_conn: Any) -> set[str]:
        return {c["name"] for c in inspect(sync_conn).get_columns("interview_sessions")}

    columns = await conn.run_sync(_column_names)
    if "owner_id" not in columns:
        await conn.exec_driver_sql(
            "ALTER TABLE interview_sessions "
            "ADD COLUMN owner_id VARCHAR(64) NOT NULL DEFAULT 'local'"
        )
    # Ensure the sentinel legacy owner exists so backfilled rows have a valid owner.
    existing = (
        await conn.execute(select(User.id).where(User.username == LEGACY_OWNER))
    ).scalar_one_or_none()
    if existing is None:
        await conn.execute(
            insert(User).values(
                id=str(uuid4()),
                username=LEGACY_OWNER,
                password_hash=hash_password(secrets.token_urlsafe(32)),
            )
        )


class Storage:
    def __init__(self, database_url: str = "sqlite+aiosqlite:///./interview_os.db") -> None:
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await _ensure_schema(conn)

    # ---- sessions ---------------------------------------------------------

    async def save_session(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        owner_id: str = LEGACY_OWNER,
    ) -> None:
        async with self.session_factory() as session:
            record = await session.get(InterviewSession, session_id)
            values = {
                "owner_id": owner_id,
                "candidate_name": state.get("candidate", {}).get("name", ""),
                "job_title": state.get("job", {}).get("title", ""),
                "company_name": state.get("company", {}).get("name", ""),
                "state_json": json.dumps(state, default=str),
            }
            if record is None:
                session.add(InterviewSession(id=session_id, **values))
            else:
                for key, value in values.items():
                    setattr(record, key, value)
            await session.commit()

    async def get_session_state(
        self, session_id: str, *, owner_id: str | None = LEGACY_OWNER
    ) -> dict[str, Any] | None:
        """Return a session's state blob, scoped to ``owner_id`` when given.

        ``owner_id=None`` disables the ownership filter (used by the localhost
        debug console to inspect any session).
        """
        async with self.session_factory() as session:
            query = select(InterviewSession).where(InterviewSession.id == session_id)
            if owner_id is not None:
                query = query.where(InterviewSession.owner_id == owner_id)
            record = await session.execute(query)
            row = record.scalar_one_or_none()
            return json.loads(row.state_json) if row is not None else None

    async def list_sessions(
        self, limit: int = 50, *, owner_id: str | None = LEGACY_OWNER
    ) -> list[dict[str, Any]]:
        """List sessions, scoped to ``owner_id`` when given (``None`` = all)."""
        bounded_limit = max(1, min(limit, 200))
        async with self.session_factory() as session:
            query = select(InterviewSession).order_by(InterviewSession.updated_at.desc())
            if owner_id is not None:
                query = query.where(InterviewSession.owner_id == owner_id)
            result = await session.execute(query.limit(bounded_limit))
            return [
                {
                    "id": record.id,
                    "candidate_name": record.candidate_name,
                    "job_title": record.job_title,
                    "company_name": record.company_name,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                }
                for record in result.scalars()
            ]

    async def save_evidence(self, session_id: str, evidence: dict[str, Any]) -> None:
        async with self.session_factory() as session:
            record = EvidenceRecord(session_id=session_id, **evidence)
            session.add(record)
            await session.commit()

    # ---- auth -------------------------------------------------------------

    async def create_user(self, username: str, password: str) -> User:
        async with self.session_factory() as session:
            user = User(username=username, password_hash=hash_password(password))
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def get_user_by_username(self, username: str) -> User | None:
        async with self.session_factory() as session:
            return (
                await session.execute(select(User).where(User.username == username))
            ).scalar_one_or_none()

    async def get_user_by_id(self, user_id: str) -> User | None:
        async with self.session_factory() as session:
            return await session.get(User, user_id)

    async def create_token(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        async with self.session_factory() as session:
            session.add(
                AuthToken(
                    user_id=user_id,
                    token_hash=hash_token(token),
                    expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + _TOKEN_TTL,
                )
            )
            await session.commit()
        return token

    async def get_user_by_token(self, token: str) -> User | None:
        async with self.session_factory() as session:
            row = await session.execute(
                select(AuthToken).where(AuthToken.token_hash == hash_token(token))
            )
            token_row = row.scalar_one_or_none()
            now = datetime.now(timezone.utc).replace(tzinfo=None)  # stored naive
            if token_row is None or token_row.expires_at < now:
                return None
            return await session.get(User, token_row.user_id)

    async def delete_token(self, token: str) -> None:
        async with self.session_factory() as session:
            await session.execute(
                delete(AuthToken).where(AuthToken.token_hash == hash_token(token))
            )
            await session.commit()

    async def close(self) -> None:
        await self.engine.dispose()
