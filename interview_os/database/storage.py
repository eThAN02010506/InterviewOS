"""Database storage operations."""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from interview_os.database.schema import Base, EvidenceRecord, InterviewSession

logger = logging.getLogger(__name__)


class Storage:
    def __init__(self, database_url: str = "sqlite+aiosqlite:///./interview_os.db") -> None:
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_session(self, session_id: str, state: dict[str, Any]) -> None:
        async with self.session_factory() as session:
            record = await session.get(InterviewSession, session_id)
            values = {
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

    async def get_session_state(self, session_id: str) -> dict[str, Any] | None:
        async with self.session_factory() as session:
            record = await session.get(InterviewSession, session_id)
            return json.loads(record.state_json) if record is not None else None

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(limit, 200))
        async with self.session_factory() as session:
            result = await session.execute(
                select(InterviewSession)
                .order_by(InterviewSession.updated_at.desc())
                .limit(bounded_limit)
            )
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

    async def close(self) -> None:
        await self.engine.dispose()
