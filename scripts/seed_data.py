"""Seed sample data for development."""
import asyncio
import json

from interview_os.database.storage import Storage


async def main():
    storage = Storage()
    await storage.init_db()
    sample_state = {
        "candidate": {"name": "John Doe", "strengths": ["LLM Engineering"]},
        "job": {"title": "ML Engineer"},
        "company": {"name": "AI Startup"},
    }
    await storage.save_session("sample-session-1", sample_state)
    print("Seed data inserted.")


if __name__ == "__main__":
    asyncio.run(main())
