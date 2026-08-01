"""Initialize the database."""
import asyncio

from interview_os.database.storage import Storage


async def main():
    storage = Storage()
    await storage.init_db()
    print("Database initialized successfully.")


if __name__ == "__main__":
    asyncio.run(main())
