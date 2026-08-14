"""Shared helpers for authenticated real-HTTP release smoke tests."""

from __future__ import annotations

import secrets
import time

import httpx


def authenticate_e2e_client(
    client: httpx.Client,
    *,
    username: str = "",
    password: str = "",
    prefix: str = "interviewos_e2e",
) -> str:
    """Authenticate a smoke-test client without exposing generated credentials.

    Supplying credentials reuses an explicit test account. Otherwise each run
    registers an isolated synthetic account so it cannot alter a user's sessions.
    """
    if bool(username) != bool(password):
        raise ValueError("--username and --password must be supplied together")
    if username:
        path = "/api/auth/login"
    else:
        suffix = f"{int(time.time())}_{secrets.token_hex(3)}"
        username = f"{prefix}_{suffix}"[:64]
        password = secrets.token_urlsafe(24)
        path = "/api/auth/register"
    response = client.post(path, json={"username": username, "password": password})
    response.raise_for_status()
    token = response.json()["token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return username
