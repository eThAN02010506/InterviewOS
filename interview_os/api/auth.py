"""Authentication: login/register/logout/me routes + get_current_user dependency.

Local multi-account auth. Passwords are hashed with PBKDF2 (stdlib); API access
uses opaque bearer tokens whose hash is stored in the ``auth_tokens`` table.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from interview_os.database.schema import LEGACY_OWNER, User
from interview_os.database.storage import verify_password

router = APIRouter()


class AuthRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=200)


class AuthResponse(BaseModel):
    token: str
    username: str


class MeResponse(BaseModel):
    id: str
    username: str


def _auth_error(message: str = "用户名或密码错误") -> HTTPException:
    return HTTPException(status_code=401, detail=message)


@router.post("/register", response_model=AuthResponse)
async def register(request: Request, body: AuthRequest) -> AuthResponse:
    storage = request.app.state.storage
    username = body.username.strip()
    if not username:
        raise HTTPException(status_code=422, detail="用户名不能为空")
    if username == LEGACY_OWNER:
        raise HTTPException(status_code=422, detail="该用户名为系统迁移保留名称")
    existing = await storage.get_user_by_username(username)
    if existing is not None:
        raise HTTPException(status_code=409, detail="用户名已存在")
    user = await storage.create_user(username, body.password)
    token = await storage.create_token(user.id)
    return AuthResponse(token=token, username=user.username)


@router.post("/login", response_model=AuthResponse)
async def login(request: Request, body: AuthRequest) -> AuthResponse:
    storage = request.app.state.storage
    user = await storage.get_user_by_username(body.username.strip())
    if user is None or not verify_password(body.password, user.password_hash):
        raise _auth_error()
    token = await storage.create_token(user.id)
    return AuthResponse(token=token, username=user.username)


@router.post("/logout")
async def logout(request: Request) -> dict[str, bool]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        await request.app.state.storage.delete_token(auth[7:].strip())
    return {"ok": True}


@router.get("/me", response_model=MeResponse)
async def me(user: Annotated[User, Depends(get_current_user)]) -> MeResponse:
    return MeResponse(id=user.id, username=user.username)


async def get_current_user(request: Request) -> User:
    """Resolve the authenticated user from the Authorization bearer header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="需要登录")
    token = auth[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="需要登录")
    user = await request.app.state.storage.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return user
