"""FastAPI dependency providers."""
from fastapi import HTTPException, Request

from interview_os.services.interview_service import InterviewService, _owner_ctx


async def get_interview_service(request: Request) -> InterviewService:
    """Return the app-wide InterviewService, scoping this request's owner.

    The service is a singleton; the owner is carried per-request via a
    contextvar so every service call (and background task spawned from it) is
    scoped to the authenticated account. Production routers require auth before
    reaching this dependency. The ``local`` fallback exists only for explicitly
    auth-disabled development/test apps; invalid bearer tokens are always rejected.
    """
    auth = request.headers.get("Authorization", "")
    owner: str | None = None
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        if not token:
            raise HTTPException(status_code=401, detail="需要登录")
        user = await request.app.state.storage.get_user_by_token(token)
        if user is None:
            raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
        owner = user.id
    _owner_ctx.set(owner if owner is not None else "local")
    return request.app.state.interview_service
