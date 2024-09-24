from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        {"errors": [{"status": exc.status_code, "detail": exc.detail}]},
        status_code=exc.status_code,
    )
