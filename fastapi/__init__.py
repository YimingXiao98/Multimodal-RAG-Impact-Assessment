"""Minimal FastAPI stub sufficient for unit tests and lightweight ASGI hosting."""
from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List


class Request:
    def __init__(self, app: "FastAPI") -> None:
        self.app = app


class Response:
    def __init__(self, content: Any, status_code: int = 200, headers: Dict[str, str] | None = None) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> Any:
        return self.content


class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class Depends:
    def __init__(self, dependency: Callable[..., Any]) -> None:
        self.dependency = dependency


class APIRouter:
    def __init__(self, prefix: str = "", tags: List[str] | None = None) -> None:
        self.prefix = prefix
        self.tags = tags or []
        self.routes: Dict[str, Dict[str, Callable[..., Any]]] = {}

    def add_api_route(self, path: str, endpoint: Callable[..., Any], methods: List[str]) -> None:
        full_path = self.prefix + path
        self.routes.setdefault(full_path, {})
        for method in methods:
            self.routes[full_path][method.upper()] = endpoint

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, ["GET"])
            return func

        return decorator

    def post(self, path: str, response_model: Any | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, ["POST"])
            return func

        return decorator


class FastAPI:
    def __init__(self, title: str = "") -> None:
        self.title = title
        self.routes: Dict[str, Dict[str, Callable[..., Any]]] = {}
        self.state = SimpleNamespace()
        self._startup_handlers: List[Callable[[], Any]] = []
        self._startup_complete = False

    def include_router(self, router: APIRouter) -> None:
        for path, methods in router.routes.items():
            self.routes.setdefault(path, {}).update(methods)

    def add_api_route(self, path: str, endpoint: Callable[..., Any], methods: List[str]) -> None:
        self.routes.setdefault(path, {})
        for method in methods:
            self.routes[path][method.upper()] = endpoint

    def get(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, ["GET"])
            return func

        return decorator

    def post(self, path: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.add_api_route(path, func, ["POST"])
            return func

        return decorator

    def on_event(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if event == "startup":
                self._startup_handlers.append(func)
            return func

        return decorator

    def add_middleware(self, middleware_class: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        return None

    def _run_startup_sync(self) -> None:
        asyncio.run(self._ensure_startup())

    async def _ensure_startup(self) -> None:
        if self._startup_complete:
            return
        for handler in self._startup_handlers:
            result = handler()
            if inspect.isawaitable(result):
                await result
        self._startup_complete = True

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[[], Awaitable[Dict[str, Any]]],
        send: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope["type"] != "http":  # pragma: no cover - not exercised in tests
            raise NotImplementedError("Only HTTP scope is supported in this stub.")

        await self._ensure_startup()

        method = scope["method"].upper()
        path = scope["path"]
        handler = self.routes.get(path, {}).get(method)
        if handler is None:
            await _send_response(send, Response({"detail": "Not Found"}, status_code=404))
            return

        body = await _receive_body(receive)
        parsed_body = _parse_body(body, scope.get("headers", []))
        request = Request(self)
        kwargs = _build_args(handler, request, parsed_body)
        result = handler(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        await _send_response(send, result)


async def _receive_body(receive: Callable[[], Awaitable[Dict[str, Any]]]) -> bytes:
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)
    return body


def _parse_body(body: bytes, headers: List[tuple[bytes, bytes]]) -> Any | None:
    if not body:
        return None
    header_map = {key.lower(): value for key, value in headers}
    content_type = header_map.get(b"content-type", b"").split(b";")[0]
    if content_type == b"application/json":
        return json.loads(body.decode("utf-8"))
    return body


async def _send_response(send: Callable[[Dict[str, Any]], Awaitable[None]], result: Any) -> None:
    if isinstance(result, Response):
        content = result.content
        status = result.status_code
        headers = result.headers
    else:
        content = result
        status = 200
        headers: Dict[str, str] = {}

    if isinstance(content, (dict, list)):
        body = json.dumps(content).encode("utf-8")
        headers = {**headers, "content-type": "application/json"}
    elif isinstance(content, bytes):
        body = content
    else:
        body = str(content).encode("utf-8")
        headers = {**headers, "content-type": "text/plain; charset=utf-8"}

    header_list = [(key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()]
    await send({"type": "http.response.start", "status": status, "headers": header_list})
    await send({"type": "http.response.body", "body": body})


def _build_args(func: Callable[..., Any], request: Request, body: Any | None) -> dict[str, Any]:
    signature = inspect.signature(func)
    bound_args: dict[str, Any] = {}
    body_consumed = False
    for name, parameter in signature.parameters.items():
        if parameter.annotation is Request or (isinstance(parameter.annotation, str) and parameter.annotation == "Request"):
            bound_args[name] = request
        elif not body_consumed and body is not None and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            annotation = parameter.annotation
            if isinstance(annotation, str):
                annotation = func.__globals__.get(annotation)
            if callable(annotation) and isinstance(body, dict):
                bound_args[name] = annotation(**body)
            else:
                bound_args[name] = body
            body_consumed = True
        elif isinstance(parameter.default, Depends):
            dependency = parameter.default.dependency()
            if hasattr(dependency, "__next__"):
                bound_args[name] = next(dependency)
            else:
                bound_args[name] = dependency
    return bound_args


__all__ = [
    "FastAPI",
    "APIRouter",
    "Request",
    "Response",
    "HTTPException",
    "Depends",
    "TestClient",
]


class TestClient:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.app._run_startup_sync()

    def get(self, path: str):
        handler = self.app.routes[path]["GET"]
        request = Request(self.app)
        result = handler(**_build_args(handler, request, None))
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return Response(result if isinstance(result, dict) else {"data": result})

    def post(self, path: str, json: Dict[str, Any]):
        handler = self.app.routes[path]["POST"]
        request = Request(self.app)
        kwargs = _build_args(handler, request, json)
        result = handler(**kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return Response(result if isinstance(result, dict) else {"data": result})

