"""Minimal FastAPI stub for offline testing."""
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Callable, Dict



class Request:
    def __init__(self, app: "FastAPI") -> None:
        self.app = app


class Response:
    def __init__(self, content: Any, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

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
    def __init__(self, prefix: str = "", tags: list[str] | None = None) -> None:
        self.prefix = prefix
        self.tags = tags or []
        self.routes: Dict[str, Dict[str, Callable[..., Any]]] = {}

    def add_api_route(self, path: str, endpoint: Callable[..., Any], methods: list[str]) -> None:
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
        self._startup_handlers: list[Callable[[], Any]] = []

    def include_router(self, router: APIRouter) -> None:
        for path, methods in router.routes.items():
            self.routes.setdefault(path, {}).update(methods)

    def add_api_route(self, path: str, endpoint: Callable[..., Any], methods: list[str]) -> None:
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

    def _run_startup(self) -> None:
        for handler in self._startup_handlers:
            result = handler()
            if inspect.iscoroutine(result):
                import asyncio
                asyncio.run(result)


def _build_args(func: Callable[..., Any], request: Request, body: Any | None) -> dict[str, Any]:
    signature = inspect.signature(func)
    bound_args: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.annotation is Request or (isinstance(parameter.annotation, str) and parameter.annotation == 'Request'):
            bound_args[name] = request
        elif body is not None and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            annotation = parameter.annotation
            if isinstance(annotation, str):
                annotation = func.__globals__.get(annotation)
            if callable(annotation) and isinstance(body, dict):
                bound_args[name] = annotation(**body)
            else:
                bound_args[name] = body
            body = None
        elif isinstance(parameter.default, Depends):
            dep = parameter.default.dependency()
            if hasattr(dep, '__next__'):
                bound_args[name] = next(dep)
            else:
                bound_args[name] = dep
        else:
            pass
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
    def __init__(self, app) -> None:
        self.app = app
        self.app._run_startup()

    def get(self, path: str):
        handler = self.app.routes[path]["GET"]
        request = Request(self.app)
        result = handler(**_build_args(handler, request, None))
        if inspect.iscoroutine(result):
            import asyncio
            result = asyncio.run(result)
        return Response(result if isinstance(result, dict) else {"data": result})

    def post(self, path: str, json: Dict[str, Any]):
        handler = self.app.routes[path]["POST"]
        request = Request(self.app)
        kwargs = _build_args(handler, request, json)
        result = handler(**kwargs)
        if inspect.iscoroutine(result):
            import asyncio
            result = asyncio.run(result)
        return Response(result if isinstance(result, dict) else {"data": result})
