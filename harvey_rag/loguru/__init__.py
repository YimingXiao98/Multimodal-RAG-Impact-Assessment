"""Minimal loguru stub."""

class Logger:
    def info(self, *args, **kwargs):  # pragma: no cover - noop
        pass

    def warning(self, *args, **kwargs):  # pragma: no cover - noop
        pass

    def error(self, *args, **kwargs):  # pragma: no cover - noop
        pass


logger = Logger()
