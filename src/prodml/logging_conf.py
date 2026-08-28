import logging
import sys

from contextvars import ContextVar

from pythonjsonlogger.json import JsonFormatter


correlation_id_context: ContextVar[str] = ContextVar(
    "correlation_id",
    default="-",
)


class CorrelationIdFilter(logging.Filter):
    """Add the current correlation ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_context.get()

        return True


def configure_logging() -> None:
    """Configure application-wide JSON logging."""

    handler = logging.StreamHandler(sys.stdout)

    handler.addFilter(CorrelationIdFilter())

    formatter = JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s"
    )

    handler.setFormatter(formatter)

    root_logger = logging.getLogger()

    root_logger.setLevel(logging.INFO)

    root_logger.handlers.clear()

    root_logger.addHandler(handler)
