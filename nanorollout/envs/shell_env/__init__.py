"""Shell/filesystem environment implementations."""

from .base import (
    DEFAULT_TRUNCATE_NOTICE,
    DEFAULT_TRUNCATE_NOTICE_WITH_PERSIST,
    GIT_COMMIT_MESSAGE,
    GIT_USER_EMAIL,
    GIT_USER_NAME,
    TOOL_LOGGER_NAME,
    BaseEnvironment,
    ExecutionResult,
    ShellEnvironment,
    extract_cwd_marker,
    maybe_truncate,
)
from .types import ToolResult

__all__ = [
    "BaseEnvironment",
    "DEFAULT_TRUNCATE_NOTICE",
    "DEFAULT_TRUNCATE_NOTICE_WITH_PERSIST",
    "ExecutionResult",
    "GIT_COMMIT_MESSAGE",
    "GIT_USER_EMAIL",
    "GIT_USER_NAME",
    "KubernetesEnvironment",
    "ShellEnvironment",
    "TOOL_LOGGER_NAME",
    "ToolResult",
    "extract_cwd_marker",
    "maybe_truncate",
]


def __getattr__(name: str):
    if name == "KubernetesEnvironment":
        from .k8s import KubernetesEnvironment

        return KubernetesEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
