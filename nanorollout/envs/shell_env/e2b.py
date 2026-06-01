"""E2B sandbox-based shell environment for NanoRollout.

Runs agent commands inside a remote E2B sandbox via the E2B SDK
(``sandbox.commands.run``). Sandbox lifecycle: create sandbox (from a template
mapped from the SWE-bench docker image) -> run commands -> kill sandbox.

Mirrors the Kubernetes backend's interface: like ``kubectl exec``, every
``commands.run`` call is a fresh process, so the working directory is tracked
across calls using a trailing PWD marker (see ``extract_cwd_marker``).

The docker-image -> E2B-template-alias mapping follows the same scheme as
siirl-agentic: an explicit ``template_map`` override, falling back to a
deterministic sanitizing transform with configurable
strip-prefix / replace-suffix / suffix adjustments.
"""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import ExecutionResult, ShellEnvironment, extract_cwd_marker

logger = logging.getLogger(__name__)

_PWD_MARKER = "__NANOROLLOUT_PWD__"


def _e2b_template_alias_for_docker_image(image_name: str) -> str:
    """Sanitize a docker image reference into an E2B template alias.

    Example::

        swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest
        -> swebench-sweb-eval-x86-64-astropy-1776-astropy-12907-latest

    Kept in sync with siirl-agentic's ``_e2b_template_alias_for_docker_image``.
    """
    return (
        image_name.replace("/", "-")
        .replace(".", "-")
        .replace("_", "-")
        .replace(":", "-")
        .lower()
    )


@dataclass
class E2BEnvironmentConfig:
    """Configuration for the E2B environment."""

    image: str
    cwd: str = "/"
    timeout: int = 120
    # Sandbox-level lifetime (seconds). The sandbox is auto-killed by E2B after
    # this elapses, acting as a safety net against leaks.
    sandbox_timeout: int = 3600
    request_timeout: Optional[float] = None
    # Template mapping
    template_map: dict[str, str] = field(default_factory=dict)
    template_strip_prefix: str = ""
    template_replace_suffix: dict[str, str] = field(default_factory=dict)
    template_suffix: str = ""
    # Connection (None -> read from E2B_* env vars by the SDK)
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    domain: Optional[str] = None
    force_http: Optional[bool] = None
    user: str = "root"
    allow_internet_access: bool = True


class E2BEnvironment(ShellEnvironment):
    """An E2B sandbox-based execution environment."""

    def __init__(
        self,
        image: str,
        instance: dict[str, Any],
        workspace_dir: str = "/",
        timeout: int = 120,
        sandbox_timeout: int = 3600,
        request_timeout: Optional[float] = None,
        template_map: Optional[dict[str, str]] = None,
        template_strip_prefix: str = "",
        template_replace_suffix: Optional[dict[str, str]] = None,
        template_suffix: str = "",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        domain: Optional[str] = None,
        force_http: Optional[bool] = None,
        user: str = "root",
        allow_internet_access: bool = True,
        logger_override: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        self.logger = logger_override or logger
        self.instance = instance

        self.config = E2BEnvironmentConfig(
            image=image,
            cwd=workspace_dir,
            timeout=timeout,
            sandbox_timeout=sandbox_timeout,
            request_timeout=request_timeout,
            template_map=template_map or {},
            template_strip_prefix=template_strip_prefix,
            template_replace_suffix=template_replace_suffix or {},
            template_suffix=template_suffix,
            api_key=api_key,
            api_url=api_url,
            domain=domain,
            force_http=force_http,
            user=user,
            allow_internet_access=allow_internet_access,
        )

        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self._cwd = workspace_dir
        self._sandbox: Any = None
        self._file_history: dict[str, list[str]] = {}

        if kwargs:
            self.logger.debug("Ignoring unknown kwargs: %s", list(kwargs.keys()))

    # ------------------------------------------------------------------
    # Template resolution
    # ------------------------------------------------------------------

    def _resolve_template(self, image_or_template: str) -> str:
        """Map a docker image reference to an E2B template alias."""
        if image_or_template in self.config.template_map:
            return str(self.config.template_map[image_or_template])

        alias = _e2b_template_alias_for_docker_image(image_or_template)

        prefix = self.config.template_strip_prefix
        if prefix and alias.startswith(prefix):
            alias = alias[len(prefix):]

        for old_suffix, new_suffix in self.config.template_replace_suffix.items():
            if alias.endswith(old_suffix):
                alias = alias[: -len(old_suffix)] + new_suffix
                break

        if self.config.template_suffix:
            alias = alias + self.config.template_suffix

        # E2B template aliases are capped at 64 chars.
        if len(alias) > 64:
            alias = alias[:64]
        return alias

    def _connect_kwargs(self) -> dict[str, Any]:
        """Connection kwargs for Sandbox.create.

        Only includes values explicitly configured; anything left out is filled
        in by the SDK from E2B_* environment variables.
        """
        kwargs: dict[str, Any] = {}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        if self.config.api_url:
            kwargs["api_url"] = self.config.api_url
        if self.config.domain:
            kwargs["domain"] = self.config.domain
        if self.config.force_http is not None:
            kwargs["force_http"] = self.config.force_http
        if self.config.request_timeout is not None:
            kwargs["request_timeout"] = self.config.request_timeout
        return kwargs

    # ------------------------------------------------------------------
    # ShellEnvironment interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        from e2b import Sandbox

        template = self._resolve_template(self.config.image)
        self.logger.info(
            "Creating E2B sandbox (image=%s -> template=%s)",
            self.config.image,
            template,
        )

        connect_kwargs = self._connect_kwargs()
        try:
            self._sandbox = Sandbox.create(
                template=template,
                timeout=self.config.sandbox_timeout,
                allow_internet_access=self.config.allow_internet_access,
                **connect_kwargs,
            )
        except TypeError:
            # Older/newer SDKs may not accept allow_internet_access; retry without.
            self._sandbox = Sandbox.create(
                template=template,
                timeout=self.config.sandbox_timeout,
                **connect_kwargs,
            )

        self.logger.info(
            "E2B sandbox %s created (template=%s)",
            getattr(self._sandbox, "sandbox_id", "?"),
            template,
        )

    def stop(self) -> None:
        if self._sandbox is not None:
            sandbox_id = getattr(self._sandbox, "sandbox_id", "?")
            self.logger.info("Killing E2B sandbox %s", sandbox_id)
            try:
                self._sandbox.kill()
            except Exception as exc:
                self.logger.warning(
                    "Failed to kill E2B sandbox %s: %s", sandbox_id, exc
                )
            finally:
                self._sandbox = None

    def is_running(self) -> bool:
        if self._sandbox is None:
            return False
        try:
            return bool(self._sandbox.is_running())
        except Exception:
            return False

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecutionResult:
        if self._sandbox is None:
            raise RuntimeError("E2B sandbox is not running")

        # Merge the command's stderr into stdout (to match the K8s pty ordering),
        # then emit a trailing PWD marker so the next call resumes in the same
        # working directory (each commands.run is a fresh process).
        safe_cwd = shlex.quote(self._cwd)
        inner = (
            f"cd {safe_cwd} && {{ {command}\n}} 2>&1\n"
            "status=$?\n"
            f"printf '\\n{_PWD_MARKER}%s\\n' \"$(pwd)\"\n"
            "exit $status"
        )
        full_command = f"bash -lc {shlex.quote(inner)}"

        timeout_s = self.config.timeout if timeout is None else timeout

        run_kwargs: dict[str, Any] = {
            "timeout": int(timeout_s),
            "user": self.config.user,
        }
        if self.config.request_timeout is not None:
            run_kwargs["request_timeout"] = self.config.request_timeout

        result = self._run_command(full_command, run_kwargs)

        stdout = _to_text(getattr(result, "stdout", ""))
        stderr = _to_text(getattr(result, "stderr", ""))
        output_str = stdout + stderr
        exit_code = int(getattr(result, "exit_code", 1) or 0)

        if exit_code == _TIMEOUT_SENTINEL:
            return ExecutionResult(
                output=output_str or f"Command timed out after {timeout_s}s",
                exit_code=124,
            )

        output_str, new_cwd = extract_cwd_marker(output_str, _PWD_MARKER)
        if new_cwd:
            self._cwd = new_cwd

        return ExecutionResult(output=output_str, exit_code=exit_code)

    def _run_command(self, command: str, run_kwargs: dict[str, Any]) -> Any:
        """Run a command, normalizing non-zero exit and timeout into a result.

        ``commands.run`` raises ``CommandExitException`` (which subclasses
        ``CommandResult``) on non-zero exit, and ``TimeoutException`` on timeout.
        We turn both into objects carrying ``stdout``/``stderr``/``exit_code``.
        """
        kwargs = dict(run_kwargs)
        while True:
            try:
                return self._sandbox.commands.run(command, **kwargs)
            except TypeError:
                # Drop unsupported kwargs progressively for SDK compatibility.
                if "request_timeout" in kwargs:
                    kwargs.pop("request_timeout")
                    continue
                if "user" in kwargs:
                    kwargs.pop("user")
                    continue
                raise
            except Exception as exc:
                name = type(exc).__name__
                if name == "CommandExitException":
                    return exc  # carries stdout/stderr/exit_code
                if name == "TimeoutException":
                    return _TimeoutResult(
                        stdout=_to_text(getattr(exc, "stdout", "")),
                        stderr=_to_text(getattr(exc, "stderr", "")),
                    )
                raise

    # ------------------------------------------------------------------
    # File operations with undo support
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str) -> ExecutionResult:
        current = self.execute(f"cat {shlex.quote(path)} 2>/dev/null")
        if current.exit_code == 0:
            self._file_history.setdefault(path, []).append(current.output)
        path_quoted = shlex.quote(path)
        cmd = (
            f"cat > {path_quoted} << 'NANOROLLOUT_EOF_31415926'\n"
            f"{content}\nNANOROLLOUT_EOF_31415926"
        )
        return self.execute(cmd)

    def undo_edit(self, path: str) -> ExecutionResult:
        if not self._file_history.get(path):
            return ExecutionResult(output="No edit history for this file", exit_code=1)
        previous_content = self._file_history[path].pop()
        path_quoted = shlex.quote(path)
        cmd = (
            f"cat > {path_quoted} << 'NANOROLLOUT_EOF_31415926'\n"
            f"{previous_content}\nNANOROLLOUT_EOF_31415926"
        )
        return self.execute(cmd)

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


_TIMEOUT_SENTINEL = -999


@dataclass
class _TimeoutResult:
    """Stand-in result for a timed-out command."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = _TIMEOUT_SENTINEL


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
