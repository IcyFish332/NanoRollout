"""Bridge between NanoRollout's ShellEnvironment and SWE-agent's SWEEnv interface.

Allows SWE-agent's DefaultAgent to run against a NanoRollout-managed container
(Docker, K8s, Enroot) without needing SWE-ReX's deployment/runtime stack.
"""

from __future__ import annotations

import base64
import io
import logging
import shlex
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from nanorollout.envs.shell_env.base import ShellEnvironment

logger = logging.getLogger(__name__)


@dataclass
class _UploadRequest:
    source_path: str
    target_path: str


@dataclass
class _Command:
    command: str
    shell: bool = True
    check: bool = False
    env: Optional[dict[str, str]] = None
    cwd: Optional[str] = None


class _BridgeRuntime:
    """Minimal SWE-ReX runtime interface for ToolHandler.install()."""

    def __init__(self, bridge: "SWEEnvShellBridge"):
        self._bridge = bridge

    async def upload(self, request: Any) -> None:
        source = getattr(request, "source_path", None) or request.get("source_path")
        target = getattr(request, "target_path", None) or request.get("target_path")
        self._bridge.upload_path(Path(source), target)

    async def execute(self, command: Any) -> Any:
        cmd_str = getattr(command, "command", None) or (command if isinstance(command, str) else "")
        check = getattr(command, "check", False)
        env = getattr(command, "env", None)
        cwd = getattr(command, "cwd", None)
        self._bridge.execute_command(cmd_str, check=check, env=env, cwd=cwd)

        @dataclass
        class _Result:
            exit_code: int = 0
        return _Result()


class _BridgeDeployment:
    """Minimal SWE-ReX deployment interface."""

    def __init__(self, bridge: "SWEEnvShellBridge"):
        self.runtime = _BridgeRuntime(bridge)

    async def is_alive(self, timeout: float = 10) -> bool:
        return True


class SWEEnvShellBridge:
    """Adapts a NanoRollout ShellEnvironment to look like SWE-agent's SWEEnv.

    SWE-agent's DefaultAgent calls:
      - env.communicate(cmd, timeout, check) -> str
      - env.read_file(path) -> str
      - env.write_file(path, content) -> None
      - env.set_env_variables(dict) -> None
      - env.execute_command(cmd, ...) -> None
      - env.interrupt_session() -> None

    ToolHandler.install() also accesses:
      - env.deployment.runtime.upload(UploadRequest)
      - env.deployment.runtime.execute(Command)

    This bridge translates these into ShellEnvironment.execute() calls.
    Environment variables are tracked locally and prepended to each command.
    """

    def __init__(
        self,
        shell_env: ShellEnvironment,
        name: str = "main",
    ):
        self._env = shell_env
        self._env_vars: dict[str, str] = {}
        self.name = name
        self.repo = None
        self.deployment = _BridgeDeployment(self)

    def communicate(
        self,
        input: str,
        timeout: int | float = 25,
        *,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        self._track_exports(input)
        export_prefix = self._build_export_prefix()
        full_cmd = f"{export_prefix}{input}" if export_prefix else input

        result = self._env.execute(full_cmd, timeout=int(timeout))

        if result.exit_code == 0 and "source " in input:
            self._sync_env_after_source(export_prefix, input, timeout)

        if check == "raise" and result.exit_code != 0:
            raise RuntimeError(f"{error_msg}: exit code {result.exit_code}\n{result.output}")
        if check == "warn" and result.exit_code != 0:
            logger.warning("%s (exit_code=%d): %s", error_msg, result.exit_code, result.output[:200])

        return result.output

    def read_file(
        self,
        path: str | Path,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
    ) -> str:
        result = self._env.execute(f"cat {shlex.quote(str(path))}")
        if result.exit_code != 0:
            raise FileNotFoundError(f"Cannot read {path}: {result.output}")
        return result.output

    def write_file(
        self,
        path: str | Path,
        content: str,
    ) -> None:
        self._env.write_file(str(path), content)

    def set_env_variables(self, env_variables: dict[str, str]) -> None:
        self._env_vars.update(env_variables)
        export_cmd = " && ".join(
            f"export {k}={shlex.quote(v)}" for k, v in env_variables.items()
        )
        if export_cmd:
            self._env.execute(export_cmd)

    def execute_command(
        self,
        command: str,
        shell: bool = True,
        check: bool = False,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str] = None,
    ) -> None:
        parts = []
        if cwd:
            parts.append(f"cd {shlex.quote(cwd)}")
        if env:
            parts.extend(f"export {k}={shlex.quote(v)}" for k, v in env.items())
        parts.append(command)
        full_cmd = " && ".join(parts)
        result = self._env.execute(full_cmd)
        if check and result.exit_code != 0:
            raise RuntimeError(f"execute_command failed (exit={result.exit_code}): {result.output[:500]}")

    def interrupt_session(self) -> None:
        pass

    def hard_reset(self) -> None:
        pass

    def upload_path(self, source: Path, target: str) -> None:
        """Upload a local directory/file to the container via tar+base64."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(source), arcname=".")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        self._env.execute(f"mkdir -p {shlex.quote(target)}")

        chunk_size = 65536
        if len(encoded) <= chunk_size:
            self._env.execute(
                f"echo '{encoded}' | base64 -d | tar xzf - -C {shlex.quote(target)}"
            )
        else:
            tmp_file = f"/tmp/_nro_upload_{id(self)}.b64"
            self._env.execute(f"rm -f {tmp_file}")
            for i in range(0, len(encoded), chunk_size):
                chunk = encoded[i:i + chunk_size]
                self._env.execute(f"echo -n '{chunk}' >> {tmp_file}")
            self._env.execute(
                f"base64 -d {tmp_file} | tar xzf - -C {shlex.quote(target)} && rm -f {tmp_file}"
            )

    def close(self) -> None:
        pass

    def add_hook(self, hook: Any) -> None:
        if hasattr(hook, "on_init"):
            hook.on_init(env=self)

    def _build_export_prefix(self) -> str:
        if not self._env_vars:
            return ""
        parts = []
        for k, v in self._env_vars.items():
            if "$" in v:
                parts.append(f'export {k}="{v}"')
            else:
                parts.append(f"export {k}={shlex.quote(v)}")
        return " && ".join(parts) + " && "

    def _track_exports(self, cmd: str) -> None:
        """Parse export statements from a command and persist them in _env_vars.

        This is critical because NanoRollout's ShellEnvironment doesn't maintain
        a persistent bash session. Without this, env vars set via communicate()
        (like PATH updates during tool installation) would be lost between calls.
        """
        import re
        for segment in cmd.split("&&"):
            segment = segment.strip()
            m = re.match(r"^export\s+(\w+)=(.+)$", segment)
            if m:
                key = m.group(1)
                value = m.group(2).strip()
                if value.startswith(("'", '"')) and value.endswith(value[0]):
                    value = value[1:-1]
                if key == "PATH" and "$PATH" in value:
                    existing = self._env_vars.get("PATH", "$PATH")
                    value = value.replace("$PATH", existing)
                elif key == "PYTHONPATH" and "$PYTHONPATH" in value:
                    existing = self._env_vars.get("PYTHONPATH", "$PYTHONPATH")
                    value = value.replace("$PYTHONPATH", existing)
                self._env_vars[key] = value

    def _sync_env_after_source(
        self, export_prefix: str, original_cmd: str, timeout: int | float
    ) -> None:
        """After sourcing a script, capture PATH and PYTHONPATH changes."""
        env_query = f"{export_prefix}{original_cmd} && echo __NRO_ENV_SYNC__ && echo PATH=$PATH && echo PYTHONPATH=$PYTHONPATH"
        result = self._env.execute(env_query, timeout=int(timeout))
        if result.exit_code != 0 or "__NRO_ENV_SYNC__" not in result.output:
            return
        _, env_section = result.output.split("__NRO_ENV_SYNC__", 1)
        for line in env_section.strip().splitlines():
            if line.startswith("PATH="):
                self._env_vars["PATH"] = line[5:]
            elif line.startswith("PYTHONPATH="):
                self._env_vars["PYTHONPATH"] = line[11:]
