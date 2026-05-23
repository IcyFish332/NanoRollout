"""Kubernetes-based shell environment for NanoRollout.

Runs agent commands inside a Kubernetes pod via ``kubectl exec``.
Pod lifecycle: create pod (sleep infinity) → kubectl exec → delete pod.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .base import ExecutionResult, ShellEnvironment, extract_cwd_marker

logger = logging.getLogger(__name__)

_PWD_MARKER = "__NANOROLLOUT_PWD__"


@dataclass
class KubernetesEnvironmentConfig:
    """Configuration for the Kubernetes environment."""

    image: str
    namespace: str = "default"
    cwd: str = "/"
    timeout: int = 180
    create_timeout: int = 600
    startup_timeout: int = 900
    use_acr: bool = False
    acr_registry: str = ""
    acr_namespace: str = ""
    resource_requests: dict[str, str] = field(
        default_factory=lambda: {"memory": "16Gi", "cpu": "2"}
    )
    resource_limits: dict[str, str] = field(
        default_factory=lambda: {"memory": "16Gi", "cpu": "2"}
    )
    active_deadline_seconds: Optional[int] = 7200
    node_selector: dict[str, str] = field(
        default_factory=lambda: {"kubernetes.io/arch": "amd64"}
    )
    labels: dict[str, str] = field(
        default_factory=lambda: {"app": "nanorollout", "role": "eval"}
    )
    image_pull_secrets: list[str] = field(default_factory=list)


def _map_image_to_acr(image: str, registry: str, namespace: str) -> str:
    """Map a Docker Hub image reference to an ACR path.

    Example: docker.io/swebench/sweb.eval.x86_64.repo__issue:latest
         --> registry/namespace:swebench--sweb.eval.x86_64.repo__issue--latest
    """
    if ":" in image:
        image_part, tag = image.rsplit(":", 1)
    else:
        image_part, tag = image, "latest"

    segments = image_part.split("/")
    if len(segments) > 1 and (
        "." in segments[0] or ":" in segments[0] or segments[0] == "localhost"
    ):
        image_part = "/".join(segments[1:])

    if "latest" not in image:
        acr_tag = image_part.replace("/", "--")
    else:
        acr_tag = image_part.replace("/", "--") + f"--{tag}"

    return f"{registry}/{namespace}:{acr_tag}"


def _generate_pod_name(image: str) -> str:
    """Generate a DNS-1123 compliant pod name from an image reference."""
    raw = image.lower()
    parts = [p for p in raw.split("/") if p]
    if len(parts) > 1 and (
        "." in parts[0] or ":" in parts[0] or parts[0] == "localhost"
    ):
        parts = parts[1:]
    last = parts[-1] if parts else raw
    last_no_tag = last.split(":", 1)[0]
    if "__" in last_no_tag:
        core = last_no_tag.rsplit("__", 1)[-1]
    else:
        core = last_no_tag
    core = re.sub(r"[^a-z0-9-]", "-", core)
    core = re.sub(r"-{2,}", "-", core).strip("-") or "pod"

    prefix = "nro-"
    suffix = uuid.uuid4().hex[:8]
    max_core_len = 63 - len(prefix) - 1 - len(suffix)
    core = core[:max_core_len]
    return f"{prefix}{core}-{suffix}"


class KubernetesEnvironment(ShellEnvironment):
    """A Kubernetes pod-based execution environment.

    Creates a pod with the given image running ``sleep infinity``, then
    executes commands via ``kubectl exec``.
    """

    def __init__(
        self,
        image: str,
        instance: dict[str, Any],
        workspace_dir: str = "/",
        timeout: int = 120,
        namespace: str = "default",
        create_timeout: int = 600,
        startup_timeout: int = 900,
        use_acr: bool = False,
        acr_registry: str = "",
        acr_namespace: str = "",
        resource_requests: Optional[dict[str, str]] = None,
        resource_limits: Optional[dict[str, str]] = None,
        active_deadline_seconds: Optional[int] = 7200,
        node_selector: Optional[dict[str, str]] = None,
        labels: Optional[dict[str, str]] = None,
        image_pull_secrets: Optional[list[str]] = None,
        logger_override: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        self.logger = logger_override or logger
        self.instance = instance

        self.config = KubernetesEnvironmentConfig(
            image=image,
            namespace=namespace,
            cwd=workspace_dir,
            timeout=timeout,
            create_timeout=create_timeout,
            startup_timeout=startup_timeout,
            use_acr=use_acr,
            acr_registry=acr_registry,
            acr_namespace=acr_namespace,
            resource_requests=resource_requests or {"memory": "16Gi", "cpu": "2"},
            resource_limits=resource_limits or {"memory": "16Gi", "cpu": "2"},
            active_deadline_seconds=active_deadline_seconds,
            node_selector=node_selector or {"kubernetes.io/arch": "amd64"},
            labels=labels or {"app": "nanorollout", "role": "eval"},
            image_pull_secrets=image_pull_secrets or [],
        )

        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self._cwd = workspace_dir
        self._pod_name: Optional[str] = None
        self._orphan_reaper_process: Optional[subprocess.Popen] = None
        self._file_history: dict[str, list[str]] = {}

        if kwargs:
            self.logger.debug("Ignoring unknown kwargs: %s", list(kwargs.keys()))

    # ------------------------------------------------------------------
    # ShellEnvironment interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        image = self.config.image
        if self.config.use_acr and (
            "swebench" in image.lower() or "sweb.eval" in image
        ):
            image = _map_image_to_acr(
                image, self.config.acr_registry, self.config.acr_namespace
            )
            self.logger.info("Mapped image to ACR: %s", image)

        self._pod_name = _generate_pod_name(self.config.image)
        self.logger.info("Creating pod %s (image=%s)", self._pod_name, image)

        manifest = self._build_pod_manifest(image)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(manifest, f)
            manifest_path = f.name

        try:
            result = self._run_kubectl(
                ["apply", "-f", manifest_path], timeout_s=30
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Failed to create pod {self._pod_name}: {result.output}"
                )
        finally:
            Path(manifest_path).unlink(missing_ok=True)

        self._wait_for_ready()
        self._start_orphan_reaper()

    def stop(self) -> None:
        self._stop_orphan_reaper()
        if self._pod_name is not None:
            self.logger.info("Deleting pod %s", self._pod_name)
            try:
                self._run_kubectl(
                    [
                        "delete", "pod", self._pod_name,
                        f"--namespace={self.config.namespace}",
                        "--force", "--grace-period=0", "--wait=false",
                    ],
                    timeout_s=15,
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to delete pod %s: %s", self._pod_name, exc
                )
            finally:
                self._pod_name = None

    def is_running(self) -> bool:
        if self._pod_name is None:
            return False
        result = self._run_kubectl(
            [
                "get", "pod", self._pod_name,
                f"--namespace={self.config.namespace}",
                "-o", "jsonpath={.status.phase}",
            ],
            timeout_s=10,
        )
        return result.exit_code == 0 and result.output.strip() == "Running"

    def execute(self, command: str, timeout: Optional[int] = None) -> ExecutionResult:
        if self._pod_name is None:
            raise RuntimeError("Pod is not running")

        wrapped_command = (
            f"{command}\n"
            "status=$?\n"
            f"printf '\\n{_PWD_MARKER}%s\\n' \"$(pwd)\"\n"
            "exit $status"
        )
        safe_cwd = shlex.quote(self._cwd)
        full_command = f"cd {safe_cwd} && {wrapped_command}"

        timeout_s = self.config.timeout if timeout is None else timeout
        cmd = [
            "kubectl", "exec", self._pod_name,
            f"--namespace={self.config.namespace}",
            "--", "bash", "-lc", full_command,
        ]

        try:
            result = self._run_subprocess(cmd, timeout_s=timeout_s)
        except OSError as exc:
            if exc.errno != errno.E2BIG:
                raise
            stdin_cmd = [
                "kubectl", "exec", "-i", self._pod_name,
                f"--namespace={self.config.namespace}",
                "--", "bash", "-l", "-s",
            ]
            result = self._run_subprocess(
                stdin_cmd, timeout_s=timeout_s, stdin_data=full_command
            )

        output_str = result.output or ""
        output_str, new_cwd = extract_cwd_marker(output_str, _PWD_MARKER)
        if new_cwd:
            self._cwd = new_cwd

        return ExecutionResult(output=output_str, exit_code=result.exit_code)

    # ------------------------------------------------------------------
    # File operations with undo support
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str) -> ExecutionResult:
        current = self.execute(f"cat {shlex.quote(path)} 2>/dev/null")
        if current.exit_code == 0:
            if path not in self._file_history:
                self._file_history[path] = []
            self._file_history[path].append(current.output)
        path_quoted = shlex.quote(path)
        cmd = (
            f"cat > {path_quoted} << 'NANOROLLOUT_EOF_31415926'\n"
            f"{content}\nNANOROLLOUT_EOF_31415926"
        )
        return self.execute(cmd)

    def undo_edit(self, path: str) -> ExecutionResult:
        if path not in self._file_history or not self._file_history[path]:
            return ExecutionResult(output="No edit history for this file", exit_code=1)
        previous_content = self._file_history[path].pop()
        path_quoted = shlex.quote(path)
        cmd = (
            f"cat > {path_quoted} << 'NANOROLLOUT_EOF_31415926'\n"
            f"{previous_content}\nNANOROLLOUT_EOF_31415926"
        )
        return self.execute(cmd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pod_manifest(self, image: str) -> dict[str, Any]:
        container: dict[str, Any] = {
            "name": "main",
            "image": image,
            "command": ["sleep", "infinity"],
            "resources": {
                "requests": dict(self.config.resource_requests),
                "limits": dict(self.config.resource_limits),
            },
        }

        spec: dict[str, Any] = {
            "containers": [container],
            "nodeSelector": dict(self.config.node_selector),
            "restartPolicy": "Never",
        }

        if self.config.active_deadline_seconds is not None:
            spec["activeDeadlineSeconds"] = self.config.active_deadline_seconds

        if self.config.image_pull_secrets:
            spec["imagePullSecrets"] = [
                {"name": s} for s in self.config.image_pull_secrets
            ]

        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self._pod_name,
                "namespace": self.config.namespace,
                "labels": dict(self.config.labels),
            },
            "spec": spec,
        }

    def _wait_for_ready(self) -> None:
        deadline = time.time() + self.config.startup_timeout
        while time.time() < deadline:
            result = self._run_kubectl(
                [
                    "get", "pod", self._pod_name,
                    f"--namespace={self.config.namespace}",
                    "-o", "json",
                ],
                timeout_s=10,
            )
            if result.exit_code == 0:
                try:
                    pod_json = json.loads(result.output)
                except json.JSONDecodeError:
                    time.sleep(2)
                    continue

                conditions = pod_json.get("status", {}).get("conditions", [])
                if any(
                    c.get("type") == "Ready" and c.get("status") == "True"
                    for c in conditions
                ):
                    self.logger.info("Pod %s is ready", self._pod_name)
                    return

                phase = pod_json.get("status", {}).get("phase", "")
                if phase in ("Failed", "Succeeded"):
                    raise RuntimeError(
                        f"Pod {self._pod_name} entered terminal phase: {phase}"
                    )

                container_statuses = pod_json.get("status", {}).get(
                    "containerStatuses", []
                )
                for cs in container_statuses:
                    waiting = cs.get("state", {}).get("waiting", {})
                    reason = waiting.get("reason", "")
                    if reason in (
                        "ImagePullBackOff",
                        "ErrImagePull",
                        "InvalidImageName",
                        "ErrImageNeverPull",
                    ):
                        raise RuntimeError(
                            f"Pod {self._pod_name} image pull failed: {reason} "
                            f"- {waiting.get('message', '')}"
                        )

            time.sleep(2)

        raise TimeoutError(
            f"Pod {self._pod_name} did not become ready within "
            f"{self.config.startup_timeout}s"
        )

    def _run_kubectl(self, args: list[str], timeout_s: int = 30) -> ExecutionResult:
        return self._run_subprocess(["kubectl", *args], timeout_s=timeout_s)

    def _run_subprocess(
        self,
        cmd: list[str],
        *,
        timeout_s: int,
        stdin_data: Optional[str] = None,
    ) -> ExecutionResult:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(input=stdin_data, timeout=timeout_s)
            return ExecutionResult(
                output=stdout or "",
                exit_code=(
                    process.returncode if process.returncode is not None else 1
                ),
            )
        except subprocess.TimeoutExpired as exc:
            self.logger.warning(
                "Command timed out after %ss, killing process group: %s",
                timeout_s,
                shlex.join(cmd[:6]),
            )
            self._terminate_process_group(process, term_timeout_s=5.0)
            partial_output = exc.stdout or ""
            timeout_msg = f"Command timed out after {timeout_s}s"
            output = (
                f"{partial_output.rstrip()}\n{timeout_msg}"
                if partial_output
                else timeout_msg
            )
            return ExecutionResult(output=output, exit_code=124)

    def _terminate_process_group(
        self, process: subprocess.Popen, *, term_timeout_s: float = 5.0
    ) -> None:
        if process.poll() is not None:
            return
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=term_timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self.logger.warning(
                "Process group did not exit after SIGKILL (pid=%s)", process.pid
            )

    # ------------------------------------------------------------------
    # Orphan reaper
    # ------------------------------------------------------------------

    def _start_orphan_reaper(self) -> None:
        self._stop_orphan_reaper()
        if self._pod_name is None:
            return

        parent_pid = os.getpid()
        pod_name = self._pod_name
        namespace = self.config.namespace

        reaper_script = textwrap.dedent(f"""\
            import os, subprocess, time
            parent_pid = {parent_pid}
            pod_name = {pod_name!r}
            namespace = {namespace!r}
            while True:
                try:
                    os.kill(parent_pid, 0)
                except OSError:
                    break
                time.sleep(2.0)
            try:
                subprocess.run(
                    ["kubectl", "delete", "pod", pod_name,
                     f"--namespace={{namespace}}",
                     "--force", "--grace-period=0", "--wait=false"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    check=False, timeout=30,
                )
            except Exception:
                pass
        """)

        python_bin = sys.executable or "python3"
        try:
            self._orphan_reaper_process = subprocess.Popen(
                [python_bin, "-c", reaper_script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.logger.debug(
                "Started orphan reaper (pid=%s) for pod %s",
                self._orphan_reaper_process.pid,
                pod_name,
            )
        except Exception as exc:
            self._orphan_reaper_process = None
            self.logger.warning("Failed to start orphan reaper: %s", exc)

    def _stop_orphan_reaper(self) -> None:
        proc = self._orphan_reaper_process
        if proc is None:
            return
        self._orphan_reaper_process = None
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
