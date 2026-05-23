"""NanoRollout agent wrapper around SWE-agent's DefaultAgent."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from nanorollout.envs.shell_env.base import ShellEnvironment
from nanorollout.harness.agents.swe.base import AgentResult

from .sweagent_bridge import SWEEnvShellBridge

logger = logging.getLogger(__name__)

# Path inside this repo to the canonical agent config (templates + tools).
# `eval.agent` block is consumed; its non-SWE-agent keys are filtered.
_DEFAULT_CONFIG_RELPATH = Path("configs/swe/default.yaml")


class SWEAgentWrapper:
    """Wraps SWE-agent's DefaultAgent for NanoRollout's lifecycle.

    LLM calls happen on the host. Only bash commands go to the pod
    via the SWEEnvShellBridge → ShellEnvironment → kubectl exec chain.
    """

    def __init__(
        self,
        shell_env: ShellEnvironment,
        model_name: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_iterations: int = 30,
        step_timeout: int = 600,
        sweagent_config_path: Optional[str] = None,
    ):
        self._shell_env = shell_env
        self._model_name = model_name
        self._api_base = api_base
        self._api_key = api_key or "EMPTY"
        self._max_iterations = max_iterations
        self._step_timeout = step_timeout
        self._config_path = sweagent_config_path

    def run(self, task: str) -> AgentResult:
        from sweagent.agent.agents import DefaultAgent
        from sweagent.agent.problem_statement import TextProblemStatement

        bridge = SWEEnvShellBridge(self._shell_env)

        agent_config = self._build_agent_config()
        agent = DefaultAgent.from_config(agent_config)

        problem = TextProblemStatement(text=task, id="instance")

        output_dir = Path("/tmp/sweagent_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = agent.run(
                env=bridge,
                problem_statement=problem,
                output_dir=output_dir,
            )
        except Exception as e:
            logger.exception("SWE-agent run failed: %s", e)
            patch = self._try_get_patch(bridge)
            return AgentResult(
                success=False,
                message=str(e),
                history=[],
                llm_metrics=[],
                llm_cost_total=0.0,
                patch=patch,
                iterations=0,
                error=str(e),
                exit_status="error",
            )

        submission = result.info.get("submission")
        patch = submission if submission else self._try_get_patch(bridge)

        exit_status = result.info.get("exit_status", "unknown")
        nro_exit_status = self._map_exit_status(exit_status)

        history = []
        for step in result.trajectory:
            if step.get("thought"):
                history.append({"role": "assistant", "content": step["thought"]})
            if step.get("action"):
                history.append({"role": "assistant", "content": step["action"]})
            if step.get("observation"):
                history.append({"role": "user", "content": step["observation"]})

        model_stats = result.info.get("model_stats", {})

        return AgentResult(
            success=nro_exit_status == "finished",
            message=patch or "",
            history=history,
            llm_metrics=[model_stats] if model_stats else [],
            llm_cost_total=model_stats.get("instance_cost", 0.0),
            patch=patch or "",
            iterations=len(result.trajectory),
            error=None,
            exit_status=nro_exit_status,
        )

    def _build_agent_config(self) -> "DefaultAgentConfig":
        from sweagent.agent.agents import DefaultAgentConfig
        from sweagent.agent.models import GenericAPIModelConfig

        agent_raw = self._load_agent_config_block()

        # SWE-agent resolves bundle paths (e.g. tools/registry) relative to
        # SWE_AGENT_CONFIG_ROOT (or sweagent's REPO_ROOT). Pin it to the
        # SWE-agent repo so the bundles in tools/ are found.
        os.environ.setdefault("SWE_AGENT_CONFIG_ROOT", self._sweagent_repo_root())

        # Project's configs/swe/default.yaml carries NRO-specific knobs in
        # `agent.model` and `agent.*` that SWE-agent's pydantic models reject
        # (extra="forbid"). Keep only the keys SWE-agent understands.
        model_raw = self._filter_model_dict(dict(agent_raw.get("model") or {}))
        model_raw["name"] = self._model_name
        if self._api_base:
            model_raw["api_base"] = self._api_base
        model_raw["api_key"] = self._api_key
        model_raw["per_instance_cost_limit"] = 0.0
        model_raw["total_cost_limit"] = 0.0
        model_raw.setdefault("temperature", 0.7)

        tools_raw = dict(agent_raw.get("tools") or {})
        tools_raw.setdefault("execution_timeout", self._step_timeout)

        agent_dict: dict[str, Any] = {
            "model": model_raw,
            "tools": tools_raw,
            "templates": dict(agent_raw.get("templates") or {}),
            "max_requeries": agent_raw.get("max_requeries", 3),
        }
        if "history_processors" in agent_raw and agent_raw["history_processors"]:
            agent_dict["history_processors"] = agent_raw["history_processors"]

        return DefaultAgentConfig.model_validate(agent_dict)

    def _load_agent_config_block(self) -> dict:
        import yaml

        if self._config_path:
            path = Path(self._config_path)
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            # Allow either a flat `agent:` doc or the project's eval.agent layout.
            if "agent" in raw:
                return raw["agent"] or {}
            if "eval" in raw and isinstance(raw["eval"], dict):
                return raw["eval"].get("agent") or {}
            return raw

        default_path = self._project_root() / _DEFAULT_CONFIG_RELPATH
        if default_path.exists():
            with open(default_path) as f:
                raw = yaml.safe_load(f) or {}
            return (raw.get("eval") or {}).get("agent") or {}
        logger.warning("No SWE-agent config found at %s; using empty config.", default_path)
        return {}

    @staticmethod
    def _filter_model_dict(model_raw: dict) -> dict:
        from sweagent.agent.models import GenericAPIModelConfig

        allowed = set(GenericAPIModelConfig.model_fields.keys())
        return {k: v for k, v in model_raw.items() if k in allowed}

    @staticmethod
    def _project_root() -> Path:
        # .../project_swe_rl/3rdparty/NanoRollout/nanorollout/harness/agents/swe/sweagent_wrapper.py
        # parents:                                  [0]=swe [1]=agents [2]=harness [3]=nanorollout
        #                                           [4]=NanoRollout [5]=3rdparty [6]=project_swe_rl
        return Path(__file__).resolve().parents[6]

    @staticmethod
    def _sweagent_repo_root() -> Path:
        import sweagent
        return str(Path(sweagent.__file__).resolve().parent.parent)

    def _try_get_patch(self, bridge: SWEEnvShellBridge) -> str:
        try:
            return bridge.read_file("/root/model.patch")
        except (FileNotFoundError, RuntimeError):
            try:
                return self._shell_env.get_git_diff()
            except Exception:
                return ""

    @staticmethod
    def _map_exit_status(sweagent_status: str) -> str:
        if not sweagent_status:
            return "unknown"
        if "submitted" in sweagent_status:
            return "finished"
        if "exit_context" in sweagent_status:
            return "max_length"
        if "exit_format" in sweagent_status:
            return "error"
        if "exit_cost" in sweagent_status:
            return "max_iterations"
        return sweagent_status
