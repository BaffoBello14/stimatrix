from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union
import os
import time


def _truthy(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off", "", "none"}:
            return False
    return default


def _safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, None)
        if cur is None:
            return default
    return cur


@dataclass
class _WandbState:
    enabled: bool = False
    run: Any = None  # wandb.sdk.wandb_run.Run, but keep untyped to avoid hard dependency
    module: Any = None  # the imported wandb module


class WandbTracker:
    """Thin wrapper around wandb to keep the core codebase decoupled.

    - Does nothing if disabled or if the module is unavailable
    - Reads configuration from config["tracking"]["wandb"] and environment variables
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config: Dict[str, Any] = config or {}
        tracking_cfg: Dict[str, Any] = self.config.get("tracking", {})
        self.wandb_cfg: Dict[str, Any] = tracking_cfg.get("wandb", {})
        # Enabled if set in config or env var
        self.state = _WandbState(
            enabled=_truthy(
                self.wandb_cfg.get("enabled", os.environ.get("WANDB_ENABLED", "false")),
                default=False,
            )
        )

    def _import(self) -> bool:
        if not self.state.enabled:
            return False
        if self.state.module is not None:
            return True
        try:
            import wandb  # type: ignore
        except Exception:
            # Disable if module not available
            self.state.enabled = False
            self.state.module = None
            return False
        self.state.module = wandb
        return True

    def start_run(self, job_type: str = "training", run_name: Optional[str] = None) -> None:
        if not self._import():
            return
        wandb = self.state.module

        # Resolve configuration values with env fallbacks
        project = str(
            self.wandb_cfg.get("project", os.environ.get("WANDB_PROJECT", "stimatrix"))
        )
        entity = self.wandb_cfg.get("entity", os.environ.get("WANDB_ENTITY", None))
        group = self.wandb_cfg.get("group", os.environ.get("WANDB_GROUP", None))
        tags = self.wandb_cfg.get("tags", os.environ.get("WANDB_TAGS", "").split(",") if os.environ.get("WANDB_TAGS") else None)
        mode = str(self.wandb_cfg.get("mode", os.environ.get("WANDB_MODE", "online"))).lower()
        dir_override = self.wandb_cfg.get("dir", os.environ.get("WANDB_DIR", None))

        # Build run name if not provided
        resolved_name = run_name or self.wandb_cfg.get("name")
        if not resolved_name:
            resolved_name = f"{job_type}_{time.strftime('%Y%m%d_%H%M%S')}"

        init_kwargs: Dict[str, Any] = {
            "project": project,
            "name": resolved_name,
            "config": self.config,
            "job_type": job_type,
        }
        if entity:
            init_kwargs["entity"] = entity
        if group:
            init_kwargs["group"] = group
        if tags:
            init_kwargs["tags"] = tags
        if mode in {"offline", "disabled"}:
            init_kwargs["mode"] = mode
        if dir_override:
            init_kwargs["dir"] = dir_override

        try:
            self.state.run = wandb.init(**init_kwargs)
        except Exception:
            # Fail safe: disable tracking if init fails
            self.state.run = None
            self.state.enabled = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.state.enabled or self.state.module is None or self.state.run is None:
            return
        try:
            if step is None:
                self.state.module.log(data)
            else:
                self.state.module.log(data, step=step)
        except Exception:
            pass

    def log_prefixed_metrics(self, prefix: str, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        payload: Dict[str, Any] = {}
        for k, v in (metrics or {}).items():
            payload[f"{prefix}/{k}"] = v
        if payload:
            self.log(payload, step=step)

    def log_image(self, key: str, image_path: Union[str, Path]) -> None:
        if not self.state.enabled or self.state.module is None or self.state.run is None:
            return
        try:
            wandb = self.state.module
            self.state.module.log({key: wandb.Image(str(image_path))})
        except Exception:
            pass

    def log_artifact(self, target: Union[str, Path], name: str, type: str = "artifact", description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.state.enabled or self.state.module is None or self.state.run is None:
            return
        try:
            wandb = self.state.module
            artifact = wandb.Artifact(name=name, type=type, description=description, metadata=metadata)
            target_path = Path(target)
            if target_path.is_dir():
                artifact.add_dir(str(target_path))
            elif target_path.is_file():
                artifact.add_file(str(target_path))
            else:
                return
            self.state.run.log_artifact(artifact)
        except Exception:
            pass

    def finish(self) -> None:
        if not self.state.enabled or self.state.module is None or self.state.run is None:
            return
        try:
            self.state.run.finish()
        except Exception:
            pass

