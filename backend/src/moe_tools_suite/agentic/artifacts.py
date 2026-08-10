from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..telemetry import DecodedRouting
from .domain import RoutingArtifactInfo, TrialRoutingSummary

_SAFE_ID = re.compile(r"^[A-Za-z0-9-]{1,128}$")


class AgentArtifactStore:
    """Atomic, content-addressed files for agent trials."""

    def __init__(self, data_dir: Path) -> None:
        self.root = (data_dir / "artifacts" / "agent-trials").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_inference_routing(
        self,
        trial_id: str,
        inference_id: str,
        routing: DecodedRouting,
    ) -> RoutingArtifactInfo:
        directory = self._trial_dir(trial_id) / "routing"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._safe_id(inference_id)}.npz"
        temporary = self._temporary(path)
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                expert_ids=routing.expert_ids,
                expert_weights=routing.expert_weights,
            )
        os.replace(temporary, path)
        payload = path.read_bytes()
        return RoutingArtifactInfo(
            inference_id=inference_id,
            relative_path=path.relative_to(self.root).as_posix(),
            sha256=hashlib.sha256(payload).hexdigest(),
            token_count=int(routing.expert_ids.shape[0]),
            total_routed_slots=int(routing.expert_ids.size),
        )

    def save_routing_summary(self, summary: TrialRoutingSummary) -> None:
        path = self._trial_dir(summary.trial_id) / "routing-summary.npz"
        temporary = self._temporary(path)
        artifact_metadata = [
            artifact.model_dump(mode="json") for artifact in summary.artifacts
        ]
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                layer_ids=np.asarray(summary.layer_ids, dtype=np.int32),
                selection_counts=np.asarray(summary.selection_counts, dtype=np.int64),
                routing_mass=np.asarray(summary.routing_mass, dtype=np.float64),
                total_routed_slots=np.asarray(
                    summary.total_routed_slots, dtype=np.int64
                ),
                inference_count=np.asarray(summary.inference_count, dtype=np.int64),
                metadata_json=np.asarray(
                    json.dumps(
                        {
                            "run_id": summary.run_id,
                            "inference_calls": summary.inference_calls,
                            "captured_inference_calls": (
                                summary.captured_inference_calls
                            ),
                            "served_tokens": summary.served_tokens,
                            "model_id": summary.model_id,
                            "model_session_id": summary.model_session_id,
                            "profile_id": summary.profile_id,
                            "profile_fingerprint": summary.profile_fingerprint,
                        },
                        separators=(",", ":"),
                    )
                ),
                artifacts_json=np.asarray(
                    json.dumps(artifact_metadata, separators=(",", ":"))
                ),
            )
        os.replace(temporary, path)

    def load_routing_summary(self, trial_id: str) -> TrialRoutingSummary | None:
        path = self._trial_dir(trial_id, create=False) / "routing-summary.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"]))
            return TrialRoutingSummary(
                trial_id=trial_id,
                run_id=metadata["run_id"],
                layer_ids=payload["layer_ids"].astype(int).tolist(),
                selection_counts=payload["selection_counts"].tolist(),
                routing_mass=payload["routing_mass"].tolist(),
                total_routed_slots=int(payload["total_routed_slots"]),
                inference_count=int(payload["inference_count"]),
                inference_calls=metadata["inference_calls"],
                captured_inference_calls=metadata["captured_inference_calls"],
                served_tokens=metadata["served_tokens"],
                model_id=metadata["model_id"],
                model_session_id=metadata["model_session_id"],
                profile_id=metadata["profile_id"],
                profile_fingerprint=metadata["profile_fingerprint"],
                artifacts=json.loads(str(payload["artifacts_json"])),
            )

    def save_json(self, trial_id: str, name: str, payload: object) -> Path:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        return self.save_bytes(trial_id, name, (encoded + "\n").encode())

    def load_json(self, trial_id: str, name: str) -> dict[str, object] | None:
        path = self.path(trial_id, name)
        if not path.is_file():
            return None
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"agent artifact {name!r} must contain an object")
        return value

    def save_text(self, trial_id: str, name: str, payload: str) -> Path:
        return self.save_bytes(trial_id, name, payload.encode())

    def save_bytes(self, trial_id: str, name: str, payload: bytes) -> Path:
        path = self.path(trial_id, name)
        temporary = self._temporary(path)
        temporary.write_bytes(payload)
        os.replace(temporary, path)
        return path

    def read_text(self, trial_id: str, name: str) -> str | None:
        path = self.path(trial_id, name)
        return path.read_text() if path.is_file() else None

    def path(self, trial_id: str, name: str) -> Path:
        if name not in {
            "trajectory.json",
            "patch.diff",
            "verifier.json",
            "manifest.json",
        }:
            raise ValueError("unsupported agent artifact name")
        return self._trial_dir(trial_id) / name

    def _trial_dir(self, trial_id: str, *, create: bool = True) -> Path:
        directory = (self.root / self._safe_id(trial_id)).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("agent artifact escaped the configured data directory")
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _safe_id(value: str) -> str:
        if not _SAFE_ID.fullmatch(value):
            raise ValueError("unsafe agent artifact identifier")
        return value

    @staticmethod
    def _temporary(path: Path) -> Path:
        return path.with_name(f".{path.name}.{uuid4().hex}.tmp")
