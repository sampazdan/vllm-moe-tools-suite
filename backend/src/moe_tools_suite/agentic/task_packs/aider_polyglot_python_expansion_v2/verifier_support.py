from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence

_decoder = json.JSONDecoder().decode
_encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":")).encode
_dev_mode = os.environ.get("MOE_TOOLS_VERIFIER_DEV_MODE") == "1"
_child_env = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
_allowed_exceptions = {"ValueError": ValueError}


def _demote() -> None:
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    elif not _dev_mode:
        os._exit(126)


def call(
    module_path: str,
    function: str,
    args: Sequence[object] = (),
    kwargs: Mapping[str, object] | None = None,
) -> object:
    payload = {
        "args": list(args),
        "function": function,
        "kwargs": dict(kwargs or {}),
        "mode": "function",
        "module_path": module_path,
    }
    completed = subprocess.run(
        [sys.executable, "-I", ".moe_tools/candidate_runner.py"],
        input=_encoder(payload),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=".",
        env=_child_env,
        preexec_fn=_demote,
    )
    if completed.returncode != 0 or completed.stderr:
        raise AssertionError(completed.stderr or "isolated candidate runner failed")
    try:
        result = _decoder(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        message = "isolated candidate runner returned invalid JSON"
        raise AssertionError(message) from error
    if result.get("status") == "raised":
        exception_type = result.get("exception_type")
        exception = _allowed_exceptions.get(exception_type)
        if exception is not None:
            raise exception(result.get("message", ""))
        raise AssertionError(
            f"candidate raised {exception_type}: {result.get('message')}"
        )
    if result.get("status") != "returned":
        raise AssertionError("isolated candidate runner returned invalid status")
    return result.get("value")
