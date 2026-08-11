from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

_decoder = json.JSONDecoder().decode
_encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":")).encode
_dev_mode = os.environ.get("MOE_TOOLS_VERIFIER_DEV_MODE") == "1"
_child_env = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


def _demote() -> None:
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    elif not _dev_mode:
        os._exit(126)


def _call(function: str) -> object:
    payload = {
        "args": [],
        "function": function,
        "mode": "function",
        "module_path": "zebra_puzzle.py",
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
        raise AssertionError(completed.stderr or "isolated zebra runner failed")
    try:
        result = _decoder(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise AssertionError("isolated zebra runner returned invalid JSON") from error
    if result.get("status") != "returned":
        raise AssertionError(
            f"candidate raised {result.get('exception_type')}: {result.get('message')}"
        )
    return result.get("value")


# Exact assertion semantics from the pinned Exercism Python tests emitted by
# Harbor dataset f30b14415dd733c83627204bad0af69a89ceb46f.
class ZebraPuzzleTest(unittest.TestCase):
    def test_resident_who_drinks_water(self):
        self.assertEqual(_call("drinks_water"), "Norwegian")

    def test_resident_who_owns_zebra(self):
        self.assertEqual(_call("owns_zebra"), "Japanese")


if __name__ == "__main__":
    unittest.main(verbosity=2)
