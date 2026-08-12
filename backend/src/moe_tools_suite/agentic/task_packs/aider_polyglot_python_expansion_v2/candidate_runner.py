from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys

_decoder = json.JSONDecoder().decode
_encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":")).encode
_read = sys.stdin.read
_write = sys.__stdout__.write


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_function(payload: dict[str, object], module: object) -> dict[str, object]:
    function = getattr(module, str(payload["function"]))
    value = function(
        *payload.get("args", []),
        **payload.get("kwargs", {}),
    )
    return {"status": "returned", "value": value}


def main() -> None:
    payload = _decoder(_read())
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            module = _load_module(str(payload["module_path"]))
            if payload["mode"] != "function":
                raise ValueError("unsupported candidate-runner mode")
            result = _run_function(payload, module)
    except BaseException as error:
        result = {
            "status": "raised",
            "exception_type": type(error).__name__,
            "message": str(error),
        }
    _write(_encoder(result))


if __name__ == "__main__":
    main()
