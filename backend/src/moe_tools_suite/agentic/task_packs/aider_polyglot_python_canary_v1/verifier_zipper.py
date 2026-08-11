from __future__ import annotations

import copy
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


def _invoke(tree: dict[str, object], steps: list[dict[str, object]]) -> dict:
    payload = {
        "mode": "zipper",
        "module_path": "zipper.py",
        "steps": steps,
        "tree": tree,
    }
    completed = subprocess.run(
        [sys.executable, "-I", ".moe_tools/candidate_runner.py"],
        input=_encoder(payload),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
        cwd=".",
        env=_child_env,
        preexec_fn=_demote,
    )
    if completed.returncode != 0 or completed.stderr:
        raise AssertionError(completed.stderr or "isolated zipper runner failed")
    try:
        result = _decoder(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise AssertionError("isolated zipper runner returned invalid JSON") from error
    if result.get("status") != "returned":
        raise AssertionError(
            f"candidate raised {result.get('exception_type')}: {result.get('message')}"
        )
    return result


class Zipper:
    def __init__(self, tree: dict[str, object], steps: list[dict[str, object]]):
        self._tree = tree
        self._steps = steps

    @staticmethod
    def from_tree(tree: dict[str, object]):
        return Zipper(copy.deepcopy(tree), [])

    def _move(self, method: str, *args: object):
        steps = [*self._steps, {"args": list(args), "method": method}]
        result = _invoke(self._tree, steps)
        if result.get("kind") == "none":
            return None
        if result.get("kind") != "zipper":
            raise AssertionError(f"{method} did not return a zipper")
        return Zipper(self._tree, steps)

    def _value(self, method: str):
        steps = [*self._steps, {"args": [], "method": method}]
        result = _invoke(self._tree, steps)
        if result.get("kind") != "value":
            raise AssertionError(f"{method} did not return a value")
        return result.get("value")

    def value(self):
        return self._value("value")

    def set_value(self, value):
        return self._move("set_value", value)

    def left(self):
        return self._move("left")

    def set_left(self, tree):
        return self._move("set_left", tree)

    def right(self):
        return self._move("right")

    def set_right(self, tree):
        return self._move("set_right", tree)

    def up(self):
        return self._move("up")

    def to_tree(self):
        return self._value("to_tree")


def _initial_tree() -> dict[str, object]:
    return {
        "value": 1,
        "left": {
            "value": 2,
            "left": None,
            "right": {"value": 3, "left": None, "right": None},
        },
        "right": {"value": 4, "left": None, "right": None},
    }


# Adapted without changing cases from the pinned Exercism Python tests emitted by
# Harbor dataset f30b14415dd733c83627204bad0af69a89ceb46f.
class ZipperTest(unittest.TestCase):
    def test_data_is_retained(self):
        initial = _initial_tree()
        self.assertEqual(Zipper.from_tree(initial).to_tree(), _initial_tree())

    def test_left_right_and_value(self):
        zipper = Zipper.from_tree(_initial_tree())
        self.assertEqual(zipper.left().right().value(), 3)

    def test_dead_end(self):
        zipper = Zipper.from_tree(_initial_tree())
        self.assertIsNone(zipper.left().left())

    def test_tree_from_deep_focus(self):
        zipper = Zipper.from_tree(_initial_tree())
        self.assertEqual(zipper.left().right().to_tree(), _initial_tree())

    def test_traversing_up_from_top(self):
        zipper = Zipper.from_tree(_initial_tree())
        self.assertIsNone(zipper.up())

    def test_left_right_and_up(self):
        zipper = Zipper.from_tree(_initial_tree())
        result = zipper.left().up().right().up().left().right().value()
        self.assertEqual(result, 3)

    def test_test_ability_to_descend_multiple_levels_and_return(self):
        zipper = Zipper.from_tree(_initial_tree())
        self.assertEqual(zipper.left().right().up().up().value(), 1)

    def test_set_value(self):
        expected = _initial_tree()
        expected["left"]["value"] = 5
        zipper = Zipper.from_tree(_initial_tree())
        self.assertEqual(zipper.left().set_value(5).to_tree(), expected)

    def test_set_value_after_traversing_up(self):
        expected = _initial_tree()
        expected["left"]["value"] = 5
        zipper = Zipper.from_tree(_initial_tree())
        result = zipper.left().right().up().set_value(5).to_tree()
        self.assertEqual(result, expected)

    def test_set_left_with_leaf(self):
        leaf = {"value": 5, "left": None, "right": None}
        expected = _initial_tree()
        expected["left"]["left"] = leaf
        zipper = Zipper.from_tree(_initial_tree())
        self.assertEqual(zipper.left().set_left(leaf).to_tree(), expected)

    def test_set_right_with_null(self):
        expected = _initial_tree()
        expected["left"]["right"] = None
        zipper = Zipper.from_tree(_initial_tree())
        self.assertEqual(zipper.left().set_right(None).to_tree(), expected)

    def test_set_right_with_subtree(self):
        subtree = {
            "value": 6,
            "left": {"value": 7, "left": None, "right": None},
            "right": {"value": 8, "left": None, "right": None},
        }
        expected = _initial_tree()
        expected["right"] = subtree
        zipper = Zipper.from_tree(_initial_tree())
        self.assertEqual(zipper.set_right(subtree).to_tree(), expected)

    def test_set_value_on_deep_focus(self):
        expected = _initial_tree()
        expected["left"]["right"]["value"] = 5
        zipper = Zipper.from_tree(_initial_tree())
        result = zipper.left().right().set_value(5).to_tree()
        self.assertEqual(result, expected)

    def test_different_paths_to_same_zipper(self):
        result = Zipper.from_tree(_initial_tree()).left().up().right().to_tree()
        expected = Zipper.from_tree(_initial_tree()).right().to_tree()
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
