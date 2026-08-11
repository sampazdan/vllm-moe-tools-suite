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


def answer(question: str) -> int:
    payload = {
        "args": [question],
        "function": "answer",
        "mode": "function",
        "module_path": "wordy.py",
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
        raise AssertionError(completed.stderr or "isolated wordy runner failed")
    try:
        result = _decoder(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise AssertionError("isolated wordy runner returned invalid JSON") from error
    if result.get("status") == "raised":
        if result.get("exception_type") == "ValueError":
            raise ValueError(result.get("message", ""))
        raise AssertionError(
            f"candidate raised {result.get('exception_type')}: {result.get('message')}"
        )
    return result.get("value")


# Adapted without changing cases from the pinned Exercism Python tests emitted by
# Harbor dataset f30b14415dd733c83627204bad0af69a89ceb46f.
class WordyTest(unittest.TestCase):
    def test_just_a_number(self):
        self.assertEqual(answer("What is 5?"), 5)

    def test_addition(self):
        self.assertEqual(answer("What is 1 plus 1?"), 2)

    def test_more_addition(self):
        self.assertEqual(answer("What is 53 plus 2?"), 55)

    def test_addition_with_negative_numbers(self):
        self.assertEqual(answer("What is -1 plus -10?"), -11)

    def test_large_addition(self):
        self.assertEqual(answer("What is 123 plus 45678?"), 45801)

    def test_subtraction(self):
        self.assertEqual(answer("What is 4 minus -12?"), 16)

    def test_multiplication(self):
        self.assertEqual(answer("What is -3 multiplied by 25?"), -75)

    def test_division(self):
        self.assertEqual(answer("What is 33 divided by -3?"), -11)

    def test_multiple_additions(self):
        self.assertEqual(answer("What is 1 plus 1 plus 1?"), 3)

    def test_addition_and_subtraction(self):
        self.assertEqual(answer("What is 1 plus 5 minus -2?"), 8)

    def test_multiple_subtraction(self):
        self.assertEqual(answer("What is 20 minus 4 minus 13?"), 3)

    def test_subtraction_then_addition(self):
        self.assertEqual(answer("What is 17 minus 6 plus 3?"), 14)

    def test_multiple_multiplication(self):
        self.assertEqual(answer("What is 2 multiplied by -2 multiplied by 3?"), -12)

    def test_addition_and_multiplication(self):
        self.assertEqual(answer("What is -3 plus 7 multiplied by -2?"), -8)

    def test_multiple_division(self):
        self.assertEqual(answer("What is -12 divided by 2 divided by -3?"), 2)

    def test_unknown_operation(self):
        with self.assertRaises(ValueError) as error:
            answer("What is 52 cubed?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "unknown operation")

    def test_non_math_question(self):
        with self.assertRaises(ValueError) as error:
            answer("Who is the President of the United States?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "unknown operation")

    def test_reject_problem_missing_an_operand(self):
        with self.assertRaises(ValueError) as error:
            answer("What is 1 plus?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "syntax error")

    def test_reject_problem_with_no_operands_or_operators(self):
        with self.assertRaises(ValueError) as error:
            answer("What is?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "syntax error")

    def test_reject_two_operations_in_a_row(self):
        with self.assertRaises(ValueError) as error:
            answer("What is 1 plus plus 2?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "syntax error")

    def test_reject_two_numbers_in_a_row(self):
        with self.assertRaises(ValueError) as error:
            answer("What is 1 plus 2 1?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "syntax error")

    def test_reject_postfix_notation(self):
        with self.assertRaises(ValueError) as error:
            answer("What is 1 2 plus?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "syntax error")

    def test_reject_prefix_notation(self):
        with self.assertRaises(ValueError) as error:
            answer("What is plus 1 2?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "syntax error")

    def test_missing_operation(self):
        with self.assertRaises(ValueError) as error:
            answer("What is 2 2 minus 3?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "syntax error")

    def test_missing_number(self):
        with self.assertRaises(ValueError) as error:
            answer("What is 7 plus multiplied by -2?")
        self.assertEqual(type(error.exception), ValueError)
        self.assertEqual(error.exception.args[0], "syntax error")


if __name__ == "__main__":
    unittest.main(verbosity=2)
