from __future__ import annotations

import unittest
from runpy import run_path

call = run_path("tests/verifier_support.py")["call"]


def can_chain(dominoes):
    return call("dominoes.py", "can_chain", [dominoes])


class DominoesTest(unittest.TestCase):
    def test_empty_input_empty_output(self):
        self.assert_correct_chain([], can_chain([]))

    def test_singleton_input_singleton_output(self):
        stones = [(1, 1)]
        self.assert_correct_chain(stones, can_chain(stones))

    def test_singleton_that_cannot_be_chained(self):
        stones = [(1, 2)]
        self.refute_correct_chain(stones, can_chain(stones))

    def test_three_elements(self):
        stones = [(1, 2), (3, 1), (2, 3)]
        self.assert_correct_chain(stones, can_chain(stones))

    def test_can_reverse_dominoes(self):
        stones = [(1, 2), (1, 3), (2, 3)]
        self.assert_correct_chain(stones, can_chain(stones))

    def test_cannot_be_chained(self):
        stones = [(1, 2), (4, 1), (2, 3)]
        self.refute_correct_chain(stones, can_chain(stones))

    def test_disconnected_simple(self):
        stones = [(1, 1), (2, 2)]
        self.refute_correct_chain(stones, can_chain(stones))

    def test_disconnected_double_loop(self):
        stones = [(1, 2), (2, 1), (3, 4), (4, 3)]
        self.refute_correct_chain(stones, can_chain(stones))

    def test_disconnected_single_isolated(self):
        stones = [(1, 2), (2, 3), (3, 1), (4, 4)]
        self.refute_correct_chain(stones, can_chain(stones))

    def test_need_backtrack(self):
        stones = [(1, 2), (2, 3), (3, 1), (2, 4), (2, 4)]
        self.assert_correct_chain(stones, can_chain(stones))

    def test_separate_loops(self):
        stones = [(1, 2), (2, 3), (3, 1), (1, 1), (2, 2), (3, 3)]
        self.assert_correct_chain(stones, can_chain(stones))

    def test_nine_elements(self):
        stones = [
            (1, 2),
            (5, 3),
            (3, 1),
            (1, 2),
            (2, 4),
            (1, 6),
            (2, 3),
            (3, 4),
            (5, 6),
        ]
        self.assert_correct_chain(stones, can_chain(stones))

    def test_separate_three_domino_loops(self):
        stones = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]
        self.refute_correct_chain(stones, can_chain(stones))

    def normalize_dominoes(self, dominoes):
        return sorted(tuple(sorted(domino)) for domino in dominoes)

    def assert_same_dominoes(self, input_dominoes, output_chain):
        self.assertEqual(
            self.normalize_dominoes(input_dominoes),
            self.normalize_dominoes(output_chain),
            "output must use exactly the input dominoes",
        )

    def assert_correct_chain(self, input_dominoes, output_chain):
        self.assertIsNotNone(output_chain)
        self.assert_same_dominoes(input_dominoes, output_chain)
        if not any(output_chain):
            return
        for left, right in zip(output_chain, output_chain[1:], strict=False):
            self.assertEqual(left[1], right[0])
        self.assertEqual(output_chain[0][0], output_chain[-1][1])

    def refute_correct_chain(self, _input_dominoes, output_chain):
        self.assertIsNone(output_chain)


if __name__ == "__main__":
    unittest.main(verbosity=2)
