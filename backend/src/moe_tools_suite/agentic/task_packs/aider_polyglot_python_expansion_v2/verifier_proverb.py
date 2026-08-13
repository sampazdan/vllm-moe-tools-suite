from __future__ import annotations

import unittest
from runpy import run_path

call = run_path("tests/verifier_support.py")["call"]


def proverb(*items: str, qualifier: str | None):
    return call(
        "proverb.py",
        "proverb",
        items,
        {"qualifier": qualifier},
    )


class ProverbTest(unittest.TestCase):
    def test_zero_pieces(self):
        self.assertEqual(proverb(qualifier=None), [])

    def test_one_piece(self):
        self.assertEqual(
            proverb("nail", qualifier=None),
            ["And all for the want of a nail."],
        )

    def test_two_pieces(self):
        self.assertEqual(
            proverb("nail", "shoe", qualifier=None),
            [
                "For want of a nail the shoe was lost.",
                "And all for the want of a nail.",
            ],
        )

    def test_three_pieces(self):
        self.assertEqual(
            proverb("nail", "shoe", "horse", qualifier=None),
            [
                "For want of a nail the shoe was lost.",
                "For want of a shoe the horse was lost.",
                "And all for the want of a nail.",
            ],
        )

    def test_full_proverb(self):
        items = ("nail", "shoe", "horse", "rider", "message", "battle", "kingdom")
        self.assertEqual(
            proverb(*items, qualifier=None),
            [
                "For want of a nail the shoe was lost.",
                "For want of a shoe the horse was lost.",
                "For want of a horse the rider was lost.",
                "For want of a rider the message was lost.",
                "For want of a message the battle was lost.",
                "For want of a battle the kingdom was lost.",
                "And all for the want of a nail.",
            ],
        )

    def test_four_pieces_modernized(self):
        self.assertEqual(
            proverb("pin", "gun", "soldier", "battle", qualifier=None),
            [
                "For want of a pin the gun was lost.",
                "For want of a gun the soldier was lost.",
                "For want of a soldier the battle was lost.",
                "And all for the want of a pin.",
            ],
        )

    def test_an_optional_qualifier_can_be_added(self):
        self.assertEqual(
            proverb("nail", qualifier="horseshoe"),
            ["And all for the want of a horseshoe nail."],
        )

    def test_an_optional_qualifier_in_the_final_consequences(self):
        items = ("nail", "shoe", "horse", "rider", "message", "battle", "kingdom")
        result = proverb(*items, qualifier="horseshoe")
        self.assertEqual(result[-1], "And all for the want of a horseshoe nail.")
        self.assertEqual(len(result), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
