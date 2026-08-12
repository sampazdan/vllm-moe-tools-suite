from __future__ import annotations

import unittest
from runpy import run_path

call = run_path("tests/verifier_support.py")["call"]


def transpose(text: str) -> str:
    return call("transpose.py", "transpose", [text])


class TransposeTest(unittest.TestCase):
    CASES = (
        ("", ""),
        ("A1", "A\n1"),
        ("A\n1", "A1"),
        ("ABC\n123", "A1\nB2\nC3"),
        ("Single line.", "S\ni\nn\ng\nl\ne\n \nl\ni\nn\ne\n."),
        (
            "The fourth line.\nThe fifth line.",
            "TT\nhh\nee\n  \nff\noi\nuf\nrt\nth\nh \n l\nli\nin\nne\ne.\n.",
        ),
        (
            "The first line.\nThe second line.",
            "TT\nhh\nee\n  \nfs\nie\nrc\nso\ntn\n d\nl \nil\nni\nen\n.e\n .",
        ),
        (
            "The longest line.\nA long line.\nA longer line.\nA line.",
            "TAAA\nh   \nelll\n ooi\nlnnn\nogge\nn e.\nglr\nei \nsnl\n"
            "tei\n .n\nl e\ni .\nn\ne\n.",
        ),
        ("HEART\nEMBER\nABUSE\nRESIN\nTREND", "HEART\nEMBER\nABUSE\nRESIN\nTREND"),
        (
            "FRACTURE\nOUTLINED\nBLOOMING\nSEPTETTE",
            "FOBS\nRULE\nATOP\nCLOT\nTIME\nUNIT\nRENT\nEDGE",
        ),
        (
            "T\nEE\nAAA\nSSSS\nEEEEE\nRRRRRR",
            "TEASER\n EASER\n  ASER\n   SER\n    ER\n     R",
        ),
        (
            "11\n2\n3333\n444\n555555\n66666",
            "123456\n1 3456\n  3456\n  3 56\n    56\n    5",
        ),
    )

    def test_pinned_cases(self):
        for text, expected in self.CASES:
            with self.subTest(text=text):
                self.assertEqual(transpose(text), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
