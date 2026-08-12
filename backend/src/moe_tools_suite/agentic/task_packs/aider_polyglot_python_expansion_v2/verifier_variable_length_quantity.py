from __future__ import annotations

import unittest
from runpy import run_path

call = run_path("tests/verifier_support.py")["call"]


def encode(numbers: list[int]) -> list[int]:
    return call("variable_length_quantity.py", "encode", [numbers])


def decode(bytes_: list[int]) -> list[int]:
    return call("variable_length_quantity.py", "decode", [bytes_])


class VariableLengthQuantityTest(unittest.TestCase):
    ENCODE_CASES = (
        ([0x0], [0x0]),
        ([0x40], [0x40]),
        ([0x7F], [0x7F]),
        ([0x80], [0x81, 0x0]),
        ([0x2000], [0xC0, 0x0]),
        ([0x3FFF], [0xFF, 0x7F]),
        ([0x4000], [0x81, 0x80, 0x0]),
        ([0x100000], [0xC0, 0x80, 0x0]),
        ([0x1FFFFF], [0xFF, 0xFF, 0x7F]),
        ([0x200000], [0x81, 0x80, 0x80, 0x0]),
        ([0x8000000], [0xC0, 0x80, 0x80, 0x0]),
        ([0xFFFFFFF], [0xFF, 0xFF, 0xFF, 0x7F]),
        ([0x10000000], [0x81, 0x80, 0x80, 0x80, 0x0]),
        ([0xFF000000], [0x8F, 0xF8, 0x80, 0x80, 0x0]),
        ([0xFFFFFFFF], [0x8F, 0xFF, 0xFF, 0xFF, 0x7F]),
        ([0x40, 0x7F], [0x40, 0x7F]),
        ([0x4000, 0x123456], [0x81, 0x80, 0x0, 0xC8, 0xE8, 0x56]),
    )
    MIXED_VALUES = [0x2000, 0x123456, 0xFFFFFFF, 0x0, 0x3FFF, 0x4000]
    MIXED_BYTES = [
        0xC0,
        0x0,
        0xC8,
        0xE8,
        0x56,
        0xFF,
        0xFF,
        0xFF,
        0x7F,
        0x0,
        0xFF,
        0x7F,
        0x81,
        0x80,
        0x0,
    ]

    def test_encode_cases(self):
        for numbers, expected in self.ENCODE_CASES:
            with self.subTest(numbers=numbers):
                self.assertEqual(encode(numbers), expected)

    def test_encode_many_multi_byte_values(self):
        self.assertEqual(encode(self.MIXED_VALUES), self.MIXED_BYTES)

    def test_decode_cases(self):
        cases = (
            ([0x7F], [0x7F]),
            ([0xC0, 0x0], [0x2000]),
            ([0xFF, 0xFF, 0x7F], [0x1FFFFF]),
            ([0x81, 0x80, 0x80, 0x0], [0x200000]),
            ([0x8F, 0xFF, 0xFF, 0xFF, 0x7F], [0xFFFFFFFF]),
        )
        for bytes_, expected in cases:
            with self.subTest(bytes_=bytes_):
                self.assertEqual(decode(bytes_), expected)

    def test_incomplete_sequence_causes_error(self):
        for bytes_ in ([0xFF], [0x80]):
            with (
                self.subTest(bytes_=bytes_),
                self.assertRaisesRegex(
                    ValueError,
                    "incomplete sequence",
                ),
            ):
                decode(bytes_)

    def test_decode_multiple_values(self):
        self.assertEqual(decode(self.MIXED_BYTES), self.MIXED_VALUES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
