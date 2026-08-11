from __future__ import annotations

import regex

_MATCH_TIMEOUT_SECONDS = 0.05


def bounded_regex_search(
    pattern: str,
    value: str,
    *,
    case_sensitive: bool = True,
) -> bool:
    flags = 0 if case_sensitive else regex.IGNORECASE
    try:
        return (
            regex.search(
                pattern,
                value,
                flags=flags,
                timeout=_MATCH_TIMEOUT_SECONDS,
            )
            is not None
        )
    except TimeoutError as error:
        raise ValueError("regular expression exceeded the 50 ms match limit") from error
    except regex.error as error:
        raise ValueError(f"invalid regular expression: {error}") from error


def validate_bounded_regex(pattern: str) -> None:
    try:
        regex.compile(pattern)
    except regex.error as error:
        raise ValueError(f"invalid regular expression: {error}") from error
