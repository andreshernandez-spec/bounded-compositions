#!/usr/bin/env python3
"""Check the kernel headers against docs/kernel-rules.md.

Usage: check_kernel.py <kernel include directory>

Exits non-zero and prints file:line for every violation. Comments and string
literals are blanked out before scanning, keeping line numbers.
"""

import pathlib
import re
import sys

ALLOWED_SYSTEM_INCLUDES = {"cstdint", "cstddef", "climits"}
ALLOWED_LOCAL_INCLUDE = re.compile(r"^bounded_compositions/kernel/[a-z0-9_]+\.hpp$")
ALLOWED_STD = {
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "size_t", "ptrdiff_t", "intptr_t", "uintptr_t",
}
FORBIDDEN_WORDS = {
    "new": "allocation",
    "delete": "allocation",
    "malloc": "allocation",
    "free": "allocation",
    "throw": "exceptions",
    "try": "exceptions",
    "catch": "exceptions",
    "virtual": "virtual calls",
    "typeid": "RTTI",
    "dynamic_cast": "RTTI",
    "thread_local": "thread-local state",
}


def blank_comments_and_strings(text):
    """Replace comments and string/char literals by spaces, keep newlines."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
        elif c == "/" and nxt == "*":
            out.append("  ")
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            out.append("  ")
            i += 2
        elif c in "\"'":
            quote = c
            out.append(" ")
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    out.append(" ")
                    i += 1
                out.append("\n" if i < n and text[i] == "\n" else " ")
                i += 1
            out.append(" ")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def check_file(path):
    raw = path.read_text()
    problems = []
    # Includes are checked on the raw text, since their names are string-like.
    for lineno, line in enumerate(raw.splitlines(), 1):
        m = re.match(r'\s*#\s*include\s*([<"])([^>"]+)[>"]', line)
        if not m:
            continue
        kind, name = m.groups()
        if kind == "<" and name not in ALLOWED_SYSTEM_INCLUDES:
            problems.append((lineno, f"include <{name}> is outside the allowlist"))
        if kind == '"' and not ALLOWED_LOCAL_INCLUDE.match(name):
            problems.append((lineno, f'include "{name}" is not a kernel header'))
    code = blank_comments_and_strings(raw)
    for lineno, line in enumerate(code.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        for word, why in FORBIDDEN_WORDS.items():
            if re.search(rf"\b{word}\b", line):
                problems.append((lineno, f"'{word}' ({why})"))
        for m in re.finditer(r"\bstatic\b(?!_)\s*(\w*)", line):
            if m.group(1) != "constexpr":
                problems.append((lineno, "'static' other than 'static constexpr'"))
        for m in re.finditer(r"\bstd\s*::\s*(\w+)", line):
            if m.group(1) not in ALLOWED_STD:
                problems.append((lineno, f"std::{m.group(1)} (standard library)"))
    return problems


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 2
    root = pathlib.Path(sys.argv[1])
    headers = sorted(root.glob("*.hpp"))
    if not headers:
        print(f"no kernel headers found in {root}")
        return 1
    failed = False
    for path in headers:
        for lineno, msg in check_file(path):
            print(f"{path}:{lineno}: {msg}")
            failed = True
    if not failed:
        print(f"kernel rules: {len(headers)} header(s) clean")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
