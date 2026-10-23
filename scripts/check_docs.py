"""Mechanical README check: commands parse, links and paths exist, test refs are real, labels agree.

    python scripts/check_docs.py [--root DIR]

Reads README.md, docs/REFERENCE.md and ios/README.md under the root and prints one line per problem.
Exit 0 when clean, 1 when anything is wrong. Nothing is executed: commands are only parsed with the
parser that arrival.cli.main() builds.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GUIDES = ("DEPLOY", "GR00T-FINETUNE", "RELEASE-QA", "SCANNING", "07-eval", "08-access")
DOCS = ("README.md", "docs/REFERENCE.md", "ios/README.md", *(f"docs/{g}.md" for g in GUIDES))
PATH_ROOTS = ("arrival", "scripts", "tests", "docs", "configs")

FENCE = re.compile(r"^\s*```")
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
TICKS = re.compile(r"`([^`\n]+)`")
PATH = re.compile(r"^((?:%s)/[\w.\-/]*)(?:::(\w+))?$" % "|".join(PATH_ROOTS))
TASK_ID = re.compile(r"\b[A-Z]{3}-\d+[a-z]?\b")
LABEL_SENTENCE = re.compile(r"`(arrival_physically_valid|joints_are_placeholder)(?::|=)\s*(true|false)`?", re.I)


def _cli_parser():
    sys.path.insert(0, str(REPO))
    from arrival.cli import build_parser

    return build_parser()


def fenced_commands(text: str) -> list[tuple[int, str]]:
    """(line number, command) for every `python -m arrival ...` line inside a fenced block."""
    out, inside, pending = [], False, None
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            inside, pending = not inside, None
            continue
        if not inside:
            continue
        s = line.strip()
        if pending:
            start, acc = pending
            s = acc + " " + s
        else:
            start = n
        if s.endswith("\\"):
            pending = (start, s[:-1].strip())
            continue
        pending = None
        s = re.sub(r"\s+#.*$", "", s)
        if s.startswith("python -m arrival"):
            out.append((start, s))
    return out


def check_command(parser, cmd: str) -> str | None:
    """None if `cmd` parses with the real parser, else the argparse complaint."""
    try:
        argv = shlex.split(cmd)[3:]
    except ValueError as e:
        return f"unparseable command ({e})"
    if not argv:
        return "no subcommand"
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            ns = parser.parse_args(argv)
    except SystemExit as e:
        if not e.code:  # --help
            return None
        msg = err.getvalue().strip().splitlines()
        return msg[-1] if msg else "does not parse"
    return None if getattr(ns, "cmd", None) else "no subcommand"


def test_defs(root: Path) -> dict[str, set[str]]:
    defs = {}
    for f in (root / "tests").glob("test_*.py"):
        defs[f"tests/{f.name}"] = set(re.findall(r"^\s*(?:async\s+)?def\s+(test\w*)", f.read_text(encoding="utf-8"), re.M))
    return defs


PLANNED_SECTION = re.compile(r"\bto-?do\b", re.I)
SECTION_HEADING = re.compile(r"^ {0,3}(#{1,2})(?:[ \t]+|$)(.*)$")
BLOCK_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def check_paths(root: Path, doc: Path, text: str, tests: dict[str, set[str]]) -> list[str]:
    """Links must resolve everywhere. Code paths must exist too, except under a `## ... to-do` heading,
    where they name files that are planned and do not exist yet."""
    rel = doc.relative_to(root).as_posix()
    probs = []
    planned = False
    fence = None
    for n, line in enumerate(text.splitlines(), 1):
        block = BLOCK_FENCE.match(line)
        if block:
            marker, info = block.groups()
            if fence is None:
                if marker[0] != "`" or "`" not in info:
                    fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not info.strip():
                fence = None
        elif fence is None:
            heading = SECTION_HEADING.match(line)
            if heading:
                level, title = heading.groups()
                planned = len(level) == 2 and bool(PLANNED_SECTION.search(title))
        for target in LINK.findall(line):
            if re.match(r"^(\w+:|#|//)", target):
                continue
            p = target.split("#")[0]
            if p and not (doc.parent / p).exists():
                probs.append(f"{rel}:{n}: broken link {target}")
        for tok in [] if planned else TICKS.findall(line):
            m = PATH.match(tok.strip())
            if not m or re.search(r"[*<>{}]", tok):
                continue
            path, name = m.groups()
            if not (root / path).exists():
                probs.append(f"{rel}:{n}: path does not exist `{path}`")
            elif name:
                if path not in tests:
                    probs.append(f"{rel}:{n}: `{path}::{name}` is not a test file")
                elif name not in tests[path]:
                    probs.append(f"{rel}:{n}: no test named {name} in {path}")
    return probs


def check_status_lists(doc: Path, root: Path, text: str) -> list[str]:
    """A paragraph with **Done:** and **Open:** lists must not name one task ID in both."""
    probs = []
    for n, line in enumerate(text.splitlines(), 1):
        parts = re.split(r"\*\*([A-Za-z ]+?):?\*\*:?", line)
        lists = {}
        for label, body in zip(parts[1::2], parts[2::2]):
            lists[label.strip().lower()] = set(TASK_ID.findall(body))
        both = lists.get("done", set()) & lists.get("open", set())
        for tid in sorted(both):
            probs.append(f"{doc.relative_to(root).as_posix()}:{n}: {tid} is listed as both Done and Open")
    return probs


def writer_constants(root: Path) -> dict[tuple[str, str], str]:
    """(writer, label) -> 'true'/'false', read from the demo and sim writers' source."""
    srcs = {"demo": root / "arrival/export/lerobot.py", "sim": root / "arrival/sim/export.py"}
    out = {}
    for who, f in srcs.items():
        if not f.exists():
            continue
        t = f.read_text(encoding="utf-8")
        for label in ("arrival_physically_valid", "joints_are_placeholder"):
            m = re.search(r"[\"']%s[\"']:\s*(True|False)" % label, t)
            if m:
                out[(who, label)] = m.group(1).lower()
    return out


def check_labels(doc: Path, root: Path, text: str, consts: dict[tuple[str, str], str]) -> list[str]:
    """A sentence that ties a label value to `python -m arrival demo|sim` must match what that writer writes."""
    probs = []
    for n, line in enumerate(text.splitlines(), 1):
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            who = {w for w in ("demo", "sim") if f"python -m arrival {w}" in sentence}
            if len(who) != 1:
                continue
            w = who.pop()
            for label, val in LABEL_SENTENCE.findall(sentence):
                want = consts.get((w, label))
                if want and want != val.lower():
                    probs.append(f"{doc.relative_to(root).as_posix()}:{n}: says {w} writes {label}={val.lower()}, "
                                 f"but the writer sets {want}")
    return probs


def check(root: Path = REPO) -> list[str]:
    parser = _cli_parser()
    tests = test_defs(root)
    consts = writer_constants(root)
    probs: list[str] = []
    for name in DOCS:
        doc = root / name
        if not doc.exists():
            probs.append(f"{name}: file missing")
            continue
        text = doc.read_text(encoding="utf-8")
        for n, cmd in fenced_commands(text):
            why = check_command(parser, cmd)
            if why:
                probs.append(f"{name}:{n}: command does not parse ({why}): {cmd}")
        probs += check_paths(root, doc, text, tests)
        probs += check_status_lists(doc, root, text)
        probs += check_labels(doc, root, text, consts)
    return probs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO), help="repo copy to check (default: this repo)")
    a = ap.parse_args(argv)
    probs = check(Path(a.root).resolve())
    for p in probs:
        print(p)
    return 1 if probs else 0


if __name__ == "__main__":
    raise SystemExit(main())
