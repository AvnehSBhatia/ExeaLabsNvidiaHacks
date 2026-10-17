"""Fetch the MuJoCo Menagerie Franka Panda (Apache-2.0) for `python -m arrival sim --arm menagerie`.

The folder is 34 MB of meshes, so it is not committed. This clones only that folder at a pinned commit into
arrival/assets/menagerie/franka_emika_panda (gitignored); the LICENSE and README travel with it.

    python scripts/fetch_menagerie.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
FOLDER = "franka_emika_panda"
DEST = Path(__file__).resolve().parents[1] / "arrival" / "assets" / "menagerie" / FOLDER


def main() -> int:
    if (DEST / "panda.xml").is_file():
        print(f"already present: {DEST}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        run = lambda *a: subprocess.run(a, check=True, cwd=repo if repo.exists() else tmp)  # noqa: E731
        run("git", "clone", "-q", "--filter=blob:none", "--sparse", "--no-checkout", REPO, str(repo))
        run("git", "sparse-checkout", "set", FOLDER)
        run("git", "checkout", "-q", COMMIT)
        src = repo / FOLDER
        DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, DEST, ignore=shutil.ignore_patterns("panda.png", "mjx_*", ".git*"))
    (DEST / "SOURCE.txt").write_text(f"{REPO}\n{FOLDER} at commit {COMMIT}\nLicense: Apache-2.0 (see LICENSE)\n")
    size = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()) / 1e6
    print(f"fetched {FOLDER} ({size:.1f} MB) into {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
