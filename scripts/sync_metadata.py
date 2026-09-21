#!/usr/bin/env python3
"""Regenerate every file that derives a value from project.toml.

Run it after editing project.toml:

    python3 scripts/sync_metadata.py

`--check` reports drift and exits non-zero instead of writing, which is what CI
uses to prove nothing was hand-edited.

quidra.package is written for the closed key set that Quidra Core's
`src/package_manifest.cpp` accepts: `name`, `version`, `repository`,
`asset.<platform>` and `requires.<dep>`. Anything else is a hard error on the
ordinary compile path, including for already-released Core binaries, so the
metadata that has no home there - the distribution name, the display name and
the ABI requirement - stays in project.toml only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toml_subset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROJECT_TOML = ROOT / "project.toml"
PACKAGE = ROOT / "quidra.package"
BUNDLE_EXAMPLE = ROOT / "nvidia" / "BUNDLE.example.json"


def asset_url(repository: str, version: str, filename: str) -> str:
    return f"{repository}/releases/download/v{version}/{filename}"


def render_package(project: dict) -> str:
    package = project["package"]
    version = package["version"]
    repository = package["repository"]
    lines = [
        f"name = {package['import']}",
        f"version = {version}",
        f"repository = {repository}",
    ]
    for platform, filename in project["assets"].items():
        lines.append(f"asset.{platform} = {asset_url(repository, version, filename)}")
    lines.append(f"requires.quidra = {project['requires']['quidra']}")
    return "\n".join(lines) + "\n"


def render_bundle_example(project: dict) -> str:
    version = project["package"]["version"]
    text = BUNDLE_EXAMPLE.read_text(encoding="utf-8")
    return re.sub(
        r'^(\s*"dnn_version": )"\d+\.\d+\.\d+",$',
        rf'\g<1>"{version}",',
        text,
        flags=re.MULTILINE,
    )


TARGETS = (
    (PACKAGE, render_package),
    (BUNDLE_EXAMPLE, render_bundle_example),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit non-zero instead of writing",
    )
    args = parser.parse_args()

    project = toml_subset.load(PROJECT_TOML)
    stale = []
    for path, render in TARGETS:
        rendered = render(project)
        if rendered == path.read_text(encoding="utf-8"):
            continue
        if args.check:
            stale.append(path.relative_to(ROOT))
        else:
            path.write_text(rendered, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}")

    if stale:
        names = ", ".join(str(path) for path in stale)
        print(
            f"{names} disagree with project.toml.\n"
            "Edit project.toml and run: python3 scripts/sync_metadata.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
