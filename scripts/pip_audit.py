#!/usr/bin/env python
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Run pip-audit against the pinned requirements, honoring time-boxed ignores.

pip-audit has no config file, so accepted-risk vulnerabilities are listed in
``requirements/pip-audit-ignore.yml`` and passed as ``--ignore-vuln`` flags.
Every ignore carries an expiry date; once it lapses this script refuses to run
until the entry is renewed (with a fresh review) or removed.

Usage::

    python scripts/pip_audit.py                      # audits requirements/base.txt
    python scripts/pip_audit.py -r requirements/development.txt -- --format json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import yaml

IGNORE_FILE = Path("requirements") / "pip-audit-ignore.yml"
DEFAULT_REQUIREMENTS = ["requirements/base.txt"]
REQUIRED_FIELDS = ("id", "package", "reason", "added", "expires")


@dataclass(frozen=True)
class Ignore:
    """A single accepted-risk vulnerability with an expiry date."""

    id: str
    package: str
    reason: str
    added: date
    expires: date
    aliases: tuple[str, ...] = field(default_factory=tuple)
    tracking: str | None = None


def _parse_date(value: Any, key: str, vuln_id: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as ex:
        raise ValueError(
            f"{vuln_id}: '{key}' must be an ISO date (YYYY-MM-DD), got {value!r}"
        ) from ex


def load_ignores(path: Path) -> list[Ignore]:
    """Load and validate the ignore entries in ``path``."""
    data = yaml.safe_load(path.read_text()) or {}
    entries = data.get("ignores") or []
    ignores: list[Ignore] = []
    for entry in entries:
        vuln_id = entry.get("id") or "<missing id>"
        missing = [key for key in REQUIRED_FIELDS if not entry.get(key)]
        if missing:
            raise ValueError(f"{vuln_id}: missing required field(s) {missing}")
        added = _parse_date(entry["added"], "added", vuln_id)
        expires = _parse_date(entry["expires"], "expires", vuln_id)
        if expires <= added:
            raise ValueError(f"{vuln_id}: 'expires' must be after 'added'")
        ignores.append(
            Ignore(
                id=str(entry["id"]),
                package=str(entry["package"]),
                reason=str(entry["reason"]).strip(),
                added=added,
                expires=expires,
                aliases=tuple(str(alias) for alias in entry.get("aliases") or ()),
                tracking=entry.get("tracking"),
            )
        )
    return ignores


def partition_ignores(
    ignores: Sequence[Ignore], today: date
) -> tuple[list[Ignore], list[Ignore]]:
    """Split ignores into (active, expired). An ignore is active through its
    expiry day."""
    active = [ignore for ignore in ignores if today <= ignore.expires]
    expired = [ignore for ignore in ignores if today > ignore.expires]
    return active, expired


def build_command(
    ignores: Sequence[Ignore],
    requirements: Sequence[str],
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Build the pip-audit command line."""
    command = ["pip-audit"]
    for requirement in requirements:
        command += ["--requirement", requirement]
    for ignore in ignores:
        command += ["--ignore-vuln", ignore.id]
    command += list(extra_args)
    return command


def main(argv: Sequence[str] | None = None, today: date | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ignore-file",
        type=Path,
        default=IGNORE_FILE,
        help=f"YAML file of time-boxed ignores (default: {IGNORE_FILE})",
    )
    parser.add_argument(
        "-r",
        "--requirement",
        action="append",
        dest="requirements",
        help="Requirements file to audit; repeatable "
        f"(default: {' '.join(DEFAULT_REQUIREMENTS)})",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra arguments passed to pip-audit (after --)",
    )
    args = parser.parse_args(argv)

    active, expired = partition_ignores(
        load_ignores(args.ignore_file), today or date.today()
    )
    if expired:
        for ignore in expired:
            print(
                f"pip-audit ignore for {ignore.id} ({ignore.package}) expired on "
                f"{ignore.expires.isoformat()}; re-review the risk and renew or "
                f"remove it in {args.ignore_file}",
                file=sys.stderr,
            )
        return 1

    command = build_command(
        active, args.requirements or DEFAULT_REQUIREMENTS, args.extra_args
    )
    # The command is built from repository files and CLI args, not untrusted input.
    return subprocess.call(command)  # noqa: S603


if __name__ == "__main__":
    sys.exit(main())
