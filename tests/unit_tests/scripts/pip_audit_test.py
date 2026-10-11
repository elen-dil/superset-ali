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

from datetime import date
from pathlib import Path

import pytest

from scripts import pip_audit

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _record_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_call(command: list[str]) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(pip_audit.subprocess, "call", fake_call)
    return calls


def _write_ignores(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pip-audit-ignore.yml"
    path.write_text(body)
    return path


def test_repository_ignore_file_is_valid() -> None:
    ignores = pip_audit.load_ignores(REPOSITORY_ROOT / pip_audit.IGNORE_FILE)

    assert ignores
    for ignore in ignores:
        assert ignore.reason
        assert ignore.expires > ignore.added


def test_paramiko_sha1_rsa_is_ignored_with_expiry() -> None:
    ignores = {
        ignore.id: ignore
        for ignore in pip_audit.load_ignores(REPOSITORY_ROOT / pip_audit.IGNORE_FILE)
    }

    ignore = ignores["PYSEC-2026-2858"]
    assert ignore.package == "paramiko"
    assert {"CVE-2026-44405", "GHSA-r374-rxx8-8654"} <= set(ignore.aliases)
    assert ignore.expires == date(2026, 12, 8)


def test_active_ignores_skip_expired_entries(tmp_path: Path) -> None:
    path = _write_ignores(
        tmp_path,
        """
ignores:
  - id: PYSEC-0000-0001
    package: example
    reason: still accepted
    added: 2026-01-01
    expires: 2026-06-01
  - id: PYSEC-0000-0002
    package: example
    reason: lapsed
    added: 2026-01-01
    expires: 2026-03-01
""",
    )
    ignores = pip_audit.load_ignores(path)

    active, expired = pip_audit.partition_ignores(ignores, today=date(2026, 4, 1))

    assert [ignore.id for ignore in active] == ["PYSEC-0000-0001"]
    assert [ignore.id for ignore in expired] == ["PYSEC-0000-0002"]


def test_ignore_expires_at_end_of_expiry_day(tmp_path: Path) -> None:
    path = _write_ignores(
        tmp_path,
        """
ignores:
  - id: PYSEC-0000-0001
    package: example
    reason: accepted
    added: 2026-01-01
    expires: 2026-06-01
""",
    )
    ignores = pip_audit.load_ignores(path)

    active, _ = pip_audit.partition_ignores(ignores, today=date(2026, 6, 1))
    _, expired = pip_audit.partition_ignores(ignores, today=date(2026, 6, 2))

    assert len(active) == 1
    assert len(expired) == 1


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (
            "package: example\n    reason: r\n    added: 2026-01-01\n"
            "    expires: 2026-06-01\n",
            r"missing required field\(s\) \['id'\]",
        ),
        (
            "id: X\n    package: example\n    added: 2026-01-01\n"
            "    expires: 2026-06-01\n",
            r"missing required field\(s\) \['reason'\]",
        ),
        (
            "id: X\n    package: example\n    reason: r\n    added: 2026-01-01\n",
            r"missing required field\(s\) \['expires'\]",
        ),
        (
            "id: X\n    package: example\n    reason: r\n    added: 2026-01-01\n"
            "    expires: not-a-date\n",
            "'expires' must be an ISO date",
        ),
        (
            "id: X\n    package: example\n    reason: r\n    added: 2026-06-01\n"
            "    expires: 2026-01-01\n",
            "'expires' must be after 'added'",
        ),
    ],
    ids=[
        "missing-id",
        "missing-reason",
        "missing-expires",
        "bad-date",
        "expires-before-added",
    ],
)
def test_load_ignores_rejects_invalid_entries(
    tmp_path: Path, entry: str, match: str
) -> None:
    path = _write_ignores(tmp_path, "ignores:\n  - " + entry)

    with pytest.raises(ValueError, match=match):
        pip_audit.load_ignores(path)


def test_build_command_passes_only_active_ignores(tmp_path: Path) -> None:
    path = _write_ignores(
        tmp_path,
        """
ignores:
  - id: PYSEC-0000-0001
    package: example
    reason: accepted
    added: 2026-01-01
    expires: 2026-06-01
""",
    )
    (active,) = pip_audit.load_ignores(path)

    command = pip_audit.build_command(
        [active], ["requirements/base.txt"], ["--format", "json"]
    )

    assert command[:3] == ["pip-audit", "--requirement", "requirements/base.txt"]
    assert command.count("--ignore-vuln") == 1
    assert command[command.index("--ignore-vuln") + 1] == "PYSEC-0000-0001"
    assert command[-2:] == ["--format", "json"]


def test_main_fails_on_expired_ignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_ignores(
        tmp_path,
        """
ignores:
  - id: PYSEC-0000-0001
    package: example
    reason: lapsed
    added: 2026-01-01
    expires: 2026-03-01
""",
    )
    calls = _record_calls(monkeypatch)

    exit_code = pip_audit.main(["--ignore-file", str(path)], today=date(2026, 4, 1))

    assert exit_code == 1
    assert calls == []


def test_main_runs_pip_audit_with_active_ignores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_ignores(
        tmp_path,
        """
ignores:
  - id: PYSEC-0000-0001
    package: example
    reason: accepted
    added: 2026-01-01
    expires: 2026-06-01
""",
    )
    calls = _record_calls(monkeypatch)

    exit_code = pip_audit.main(
        ["--ignore-file", str(path), "-r", "requirements/base.txt"],
        today=date(2026, 4, 1),
    )

    assert exit_code == 0
    assert calls == [
        [
            "pip-audit",
            "--requirement",
            "requirements/base.txt",
            "--ignore-vuln",
            "PYSEC-0000-0001",
        ]
    ]
