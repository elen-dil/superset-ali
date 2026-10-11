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
import os
from typing import Any
from unittest.mock import patch

import pytest
import werkzeug.security
from werkzeug.utils import safe_join

TRANSLATIONS_DIR = "/srv/superset/translations"


@pytest.mark.parametrize(
    "lang",
    [
        "..",
        "..%2F..%2Fetc",
        "%2E%2E",
        "fr%2F..%2F..",
        "fr%5C..%5C..",
        "%2Fetc%2Fpasswd",
        "messages.json",
        "fr.json",
        "CON",
        "CON:",
        "nul.txt",
        "LPT1",
        "fr%00",
    ],
)
def test_language_pack_rejects_unsafe_lang(client: Any, lang: str) -> None:
    """Traversal sequences and unsafe filenames must never reach the
    filesystem lookup."""
    with patch("superset.views.core.send_file") as mock_send_file:
        response = client.get(f"/language_pack/{lang}/")

    assert response.status_code in (400, 404)
    mock_send_file.assert_not_called()


def test_language_pack_resolves_path_with_safe_join(client: Any) -> None:
    """The requested locale is joined under the translations directory via
    ``safe_join`` so it cannot escape it."""
    with patch("superset.views.core.safe_join", wraps=safe_join) as mock_safe_join:
        client.get("/language_pack/pt_BR/")

    mock_safe_join.assert_called_once()
    base_dir, *untrusted = mock_safe_join.call_args.args
    assert os.path.normpath(base_dir).endswith(os.path.join("superset", "translations"))
    assert untrusted == ["pt_BR", "LC_MESSAGES", "messages.json"]


def test_language_pack_404_when_safe_join_rejects_path(client: Any) -> None:
    """A path rejected by ``safe_join`` is reported as missing, not served."""
    with (
        patch("superset.views.core.safe_join", return_value=None),
        patch("superset.views.core.send_file") as mock_send_file,
    ):
        response = client.get("/language_pack/fr/")

    assert response.status_code == 404
    mock_send_file.assert_not_called()


@pytest.mark.parametrize(
    "untrusted",
    [
        ("..", "etc", "passwd"),
        ("../etc/passwd",),
        ("fr/../../etc/passwd",),
        ("/etc/passwd",),
    ],
)
def test_safe_join_rejects_path_traversal(untrusted: tuple[str, ...]) -> None:
    assert safe_join(TRANSLATIONS_DIR, *untrusted) is None


@pytest.mark.parametrize(
    "untrusted",
    [
        ("CON",),
        ("nul.txt",),
        ("fr", "LPT1"),
        ("CON:",),
        ("aux:", "LC_MESSAGES", "messages.json"),
        ("fr", "NUL:.json"),
    ],
)
def test_safe_join_rejects_windows_device_names(untrusted: tuple[str, ...]) -> None:
    """Regression guard for CVE-2026-102598: on Windows, ``safe_join`` must
    reject device names, including ones with an empty ADS stream marker
    (fixed in werkzeug 3.1.9)."""
    with patch.object(werkzeug.security.os, "name", "nt"):
        assert safe_join(TRANSLATIONS_DIR, *untrusted) is None


def test_safe_join_allows_language_pack_path() -> None:
    assert (
        safe_join(TRANSLATIONS_DIR, "pt_BR", "LC_MESSAGES", "messages.json")
        == f"{TRANSLATIONS_DIR}/pt_BR/LC_MESSAGES/messages.json"
    )
